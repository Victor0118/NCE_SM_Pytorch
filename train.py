import time
import os
import numpy as np
import random

import torch
import torch.nn as nn
from torchtext import data

from args import get_args
from model import SmPlusPlus, PairwiseConv
from trec_dataset import TrecDataset
import operator
import heapq
from torch.nn import functional as F

from utils.relevancy_metrics import get_map_mrr

from datasets.trecqa import TRECQA



class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].normal_(0, 0.01)
        return cls.cache[size_tup]


def train_sm():
    args = get_args()
    config = args
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print("Note: You are using GPU for training")
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print("You have Cuda but you're using CPU for training.")
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_root = os.path.join(os.pardir, 'data', 'TrecQA/')
    train_iter, dev_iter, test_iter = TRECQA.iters(dataset_root, args.vector_cache, args.wordvec_dir, batch_size=args.batch_size,
                                                   pt_file=True, device=args.gpu, unk_init=UnknownWordVecCache.unk) #

    index2text = np.array(TRECQA.TEXT_FIELD.vocab.itos)


    config.target_class = 2
    config.questions_num = TRECQA.VOCAB_NUM
    config.answers_num = TRECQA.VOCAB_NUM

    print("index2text:", index2text)
    print("Dataset {}    Mode {}".format(args.dataset, args.mode))
    print("VOCAB num", TRECQA.VOCAB_NUM)
    print("LABEL.target_class:", config.target_class)

    if args.resume_snapshot:
        if args.cuda:
            pw_model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            pw_model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
    else:
        model = SmPlusPlus(config)
        model.static_question_embed.weight.data.copy_(TRECQA.TEXT_FIELD.vocab.vectors)
        model.nonstatic_question_embed.weight.data.copy_(TRECQA.TEXT_FIELD.vocab.vectors)
        model.static_answer_embed.weight.data.copy_(TRECQA.TEXT_FIELD.vocab.vectors)
        model.nonstatic_answer_embed.weight.data.copy_(TRECQA.TEXT_FIELD.vocab.vectors)

        if args.cuda:
            model.cuda()
            print("Shift model to GPU")

        pw_model = PairwiseConv(model)

    parameter = filter(lambda p: p.requires_grad, pw_model.parameters())

    # the SM model originally follows SGD but Adadelta is used here
    optimizer = torch.optim.Adadelta(parameter, lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    # A good lr is required to use Adam
    # optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    # optimizer = torch.optim.SGD(parameter, lr=0.001, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = torch.optim.RMSprop(parameter, lr=0.0001, weight_decay=args.weight_decay)

    marginRankingLoss = nn.MarginRankingLoss(margin = 1, size_average = True)

    early_stop = False
    iterations = 0
    iters_not_improved = 0
    epoch = 0
    q2neg = {} # a dict from qid to a list of aid
    question2answer = {} # a dict from qid to the information of both pos and neg answers
    best_dev_map = 0
    best_dev_mrr = 0
    false_samples = {}

    start = time.time()
    header = '  Time Epoch Iteration Progress    (%Epoch)  Average_Loss Train_Accuracy Dev/MAP  Dev/MRR'
    dev_log_template = ' '.join(
        '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>11.6f},{:>11.6f},{:12.6f},{:8.4f}'.split(','))
    log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>11.6f},{:>11.6f},'.split(','))
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.dataset), exist_ok=True)
    print(header)



    # get the nearest negative samples to the positive sample by computing the feature difference
    def get_nearest_neg_id(pos_feature, neg_dict, distance="cosine", k=1):
        dis_list = []
        pos_feature = pos_feature.data.cpu().numpy()
        pos_feature_norm = pos_feature / np.sqrt(sum(pos_feature ** 2))
        neg_list = []
        for key in neg_dict:
            if distance == "l2":
                dis = np.sqrt(np.sum((np.array(pos_feature) - neg_dict[key]["feature"]) ** 2))
            elif distance == "cosine":
                neg_feature = np.array(neg_dict[key]["feature"])
                feat_norm = neg_feature / np.sqrt(sum(neg_feature ** 2))
                dis = 1 - feat_norm.dot(pos_feature_norm)
            dis_list.append(dis)
            neg_list.append(key)

        k = min(k, len(neg_dict))
        min_list = heapq.nsmallest(k, enumerate(dis_list), key=operator.itemgetter(1))
        min_id_list = [neg_list[x[0]] for x in min_list]
        return min_id_list

    # get the negative samples randomly
    def get_random_neg_id(q2neg, qid_i, k=5):
        # question 1734 has no neg answer
        if qid_i not in q2neg:
            return []
        k = min(k, len(q2neg[qid_i]))
        ran = random.sample(q2neg[qid_i], k)
        return ran

    # pack the lists of question/answer/ext_feat into a torchtext batch
    def get_batch(question, answer, ext_feat, size):
        new_batch = data.Batch()
        new_batch.batch_size = size
        new_batch.dataset = batch.dataset
        setattr(new_batch, "sentence_2", torch.stack(answer))
        setattr(new_batch, "sentence_1", torch.stack(question))
        setattr(new_batch, "ext_feats", torch.stack(ext_feat))
        return new_batch

    while True:
        if early_stop:
            print("Early Stopping. Epoch: {}, Best Dev Loss: {}".format(epoch, best_dev_loss))
            break
        epoch += 1
        train_iter.init_epoch()
        '''
        batch size issue: padding is a choice (add or delete them in both train and test)
                        associated with the batch size. Currently, it seems to affect the result a lot.
        '''
        acc = 0
        tot = 0
        for batch_idx, batch in enumerate(iter(train_iter)):
            if epoch != 1:
                iterations += 1
            loss_num = 0
            pw_model.train()

            new_train = {"ext_feat": [], "question": [], "answer": [], "label": []}
            features = pw_model.convModel(batch)
            new_train_pos = {"answer": [], "question": [], "ext_feat": []}
            new_train_neg = {"answer": [], "question": [], "ext_feat": []}
            max_len_q = 0
            max_len_a = 0

            batch_near_list = []
            batch_qid = []
            batch_aid = []
            # print(index2text[batch.sentence_1.cpu().data.numpy()])

            for i in range(batch.batch_size):
                label_i = batch.label[i].cpu().data.numpy()[0]
                question_i = batch.sentence_1[i]

                # question_i = question_i[question_i!=1] # remove padding 1 <pad>
                answer_i = batch.sentence_2[i]
                # answer_i = answer_i[answer_i!=1] # remove padding 1 <pad>
                ext_feat_i = batch.ext_feats[i]
                qid_i = batch.id[i].data.cpu().numpy()[0]
                aid_i = batch.aid[i].data.cpu().numpy()[0]

                if qid_i not in question2answer:
                    question2answer[qid_i] = {"question": question_i, "pos": {}, "neg": {}}
                if label_i == 1:

                    if aid_i not in question2answer[qid_i]["pos"]:
                        question2answer[qid_i]["pos"][aid_i] = {}

                    question2answer[qid_i]["pos"][aid_i]["answer"] = answer_i
                    question2answer[qid_i]["pos"][aid_i]["ext_feat"] = ext_feat_i

                    # get neg samples in the first epoch but do not train
                    if epoch == 1:
                        continue
                    # random generate sample in the first training epoch
                    elif epoch == 2 or args.neg_sample == "random":
                        near_list = get_random_neg_id(q2neg, qid_i, k=args.neg_num)
                    else:
                        debug_qid = qid_i
                        near_list = get_nearest_neg_id(features[i], question2answer[qid_i]["neg"], distance="cosine", k=args.neg_num)

                    batch_near_list.extend(near_list)

                    neg_size = len(near_list)
                    if neg_size != 0:
                        answer_i = answer_i[answer_i != 1] # remove padding 1 <pad>
                        question_i = question_i[question_i != 1] # remove padding 1 <pad>
                        for near_id in near_list:
                            batch_qid.append(qid_i)
                            batch_aid.append(aid_i)

                            new_train_pos["answer"].append(answer_i)
                            new_train_pos["question"].append(question_i)
                            new_train_pos["ext_feat"].append(ext_feat_i)

                            near_answer = question2answer[qid_i]["neg"][near_id]["answer"]
                            if question_i.size()[0] > max_len_q:
                                max_len_q = question_i.size()[0]
                            if near_answer.size()[0] > max_len_a:
                                max_len_a = near_answer.size()[0]
                            if answer_i.size()[0] > max_len_a:
                                max_len_a = answer_i.size()[0]

                            ext_feat_neg = question2answer[qid_i]["neg"][near_id]["ext_feat"]
                            new_train_neg["answer"].append(near_answer)
                            new_train_neg["question"].append(question_i)
                            new_train_neg["ext_feat"].append(ext_feat_neg)

                elif label_i == 0:

                    if aid_i not in question2answer[qid_i]["neg"]:
                        answer_i = answer_i[answer_i != 1]
                        question2answer[qid_i]["neg"][aid_i] = {"answer": answer_i}

                    question2answer[qid_i]["neg"][aid_i]["feature"] = features[i].data.cpu().numpy()
                    question2answer[qid_i]["neg"][aid_i]["ext_feat"] = ext_feat_i

                    if epoch == 1:
                        if qid_i not in q2neg:
                            q2neg[qid_i] = []

                        q2neg[qid_i].append(aid_i)

            # pack the selected pos and neg samples into the torchtext batch and train
            if epoch != 1:
                true_batch_size = len(new_train_neg["answer"])
                if true_batch_size != 0:
                    for j in range(true_batch_size):
                        new_train_neg["answer"][j] = F.pad(new_train_neg["answer"][j],
                                                           (0, max_len_a - new_train_neg["answer"][j].size()[0]), value=1)
                        new_train_pos["answer"][j] = F.pad(new_train_pos["answer"][j],
                                                           (0, max_len_a - new_train_pos["answer"][j].size()[0]), value=1)
                        new_train_pos["question"][j] = F.pad(new_train_pos["question"][j],
                                                           (0, max_len_q - new_train_pos["question"][j].size()[0]), value=1)
                        new_train_neg["question"][j] = F.pad(new_train_neg["question"][j],
                                                           (0, max_len_q - new_train_neg["question"][j].size()[0]), value=1)

                    pos_batch = get_batch(new_train_pos["question"], new_train_pos["answer"], new_train_pos["ext_feat"],
                                          true_batch_size)
                    neg_batch = get_batch(new_train_neg["question"], new_train_neg["answer"], new_train_neg["ext_feat"],
                                          true_batch_size)

                    optimizer.zero_grad()
                    output = pw_model([pos_batch, neg_batch])

                    '''
                    debug code
                    '''
                    cmp = output[:, 0] <= output[:, 1]
                    cmp = np.array(cmp.data.cpu().numpy(), dtype=bool)
                    batch_near_list = np.array(batch_near_list)
                    batch_aid = np.array(batch_aid)
                    batch_qid = np.array(batch_qid)
                    qlist = batch_qid[cmp]
                    alist = batch_aid[cmp]
                    nlist = batch_near_list[cmp]
                    for k in range(len(batch_qid[cmp])):
                        pair = (qlist[k], alist[k], nlist[k])
                        if pair in false_samples:
                            false_samples[pair] += 1
                        else:
                            false_samples[pair] = 1

                    cmp = output[:, 0] > output[:, 1]
                    acc += sum(cmp.data.cpu().numpy())
                    tot += true_batch_size


                    loss = marginRankingLoss(output[:, 0], output[:, 1], torch.autograd.Variable(torch.ones(1)))
                    loss_num = loss.data.numpy()[0]
                    loss.backward()
                    optimizer.step()

            # Evaluate performance on validation set
            if iterations % args.dev_every == 1 and epoch != 1:
                # switch model into evaluation mode
                pw_model.eval()
                dev_iter.init_epoch()
                n_dev_correct = 0
                n_dev_total = 0
                dev_losses = []
                qids = []
                predictions = []
                labels = []

                '''
                debug code
                '''
                if 'false_samples' in locals():
                    print("false_samples:", len(false_samples), end=' ')
                    false_samples_sorted = sorted(false_samples.items(), key=lambda t: t[1], reverse=True)
                    for k in range(min(4,len(false_samples))):
                        print(false_samples_sorted[k][0], false_samples_sorted[k][1], end=" ")
                    print()

                    # new_train_pos = {"answer": [], "question": [], "ext_feat": []}
                    # new_train_neg = {"answer": [], "question": [], "ext_feat": []}
                    # max_len_a = 0
                    # max_len_q = 0
                    # for s in false_samples:
                    #     qid = s[0]
                    #     aid = s[1]
                    #     nid = s[2]
                    #     q = question2answer[qid]["question"]
                    #     pa = question2answer[qid]["pos"][aid]["answer"]
                    #     na = question2answer[qid]["neg"][nid]["answer"]
                    #     if q.size()[0] > max_len_q:
                    #         max_len_q = q.size()[0]
                    #     if pa.size()[0] > max_len_a:
                    #         max_len_a = pa.size()[0]
                    #     if na.size()[0] > max_len_a:
                    #         max_len_a = na.size()[0]
                    #
                    #     new_train_pos["answer"].append(pa)
                    #     new_train_pos["question"].append(q)
                    #     new_train_pos["ext_feat"].append(question2answer[qid]["pos"][aid]["ext_feat"])
                    #     new_train_neg["answer"].append(na)
                    #     new_train_neg["question"].append(q)
                    #     new_train_neg["ext_feat"].append(question2answer[qid]["neg"][nid]["ext_feat"])
                    #
                    # for j in range(len(false_samples)):
                    #     new_train_neg["answer"][j] = F.pad(new_train_neg["answer"][j],
                    #                                        (0, max_len_a - new_train_neg["answer"][j].size()[0]), value=1)
                    #     new_train_pos["answer"][j] = F.pad(new_train_pos["answer"][j],
                    #                                        (0, max_len_a - new_train_pos["answer"][j].size()[0]), value=1)
                    #     new_train_pos["question"][j] = F.pad(new_train_pos["question"][j],
                    #                                        (0, max_len_q - new_train_pos["question"][j].size()[0]), value=1)
                    #     new_train_neg["question"][j] = F.pad(new_train_neg["question"][j],
                    #                                        (0, max_len_q - new_train_neg["question"][j].size()[0]), value=1)
                    #
                    # pos_batch = get_batch(new_train_pos["question"], new_train_pos["answer"], new_train_pos["ext_feat"],
                    #                       true_batch_size)
                    # neg_batch = get_batch(new_train_neg["question"], new_train_neg["answer"], new_train_neg["ext_feat"],
                    #                       true_batch_size)
                    #
                    # optimizer.zero_grad()
                    # output = pw_model([pos_batch, neg_batch])
                    # loss = marginRankingLoss(output[:, 0], output[:, 1], torch.autograd.Variable(torch.ones(1)))
                    # loss.backward()
                    # optimizer.step()
                    # false_samples = {}


                    # if epoch >= 3:
                    #     print("qid:", debug_qid, " near_list:", [x for x in near_list])

                # print("============output:============")
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    '''
                    # dev singlely or in a batch? -> in a batch
                    but dev singlely is equal to dev_size = 1
                    '''
                    scores = pw_model.convModel(dev_batch)
                    scores = pw_model.linearLayer(scores)
                    qid_array = np.transpose(dev_batch.id.cpu().data.numpy())
                    score_array = scores.cpu().data.numpy().reshape(-1)
                    true_label_array = np.transpose(dev_batch.label.cpu().data.numpy())

                    qids.extend(qid_array.tolist())
                    predictions.extend(score_array.tolist())
                    labels.extend(true_label_array.tolist())

                dev_map, dev_mrr = get_map_mrr(qids, predictions, labels)
                print(dev_log_template.format(time.time() - start,
                                              epoch, iterations, 1 + batch_idx, len(train_iter),
                                              100. * (1 + batch_idx) / len(train_iter),
                                              loss_num, acc / tot, dev_map, dev_mrr))
                if best_dev_mrr < dev_mrr:
                    snapshot_path = os.path.join(args.save_path, args.dataset, args.mode + '_best_model.pt')
                    torch.save(pw_model, snapshot_path)
                    iters_not_improved = 0
                    best_dev_mrr = dev_mrr
                else:
                    iters_not_improved += 1
                    if iters_not_improved >= args.patience:
                        early_stop = True
                        break

            if iterations % args.log_every == 1 and epoch != 1:
                # print progress message
                print(log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter),
                                         loss_num,  acc / tot))
                acc = 0
                tot = 0

if __name__ == '__main__':
    train_sm()