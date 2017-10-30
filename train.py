import time
import os
import numpy as np
import random

import torch
import torch.nn as nn
from torchtext import data

from args import get_args
from model import SmPlusPlus, PairwiseLossCriterion, PairwiseConv
from trec_dataset import TrecDataset
from sklearn.preprocessing import normalize
import operator
import heapq
from torch.autograd import Variable

from evaluate import evaluate
import sys

args = get_args()
config = args

torch.manual_seed(args.seed)


def set_vectors(field, vector_path):
    if os.path.isfile(vector_path):
        stoi, vectors, dim = torch.load(vector_path)
        field.vocab.vectors = torch.Tensor(len(field.vocab), dim)

        for i, token in enumerate(field.vocab.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                field.vocab.vectors[i] = vectors[wv_index]
            else:
                # initialize <unk> with U(-0.25, 0.25) vectors
                field.vocab.vectors[i] = torch.FloatTensor(dim).uniform_(-0.25, 0.25)
    else:
        print("Error: Need word embedding pt file")
        print("Error: Need word embedding pt file")
        exit(1)
    return field


args = get_args()
config = args

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

QID = data.Field(sequential=False)
AID = data.Field(sequential=False)
QUESTION = data.Field(batch_first=True)
ANSWER = data.Field(batch_first=True)
LABEL = data.Field(sequential=False)
EXTERNAL = data.Field(sequential=False, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False,
                      preprocessing=data.Pipeline(lambda x: x.split()),
                      postprocessing=data.Pipeline(lambda x, train: [float(y) for y in x]))

train, dev, test = TrecDataset.splits(QID, QUESTION, AID, ANSWER, EXTERNAL, LABEL)

QID.build_vocab(train, dev, test)
AID.build_vocab(train, dev, test)
QUESTION.build_vocab(train, dev, test)
ANSWER.build_vocab(train, dev, test)
# POS.build_vocab(train, dev, test)
# NEG.build_vocab(train, dev, test)
LABEL.build_vocab(train, dev, test)

QUESTION = set_vectors(QUESTION, args.vector_cache)
ANSWER = set_vectors(ANSWER, args.vector_cache)

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                           sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                         sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                          sort=False, shuffle=False)

config.target_class = len(LABEL.vocab)
config.questions_num = len(QUESTION.vocab)
config.answers_num = len(ANSWER.vocab)

print("Dataset {}    Mode {}".format(args.dataset, args.mode))
print("VOCAB num", len(QUESTION.vocab))
print("LABEL.target_class:", len(LABEL.vocab))
print("LABELS:", LABEL.vocab.itos)
print("Train instance", len(train))
print("Dev instance", len(dev))
print("Test instance", len(test))

if args.resume_snapshot:
    if args.cuda:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = SmPlusPlus(config)
    model.static_question_embed.weight.data.copy_(QUESTION.vocab.vectors)
    model.nonstatic_question_embed.weight.data.copy_(QUESTION.vocab.vectors)
    model.static_answer_embed.weight.data.copy_(ANSWER.vocab.vectors)
    model.nonstatic_answer_embed.weight.data.copy_(ANSWER.vocab.vectors)

    if args.cuda:
        model.cuda()
        print("Shift model to GPU")


pw_model = PairwiseConv(model)

parameter = filter(lambda p: p.requires_grad, pw_model.parameters())

# the SM model originally follows SGD but Adadelta is used here
optimizer = torch.optim.Adadelta(parameter, lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
pairwiseLoss = PairwiseLossCriterion()
marginRankingLoss = nn.MarginRankingLoss(margin = 1)

early_stop = False
best_dev_map = 0
best_dev_loss = 0
iterations = 0
iters_not_improved = 0
epoch = 0
start = time.time()
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.join(args.save_path, args.dataset), exist_ok=True)
print(header)

index2label = np.array(LABEL.vocab.itos)
index2qid = np.array(QID.vocab.itos)
index2question = np.array(ANSWER.vocab.itos)


def get_nearest_neg_id(pos_feature, neg_dict, distance="cosine", k=1):
    dis_list = []
    pos_feature_norm = pos_feature / np.sqrt(sum(pos_feature ** 2))
    for key in neg_dict:
        if distance == "l2":
            dis = np.sqrt(np.sum((np.array(pos_feature) - neg_dict[key]) ** 2))
        elif distance == "cosine":
            feat_norm = normalize(np.array(neg_dict[key]), norm='l2')
            dis = 1 - feat_norm.dot(pos_feature_norm)
        dis_list.append(dis)

    k = min(k, len(neg_dict))
    min_list = heapq.nsmallest(k, enumerate(dis_list), key=operator.itemgetter(1))
    min_id_list = [x[0] for x in min_list]
    return min_id_list


def get_random_neg_id(q2neg, qid_i, k=5):
    k = min(k, len(q2neg[qid_i]))
    ran = random.sample(q2neg[qid_i], k)
    return ran


def get_batch(question, answer, ext_feat):
    new_batch = data.Batch()
    new_batch.batch_size = 1
    new_batch.dataset = batch.dataset
    setattr(new_batch, "answer", torch.stack([answer]))
    setattr(new_batch, "question", torch.stack([question]))
    setattr(new_batch, "ext_feat", torch.stack([ext_feat]))
    return new_batch


q2neg = {}
question2answer = {}
best_dev_correct = 0

while True:
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev Loss: {}".format(epoch, best_dev_loss))
        break
    epoch += 1
    train_iter.init_epoch()
    for batch_idx, batch in enumerate(train_iter):
        iterations += 1
        loss_num = 0
        # model.train();
        pw_model.train();
        optimizer.zero_grad()

        new_train = {"ext_feat": [], "question": [], "answer": [], "label": []}
        features = pw_model.convModel(batch)
        # print(batch.label)
        # exit(1)
        for i in range(batch.batch_size):
            label_i = batch.label[i].cpu().data.numpy()[0]
            question_i = batch.question[i]
            # question_i = question_i[question_i!=1] # remove padding 1 <pad>
            # print(torch.max(question_i))
            # exit(1)
            answer_i = batch.answer[i]
            # answer_i = answer_i[answer_i!=1] # remove padding 1 <pad>
            ext_feat_i = batch.ext_feat[i]
            qid_i = batch.qid[i].data.cpu().numpy()[0]
            aid_i = batch.aid[i].data.cpu().numpy()[0]

            if qid_i not in question2answer:
                question2answer[qid_i] = {"question": question_i, "pos": {}, "neg": {}}

            # in the dataset, "1" for positive, "0" for negative
            # in the code, 2 for positive and 1 for negative?
            if label_i == 2:

                if aid_i not in question2answer[qid_i]["pos"]:
                    question2answer[qid_i]["pos"][aid_i] = {}

                question2answer[qid_i]["pos"][aid_i]["answer"] = answer_i
                question2answer[qid_i]["pos"][aid_i]["ext_feat"] = ext_feat_i

                # get neg samples
                if epoch == 1:
                    continue
                # random generate sample in the first training epoch
                elif epoch == 2:
                    near_list = get_random_neg_id(q2neg, qid_i, k=8)
                else:
                    near_list = get_nearest_neg_id(features[i], question2answer[qid_i]["neg"], distance="cosine", k=8)
                # print(near_list)
                new_pos = get_batch(question_i, answer_i, ext_feat_i)
                for near_id in near_list:
                    near_answer = question2answer[qid_i]["neg"][near_id]["answer"]
                    near_answer = near_answer[near_answer != 1]
                    ext_feat_neg = question2answer[qid_i]["neg"][near_id]["ext_feat"]
                    new_neg = get_batch(question_i, near_answer, ext_feat_neg)
                    # print(new_pos.answer.size()) # [1, 17]
                    # print(new_pos.batch_size)
                    # print(new_pos.question.size())  # [1, 50]
                    # print(new_pos.ext_feat.size())  # [1, 4]
                    output = pw_model([new_pos, new_neg])
                    # print("output:",output.data.numpy()[0][0], output.data.numpy()[1][0])
                    # loss = pairwiseLoss(output)
                    loss = marginRankingLoss(output[0], output[1], 1)
                    # print("loss:",loss.data.numpy()[0])
                    loss_num += loss.data.numpy()[0]
                    # print(loss_num)
                    loss.backward()
                    optimizer.step()

            elif label_i == 1:

                if aid_i not in question2answer[qid_i]["neg"]:
                    question2answer[qid_i]["neg"][aid_i] = {}

                question2answer[qid_i]["neg"][aid_i]["answer"] = answer_i
                question2answer[qid_i]["neg"][aid_i]["ext_feat"] = ext_feat_i

                if epoch == 1:
                    if qid_i not in q2neg:
                        q2neg[qid_i] = []

                    q2neg[qid_i].append(aid_i)

        # Evaluate performance on validation set
        if iterations % args.dev_every == 1 and epoch != 1:
            # switch model into evaluation mode
            # model.eval()
            pw_model.eval()
            dev_iter.init_epoch()
            n_dev_correct = 0
            n_dev_total = 0
            dev_losses = []
            instance = []
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                # qid_array = index2qid[np.transpose(dev_batch.qid.cpu().data.numpy())]
                output = pw_model.convModel(dev_batch)
                output = pw_model.linearLayer(output)
                print("============output:============")
                print(output.data.numpy()[0], "label: ",dev_batch.label.data.numpy()[0])
                output[output>0.5] = 2
                output[output<=0.5] = 1
                output = Variable(output.data.long())
                # print(output.size())
                # print(dev_batch.label.size())
                # print(dev_batch.label)
                n_dev_correct += (output.view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                n_dev_total += dev_batch.batch_size
                # dev_loss_num += loss.data[0]
                # true_label_array = index2lab
                # el[np.transpose(dev_batch.label.cpu().data.numpy())]
                # scores = model(dev_batch)
                # n_dev_correct += (torch.max(scores, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                # dev_loss = criterion(scores, dev_batch.label)
                # dev_losses.append(dev_loss.data[0])
                # index_label = np.transpose(torch.max(scores, 1)[1].view(dev_batch.label.size()).cpu().data.numpy())
                # label_array = index2label[index_label]
                # get the relevance scores
                # score_array = scores[:, 2].cpu().data.numpy()
                # for i in range(dev_batch.batch_size):
                #     this_qid, predicted_label, score, gold_label = qid_array[i], label_array[i], score_array[i], true_label_array[i]
                #     instance.append((this_qid, predicted_label, score, gold_label))

            # dev_map, dev_mrr = evaluate(instance, 'valid', config.mode)

            # print(dev_log_template.format(time.time() - start,
            #                               epoch, iterations, 1 + batch_idx, len(train_iter),
            #                               100. * (1 + batch_idx) / len(train_iter), loss.data[0],
            #                               sum(dev_losses) / len(dev_losses), train_acc, dev_map))

            # Update validation results
            print(best_dev_correct/n_dev_total)
            if best_dev_correct < n_dev_correct:
                iters_not_improved = 0
                best_dev_correct = n_dev_correct
                snapshot_path = os.path.join(args.save_path, args.dataset, args.mode + '_best_model.pt')
                torch.save(model, snapshot_path)
            else:
                iters_not_improved += 1
                if iters_not_improved >= args.patience:
                    early_stop = True
                    break

        if iterations % args.log_every == 1 and epoch != 1:
            # print progress message
            print(log_template.format(time.time() - start,
                                      epoch, iterations, 1 + batch_idx, len(train_iter),
                                      100. * (1 + batch_idx) / len(train_iter), loss_num, ' ' * 0,
                                      best_dev_correct, ' ' * 0))
            # print(log_template.format(time.time() - start,
            #                           epoch, iterations, 1 + batch_idx, len(train_iter),
            #                           100. * (1 + batch_idx) / len(train_iter), loss_num, ' ' * 8,
            #                           dev_loss_num, ' ' * 12))
            # print(log_template.format(time.time() - start,
            #                           epoch, iterations, 1 + batch_idx, len(train_iter),
            #                           100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8,
            #                           n_correct / n_total * 100, ' ' * 12))
