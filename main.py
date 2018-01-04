import numpy as np
import random
import logging

import torch
from torchtext import data

from args import get_args
from trec_dataset import TrecDataset, WikiDataset
from evaluate import evaluate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

args = get_args()
config = args

torch.manual_seed(args.seed)

if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    logger.info("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    logger.info("Warning: You have Cuda but do not use it. You are using CPU for training")
np.random.seed(args.seed)
random.seed(args.seed)

QID = data.Field(sequential=False)
AID = data.Field(sequential=False)
QUESTION = data.Field(batch_first=True)
ANSWER = data.Field(batch_first=True)
LABEL = data.Field(sequential=False)
EXTERNAL = data.Field(sequential=True, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False,
                      postprocessing=data.Pipeline(lambda arr, _, train: [float(y) for y in arr]))
if config.dataset == 'trecqa':
    train, dev, test = TrecDataset.splits(QID, QUESTION, AID, ANSWER, EXTERNAL, LABEL)
elif config.dataset == 'wikiqa':
    train, dev, test = WikiDataset.splits(QID, QUESTION, AID, ANSWER, EXTERNAL, LABEL)
else:
    print("Unsupported dataset")
    exit()

QID.build_vocab(train, dev, test)
AID.build_vocab(train, dev, test)
QUESTION.build_vocab(train, dev, test)
ANSWER.build_vocab(train, dev, test)
LABEL.build_vocab(train, dev, test)

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                           sort_key=lambda x: len(x.question), sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                         sort_key=lambda x: len(x.question), sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                          sort_key=lambda x: len(x.question), sort=False, shuffle=False)

config.target_class = len(LABEL.vocab)
config.questions_num = len(QUESTION.vocab)
config.answers_num = len(ANSWER.vocab)
print("Label dict:", LABEL.vocab.itos)

if args.cuda:
    model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    model = torch.load(args.trained_model, map_location=lambda storage, location: storage)

index2label = np.array(LABEL.vocab.itos)
index2qid = np.array(QID.vocab.itos)


def predict(test_mode, dataset_iter):
    model.eval()
    dataset_iter.init_epoch()
    instance = []
    for dev_batch_idx, dev_batch in enumerate(dataset_iter):
        qid_array = index2qid[np.transpose(dev_batch.qid.cpu().data.numpy())]
        true_label_array = index2label[np.transpose(dev_batch.label.cpu().data.numpy())]

        output = model.convModel(dev_batch)

        scores = model.linearLayer(output)
        score_array = scores.cpu().data.numpy().reshape(-1)

        # print and write the result
        for i in range(dev_batch.batch_size):
            this_qid, score, gold_label = qid_array[i], score_array[i], true_label_array[i]
            instance.append((this_qid, score, gold_label))

    dev_map, dev_mrr = evaluate(instance, test_mode, config.mode)
    print(dev_map, dev_mrr)


# Run the model on the dev set
predict('dev', dataset_iter=dev_iter)

# Run the model on the test set
predict('test', dataset_iter=test_iter)
