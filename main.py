import argparse
import logging
import os
import pprint
import random

import numpy as np
import torch
import torch.optim as optim

from sample_sm.dataset import SMCNNDatasetFactory
from sample_sm.evaluation import SMCNNEvaluatorFactory
from sample_sm.model import SmPlusPlus
from sample_sm.train import SMCNNTrainerFactory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of Multi-Perspective CNN')
    parser.add_argument('model_outfile', help='file to save final model')
    parser.add_argument('--dataset', help='dataset to use, one of [sick, msrvid, trecqa, wikiqa]', default='sick')
    parser.add_argument('--word-vectors-dir', help='word vectors directory', default=os.path.join(os.pardir, os.pardir, 'data', 'GloVe'))
    parser.add_argument('--word-vectors-file', help='word vectors filename', default='glove.840B.300d.txt')
    parser.add_argument('--skip-training', help='will load pre-trained model', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')
    parser.add_argument('--sparse-features', action='store_true', default=False, help='use sparse features (default: false)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use: adam or sgd (default: adam)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr-reduce-factor', type=float, default=0.3, help='learning rate reduce factor after plateau (default: 0.3)')
    parser.add_argument('--patience', type=float, default=2, help='learning rate patience after seeing plateau (default: 2)')
    parser.add_argument('--momentum', type=float, default=0, help='momentum (default: 0)')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Adam epsilon (default: 1e-8)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--regularization', type=float, default=0.0001, help='Regularization for the optimizer (default: 0.0001)')
    parser.add_argument('--max-window-size', type=int, default=3, help='windows sizes will be [1,max_window_size] and infinity (default: 300)')
    parser.add_argument('--holistic-filters', type=int, default=300, help='number of holistic filters (default: 300)')
    parser.add_argument('--per-dim-filters', type=int, default=20, help='number of per-dimension filters (default: 20)')
    parser.add_argument('--hidden-units', type=int, default=150, help='number of hidden units in each of the two hidden layers (default: 150)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='use TensorBoard to visualize training (default: false)')
    parser.add_argument('--run-label', type=str, help='label to describe run')
    parser.add_argument('--dev_log_interval', type=int, default=100, help='how many batches to wait before logging validation status (default: 100)')
    parser.add_argument('--neg_num', type=int, default=8, help='number of negative samples for each question')
    parser.add_argument('--neg_sample', type=str, default="random", help='strategy of negative sampling, random or max')
    parser.add_argument('--castor_dir', help='castor directory', default=os.path.join(os.pardir))
    parser.add_argument('--utils_trecqa', help='trecqa util file', default="utils/trec_eval-9.0.5/trec_eval")

    parser.add_argument('--output_channel', type=int, default=150)
    parser.add_argument('--words_dim', type=int, default=50)
    parser.add_argument('--epoch_decay', type=int, default=15)
    parser.add_argument('--vector_cache', type=str, default='../word2vec/word2vec.trecqa.pt')
    parser.add_argument('--filter_width', type=int, default=5)
    parser.add_argument('--mode', type=str, default='static')
    parser.add_argument('--ext_feats_size', type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != -1:
        torch.cuda.manual_seed(args.seed)

    # logging setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(pprint.pformat(vars(args)))

    dataset_cls, train_loader, test_loader, dev_loader \
        = SMCNNDatasetFactory.get_dataset(args.dataset, args.word_vectors_dir, args.vector_cache, args.batch_size, args.device, pt_file=True)

    filter_widths = list(range(1, args.max_window_size + 1)) + [np.inf]

    config = args
    config.questions_num = config.answers_num = len(dataset_cls.TEXT_FIELD.vocab)
    config.target_class = len(dataset_cls.LABEL_FIELD.vocab)

    # model = SmPlusPlus(embedding, args.holistic_filters, args.per_dim_filters, filter_widths,
    #                 args.hidden_units, dataset_cls.NUM_CLASSES, args.dropout, args.sparse_features)
    model = SmPlusPlus(config)

    if args.device != -1:
        with torch.cuda.device(args.device):
            model.cuda()

    optimizer = None
    parameter = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == 'adam':
        optimizer = optim.Adam(parameter, lr=args.lr, weight_decay=args.regularization, eps=args.epsilon)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(parameter, lr=args.lr, momentum=args.momentum, weight_decay=args.regularization)
    elif args.optimizer == "adadelta":
        # the SM model originally follows SGD but Adadelta is used here
        optimizer = torch.optim.Adadelta(parameter, lr=args.lr, weight_decay=args.regularization, eps=args.epsilon)
    else:
        raise ValueError('optimizer not recognized: it should be either adam or sgd')

    train_evaluator = SMCNNEvaluatorFactory.get_evaluator(dataset_cls, model, train_loader, args.batch_size, args.device)
    test_evaluator = SMCNNEvaluatorFactory.get_evaluator(dataset_cls, model, test_loader, args.batch_size, args.device)
    dev_evaluator = SMCNNEvaluatorFactory.get_evaluator(dataset_cls, model, dev_loader, args.batch_size, args.device)

    if args.device != -1:
        margin_label = torch.autograd.Variable(torch.ones(1).cuda(device=args.device))
    else:
        margin_label = torch.autograd.Variable(torch.ones(1))

    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_interval,
        'dev_log_interval': args.dev_log_interval,
        'model_outfile': args.model_outfile,
        'lr_reduce_factor': args.lr_reduce_factor,
        'patience': args.patience,
        'tensorboard': args.tensorboard,
        'run_label': args.run_label,
        'logger': logger,
        'neg_num': args.neg_num,
        'neg_sample': args.neg_sample,
        'margin_label': margin_label
    }
    trainer = SMCNNTrainerFactory.get_trainer(args.dataset, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)

    if not args.skip_training:
        total_params = 0
        for param in model.parameters():
            size = [s for s in param.size()]
            total_params += np.prod(size)
        logger.info('Total number of parameters: %s', total_params)
        trainer.train(args.epochs)

    model = torch.load(args.model_outfile)
    saved_model_evaluator = SMCNNEvaluatorFactory.get_evaluator(dataset_cls, model, test_loader, args.batch_size, args.device)
    scores, metric_names = saved_model_evaluator.get_scores()
    logger.info('Evaluation metrics for test')
    logger.info('\t'.join([' '] + metric_names))
    logger.info('\t'.join(['test'] + list(map(str, scores))))
