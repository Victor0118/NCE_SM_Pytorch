import os

import torch

from datasets.trecqa import TRECQA
from datasets.wikiqa import WikiQA


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
            # cls.cache[size_tup].normal_(0, 0.01)
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]


class SMCNNDatasetFactory(object):
    """
    Get the corresponding Dataset class for a particular dataset.
    """
    @staticmethod
    def get_dataset(dataset_name, word_vectors_dir, word_vectors_file, batch_size, device, pt_file=True):
        if dataset_name == 'trecqa':
            if not os.path.exists('../utils/trec_eval-9.0.5/trec_eval'):
                raise FileNotFoundError('TrecQA requires the trec_eval tool to run. Please run get_trec_eval.sh inside Castor/utils (as working directory) before continuing.')
            dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'TrecQA/')
            train_loader, dev_loader, test_loader = TRECQA.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk, pt_file=pt_file)
            return TRECQA, train_loader, test_loader, dev_loader
        elif dataset_name == 'wikiqa':
            if not os.path.exists('../utils/trec_eval-9.0.5/trec_eval'):
                raise FileNotFoundError('TrecQA requires the trec_eval tool to run. Please run get_trec_eval.sh inside Castor/utils (as working directory) before continuing.')
            dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'WikiQA/')
            train_loader, dev_loader, test_loader = WikiQA.iters(dataset_root, word_vectors_file, word_vectors_dir, batch_size, device=device, unk_init=UnknownWordVecCache.unk, pt_file=pt_file)
            return WikiQA, train_loader, test_loader, dev_loader
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_name))

