from sample_sm.trainers.sick_trainer import SICKTrainer
from sample_sm.trainers.msrvid_trainer import MSRVIDTrainer
from sample_sm.trainers.trecqa_trainer import TRECQATrainer
from sample_sm.trainers.wikiqa_trainer import WikiQATrainer


class SMCNNTrainerFactory(object):
    """
    Get the corresponding Trainer class for a particular dataset.
    """
    trainer_map = {
        'sick': SICKTrainer,
        'msrvid': MSRVIDTrainer,
        'trecqa': TRECQATrainer,
        'wikiqa': WikiQATrainer
    }

    @staticmethod
    def get_trainer(dataset_name, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        if dataset_name not in SMCNNTrainerFactory.trainer_map:
            raise ValueError('{} is not implemented.'.format(dataset_name))

        return SMCNNTrainerFactory.trainer_map[dataset_name](
            model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator
        )
