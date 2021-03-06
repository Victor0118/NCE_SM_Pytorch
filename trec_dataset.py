from torchtext import data
import os

class TrecDataset(data.TabularDataset):
    dirname = 'data'
    @classmethod

    def splits(cls, question_id, question_field, answer_id, answer_field, external_field, label_field,
               train='trecqa.train.tsv', validation='trecqa.dev.tsv', test='trecqa.test.tsv'):
        path = './data/'
        return super(TrecDataset, cls).splits(
            path=path, train=train, validation=validation, test=test,
            format='TSV', fields=[('qid', question_id), ('aid', answer_id), ('label', label_field), ('question', question_field),
                                  ('answer', answer_field), ('ext_feat', external_field)]
        )

class WikiDataset(data.TabularDataset):
    dirname = 'data'
    @classmethod

    def splits(cls, question_id, question_field, answer_id, answer_field, external_field, label_field,
               train='wikiqa.train.tsv', validation='wikiqa.dev.tsv', test='wikiqa.test.tsv'):
        path = './data'
        return super(WikiDataset, cls).splits(
            path=os.path.join(path), train=train, validation=validation, test=test,
            format='TSV', fields=[('qid', question_id), ('aid', answer_id), ('label', label_field), ('question', question_field),
                                  ('answer', answer_field), ('ext_feat', external_field)]
        )