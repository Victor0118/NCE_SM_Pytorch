import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
import math
import copy

# class PairwiseLossCriterion(Function):
#     """docstring for PairwiseLossCriterion"""
#     # def __init__(self):
#     #     super(PairwiseLossCriterion, self).__init__()
#     #     # self.arg = arg
#     #     self.gradInput = torch.zeros(2)
#     #     self.input = None
#     @classmethod
#     def forward(ctx, input, weight = None, bias = None):
#         ctx.save_for_backward(input)
#         return torch.max(torch.zeros(1), 1 - (input[0] - input[1]))/2
#
#     @classmethod
#     def backward(ctx, gradInput):
#         input = ctx.saved_variables
#         diff = 1 - (input[0] - input[1])
#         print(diff)
#         gradInput = torch.zeros(2)
#         if diff > 0:
#             gradInput[0] = -0.5
#             gradInput[1] = 0.5
#         return gradInput

class PairwiseLossCriterion(Function):
    """docstring for PairwiseLossCriterion"""
    def __init__(self):
        super(PairwiseLossCriterion, self).__init__()
        self.input = None

    def forward(self, input, weight = None, bias = None):
        self.input = input
        # print(self.input)
        return torch.max(torch.zeros(1), 1 - (input[0] - input[1]))/2

    def backward(self, grad_output):
        diff = 1 - (self.input[0] - self.input[1])

        # print("diff:",diff) #, "grad_output:",grad_output, "type of grad_output:", type(grad_output)
        grad_output = torch.zeros(2)
        if diff.numpy()[0] > 0:
            grad_output[0] = -0.5
            grad_output[1] = 0.5
        # else:
        #     grad_output[0] = 0
        #     grad_output[1] = 0

        print(grad_output)
        return grad_output.view(2, 1)

class PairwiseConv(nn.Module):
    """docstring for PairwiseConv"""
    def __init__(self, model):
        super(PairwiseConv, self).__init__()
        # self.linearLayer = self:LinearLayer() ??
        # self.convModel = SmPlusPlus(config)
        self.convModel = model
        self.dropout = nn.Dropout(0.5)
        self.linearLayer = nn.Linear(model.n_hidden, 1)
        self.posModel = self.convModel
        # share or copy ??
        # https://discuss.pytorch.org/t/copying-nn-modules-without-shared-memory/113
        # self.negModel = copy.deepcopy(self.posModel)
        self.negModel = self.convModel

    def forward(self, input):
        pos = self.convModel(input[0])
        neg = self.convModel(input[1])
        pos = self.dropout(pos)
        neg = self.dropout(neg)
        pos = self.linearLayer(pos)
        neg = self.linearLayer(neg)
        combine = torch.cat([pos, neg], 0)
        return combine

class SmPlusPlus(nn.Module):
    def __init__(self, config):
        super(SmPlusPlus, self).__init__()
        output_channel = config.output_channel
        questions_num = config.questions_num
        answers_num = config.answers_num
        words_dim = config.words_dim
        filter_width = config.filter_width
        self.mode = config.mode

        n_classes = config.target_class
        ext_feats_size = 4

        if self.mode == 'multichannel':
            input_channel = 2
        else:
            input_channel = 1

        self.question_embed = nn.Embedding(questions_num, words_dim)
        self.answer_embed = nn.Embedding(answers_num, words_dim)
        self.static_question_embed = nn.Embedding(questions_num, words_dim)
        self.nonstatic_question_embed = nn.Embedding(questions_num, words_dim)
        self.static_answer_embed = nn.Embedding(answers_num, words_dim)
        self.nonstatic_answer_embed = nn.Embedding(answers_num, words_dim)
        self.static_question_embed.weight.requires_grad = False
        self.static_answer_embed.weight.requires_grad = False

        self.conv_q = nn.Conv2d(input_channel, output_channel, (filter_width, words_dim), padding=(filter_width - 1, 0))
        self.conv_a = nn.Conv2d(input_channel, output_channel, (filter_width, words_dim), padding=(filter_width - 1, 0))
        self.conv_qa = nn.Conv2d(2, output_channel, (filter_width, words_dim), padding=(7, 0))

        self.dropout = nn.Dropout(config.dropout)
        self.n_hidden = 3 * output_channel + ext_feats_size

        self.combined_feature_vector = nn.Linear(self.n_hidden, self.n_hidden)
        self.hidden = nn.Linear(self.n_hidden, n_classes)

    def forward(self, x):
        x_question = x.question
        x_answer = x.answer
        x_ext = x.ext_feat

        if self.mode == 'rand':
            question = self.question_embed(x_question).unsqueeze(1)
            answer = self.answer_embed(x_answer).unsqueeze(1) # (batch, sent_len, embed_dim)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        # actual SM model mode (Severyn & Moschitti, 2015)
        elif self.mode == 'static':
            question = self.static_question_embed(x_question).unsqueeze(1)
            answer = self.static_answer_embed(x_answer).unsqueeze(1) # (batch, sent_len, embed_dim)
            padding = Variable(torch.zeros(answer.size(0), answer.size(1), answer.size(2) - question.size(2), answer.size(3)))
            padded_question = torch.cat([question, padding], 2)
            qa_combined = torch.stack([padded_question, answer], dim=1).squeeze(2)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3), F.tanh(self.conv_qa(qa_combined)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        elif self.mode == 'non-static':
            question = self.nonstatic_question_embed(x_question).unsqueeze(1)
            answer = self.nonstatic_answer_embed(x_answer).unsqueeze(1) # (batch, sent_len, embed_dim)
            padding = Variable(torch.zeros(answer.size(0), answer.size(1), answer.size(2) - question.size(2), answer.size(3)))
            padded_question = torch.cat([question, padding], 2)
            qa_combined = torch.stack([padded_question, answer], dim=1).squeeze(2)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3), F.tanh(self.conv_qa(qa_combined)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        elif self.mode == 'multichannel':
            question_static = self.static_question_embed(x_question)
            answer_static = self.static_answer_embed(x_answer)
            question_nonstatic = self.nonstatic_question_embed(x_question)
            answer_nonstatic = self.nonstatic_answer_embed(x_answer)
            question = torch.stack([question_static, question_nonstatic], dim=1)
            answer = torch.stack([answer_static, answer_nonstatic], dim=1)

            question_comb_static = self.static_question_embed(x_question).unsqueeze(1)
            answer_comb_static = self.static_answer_embed(x_answer).unsqueeze(1)  # (batch, sent_len, embed_dim)

            padded_question_static = question_comb_static
            padded_answer_static = answer_comb_static
            padding_num = answer_comb_static.size(2) - question_comb_static.size(2)
            if padding_num > 0:
                padding = Variable(torch.zeros(answer_comb_static.size(0), answer_comb_static.size(1),
                                               padding_num, answer_comb_static.size(3)))

                padded_question_static = torch.cat([question_comb_static, padding], 2)
            elif padding_num < 0:
                padding = Variable(torch.zeros(answer_comb_static.size(0), answer_comb_static.size(1),
                                               -padding_num, answer_comb_static.size(3)))

                padded_answer_static = torch.cat([answer_comb_static, padding], 2)

            qa_combined_static = torch.stack([padded_question_static, padded_answer_static], dim=1).squeeze(2)

            # question_comb = self.nonstatic_question_embed(x_question).unsqueeze(1)
            # answer_comb = self.nonstatic_answer_embed(x_answer).unsqueeze(1)  # (batch, sent_len, embed_dim)
            # padding = Variable(torch.zeros(answer_comb.size(0), answer_comb.size(1), answer_comb.size(2)
            #                                - question_comb.size(2), answer_comb.size(3)))
            # padded_question = torch.cat([question_comb, padding], 2)
            # qa_combined_nonstatic = torch.stack([padded_question, answer_comb], dim=1).squeeze(2)
            # print(qa_combined_static.size(), qa_combined_nonstatic.static())
            # qa_multichannel = torch.stack([qa_combined_static, qa_combined_nonstatic], dim=1).squeeze(2)
            # print(qa_multichannel.size())

            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3),
                 F.tanh(self.conv_qa(qa_combined_static)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        else:
            print("Unsupported Mode")
            exit()

        # append external features and feed to fc
        x.append(x_ext)
        x = torch.cat(x, 1)
        x = F.tanh(self.combined_feature_vector(x))

        '''
        whether and where I add the dropout layer is a question
        '''
        # x = self.dropout(x)
        # x = self.hidden(x)
        return x