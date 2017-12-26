import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

from misc.Decoder import DecoderModel
from misc.lstm import AdaptiveAttentionLSTM, AdaptiveAttention


class AdaAttCore(nn.Module):
    def __init__(self, opt):
        super(AdaAttCore, self).__init__()
        self.lstm = AdaptiveAttentionLSTM(opt)
        self.attention = AdaptiveAttention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, step):
        top_h, sentinel, state = self.lstm(xt, fc_feats, state, step)
        atten_out = self.attention(top_h, sentinel, att_feats, p_att_feats)
        return atten_out, state


class AdaptiveAttentionModel(DecoderModel):

    def __init__(self, opt):
        super(AdaptiveAttentionModel, self).__init__(opt)
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.drop_feat_im = opt.drop_feat_im

        # Word embedding:
        self.embed_1 = nn.Embedding(self.vocab_size, self.input_encoding_size)
        self.embed = nn.Sequential(self.embed_1,
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_x_lm))
        # image embedding
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_feat_im))
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                       nn.ReLU(),
                                       nn.Dropout(self.drop_feat_im))
        # The core RNN
        self.core = AdaAttCore(opt)

        self.logit = nn.Linear(self.rnn_size, self.vocab_size)
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.init_weights()
        opt.logger.warn('Show Attend & Tell : %s' % str(self._modules))

    def init_weights(self):
        """
        Control intial model weights
        """
        initrange = 0.1
        if self.W is not None:
            self.embed_1.weight = nn.Parameter(torch.from_numpy(self.W),
                                               requires_grad=self.require_W_grad)
        else:
            self.embed_1.weight.data.uniform_(-initrange, initrange)
        # self.logit.bias.data.fill_(0)
        # self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        """
        Construct the first hidden and memory cells
        """
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def forward(self, fc_feats, att_feats, seq):
        """
        Given fc & att features of the image & ground truth sequence
        get log p(y_t|h_t), t=1..T
        """
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        # embed fc and att feats
        # print('Fc feats:', fc_feats.size())
        # print('Att feats:', att_feats.size())
        fc_feats = self.fc_embed(fc_feats)
        # print("Embedded fc feats:", fc_feats.size())
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        # print('Embedded att feats:', _att_feats.size())
        # print('Resizing into:', att_feats.size()[:-1], self.rnn_size)
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))
        # print('Embedded att feats:', att_feats.size())
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        # print('Projecting into p_att_feats:', p_att_feats.size())
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))
        # print('Reformatted p_att:', p_att_feats.size())
        for i in range(seq.size(1)):
            if self.training and i >= 2 and self.ss_prob > 0.0:
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[-1].data)
                    it.index_copy_(0,
                                   sample_ind,
                                   torch.multinomial(prob_prev,
                                                     1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 2 and seq[:, i].data.sum() == 0:
                # print('Breaking at :', i)
                break
            xt = self.embed(it)
            # print('token w:', xt.size())
            output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, i)
            output = F.log_softmax(self.logit(output))
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous()

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state, step):
        """
        Required by beam_search (cf DecoderModel)
        """
        # 'it' is Variable contraining a word index
        xt = self.embed(it)
        output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state, step)
        logprobs = F.log_softmax(self.logit(output))

        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.contiguous().view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))

                output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state, t)
                logprobs = F.log_softmax(self.logit(output))

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.contiguous().view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step

                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, t)
            logprobs = F.log_softmax(self.logit(output))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


