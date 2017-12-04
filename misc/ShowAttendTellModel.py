from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from misc.Decoder import DecoderModel
from misc.lstm import MultiLayerLSTMCells, LSTMAttnDecoder
_BOS = 0
_EOS = 0

class ShowAttendTellModel(DecoderModel):
    def __init__(self, opt):
        super(ShowAttendTellModel, self).__init__(opt)
        # Encoder
        self.img_embed = nn.Conv2d(self.att_feat_size, self.rnn_size, kernel_size=1)
        # Word embedding:
        self.embed = nn.Embedding(self.vocab_size, self.input_encoding_size)
        # self.drop_x_lm = nn.Dropout(p=opt.drop_x_lm)
        # intialize LSTM states:
        self.init_h = nn.Linear(self.rnn_size, self.rnn_size)
        self.init_c = nn.Linear(self.rnn_size, self.rnn_size)
        self.decoder_lstm = MultiLayerLSTMCells(self.rnn_size + self.input_encoding_size,
                                                self.rnn_size,
                                                self.num_layers)

        # Attention
        self.attend = nn.Linear(self.rnn_size,
                                self.rnn_size)
        self.project = nn.Linear(self.input_encoding_size + 2 * self.rnn_size,
                                 self.input_encoding_size)
        self.decoder = LSTMAttnDecoder(self.embed, # what about drop_x_lm FIXME
                                       self.decoder_lstm,
                                       self.attend,
                                       self.project)

        # self.Softmax = nn.Softmax()
        # self.logit = nn.Linear(self.rnn_size, self.vocab_size)
        self.init_weights()
        opt.logger.warn('Show, attend & Tell : %s' % str(self._modules))


    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

        self.init_h.bias.data.fill_(0)
        self.init_h.weight.data.uniform_(-initrange, initrange)
        self.init_c.bias.data.fill_(0)
        self.init_c.weight.data.uniform_(-initrange, initrange)

        self.attend.bias.data.fill_(0)
        self.attend.weight.data.uniform_(-initrange, initrange)
        self.project.bias.data.fill_(0)
        self.project.weight.data.uniform_(-initrange, initrange)

        # self.logit.bias.data.fill_(0)
        # self.logit.weight.data.uniform_(-initrange, initrange)

    def encode(self, att_feats):
        att_feats = self.img_embed(att_feats)
        avg_attn = att_feats.mean(dim=3,
                                  keepdim=False).mean(dim=2, keepdim=False)
        N = att_feats.size(0)
        L = self.num_layers
        D = self.rnn_size
        init_states = (self.init_h(avg_attn).unsqueeze(0).expand(L, N, D),
                       self.init_c(avg_attn).unsqueeze(0).expand(L, N, D))
        return att_feats, init_states

    def forward(self, fc_feats, att_feats, seq):
        att_feats, init_states = self.encode(att_feats)
        logit = self.decoder(att_feats, seq, init_states)  # [:, 1:, :]
        # Reshape:
        logit_flat = logit.resize(logit.size(0) * logit.size(1), logit.size(2))
        output = F.log_softmax(logit_flat)
        # print('out:', torch.sum(output, dim=1))  # tested with softmax sum==1
        output = output.resize(logit.size(0), logit.size(1), logit.size(2))
        return output

    def sample(self, fc_feats, att_feats, opt={}):
        """
        Sampling without beam search
        """
        batch_size = att_feats.size(0)
        att_feats, init_states = self.encode(att_feats)
        tok = Variable(torch.LongTensor([[_BOS] for i in range(batch_size)])).cuda()
        seq = []
        logprobs = []
        attns = []
        states = init_states
        for _ in range(self.seq_length):
            out, prob, states, attn = self.decoder.decode_step(tok, states, att_feats)
            if out.data[0, 0] == _EOS and _:  #FIXME may be the index as is
                break
            seq.append(out)
            logprobs.append(prob)
            attns.append(attn)
            tok = out
        # FIXME return attns for viz
        return torch.cat(seq, 1).data, torch.cat(logprobs, 1).data


    def forward_old(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        for i in range(seq.size(1)):
            if i == 0:
                # MLP(mean(att_feats))
                mean_region = torch.mean(att_feats.view(batch_size, -1, self.att_feat_size), dim=1)
                # print('mean feat shape:', mean_region.size())
                xt = self.img_embed(mean_region)
            else:
                if i >= 2 and self.ss_prob > 0.0: # otherwiste no need to sample
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i-1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i-1].data.clone()
                        #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                        #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                        prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, i-1].clone()
                # break if all the sequences end
                if i >= 2 and seq[:, i-1].data.sum() == 0:
                    break
                # The word embedding
                xt = self.embed(it)
                # -------------------------------------------------------------------------------
                # The context embedding
                ctx = self.get_ctx(att_feats, state)
                # Concat xt and weighted_ctx:
                xt += ctx
                xt = self.drop_x_lm(xt)
            output, state = self.core(xt.unsqueeze(0), state)
            output = F.log_softmax(self.logit(output.squeeze(0)))
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
            for t in range(self.seq_length + 2):
                if t == 0:
                    # MLP(mean(att_feats))
                    mean_regions = torch.mean(att_feats[k:k+1].view(batch_size, -1, self.att_feat_size), dim=1)
                    # print('mean feat shape:', mean_regions.size())
                    xt = self.img_embed(mean_regions).expand(beam_size, self.input_encoding_size)
                    # xt = self.img_embed(fc_feats[k:k+1]).expand(beam_size, 2 * self.input_encoding_size)
                elif t == 1: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float() # lets go to CPU for more efficiency in indexing operations
                    ys,ix = torch.sort(logprobsf,1,True) # sorted array of logprobs along each previous beam (last true = descending)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if t == 2:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in (sorted) position c
                            local_logprob = ys[q,c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append({'c':ix.data[q,c], 'q':q, 'p':candidate_logprob.data[0], 'r':local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if t > 2:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t-2].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-2].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 2:
                            beam_seq[:t-2, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-2, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']] # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[t-2, vix] = v['c'] # c'th word is the continuation
                        beam_seq_logprobs[t-2, vix] = v['r'] # the raw logprob here
                        beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == self.seq_length + 1:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                'logps': beam_seq_logprobs[:, vix].clone(),
                                                'p': beam_logprobs_sum[vix]
                                                })
                    # encode as vectors
                    it = beam_seq[t-2]
                    xt = self.embed(Variable(it.cuda()))
                    weighted_ctx = self.get_ctx(att_feats, state)
                    xt += weighted_ctx

                if t >= 2:
                    state = new_state

                output, state = self.core(xt.unsqueeze(0), state)
                logprobs = F.log_softmax(self.logit(output.squeeze(0)))

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample_old(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 2):
            if t == 0:
                mean_region = torch.mean(att_feats.view(batch_size, -1, self.att_feat_size), dim=1)
                # print('mean feat shape:', mean_region.size())
                xt = self.img_embed(mean_region)
            else:
                if t == 1: # input <bos>
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
                # The context embedding
                weighted_ctx = self.get_ctx(att_feats, state)
                xt += weighted_ctx
            if t >= 2:
                # stop when all finished
                if t == 2:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt.unsqueeze(0), state)
            logprobs = F.log_softmax(self.logit(output.squeeze(0)))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
