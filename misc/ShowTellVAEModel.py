from __future__ import absolute_import
from __future__ import division
#  from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from misc.cvae import CVAE, sample_z
from misc.lm import LM_encoder

def nonlin(choice='ReLU'):
    if choice.lower() == "relu":
        return nn.ReLU()
    elif choice.lower() == "sigmoid":
        return nn.Sigmoid()
    elif chocie.lower() == "tanh":
        return nn.Tanh()
    elif choice.lower() == "softmax":
        return nn.Softmax()
    elif choice.lower() == "leakyrelu":
        return nn.LeakyReLU()
    else:
        raise ValueError("Unknown non-linearity %s" % choice)


class ShowTellVAEModel(nn.Module):
    def __init__(self, opt):
        super(ShowTellVAEModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.z_size = opt.z_size
        self.z_interm_size = opt.z_interm_size
        self.ss_prob = 0.0 # Schedule sampling probability
        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        # rnn_size = the encoder/decoder hidden state size (for now similar
        # sizes)
        self.lm_encoder = LM_encoder(opt)
        self.encoder_nonlin = nonlin(opt.vae_nonlin)
        self.cvae = CVAE(self.rnn_size, self.fc_feat_size, self.z_size, self.z_interm_size)
        self.core = getattr(nn, self.rnn_type.upper())(self.input_encoding_size, self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        self.latent_embed = nn.Linear(self.z_size, self.input_encoding_size)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.drop_x_lm = nn.Dropout(p=opt.drop_x_lm)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.init_weights()
        opt.logger.warn('Show & Tell : %s' % str(self._modules))

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)
        self.img_embed.bias.data.fill_(0)
        self.img_embed.weight.data.uniform_(-initrange, initrange)
        self.latent_embed.bias.data.fill_(0)
        self.latent_embed.weight.data.uniform_(-initrange, initrange)
        self.lm_encoder.init_weights()
        self.cvae.init_weights()

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                    Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_())

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        for i in range(seq.size(1) + 1):
            if i == 0:
                xt = self.img_embed(fc_feats)
            elif i == 1:
                text_code = self.lm_encoder(seq)
                text_code = self.encoder_nonlin(text_code)
                z_mu, z_var, text_code_recon = self.cvae(text_code, fc_feats)
                #  print 'text_code', text_code
                #  print 'text_code_recon', text_code_recon
                #  recon_loss = nn.functional.smooth_l1_loss(text_code, text_code_recon, size_average=False) / batch_size
                recon_loss = nn.functional.binary_cross_entropy(text_code_recon, text_code, size_average=False) / self.z_size/ batch_size
                kld_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
                xt = self.latent_embed(sample_z(z_mu, z_var))
            else:
                if i >= 3 and self.ss_prob > 0.0: # otherwiste no need to sample
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i-2].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i-2].data.clone()
                        #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                        #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                        prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, i-2].clone()
                # break if all the sequences end
                if i >= 3 and seq[:, i-2].data.sum() == 0:
                    break
                xt = self.embed(it)
                xt = self.drop_x_lm(xt)
            output, state = self.core(xt.unsqueeze(0), state)
            output = F.log_softmax(self.logit(output.squeeze(0)))
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs[2:]], 1).contiguous(), recon_loss, kld_loss

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
            for t in range(self.seq_length + 3):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k+1]).expand(beam_size, self.input_encoding_size)
                elif t == 1:
                    xt = self.latent_embed(Variable(torch.randn(beam_size, self.z_size)))
                elif t == 2: # input <bos>
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
                    if t == 3:  # at first time step only the first beam is active
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
                    if t > 3:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t-3].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-3].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 3:
                            beam_seq[:t-3, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-3, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']] # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[t-3, vix] = v['c'] # c'th word is the continuation
                        beam_seq_logprobs[t-3, vix] = v['r'] # the raw logprob here
                        beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == self.seq_length + 2:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                'logps': beam_seq_logprobs[:, vix].clone(),
                                                'p': beam_logprobs_sum[vix]
                                                })
                    # encode as vectors
                    it = beam_seq[t-3]
                    xt = self.embed(Variable(it.cuda()))

                if t >= 3:
                    state = new_state

                output, state = self.core(xt.unsqueeze(0), state)
                logprobs = F.log_softmax(self.logit(output.squeeze(0)))

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
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
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            elif t == 1:
                xt = self.latent_embed(Variable(torch.randn(batch_size, self.z_size)))
            else:
                if t == 2: # input <bos>
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

            if t >= 3:
                # stop when all finished
                if t == 3:
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
