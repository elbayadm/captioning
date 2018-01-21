"""
Evaluate an ensemble of models (Show & Tell)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
import _pickle as pickle
import numpy as np
from misc.ShowTellModel import ShowTellModel
import misc.loss as loss

class Ensemble(nn.Module):
    def __init__(self, models, cnn_models, opt):
        super(Ensemble, self).__init__()
        self.opt = opt
        for model in models:
            assert(type(model) == ShowTellModel)
        # Store each model as a component of the overall network
        self.models = models
        self.cnn_models = cnn_models
        self.n_models = len(models)
        assert self.n_models == len(self.cnn_models)
        self.seq_length = models[0].seq_length
        self.ss_prob = 0
        self.ss_vocab = 0
        self.logger = opt.logger

    def define_loss(self, loader_vocab):
        opt = self.opt
        if opt.raml_loss:
            # D = np.eye(opt.vocab_size + 1, dtype="float32")
            # D = np.random.uniform(size=(opt.vocab_size + 1,
                                        # opt.vocab_size + 1)).astype(np.float32)
            # D = pickle.load(open('data/Glove/cocotalk_similarities_v2.pkl', 'rb'),
                            # encoding='iso-8859-1')
            D = pickle.load(open(opt.similarity_matrix, 'rb'), encoding='iso-8859-1')

            D = D.astype(np.float32)
            D = Variable(torch.from_numpy(D)).cuda()
            crit = loss.SmoothLanguageModelCriterion(Dist=D,
                                                      loader_vocab=loader_vocab,
                                                      opt=opt)
        elif opt.bootstrap_loss:
            # Using importance sampling loss:
            crit = loss.ImportanceLanguageModelCriterion(opt)

        elif opt.combine_caps_losses:
            crit = loss.MultiLanguageModelCriterion(opt.seq_per_img)
        else:
            opt.logger.warn('Using baseline loss criterion')
            crit = loss.LanguageModelCriterion(opt)
        self.crit = crit

    def get_feats(self, images):
        att_feats_ens = []
        fc_feats_ens = []
        for cnn_model in self.cnn_models:
            att_feats, fc_feats = cnn_model.forward(images)
            att_feats_ens.append(att_feats)
            fc_feats_ens.append(fc_feats)
        return att_feats_ens, fc_feats_ens

    def forward(self, images, seq):
        batch_size = seq.size(0)
        # Get the image features first
        att_feats_ens = []
        fc_feats_ens = []
        for cnn_model in self.cnn_models:
            att_feats, fc_feats = cnn_model.forward_caps(images, self.opt.seq_per_img)
            att_feats_ens.append(att_feats)
            fc_feats_ens.append(fc_feats)

        outputs = [[] for _ in self.models]
        state = []
        for model in self.models:
            model_state = model.init_hidden(batch_size)
            # print('Initial length:', [torch.sum(st).data[0] for st in model_state])
            state.append(model_state)

        for i in range(seq.size(1)):
            xt = []
            if i == 0:
                for e, model in enumerate(self.models):
                    xt.append(model.img_embed(fc_feats_ens[e]))
            else:
                if i >= 2 and self.ss_prob > 0.0: # otherwiste no need to sample
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i-1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i-1].data.clone()
                        for e, model in enumerate(self.models):
                            if not e:
                                prob_prev = torch.exp(outputs[e][-1].data) # fetch prev distribution: shape Nx(M+1)
                            else:
                                prob_prev += torch.exp(outputs[e][-1].data)
                        if self.ss_vocab:
                            for token in sample_vocab:
                                prob_prev[:, token] += 0.5
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, i-1].clone()
                # break if all the sequences end
                if i >= 2 and seq[:, i-1].data.sum() == 0:
                    break
                for model in self.models:
                    xtm = model.embed(it)
                    xtm = model.drop_x_lm(xtm)
                    xt.append(xtm)
            for e, model in enumerate(self.models):
                output, state[e] = model.core(xt[e].unsqueeze(0), state[e])
                output = F.log_softmax(model.logit(output.squeeze(0)))
                outputs[e].append(output)

        logprobs = [torch.cat([_.unsqueeze(1) for _ in outputs[e][1:]], 1).contiguous() for e in range(self.n_models)]
        # combine the probabilities:
        logprobs = torch.stack(logprobs, dim=1)
        logprobs = torch.mean(logprobs, dim=1).squeeze(1)
        return  logprobs


    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        # print('Sample max:', sample_max)
        beam_size = opt.get('beam_size', 1)
        # print('Beam size:', beam_size)
        temperature = opt.get('temperature', 1.0)
        forbid_unk = opt.get('forbid_unk', 1)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)


        batch_size = fc_feats[0].size(0)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

        # lets process every image independently for now, for simplicity
        state = []
        for model in self.models:
            model_state = model.init_hidden(batch_size)
            # print('Initial length:', [torch.sum(st).data[0] for st in model_state])
            state.append(model_state)
        for t in range(self.seq_length + 2):
            # print('At step ',t, ' states:', [torch.sum(ss).data[0] for st in state for ss in st])
            xt = []
            if t == 0:
                for e, model in enumerate(self.models):
                    xt.append(model.img_embed(fc_feats[e]))
            else:
                if t == 1:
                    it = fc_feats[e].data.new(batch_size).long().zero_()
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

                for model in self.models:
                    xt.append(model.embed(Variable(it, requires_grad=False)))
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

            for e, model in enumerate(self.models):
                if not e:
                    output, state[e] = model.core(xt[e].unsqueeze(0), state[e])
                else:
                    out, state[e] = model.core(xt[e].unsqueeze(0), state[e])
                    output += out_
            logprobs = F.log_softmax(self.logit(output.squeeze(0)))
            if forbid_unk:
                logprobs = logprobs[:, :-1]
        try:
            return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1).data
        except:  # Fix issue with Variable v. Tensor depending on the mode.
            return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats[0].size(0)
        forbid_unk = opt.get('forbid_unk', 1)
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = []
            for model in self.models:
                model_state = model.init_hidden(beam_size)
                # print('Initial length:', [torch.sum(st).data[0] for st in model_state])
                state.append(model_state)
            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
            for t in range(self.seq_length + 2):
                # print('At step ',t, ' states:', [torch.sum(ss).data[0] for st in state for ss in st])
                xt = []
                if t == 0:
                    for e, model in enumerate(self.models):
                        xt.append(model.img_embed(fc_feats[e][k:k+1]).expand(beam_size, model.input_encoding_size))
                elif t == 1: # input <bos>
                    it = fc_feats[e].data.new(beam_size).long().zero_()
                    for model in self.models:
                        xt.append(model.embed(Variable(it, requires_grad=False)))
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float() # lets go to CPU for more efficiency in indexing operations
                    ys, ix = torch.sort(logprobsf,1,True) # sorted array of logprobs along each previous beam (last true = descending)
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
                    new_state = [[_.clone() for _ in state_] for state_ in state]
                    if t > 2:
                        # we'll need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t-2].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-2].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 2:
                            beam_seq[:t-2, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-2, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for e in range(self.n_models):
                            for state_ix in range(len(new_state[e])):
                                # copy over state in previous beam q to new beam at vix
                                new_state[e][state_ix][0, vix] = state[e][state_ix][0, v['q']] # dimension one is time step

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
                    for model in self.models:
                        xt.append(model.embed(Variable(it.cuda())))
                        # print("next token embedding:", xt)

                if t >= 2:
                    for e in range(self.n_models):
                        state[e] = new_state[e]
                #  print "State:", state
                logprobs = []
                for e, model in enumerate(self.models):
                    out_, st_ = model.core(xt[e].unsqueeze(0), state[e])
                    state[e] = st_
                    # print('Output proba:', torch.sum(out_))
                    # print('state updated:', [torch.sum(tok).data[0] for tok in st_])
                    probs_ = F.log_softmax(model.logit(out_.squeeze(0)))
                    if forbid_unk:
                        probs_ = probs_[:, :-1]
                    probs_ = probs_.unsqueeze(1)
                    logprobs.append(probs_)
                # either take the max or the average ( for now the max):
                # print("Logprobs:", logprobs)
                logprobs = torch.stack(logprobs, dim=1)
                # Max of probas
                # logprobs, _ = torch.max(logprobs, dim=0)
                # Mean of probas
                logprobs = torch.mean(logprobs, dim=1).squeeze(1)
                # Sum of probas:
                # logprobs = torch.log(torch.sum(torch.exp(logprobs), dim=0))
                # logprobs = logprobs[0]
                # print('logprobs:', logprobs)
            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def step(self, data):
        opt = self.opt
        if opt.bootstrap_loss:
            if opt.bootstrap_version in ["cider", "cider-exp"]:
                tmp = [data['images'], data['labels'], data['masks'], data['scores'], data['cider']]
                tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
                images, labels, masks, scores, s_scores= tmp

            elif opt.bootstrap_version in ["bleu4", "bleu4-exp"]:
                tmp = [data['images'], data['labels'], data['masks'], data['scores'], data['bleu']]
                tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
                images, labels, masks, scores, s_scores = tmp
            elif opt.bootstrap_version in ["infersent", "infersent-exp"]:
                tmp = [data['images'], data['labels'], data['masks'], data['scores'], data['infersent']]
                tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
                images, labels, masks, scores, s_scores = tmp
            else:
                raise ValueError('Unknown bootstrap distribution %s' % opt.bootstrap_version)
            if "exp" in opt.bootstrap_version:
                print('Original rewards:', torch.mean(s_scores))
                s_scores = torch.exp(torch.div(s_scores, opt.raml_tau))
                print('Tempering the reward:', torch.mean(s_scores))
            r_scores = torch.div(s_scores, torch.exp(scores))
            opt.logger.debug('Mean importance scores: %.3e' % torch.mean(r_scores).data[0])
        else:
            tmp = [data['images'], data['labels'], data['masks'], data['scores']]
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            images, labels, masks, scores = tmp

        if opt.caption_model == 'show_tell_raml':
            probs, reward = self.forward(images, labels)
            raml_scores = reward * Variable(torch.ones(scores.size()))
            # raml_scores = Variable(torch.ones(scores.size()))
            print('Raml reward:', reward)
            real_loss, loss = self.crit(probs, labels[:, 1:], masks[:, 1:], raml_scores)
        else:
            if opt.bootstrap_loss:
                real_loss, loss = self.crit(self.forward(images, labels),
                                            labels[:, 1:], masks[:, 1:], r_scores)
            else:
                real_loss, loss = self.crit(self.forward(images, labels),
                                            labels[:, 1:], masks[:, 1:], scores)
        return real_loss, loss

