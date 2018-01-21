import os.path as osp
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable, gradcheck
import loss


class DecoderModel(nn.Module):
    def __init__(self, opt):
        super(DecoderModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.logger = opt.logger
        self.input_encoding_size = opt.input_encoding_size
        if len(opt.init_decoder_W):
            # Load W intializer:
            self.W = pickle.load(open(opt.init_decoder_W, 'rb'),
                                 encoding="iso-8859-1")
            self.logger.info('Loading weights to initialize W')
            self.input_encoding_size = self.W.shape[1]
        else:
            self.W = None
        self.require_W_grad = not bool(opt.freeze_decoder_W)
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_x_lm = opt.drop_x_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.region_size = opt.region_size
        self.opt = opt
        self.ss_prob = 0.0  # Schedule sampling probability
        self.ss_vocab = opt.scheduled_sampling_vocab
        self.cnn_finetuning = 0

    def load(self):
        opt = self.opt
        if vars(opt).get('start_from', None) is not None:
            # check if all necessary files exist
            assert osp.isfile(opt.infos_start_from),\
                    "infos file %s does not exist" % opt.start_from
            saved = torch.load(opt.start_from)
            for k in list(saved):
                # issue with InferSent
                if 'crit' in k:
                    self.logger.warn('Deleting key %s' % k)
                    del saved[k]
                # issue with changes in AdaptiveAttention
                if 'fr_' in k:
                    newk = k.replace('fr_', 'sentinel_')
                    saved[newk] = saved[k]
                    self.logger.warn('Subbing %s >> %s' % (k, newk))
                    del saved[k]

            self.logger.warn('Loading the model dict (last checkpoint) %s'\
                             % str(list(saved.keys())))
            self.load_state_dict(saved)

    def define_loss(self, vocab):
        opt = self.opt
        ver = opt.loss_version.lower()
        if ver == 'ml':
            crit = loss.MLCriterion(opt)
        elif ver == 'word':
            crit = loss.WordSmoothCriterion(opt)
        elif ver == "seq":
            if opt.stratify_reward:
                crit = loss.RewardSampler(opt, vocab)
            else:
                crit = loss.ImportanceSampler(opt, vocab)
        else:
            raise ValueError('unknown loss mode %s' % ver)
        crit.log()
        self.crit = crit

    def step(self, data, att_feats, fc_feats, batch=None, epoch=None, train=True):
        opt = self.opt
        tmp = [data['labels'], data['masks'], data['scores']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        labels, masks, scores = tmp
        if not opt.scale_loss:
            scores = None
        stats = None
        seq = labels[:, 1:]
        msk = masks[:, 1:]
        if opt.loss_version == 'seq':
            ml_loss, reward_loss, stats = self.crit(self, fc_feats, att_feats, labels, msk, scores)
        else:
            logprobs = self.forward(fc_feats, att_feats, labels)
            ml_loss, reward_loss, stats = self.crit(logprobs,
                                                    seq,
                                                    msk,
                                                    scores)
        if opt.loss_version.lower() == "ml":
            final_loss = ml_loss
        else:
            final_loss = self.crit.alpha * reward_loss + (1 - self.crit.alpha) * ml_loss
        return ml_loss, final_loss, stats

    def step_track(self, data, att_feats, fc_feats, add_dirac=False):
        tmp = [data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        labels, masks = tmp
        seq = labels[:, 1:]
        msk = masks[:, 1:]
        logprobs = self.forward(fc_feats, att_feats, labels)
        input, target = self.crit.track(logprobs,
                                        seq,
                                        msk,
                                        add_dirac)
        return input, target

    def beam_search(self, state, logprobs, *args, **kwargs):
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys,ix = torch.sort(logprobsf,1,True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q,c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])

            new_state = [_.clone() for _ in state]
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        # start beam search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)

        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
        done_beams = []

        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            logprobsf = logprobs.data.float() # lets go to CPU for more efficiency in indexing operations
            # suppress UNK tokens in the decoding
            logprobsf[:,logprobsf.size(1)-1] =  logprobsf[:, logprobsf.size(1)-1] - 1000

            beam_seq,\
            beam_seq_logprobs,\
            beam_logprobs_sum,\
            state,\
            candidates_divm = beam_step(logprobsf,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum,
                                        state)

            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # encode as vectors
            it = beam_seq[t]
            # TODO check that every submodel has get_logprobs defined
            logprobs, state = self.get_logprobs_state(Variable(it.cuda()), *(args + (state,)), t)

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams
