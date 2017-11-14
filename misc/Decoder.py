import os.path as osp
from six.moves import cPickle as pickle
import torch
import torch.nn as nn
from torch.autograd import Variable, gradcheck
import numpy as np
import misc.loss as loss


class DecoderModel(nn.Module):
    def __init__(self, opt):
        super(DecoderModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.use_glove = opt.use_glove
        if self.use_glove:
            self.input_encoding_size = 300
        else:
            self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.num_regions = opt.num_regions  # Or the number of candidate regions
        self.att_feat_size = opt.att_feat_size
        self.opt = opt
        self.logger = opt.logger
        self.ss_prob = 0.0 # Schedule sampling probability
        self.ss_vocab = opt.scheduled_sampling_vocab

    def load(self):
        opt = self.opt
        if vars(opt).get('start_from', None) is not None:
            # check if all necessary files exist
            assert osp.isfile(opt.infos_start_from),\
                    "infos file %s does not exist" % opt.start_from
            saved = torch.load(opt.start_from)
            for k in list(saved):
                if 'crit' in k:
                    self.logger.warn('Deleting key %s' % k)
                    del saved[k]
            self.logger.warn('Loading the model dict (last checkpoint) %s'\
                             % str(list(saved.keys())))
            self.load_state_dict(saved)

    def define_loss(self, vocab):
        opt = self.opt
        if opt.sample_cap:
            # Sampling from the captioning model itself
            if 'dummy' in opt.loss_version:
                crit = loss.AllIsGoodCriterion(opt, vocab)
            elif 'cider' in opt.loss_version:
                crit = loss.CiderRewardCriterion(opt, vocab)
            elif 'hamming' in opt.loss_version:
                crit = loss.HammingRewardCriterion(opt)
            elif 'infersent' in opt.loss_version:
                crit = loss.InfersentRewardCriterion(opt, vocab)
            elif 'bleu' in opt.loss_version:
                crit = loss.BleuRewardCriterion(opt, vocab)
            elif opt.loss_version == "word":
                crit = loss.WordSmoothCriterion(opt)
            elif opt.loss_version == "word2":
                crit = loss.WordSmoothCriterion2(opt)
            else:
                raise ValueError('Loss function %s in sample_cap mode unknown' % (opt.loss_version))

        elif opt.bootstrap:
            crit = loss.DataAugmentedCriterion(opt)
        # elif opt.combine_caps_losses:
            # crit = loss.MultiLanguageModelCriterion(opt.seq_per_img)
        elif opt.sample_reward:
            if 'hamming' in opt.loss_version:
                crit = loss.HammingRewardSampler(opt, vocab)
            else:
                raise ValueError('Loss function %s in sample_reward mode unknown' % (opt.loss_version))
        else:
            # The defualt ML
            opt.logger.warn('Using baseline loss criterion')
            crit = loss.LanguageModelCriterion(opt)
        self.crit = crit

    def step(self, data, att_feats, fc_feats):
        opt = self.opt
        if opt.bootstrap:
            assert opt.bootstrap_score in ['cider', 'bleu2', 'bleu3', 'bleu4', 'infersent']
            tmp = [data['labels'], data['masks'], data['scores'], data[opt.bootstrap_score]]
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            labels, masks, scores, s_scores = tmp
            # Exponentiate the scores
            if opt.tau_sent:
                print('Original mean reward:', torch.mean(s_scores).data[0])
                s_scores = torch.exp(torch.div(s_scores, opt.tau_sent))
                print('Tempering the reward (new mean):', torch.mean(s_scores).data[0])
            scores = torch.div(s_scores, torch.exp(scores))
            opt.logger.debug('Mean importance scores: %.3e' % torch.mean(scores).data[0])
        else:
            tmp = [data['labels'], data['masks'], data['scores']]
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            labels, masks, scores = tmp

        # if opt.caption_model == "show_tell_vae":
                # preds, recon_loss, kld_loss = self.forward(fc_feats, att_feats, labels)
                # real_loss, loss = self.crit(preds, labels[:, 1:], masks[:, 1:])
                # loss += opt.vae_weight * (recon_loss + opt.kld_weight * kld_loss)
                # #FIXME add the scaling as parameter

        stats = None
        # FIXME Deprecated
        if opt.caption_model == 'show_tell_raml':
            probs, reward = self.forward(fc_feats, att_feats, labels)
            raml_scores = reward * Variable(torch.ones(scores.size()))
            # raml_scores = Variable(torch.ones(scores.size()))
            print('Raml reward:', reward)
            ml_loss, loss = self.crit(probs, labels[:, 1:], masks[:, 1:], raml_scores)
        else:
            if self.opt.sample_reward:
                ml_loss, loss, stats = self.crit(self, fc_feats, att_feats, labels, masks[:, 1:], scores)
            else:
                logprobs = self.forward(fc_feats, att_feats, labels)
                ml_loss, loss, stats = self.crit(logprobs,
                                                 labels[:, 1:],
                                                 masks[:, 1:],
                                                 scores)
        # print('Grad check:', gradcheck(self.crit, [logprobs,
                                                   # labels[:, 1:],
                                                   # masks[:, 1:],
                                                   # scores]))
        return ml_loss, loss, stats


