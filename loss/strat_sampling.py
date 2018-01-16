import random
import math
from collections import OrderedDict
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .word import WordSmoothCriterion
from .utils import decode_sequence, get_ml_loss
from .samplers import init_sampler


class RewardSampler(nn.Module):
    """
    Sampling the sentences wtr the reward distribution
    instead of the captionig model itself
    """
    def __init__(self, opt, vocab):
        super(RewardSampler, self).__init__()
        self.logger = opt.logger
        self.penalize_confidence = opt.penalize_confidence  #FIXME
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha_sent
        self.combine_loss = opt.combine_loss
        self.vocab = vocab
        self.verbose = opt.verbose
        self.mc_samples = opt.mc_samples
        if self.combine_loss:
            self.loss_sampled = WordSmoothCriterion(opt)
            self.loss_gt = WordSmoothCriterion(opt)
        self.sampler = init_sampler(opt.reward.lower(), opt)

    def log(self):
        # FIXME
        self.logger.info('RewardSampler (stratified sampling), r=%s' % self.sampler.version)
        if self.combine_loss:
            self.logger.info('GT loss:')
            self.loss_gt.log()
            self.logger.info('Sampled loss:')
            self.loss_sampled.log()


    def forward(self, model, fc_feats, att_feats, labels, mask, scores=None):
        # truncate
        logp = model.forward(fc_feats, att_feats, labels)
        target = labels[:, 1:]
        seq_length = logp.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if scores is not None:
            print('scaling the masks')
            # FIXME see to it that i do not normalize with scaled masks
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        if self.combine_loss:
            ml_gt, wl_gt, _ = self.loss_gt(logp, target, mask)
            loss_gt = self.loss_gt.alpha * wl_gt + (1 - self.loss_gt.alpha) * ml_gt
        else:
            ml_gt = get_ml_loss(logp, target, mask, penalize=self.penalize_confidence)
            loss_gt = ml_gt
        if self.training:
            MC = self.mc_samples
        else:
            MC = 1
        for ss in range(MC):
            preds_matrix, _, stats = self.sampler.sample(logp, target)
            if 0:
                gt_s = decode_sequence(self.vocab, preds_matrix.data[:, 1:])
                gt = decode_sequence(self.vocab, labels.data[:, 1:])
                for s, ss in zip(gt, gt_s):
                    print('GT:', s, '\nSA:', ss)
            # Forward the sampled captions
            sample_logp = model.forward(fc_feats, att_feats, preds_matrix)
            if model.opt.caption_model in ['adaptive_attention', 'top_down']:
                sample_logp = sample_logp[:, :-1]
            if self.combine_loss:
                ml_sampled, wl_sampled, stats_sampled = self.loss_sampled(sample_logp,
                                                                          preds_matrix[:, 1:],
                                                                          mask)
                stats.update(stats_sampled)
                mc_output = (self.loss_sampled.alpha * wl_sampled +
                             (1 - self.loss_sampled.alpha) * ml_sampled)
            else:
                ml_sampled = get_ml_loss(sample_logp,
                                         preds_matrix[:, 1:],
                                         mask,
                                         scores,
                                         penalize=self.penalize_confidence)
                mc_output = ml_sampled
            if not ss:
                output = mc_output
            else:
                output += mc_output
        output /= MC
        # output = torch.sum(torch.log(r) - logprob) / N
        return loss_gt, output, stats



