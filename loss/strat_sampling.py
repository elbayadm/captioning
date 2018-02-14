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
        self.caption_model = opt.caption_model
        self.penalize_confidence = opt.penalize_confidence  #FIXME
        self.lazy_rnn = opt.lazy_rnn
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

        # labels should bet trimmed to seqlength after 1:
        loss_gt, stats = self.batch_loss_lazy(logp, labels, mask, scores)
        if self.training:
            MC = self.mc_samples
        else:
            MC = 1
        for mci in range(MC):
            sampled, _, stats = self.sampler.sample(logp, target)
            if 0:
                gt_s = decode_sequence(self.vocab, preds_matrix.data[:, 1:])
                gt = decode_sequence(self.vocab, labels.data[:, 1:])
                for s, ss in zip(gt, gt_s):
                    print('GT:', s, '\nSA:', ss)
            # Forward the sampled captions
            if self.lazy_rnn:
                mc_output, stats_sampled = self.batch_loss_lazy(logp,
                                                                sampled,
                                                                mask,
                                                                scores)

            else:
                mc_output, stats_sampled = self.batch_loss(model,
                                                           fc_feats,
                                                           att_feats,
                                                           sampled,
                                                           mask,
                                                           scores)

            if stats_sampled is not None:
                stats.update(stats_sampled)

            if not mci:
                output = mc_output
            else:
                output += mc_output
        output /= MC
        return loss_gt, output, stats

    def batch_loss(self, model, fc_feats, att_feats, labels, mask, scores):
        """
        forward the new sampled labels and return the loss
        """
        logp = model.forward(fc_feats, att_feats, labels)
        if self.caption_model in ['adaptive_attention', 'top_down']:
            logp = logp[:, :-1]
        if self.combine_loss:
            ml, wl, stats = self.loss_sampled(logp,
                                              labels[:, 1:],
                                              mask,
                                              scores)
            loss = (self.loss_sampled.alpha * wl +
                    (1 - self.loss_sampled.alpha) * ml)
        else:
            ml = get_ml_loss(logp,
                             labels[:, 1:],
                             mask,
                             scores,
                             penalize=self.penalize_confidence)
            loss = ml
            stats = None
        return loss, stats

    def batch_loss_lazy(self, logp, labels, mask, scores):
        """
        Evaluate the oss ov the new labels given the gt logits
        """
        if self.caption_model in ['adaptive_attention', 'top_down']:
            logp = logp[:, :-1]
        if self.combine_loss:
            ml, wl, stats = self.loss_sampled(logp,
                                              labels[:, 1:],
                                              mask,
                                              scores)
            loss = (self.loss_sampled.alpha * wl +
                    (1 - self.loss_sampled.alpha) * ml)
        else:
            ml = get_ml_loss(logp,
                             labels[:, 1:],
                             mask,
                             scores,
                             penalize=self.penalize_confidence)
            loss = ml
            stats = None
        return loss, stats

    def track(self, model, fc_feats, att_feats, labels, mask, scores=None):
        # truncate
        sampled_list = []
        logp_list = []
        logp = model.forward(fc_feats, att_feats, labels)
        target = labels[:, 1:]
        seq_length = logp.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        # labels should bet trimmed to seqlength after 1:
        if self.training:
            MC = self.mc_samples
        else:
            MC = 1
        for mci in range(MC):
            sampled, _, _ = self.sampler.sample(logp, target)
            sampled_list.append(sampled.data.cpu().numpy())
            if 1:
                gt_s = decode_sequence(self.vocab, sampled.data[:, 1:])
                gt = decode_sequence(self.vocab, labels.data[:, 1:])
                for s, ss in zip(gt, gt_s):
                    print('GT:', s, '\nSA:', ss)
            # Forward the sampled captions
            if not self.lazy_rnn:
                logp_s = model.forward(fc_feats, att_feats, sampled)
                logp_list.append(torch.exp(logp_s).data.cpu().numpy())

        logp = torch.exp(logp).data.cpu().numpy()
        return logp, logp_list, sampled_list


