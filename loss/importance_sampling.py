import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .samplers import init_sampler
from .scorers import init_scorer
from .utils import decode_sequence, get_ml_loss


class ImportanceSampler(nn.Module):
    """
    Apply sentence level loss smoothing
    with importance sampling
    q=p_\theta or hamming
    r = Cider or Bleu
    """
    def __init__(self, opt, vocab):
        nn.Module.__init__(self)
        self.logger = opt.logger
        self.penalize_confidence = opt.penalize_confidence  #FIXME
        self.alpha = opt.alpha_sent
        self.mc_samples = opt.mc_samples
        self.combine_loss = opt.combine_loss
        self.sampler = init_sampler(opt.importance_sampler.lower(),
                                    opt)
        self.scorer = init_scorer(opt.reward.lower(),
                                  opt, vocab)
        self.vocab = vocab

    def log(self):
        self.logger.info('using importance sampling r=%s and q=%s' % (self.scorer.version,
                                                                      self.sampler.version))

    def forward(self, model, fc_feats, att_feats, labels, mask, scores=None):
        # truncate
        logp = model.forward(fc_feats, att_feats, labels)
        target = labels[:, 1:]
        seq_length = logp.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if scores is not None:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        if self.combine_loss:
            ml_gt, wl_gt, _ = self.loss_gt(logp, target, mask, scores)
            loss_gt = self.loss_gt.alpha * wl_gt + (1 - self.loss_gt.alpha) * ml_gt
        else:
            ml_gt = get_ml_loss(logp, target, mask, scores,
                                penalize=self.penalize_confidence)
            loss_gt = ml_gt
        if self.training:
            MC = self.mc_samples
        else:
            MC = 1
        for mci in range(MC):
            sampled, sampled_q, stats = self.sampler.sample(logp, target)
            if 0:
                gt_s = decode_sequence(self.vocab, sampled.data[:, 1:])
                gt = decode_sequence(self.vocab, labels.data[:, 1:])
                for s, ss in zip(gt, gt_s):
                    print('GT:', s, '\nSA:', ss)
            # Forward the sampled captions
            sampled_logp = model.forward(fc_feats, att_feats, sampled)
            sampled_rewards, rstats = self.scorer.get_scores(sampled[:,1:], target)
            stats.update(rstats)
            importance = sampled_rewards / sampled_q
            # normalize:
            importance = importance / np.mean(importance)  # FIXME chekc how to estimate Z_q & Z_r
            stats['importance_mean'] = np.mean(importance)
            stats['importance_std'] = np.std(importance)
            importance = Variable(torch.from_numpy(importance).float(),
                                  requires_grad=False).cuda().view(-1, 1)
            if model.opt.caption_model in ['adaptive_attention', 'top_down']:
                sampled_logp = sampled_logp[:, :-1]
            if self.combine_loss:
                ml_sampled, wl_sampled, stats_sampled = self.loss_sampled(sampled_logp,
                                                                          sampled[:, 1:],
                                                                          mask,
                                                                          scores=importance)
                stats.update(stats_sampled)
                mc_output = (self.loss_sampled.alpha * wl_sampled +
                             (1 - self.loss_sampled.alpha) * ml_sampled)
            else:
                ml_sampled = get_ml_loss(sampled_logp,
                                         sampled[:, 1:],
                                         mask,
                                         scores=importance,
                                         penalize=self.penalize_confidence)
                mc_output = ml_sampled
            if not mci:
                output = mc_output
            else:
                output += mc_output
        output /= MC
        # output = torch.sum(torch.log(r) - logprob) / N
        return loss_gt, output, stats

