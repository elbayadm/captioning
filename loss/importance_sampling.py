"""
Importance sampling of the sentence smoothed loss:
    Loss = E_r[-log p] = E_q[-r/q log p]
    r referred to as scorer
    q referred to as sampler
    prior to normalization : q = ~q / Z_q and r = ~r / Z_r
    except fot q=hamming, the Z_q is untractable
    the importance ratios  w = r/q are approximated
    ~ w = ~r / ~q / (sum ~r / ~ q over MC)
    or
    ~ w = ~r / q / (sum ~r / q over MC)
"""
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
        self.caption_model = opt.caption_model
        self.penalize_confidence = opt.penalize_confidence  #FIXME
        self.lazy_rnn = opt.lazy_rnn
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
        monte_carlo = []
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
            sampled_rewards, rstats = self.scorer.get_scores(sampled[:,1:], target)
            stats.update(rstats)
            importance = sampled_rewards / sampled_q
            monte_carlo.append([sampled, importance])
            # normalize:
        if MC == 1:
            # incorrect estimation of Z_r/Z_q
            monte_carlo = monte_carlo[0]
            importance_normalized = monte_carlo[1] / np.mean(monte_carlo[1])
            stats['importance_mean'] = np.mean(importance_normalized)
            stats['importance_std'] = np.std(importance_normalized)
            importance_normalized = Variable(torch.from_numpy(importance_normalized).float(),
                                             requires_grad=False).cuda().view(-1, 1)
            if self.lazy_rnn:
                mc_output, stats_sampled = self.batch_loss_lazy(logp,
                                                                monte_carlo[0],
                                                                mask,
                                                                importance_normalized)
            else:
                mc_output, stats_sampled = self.batch_loss(model,
                                                           fc_feats,
                                                           att_feats,
                                                           monte_carlo[0],
                                                           mask,
                                                           importance_normalized)

            if stats_sampled is not None:
                stats.update(stats_sampled)
            output = mc_output
        else:
            # correct estimation of Z_r/Z_q
            imp = np.vstack([_[1] for _ in monte_carlo]).T
            imp = imp/imp.sum(axis=1)[:, None]
            stats['importance_mean'] = np.mean(imp)
            stats['importance_std'] = np.std(imp)
            for mci in range(MC):
                importance_normalized = imp[:, mci]
                importance_normalized = Variable(torch.from_numpy(importance_normalized).float(),
                                                 requires_grad=False).cuda().view(-1,1)
                # print('imp:', list(imp[:, mci].T))
                if self.lazy_rnn:
                    mc_output, stats_sampled = self.batch_loss_lazy(logp,
                                                                    monte_carlo[mci][0],
                                                                    mask,
                                                                    importance_normalized)

                else:
                    mc_output, stats_sampled = self.batch_loss(model,
                                                               fc_feats,
                                                               att_feats,
                                                               monte_carlo[mci][0],
                                                               mask,
                                                               importance_normalized)
                if stats_sampled is not None:
                    stats.update(stats_sampled)
                if not mci:
                    output = mc_output
                else:
                    output += mc_output
            output /= MC
        return loss_gt, output, stats

    def batch_loss(self, model, fc_feats, att_feats, labels, mask, scores):
        logp = model.forward(fc_feats, att_feats, labels)
        if self.caption_model in ['adaptive_attention', 'top_down']:
            logp = logp[:, :-1]
        if self.combine_loss:
            ml, wl, stats = self.loss_sampled(logp,
                                              labels[:, 1:],
                                              mask,
                                              scores)
            mc_output = (self.loss_sampled.alpha * wl +
                         (1 - self.loss_sampled.alpha) * ml)
        else:
            ml_sampled = get_ml_loss(logp,
                                     labels[:, 1:],
                                     mask,
                                     scores,
                                     penalize=self.penalize_confidence)
            mc_output = ml_sampled
            stats = None
        return mc_output, stats

    def batch_loss_lazy(self, logp, labels, mask, scores):
        if self.caption_model in ['adaptive_attention', 'top_down']:
            logp = logp[:, :-1]
        if self.combine_loss:
            ml, wl, stats = self.loss_sampled(logp,
                                              labels[:, 1:],
                                              mask,
                                              scores)
            mc_output = (self.loss_sampled.alpha * wl +
                         (1 - self.loss_sampled.alpha) * ml)
        else:
            ml_sampled = get_ml_loss(logp,
                                     labels[:, 1:],
                                     mask,
                                     scores,
                                     penalize=self.penalize_confidence)
            mc_output = ml_sampled
            stats = None
        return mc_output, stats


