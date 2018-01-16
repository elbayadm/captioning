import random
import math
from collections import OrderedDict
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .word import WordSmoothCriterion
from .utils import decode_sequence, get_ml_loss, hamming_distrib


class RewardSampler(nn.Module):
    """
    Sampling the sentences wtr the reward distribution
    instead of the captionig model itself
    """
    def __init__(self, opt, vocab):
        super(RewardSampler, self).__init__()
        self.logger = opt.logger
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha_sent
        assert self.alpha > 0, 'set alpha to a nonzero value, otherwise use the default loss'
        self.tau = opt.tau_sent
        self.combine_loss = opt.combine_loss
        self.scale_loss = opt.scale_loss
        self.vocab_size = opt.vocab_size
        self.vocab = vocab
        self.limited = opt.limited_vocab_sub
        self.verbose = opt.verbose
        self.mc_samples = opt.mc_samples
        self.seq_per_img = opt.seq_per_img
        self.penalize_confidence = opt.penalize_confidence  #FIXME
        # print('Training:', self.training)
        if self.combine_loss:
            # Instead of ML(sampled) return WL(sampled)
            self.loss_sampled = WordSmoothCriterion(opt)
            # self.loss_sampled.alpha = .7
            self.loss_gt = WordSmoothCriterion(opt)

    def forward(self, model, fc_feats, att_feats, labels, mask, scores=None):
        # truncate
        logp = model.forward(fc_feats, att_feats, labels)
        target = labels[:, 1:]
        seq_length = logp.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
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
            preds_matrix, stats = self.alter(target)
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
                ml_sampled, wl_sampled, stats_sampled = self.loss_sampled(sample_logp, preds_matrix[:, 1:], mask)
                stats.update(stats_sampled)
                mc_output = self.loss_sampled.alpha * wl_sampled + (1 - self.loss_sampled.alpha) * ml_sampled
            else:
                ml_sampled = get_ml_loss(sample_logp, preds_matrix[:, 1:], mask, penalize=self.penalize_confidence)
                mc_output = ml_sampled
            if not ss:
                output = mc_output
            else:
                output += mc_output
        output /= MC
        # output = torch.sum(torch.log(r) - logprob) / N
        return loss_gt, output, stats


class HammingRewardSampler(RewardSampler):
    """
    Sample a hamming distance and alter the truth
    """
    def __init__(self, opt, vocab):
        RewardSampler.__init__(self, opt, vocab)

    def log(self):
        self.logger.info('Initialized hamming reward sampler tau = %.2f, alpha= %.1f limited=%d' % (self.tau, self.alpha, self.limited))
        if self.combine_loss:
            self.logger.info('GT loss:')
            self.loss_gt.log()
            self.logger.info('Sampled loss:')
            self.loss_sampled.log()

    def alter(self, labels):
        # print('Altering:', labels.size())
        N = labels.size(0)
        seq_length = labels.size(1)
        # get batch vocab size
        refs = labels.cpu().data.numpy()
        if self.limited == 1:  # In-batch vocabulary substitution
            batch_vocab = np.delete(np.unique(refs), 0)
            lv = len(batch_vocab)
        elif self.limited == 2:  # In-image vocabulary substitution
            num_img = N // self.seq_per_img
            refs_per_image = np.split(refs, num_img)
            im_vocab = [np.delete(np.unique(chunk), 0) for chunk in refs_per_image]
            del refs_per_image
            lv = np.max([len(chunk) for chunk in im_vocab])
            # print('im_vocab:', im_vocab, "im_lv:", lv)
        else:  # Full vocabulary substitution
            lv = self.vocab_size
        # print('batch vocab:', len(batch_vocab), batch_vocab)
        distrib, Z = hamming_distrib(seq_length, lv, self.tau)
        if self.training:
            print('Sampling distrib:', distrib, "Z:", Z)
        # Sample a distance i.e. a reward
        select = np.random.choice(a=np.arange(seq_length + 1),
                                  p=distrib)
        # score = math.exp(-select / self.tau)
        # score = distrib[select]
        score = math.exp(-select / self.tau) / Z
        if self.training:
            self.logger.debug("reward (d=%d): %.2e" %
                              (select, score))
        stats = {"sent_mean": score,
                 "sent_std": 0}

        # Format preds by changing d=select tokens at random
        preds = refs
        # choose tokens to replace
        change_index = np.random.randint(seq_length, size=(N, select))
        rows = np.arange(N).reshape(-1, 1).repeat(select, axis=1)
        # select substitutes
        if self.limited == 1:
            select_index = np.random.choice(batch_vocab, size=(N, select))
        elif self.limited == 2:
            select_index = np.vstack([np.random.choice(chunk, size=(self.seq_per_img, select)) for chunk in im_vocab])
            # print("selected:", select_index)
        else:
            select_index = np.random.randint(low=4, high=self.vocab_size, size=(N, select))
        # print("Selected:", select_index)
        preds[rows, change_index] = select_index
        preds_matrix = np.hstack((np.zeros((N, 1)), preds))  # padd <BOS>
        preds_matrix = Variable(torch.from_numpy(preds_matrix)).cuda().type_as(labels)
        # print('yielding:', preds_matrix.size())
        return preds_matrix, stats


class TFIDFRewardSampler(RewardSampler):
    """
    Sample an edit distance and alter the ground truth
    """
    def __init__(self, opt, vocab):
        RewardSampler.__init__(self, opt, vocab)
        self.ngrams = pickle.load(open('data/coco-train-tok-ng-df.p', 'rb'))
        self.select_rare = opt.rare_tfidf
        self.tau = opt.tau_sent
        self.n = opt.ngram_length
        self.sub_idf = opt.sub_idf
        if self.select_rare:
            self.ngrams = OrderedDict(self.ngrams[self.n])
            freq = np.array([1/c for c in list(self.ngrams.values())])
            if self.tau:
                freq = np.exp(freq/self.tau)
            freq /= np.sum(freq)
            self.ngrams = OrderedDict({k: v for k,v in zip(list(self.ngrams), freq)})
            # print('self.ngrams:', self.ngrams)
        else:
            self.ngrams = list(self.ngrams[self.n])

    def log(self):
        sl = "ML" if not self.combine_loss else "Word"
        self.logger.info('Initialized IDF reward sampler tau = %.2f, alpha= %.1f select_rare=%d, sub_rare=%d sampled loss = %s' % (self.tau, self.alpha, self.select_rare, self.sub_idf, sl))

    def alter(self, labels):
        N = labels.size(0)
        seq_length = labels.size(1)
        # get batch vocab size
        refs = labels.cpu().data.numpy()
        # ng = 1 + np.random.randint(4)
        ng = self.n
        stats = {"sent_mean": ng,
                 "sent_std": 0}
        # Format preds by changing d=select tokens at random
        preds = refs
        # choose an n-consecutive words to replace
        if self.sub_idf:
            # get current ngrams dfs:
            change_index = np.zeros((N, 1), dtype=np.int32)
            for i in range(N):
                p = np.array([self.ngrams.get(tuple(refs[i, j:j+ng]),
                                              1) for j in range(seq_length - ng)])
                p = 1/p
                p /= np.sum(p)
                change_index[i] = np.random.choice(seq_length - ng,
                                                   p=p,
                                                   size=1)
        else:
            change_index = np.random.randint(seq_length - ng, size=(N, 1))
        change_index = np.hstack((change_index + k for k in range(ng)))
        rows = np.arange(N).reshape(-1, 1).repeat(ng, axis=1)
        # select substitutes from occuring n-grams in the training set:
        if self.select_rare:
            picked = np.random.choice(np.arange(len(self.ngrams)),
                                      p=list(self.ngrams.values()),
                                      size=(N,))
            picked_ngrams = [list(self.ngrams)[k] for k in picked]
        else:
            picked_ngrams = random.sample(self.ngrams, N)
        preds[rows, change_index] = picked_ngrams
        preds_matrix = np.hstack((np.zeros((N, 1)), preds))  # padd <BOS>
        preds_matrix = Variable(torch.from_numpy(preds_matrix)).cuda().type_as(labels)
        return preds_matrix, stats


