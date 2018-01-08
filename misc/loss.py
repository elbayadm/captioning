import os
import os.path as osp
import sys
import random
import collections
import math
import pickle
from scipy.special import binom
import numpy as np
from scipy.spatial.distance import hamming
from collections import Counter, OrderedDict
from scipy.misc import comb

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
sys.path.append("coco-caption")
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from misc.utils import to_contiguous, decode_sequence, sentence_bleu, group_similarity


def hamming_distrib_soft(m, v, tau):
    x = [np.log(comb(m, d, exact=False)) + d * np.log(v) - d/tau * np.log(v) - d/tau for d in range(m + 1)]
    x = np.array(x)
    p = np.exp(x)
    p /= np.sum(p)
    return p


def hamming_distrib(m, v, tau):
    x = [comb(m, d, exact=False) * (v-1)**d * math.exp(-d/tau) for d in range(m+1)]
    x = np.array(x)
    Z = np.sum(x)
    return x/Z, Z


def hamming_Z(m, v, tau):
    pd = hamming_distrib(m, v, tau)
    popul = v ** m
    print('popul:', popul)
    print('pd:', pd)
    Z = np.sum(pd * popul * np.exp(-np.arange(m+1)/tau))
    print('Pre-clipping:', Z)
    return np.clip(Z, a_max=1e30, a_min=1)


def rows_entropy(distrib):
    """
    return the entropy of each row in the given distributions
    """
    return torch.sum(distrib * torch.log(distrib), dim=1)

def normalize_reward(distrib):
    """
    Normalize so that each row sum to one
    """
    sums = torch.sum(distrib, dim=1).unsqueeze(1)
    return  distrib / sums.repeat(1, distrib.size(1))


class LanguageModelCriterion(nn.Module):
    """
    The defaul cross entropy loss with the option
    of scaling the sentence loss
    """
    def __init__(self, opt):
        super().__init__()
        self.logger = opt.logger
        self.scale_loss = opt.scale_loss
        self.normalize_batch = opt.normalize_batch

    def log(self):
        self.logger.info('Default ML loss')

    def forward(self, input, target, mask, scores=None):
        """
        input : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        mask : the ground truth mask to ignore UNK tokens (N, seq_length)
        scores: scalars to scale the loss of each sentence (N, 1)
        """
        # truncate to the same size
        # print('Logprobs:', input.size(), "labels:", target.size(), mask.size())
        seq_length = input.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.unsqueeze(1).repeat(1, seq_length)
            # print('mask:', mask.size(), 'row_scores:', row_scores.size())
            mask = torch.mul(mask, row_scores)
        # print('probs:', input.data[0,0,:30])
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output)
        # print('out:', output.data[0], 'mask:', torch.sum(mask).data[0])
        if self.normalize_batch:
            output /= torch.sum(mask)
        stats = None
        return output, output, stats

    def track(self, input, target, mask, scores=None):
        """
        input : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        mask : the ground truth mask to ignore UNK tokens (N, seq_length)
        scores: scalars to scale the loss of each sentence (N, 1)
        """
        # truncate to the same size
        # print('Logprobs:', input.size(), "labels:", target.size(), mask.size())
        N = input.size(0)
        seq_length = input.size(1)
        target = target[:, :seq_length].data.cpu().numpy()
        input = torch.exp(input).data.cpu().numpy()
        target_d = np.zeros_like(input)
        rows = np.arange(N).reshape(-1, 1).repeat(seq_length, axis=1)
        cols = np.arange(seq_length).reshape(1, -1).repeat(N, axis=0)
        target_d[rows, cols, target] = 1
        return input, target_d

def get_ml_loss(input, target, mask, norm=True):
    """
    Compute the usual ML loss
    """
    target = target[:, :input.size(1)]
    mask = mask[:, :input.size(1)]
    # print('input:', input.size(), target.size(), mask.size())
    input = to_contiguous(input).view(-1, input.size(2))
    target = to_contiguous(target).view(-1, 1)
    mask = to_contiguous(mask).view(-1, 1)
    ml_output = - input.gather(1, target) * mask
    ml_output = torch.sum(ml_output)
    if norm:
        ml_output /= torch.sum(mask)
    return ml_output

def get_indices_vocab(target, seq_per_img):
    seq_length = target.size(1)
    num_img = target.size(0) // seq_per_img
    vocab_per_image = target.chunk(num_img)
    vocab_per_image = [np.unique(to_contiguous(t).data.cpu().numpy())
                       for t in vocab_per_image]
    max_vocab = max([len(t) for t in vocab_per_image])
    vocab_per_image = [np.pad(t, (0, max_vocab - len(t)), 'constant')
                       for t in vocab_per_image]
    indices_vocab = Variable(torch.cat([torch.from_numpy(t).\
                             repeat(seq_per_img * seq_length, 1)
                             for t in vocab_per_image], dim=0)).cuda()
    return indices_vocab


class WordSmoothCriterion(nn.Module):
    """
    Apply word level loss smoothing given a similarity matrix
    the two versions are:
        full : to take into account the whole vocab
        limited: to consider only the ground truth vocab
    """
    def __init__(self, opt):
        super().__init__()
        self.logger = opt.logger
        self.seq_per_img = opt.seq_per_img
        self.scale_loss = opt.scale_loss
        self.smooth_remove_equal = opt.smooth_remove_equal
        self.clip_sim = opt.clip_sim
        if self.clip_sim:
            self.margin = opt.margin
            self.logger.warn('Clipping similarities below %.2f' % self.margin)
        self.limited = opt.limited_vocab_sim
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha_word
        assert self.alpha > 0, 'set alpha to a nonzero value, otherwise use the default loss'
        self.tau_word = opt.tau_word
        # Load the similarity matrix:
        M = pickle.load(open(opt.similarity_matrix, 'rb'), encoding='iso-8859-1')
        M = M.astype(np.float32)
        n, d = M.shape
        assert n == d and n == opt.vocab_size, 'Similarity matrix has incompatible shape'
        M = Variable(torch.from_numpy(M)).cuda()
        self.Sim_Matrix = M

    def forward(self, input, target, mask, scores=None):
        # truncate to the same size
        seq_length = input.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        ml_output = get_ml_loss(input, target, mask)
        # Get the similarities of the words in the batch (Vb, V)
        sim = self.Sim_Matrix[to_contiguous(target).view(-1, 1).squeeze().data]
        # print('raw sim:', sim)
        if self.clip_sim:
            # keep only the similarities larger than the margin
            # self.logger.warn('Clipping the sim')
            sim = sim * sim.ge(self.margin).float()
        if self.limited:
            # self.logger.warn('Limitig smoothing to the gt vocab')
            indices_vocab = get_indices_vocab(target, self.seq_per_img)
            sim = sim.gather(1, indices_vocab)
            input = input.gather(1, indices_vocab)

        if self.tau_word:
            smooth_target = torch.exp(torch.mul(torch.add(sim, -1.), 1/self.tau_word))
        else:
            # Do not exponentiate
            smooth_target = torch.add(sim, -1.)
        if self.smooth_remove_equal:
            smooth_target = smooth_target * sim.lt(1.0).float()
        # Store some stats about the sentences scores:
        scalars = smooth_target.data.cpu().numpy()[:]
        stats = {"word_mean": np.mean(scalars),
                 "word_std": np.std(scalars)}

        # print('smooth_target:', smooth_target)
        # Format
        mask = to_contiguous(mask).view(-1, 1)
        mask = mask.repeat(1, sim.size(1))
        input = to_contiguous(input).view(-1, input.size(2))
        # print('in:', input.size(), 'mask:', mask.size(), 'smooth:', smooth_target.size())
        output = - input * mask * smooth_target

        if torch.sum(smooth_target * mask).data[0] > 0:
            output = torch.sum(output) / torch.sum(smooth_target * mask)
        else:
            self.logger.warn("Smooth targets weights sum to 0")
            output = torch.sum(output)

        return ml_output, output, stats


class WordSmoothCriterion2(nn.Module):
    """
    Apply word level loss smoothing given a similarity matrix
    the two versions are:
        full : to take into account the whole vocab
        limited: to consider only the ground truth vocab
    """
    def __init__(self, opt):
        super().__init__()
        self.logger = opt.logger
        self.seq_per_img = opt.seq_per_img
        self.scale_loss = opt.scale_loss
        self.smooth_remove_equal = opt.smooth_remove_equal
        self.clip_sim = opt.clip_sim
        self.add_entropy = opt.word_add_entropy
        self.normalize_batch = opt.normalize_batch
        self.scale_wl = opt.scale_wl
        self.use_cooc = opt.use_cooc
        if self.clip_sim:
            self.margin = opt.margin
            self.logger.warn('Clipping similarities below %.2f' % self.margin)
        self.limited = opt.limited_vocab_sim
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha_word
        # assert self.alpha > 0, 'set alpha to a nonzero value, otherwise use the default loss'
        self.tau_word = opt.tau_word
        # Load the similarity matrix:
        M = pickle.load(open(opt.similarity_matrix, 'rb'), encoding='iso-8859-1')
        if not self.use_cooc:
            M = M - 1
        if opt.rare_tfidf:
            IDF = pickle.load(open('data/coco/idf_coco.pkl', 'rb'))
            M += self.tau_word * IDF  # old versions IDF/1.8
        M = M.astype(np.float32)
        n, d = M.shape
        print('Sim matrix:', n, 'x', d, ' V=', opt.vocab_size)
        assert n == d and n == opt.vocab_size, 'Similarity matrix has incompatible shape'
        self.vocab_size = opt.vocab_size
        M = Variable(torch.from_numpy(M)).cuda()
        self.Sim_Matrix = M
        self.exact = opt.exact_dkl

    def log(self):
        self.logger.info("Initialized Word2 loss tau=%.3f, alpha=%.1f" % (self.tau_word, self.alpha))

    def forward(self, input, target, mask, scores=None):
        # truncate to the same size
        seq_length = input.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        ml_output = get_ml_loss(input, target, mask,
                                norm=self.normalize_batch)
        # Get the similarities of the words in the batch (NxL, V)
        indices = to_contiguous(target).view(-1, 1).squeeze().data
        sim = self.Sim_Matrix[indices]
        # print('raw sim:', sim)
        if self.clip_sim:
            # keep only the similarities larger than the margin
            # self.logger.warn('Clipping the sim')
            sim = sim * sim.ge(self.margin).float()
        if self.limited:
            # self.logger.warn('Limitig smoothing to the gt vocab')
            indices_vocab = get_indices_vocab(target, self.seq_per_img)
            sim = sim.gather(1, indices_vocab)
            input = input.gather(1, indices_vocab)

        if self.tau_word:
            smooth_target = torch.exp(torch.mul(sim, 1/self.tau_word))
        else:
            # Do not exponentiate
            smooth_target = sim
        # Normalize the word reward distribution:
        smooth_target = normalize_reward(smooth_target)

        if self.exact:
            delta = Variable(torch.eye(self.vocab_size)[indices.cpu()]).cuda()
            smooth_target = torch.mul(smooth_target, self.alpha) + torch.mul(delta, (1 - self.alpha))
            # print("Smooth:", smooth_target)

        # Store some stats about the sentences scores:
        scalars = smooth_target.data.cpu().numpy()
        # print("Reward multip:", scalars[0][:10])

        stats = {"word_mean": np.mean(scalars),
                 "word_std": np.std(scalars)}

        # print('smooth_target:', smooth_target)
        # Format
        mask = to_contiguous(mask).view(-1, 1)
        input = to_contiguous(input).view(-1, input.size(2))
        # print('in:', input.size(), 'mask:', mask.size(), 'smooth:', smooth_target.size())
        output = - input * mask.repeat(1, sim.size(1)) * smooth_target

        if self.normalize_batch:
            if torch.sum(mask).data[0] > 0:
                output = torch.sum(output) / torch.sum(mask)
            else:
                self.logger.warn("Smooth targets weights sum to 0")
                output = torch.sum(output)
        else:
            output = torch.sum(output)

        if self.add_entropy:
            H = rows_entropy(smooth_target).unsqueeze(1)
            entropy = torch.sum(H * mask)
            if self.normalize_batch:
                entropy /= torch.sum(mask)
            # print('Entropy:', entropy.data[0])
            output += entropy

        if self.scale_wl:
            if self.scale_wl == -1:
                # dynamic scaling:
                sc = float(ml_output.data.cpu().numpy() / output.data.cpu().numpy())
            else:
                sc = self.scale_wl
            self.logger.warn('Scaling the pure WL RAML by %.3f' % sc)
            output = sc * output
        return ml_output, output, stats

    def track(self, input, target, mask, scores=None):
        """
        input : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        mask : the ground truth mask to ignore UNK tokens (N, seq_length)
        scores: scalars to scale the loss of each sentence (N, 1)
        """
        # truncate to the same size
        # print('Logprobs:', input.size(), "labels:", target.size(), mask.size())
        N = input.size(0)
        seq_length = input.size(1)
        target = target[:, :seq_length]
        indices = to_contiguous(target).view(-1, 1).squeeze().data
        sim = self.Sim_Matrix[indices]
        # print('raw sim:', sim)
        if self.clip_sim:
            # keep only the similarities larger than the margin
            # self.logger.warn('Clipping the sim')
            sim = sim * sim.ge(self.margin).float()
        if self.limited:
            # self.logger.warn('Limitig smoothing to the gt vocab')
            indices_vocab = get_indices_vocab(target, self.seq_per_img)
            sim = sim.gather(1, indices_vocab)
            input = input.gather(1, indices_vocab)

        if self.tau_word:
            smooth_target = torch.exp(torch.mul(sim, 1/self.tau_word))
        else:
            # Do not exponentiate
            smooth_target = sim
        # Normalize the word reward distribution:
        target_d = normalize_reward(smooth_target).data.cpu().numpy()
        input = torch.exp(input).data.cpu().numpy()
        return input, target_d

class SentSmoothCriterion(nn.Module):
    """
    Apply sentence level loss smoothing
    """
    def __init__(self, opt):
        nn.Module.__init__(self)
        self.logger = opt.logger
        self.logger.warn('Sentence level v1')
        self.seq_per_img = opt.seq_per_img
        self.version = opt.loss_version
        self.clip_scores = opt.clip_scores
        # TODO assert type is defined
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha_sent
        assert self.alpha > 0, 'set alpha to a nonzero value, otherwise use the default loss'
        # tau set to zero means no exponentiation
        self.tau_sent = opt.tau_sent
        self.scale_loss = opt.scale_loss
        self.normalize_batch = opt.normalize_batch

    def forward(self, input, target, mask, scores=None):
        # truncate
        seq_length = input.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        ml_output = get_ml_loss(input, target, mask)
        preds = torch.max(input, dim=2)[1].squeeze().cpu().data
        sent_scores = self.get_scores(preds, target)
        # Store some stats about the sentences scores:
        stats = {"sent_mean": np.mean(sent_scores),
                 "sent_std": np.std(sent_scores)}

        sent_scores = np.repeat(sent_scores, seq_length)
        smooth_target = Variable(torch.from_numpy(sent_scores).view(-1, 1)).cuda().float()
        # substitute target with the prediction (aka sampling wrt p_\theta)
        preds = Variable(preds[:, :seq_length]).cuda()
        # Flatten
        preds = to_contiguous(preds).view(-1, 1)
        input = to_contiguous(input).view(-1, input.size(2))
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, preds) * mask * smooth_target
        if self.normalize_batch:
            weights = smooth_target * mask
        else:
            # Only cider_tsent09_a03_nonorm
            weights = mask
        if torch.sum(weights).data[0] > 0:
            output = torch.sum(output) / torch.sum(weights)
        else:
            self.logger.warn('Weights sum to zero')
            output = sum(output)
        return ml_output, output, stats


class SentSmoothCriterion2(nn.Module):
    """
    Apply sentence level loss smoothing
    """
    def __init__(self, opt):
        nn.Module.__init__(self)
        self.logger = opt.logger
        self.logger.warn('Sentence level v2')
        self.seq_per_img = opt.seq_per_img
        self.version = opt.loss_version
        self.clip_scores = opt.clip_scores
        # TODO assert type is defined
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha_sent
        assert self.alpha > 0, 'set alpha to a nonzero value, otherwise use the default loss'
        # tau set to zero means no exponentiation
        self.tau_sent = opt.tau_sent
        self.scale_loss = opt.scale_loss

    def forward(self, input, target, mask, scores=None):
        # truncate
        batch_size = input.size(0)
        seq_length = input.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        ml_output = get_ml_loss(input, target, mask)
        preds = torch.max(input, dim=2)[1].squeeze().cpu().data
        sent_scores = self.get_scores(preds, target)

        # scale scores:
        sent_scores = np.array(sent_scores)
        stats = {"sent_mean": np.mean(sent_scores),
                 "sent_std": np.std(sent_scores)}
        # sent_scores from (N, 1) to (N, seq_length)
        # sent_scores = np.repeat(sent_scores, seq_length)
        # smooth_target = Variable(torch.from_numpy(sent_scores).view(-1, 1)).cuda().float()
        # substitute target with the prediction (aka sampling wrt p_\theta)
        preds = Variable(preds[:, :seq_length]).cuda()
        # Flatten
        preds = to_contiguous(preds).view(-1, 1)
        input = to_contiguous(input).view(-1, input.size(2))
        flat_mask = to_contiguous(mask).view(-1, 1)
        logprob = input.gather(1, preds) * flat_mask
        logprob = logprob.view(batch_size, seq_length)
        logprob = torch.sum(logprob, dim=1).unsqueeze(1) / seq_length
        # print('Logprobs', logprob.size(), logprob.data)
        # print('sent scores:', sent_scores)
        importance = Variable(torch.from_numpy(sent_scores).view(-1, 1)).cuda().float()
        # print('importance:', importance)
        importance = importance / torch.exp(logprob).float()
        # print('Importance:', importance)
        if self.sentence_version == 2:
            output = torch.sum(importance * torch.log(importance)) / batch_size
        elif self.sentence_version == 3:
            output = - torch.sum(importance * logprob) / batch_size
        return ml_output, output, stats



class WordSentSmoothCriterion(nn.Module):
    """
    Combine a word level smothing with sentence scores
    """
    def __init__(self, opt):
        nn.Module.__init__(self)
        self.logger = opt.logger
        self.seq_per_img = opt.seq_per_img
        self.version = opt.loss_version
        self.scale_loss = opt.scale_loss
        self.clip_sim = opt.clip_sim
        if self.clip_sim:
            self.margin = opt.margin
            self.logger.warn('Clipping similarities below %.2f' % self.margin)
        self.limited = opt.limited_vocab_sim
        # the final loss is (1-alpha) ML + alpha * RewardLoss
        self.alpha = opt.alpha_sent  # FIXME
        assert self.alpha > 0, 'set alpha to a nonzero value, otherwise use the default loss'
        self.tau_sent = opt.tau_sent
        self.tau_word = opt.tau_word
        # Load the similarity matrix:
        M = pickle.load(open(opt.similarity_matrix, 'rb'), encoding='iso-8859-1')
        M = M.astype(np.float32)
        n, d = M.shape
        assert n == d and n == opt.vocab_size, \
                'Similarity matrix has incompatible shape %d x %d \
                whilst vocab size is %d' % (n, d, opt.vocab_size)
        M = Variable(torch.from_numpy(M)).cuda()
        self.Sim_Matrix = M

    def forward(self, input, target, mask, scores=None):
        # truncate
        seq_length = input.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        ml_output = get_ml_loss(input, target, mask)
        # Sentence level
        preds = torch.max(input, dim=2)[1].squeeze().cpu().data
        sent_scores = self.get_scores(preds, target)
        # Store some stats about the sentences scores:
        stats = {"sent_mean": np.mean(sent_scores),
                 "sent_std": np.std(sent_scores)}

        sent_scores = np.repeat(sent_scores, seq_length)
        smooth_target = Variable(torch.from_numpy(sent_scores).view(-1, 1)).cuda().float()

        # Word level
        preds = Variable(preds[:, :input.size(1)]).cuda()
        preds = to_contiguous(preds).view(-1, 1)
        sim = self.Sim_Matrix[preds.squeeze().data]
        if self.tau_word:
            smooth_target_wl = torch.exp(torch.mul(torch.add(sim, -1.), 1/self.tau_word))
        else:
            smooth_target_wl = torch.add(sim, -1.)
        scalars = smooth_target_wl.data.cpu().numpy()[:]
        stats["word_mean"] = np.mean(scalars)
        stats["word_std"] = np.std(scalars)

        mask_wl = mask.repeat(1, sim.size(1))
        # format the sentence scores
        smooth_target = smooth_target.repeat(1, sim.size(1))
        output_wl = - input * smooth_target_wl * mask_wl * smooth_target
        norm = torch.sum(smooth_target_wl * mask_wl * smooth_target)
        if norm.data[0] > 0:
            output = torch.sum(output_wl) / norm
        else:
            self.logger.warn("Smooth targets weights sum to 0")
            output = torch.sum(output_wl)
        return ml_output, output, stats


class RewardCriterion(SentSmoothCriterion2,
                      SentSmoothCriterion,
                      WordSentSmoothCriterion):
    def __init__(self, opt):
        if 'word' in opt.loss_version:
            WordSentSmoothCriterion.__init__(self, opt)
        else:
            if opt.sentence_loss_version == 1:
                SentSmoothCriterion.__init__(self, opt)
            else:
                SentSmoothCriterion2.__init__(self, opt)

        self.sentence_version = opt.sentence_loss_version

    def forward(self, input, target, mask, scores=None):
        if 'word' in self.version:
            return WordSentSmoothCriterion.forward(self, input, target, mask, scores)
        else:
            if self.sentence_version == 1:
                return SentSmoothCriterion.forward(self, input, target, mask, scores)
            else:
                return SentSmoothCriterion2.forward(self, input, target, mask, scores)


class AllIsGoodCriterion(RewardCriterion):
    def __init__(self, opt, vocab):
        RewardCriterion.__init__(self, opt)

    def get_scores(self, preds, target):
        return np.ones(target.size(0))


class CiderRewardCriterion(RewardCriterion):
    def __init__(self, opt, vocab):
        RewardCriterion.__init__(self, opt)
        self.vocab = vocab
        DF = pickle.load(open(opt.cider_df, 'rb'),  encoding="iso-8859-1")
        if isinstance(DF, dict):
            self.DF = DF['freq']
            self.DF_len = DF['length']
        else:
            self.DF = DF
            self.DF_len = 40504

    def log(self):
        self.logger.info('CIDEr sampleP loss, tau=%.3f & alpha=%.2f' % (self.tau_sent, self.alpha))

    def get_scores(self, preds, target):
        # The reward loss:
        cider_scorer = CiderScorer(n=4, sigma=6)
        # Go to sentence space to compute scores:
        hypo = decode_sequence(self.vocab, preds)  # candidate
        refs = decode_sequence(self.vocab, target.data)  # references
        num_img = target.size(0) // self.seq_per_img
        for e, h in enumerate(hypo):
            ix_start = e // self.seq_per_img * self.seq_per_img
            ix_end = ix_start + 5  # self.seq_per_img
            cider_scorer += (h, refs[ix_start : ix_end])
        (score, scores) = cider_scorer.compute_score(df_mode=self.DF,
                                                     df_len=self.DF_len)
        self.logger.debug("CIDEr score: %s" %  str(scores))
        # scale scores:
        scores = np.array(scores)
        if self.clip_scores:
            scores = np.clip(scores, 0, 1) - 1
        # Process scores:
        if self.tau_sent:
            scores = np.exp(scores / self.tau_sent)
        if not np.sum(scores):
            self.logger.warn('Adding +1 to the zero scores')
            scores += 1
        self.logger.warn('Scores after processing: %s' % str(scores))
        return scores


class BleuRewardCriterion(RewardCriterion):
    def __init__(self, opt, vocab):
        RewardCriterion.__init__(self, opt)
        self.vocab = vocab
        self.bleu_order = int(self.version[-1])
        self.bleu_scorer = opt.bleu_version
        assert self.bleu_scorer in ['coco', 'soft'], "Unknown bleu scorer %s" % self.bleu_scorer

    def get_scores(self, preds, target):
        if self.bleu_scorer == 'coco':
            bleu_scorer = BleuScorer(n=self.bleu_order)
            coco = True
        else:
            coco = False
            scores = []
        # Go to sentence space to compute scores:
        hypo = decode_sequence(self.vocab, preds)  # candidate
        refs = decode_sequence(self.vocab, target.data)  # references
        num_img = target.size(0) // self.seq_per_img
        for e, h in enumerate(hypo):
            ix_start =  e // self.seq_per_img * self.seq_per_img
            ix_end = ix_start + 5  # self.seq_per_img
            if coco:
                bleu_scorer += (h, refs[ix_start : ix_end])
            else:
                scores.append(sentence_bleu(h, ' '.join(refs[ix_start: ix_end]),
                                            order=self.bleu_order))
        if coco:
            (score, scores) = bleu_scorer.compute_score()
            scores = scores[-1]
        self.logger.debug("Bleu scores: %s" %  str(scores))
        # scale scores:
        scores = np.array(scores)
        if self.clip_scores:
            scores = np.clip(scores, 0, 1) - 1
        # Process scores:
        if self.tau_sent:
            scores = np.exp(scores / self.tau_sent)
        if not np.sum(scores):
            self.logger.warn('Adding +1 to the zero scores')
            scores += 1
        self.logger.warn('Scores after processing: %s' % str(scores))

        return scores


class InfersentRewardCriterion(RewardCriterion):
    def __init__(self, opt, vocab):
        RewardCriterion.__init__(self, opt)
        self.vocab = vocab
        self.logger.info('loading the infersent pretrained model')
        glove_path = '../infersent/dataset/glove/glove.840b.300d.txt'
        self.infersent = torch.load('../infersent/infersent.allnli.pickle',
                                    map_location=lambda storage, loc: storage)
        self.infersent.set_glove_path(glove_path)
        self.infersent.build_vocab_k_words(k=100000)
        # freeze infersent params:
        for p in self.infersent.parameters():
            p.requires_grad = false

    def get_scores(self, preds, target):
        hypo = decode_sequence(self.vocab, preds)  # candidate
        refs = decode_sequence(self.vocab, target.data)  # references
        num_img = target.size(0) // self.seq_per_img
        scores = []
        lr = len(refs)
        codes = self.infersent.encode(refs + hypo)
        refs = codes[:lr]
        hypo = codes[lr:]
        for e, h in enumerate(hypo):
            ix_start =  e // self.seq_per_img * self.seq_per_img
            ix_end = ix_start + 5  # self.seq_per_img
            scores.append(group_similarity(h, refs[ix_start : ix_end]))
        self.logger.debug("infersent similairities: %s" %  str(scores))
        # scale scores:
        scores = np.array(scores)
        if self.clip_scores:
            scores = np.clip(scores, 0, 1) - 1
        # Process scores:
        if self.tau_sent:
            scores = np.exp(scores / self.tau_sent)
        if not np.sum(scores):
            self.logger.warn('Adding +1 to the zero scores')
            scores += 1
        self.logger.warn('Scores after processing: %s' % str(scores))

        return scores


class HammingRewardCriterion(RewardCriterion):
    def __init__(self, opt):
        RewardCriterion.__init__(self, opt)
        self.vocab_size = opt.vocab_size
        assert self.tau_sent > 0, "Hamming requires exponentiation"

    def log(self):
        self.logger.info('Hamming sampleP loss, tau=%.3f & alpha=%.2f' % (self.tau_sent, self.alpha))

    def get_scores(self, preds, target):
        seq_length = target.size(1)
        refs = target.cpu().data.numpy()
        # Hamming distances
        scores = np.array([- hamming(u, v) for u, v in zip(preds.numpy(), refs)])
        # turn r into a reward distribution:
        # check if hamming distance should be simply count of mismatches.
        self.logger.debug("Negative hamming distances: %s" %  str(scores))
        # scale scores:
        scores = np.array(scores)
        # Process scores:
        scores = np.exp(scores / self.tau_sent)
        # Normalizing:
        # v = np.unique(target.data.cpu().numpy())
        # print('vocab indices: (size):', len(v), v)
        # Z = hamming_Z(seq_length // 2, len(v), self.tau_sent)
        # print('Sum of rewards:', Z)
        # scores /= Z
        self.logger.warn('Scores after processing: %s' % str(scores))
        return scores


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
        # print('Training:', self.training)
        if self.combine_loss:
            # Instead of ML(sampled) return WL(sampled)
            self.loss_sampled = WordSmoothCriterion2(opt)
            # self.loss_sampled.alpha = .7
            self.loss_gt = WordSmoothCriterion2(opt)

    def forward(self, model, fc_feats, att_feats, labels, mask, scores=None):
        # truncate
        input = model.forward(fc_feats, att_feats, labels)
        target = labels[:, 1:]
        N = input.size(0)
        seq_length = input.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        if self.combine_loss:
            ml_gt, wl_gt, stats_gt = self.loss_gt(input, target, mask)
            loss_gt = self.loss_gt.alpha * wl_gt + (1 - self.loss_gt.alpha) * ml_gt
        else:
            ml_gt = get_ml_loss(input, target, mask)
            loss_gt = ml_gt
        if self.training:
            MC = self.mc_samples
        else:
            MC = 1
        for ss in range(MC):
            preds_matrix, stats = self.alter(target)
            # gt_s = decode_sequence(self.vocab, preds_matrix.data[:, 1:])
            # gt = decode_sequence(self.vocab, labels.data[:, 1:])
            # for s, ss in zip(gt, gt_s):
                # print('GT:', s, '\nSA:', ss)
            # Forward the sampled captions
            sample_input = model.forward(fc_feats, att_feats, preds_matrix)
            # print('Sample input:', sample_input.size(), 'labels:', preds_matrix.size())
            if model.opt.caption_model in ['adaptive_attention', 'top_down']:
                sample_input = sample_input[:, :-1]
            if self.combine_loss:
                ml_sampled, wl_sampled, stats_sampled = self.loss_sampled(sample_input, preds_matrix[:, 1:], mask)
                stats.update(stats_sampled)
                mc_output = self.loss_sampled.alpha * wl_sampled + (1 - self.loss_sampled.alpha) * ml_sampled
            else:
                ml_sampled = get_ml_loss(sample_input, preds_matrix[:, 1:], mask)
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
        print('Altering:', labels.size())
        N = labels.size(0)
        seq_length = labels.size(1)
        # get batch vocab size
        refs = labels.cpu().data.numpy()
        if self.limited:
            batch_vocab = np.delete(np.unique(refs), 0)
            lv = len(batch_vocab)
        else:
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
        if self.limited:
            select_index = np.random.choice(batch_vocab, size=(N, select))
        else:
            select_index = np.random.randint(low=4, high=self.vocab_size, size=(N, select))
        # print("Selected:", select_index)
        preds[rows, change_index] = select_index
        preds_matrix = np.hstack((np.zeros((N, 1)), preds))  # padd <BOS>
        preds_matrix = Variable(torch.from_numpy(preds_matrix)).cuda().type_as(labels)
        print('yielding:', preds_matrix.size())
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

class MultiLanguageModelCriterion(nn.Module):
    def __init__(self, seq_per_img=5):
        super(MultiLanguageModelCriterion, self).__init__()
        self.seq_per_img = seq_per_img

    def forward(self, input, target, mask):
        # truncate to the same size
        max_length = input.size(1)
        num_img = input.size(0) // self.seq_per_img
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        mask_ = mask
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        real_output = torch.sum(output) / torch.sum(mask)
        # ------------------------------------------------
        output = output.view(-1, max_length)
        sent_scores = output.sum(dim=1) / mask_.sum(dim=1)
        sent_scores_per_image = sent_scores.chunk(num_img)
        output = torch.sum(torch.cat([t.max() for t in sent_scores_per_image], dim=0))
        output = output / num_img
        return real_output, output


class DataAugmentedCriterion(nn.Module):
    """
    Treat the augmented captions separately
    """
    def __init__(self, opt):
        super(DataAugmentedCriterion, self).__init__()
        self.opt = opt
        self.beta = opt.beta
        self.seq_per_img = opt.seq_per_img
        assert self.seq_per_img > 5, 'Captions per image is seq than 5'
        # The GT loss
        if opt.gt_loss_version == 'word':
            self.crit_gt = WordSmoothCriterion(opt)
        else:
            # The usual ML
            self.crit_gt = LanguageModelCriterion(opt)
            # Ensure loss scaling with the imprtance sampling ratios
            self.crit_gt.scale_loss = 0

        # The augmented loss
        if opt.augmented_loss_version == 'word':
            self.crit_augmented = WordSmoothCriterion(opt)
        else:
            # The usual ML
            self.crit_augmented = LanguageModelCriterion(opt)
            # Ensure loss scaling with the imprtance sampling ratios
            self.crit_augmented.scale_loss = 1

    def forward(self, input, target, mask, scores):
        seq_length = input.size(1)
        batch_size = input.size(0)
        # truncate
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        # Separate gold from augmented
        num_img = batch_size // self.seq_per_img
        input_per_image = input.chunk(num_img)
        mask_per_image = mask.chunk(num_img)
        target_per_image = target.chunk(num_img)
        scores_per_image = scores.chunk(num_img)

        input_gt = torch.cat([t[:5] for t in input_per_image], dim=0)
        target_gt = torch.cat([t[:5] for t in target_per_image], dim=0)
        mask_gt = torch.cat([t[:5] for t in mask_per_image], dim=0)

        input_gen = torch.cat([t[5:] for t in input_per_image], dim=0)
        target_gen = torch.cat([t[5:] for t in target_per_image], dim=0)
        mask_gen = torch.cat([t[5:] for t in mask_per_image], dim=0)
        scores_gen = torch.cat([t[5:] for t in scores_per_image], dim=0)
        # print('Splitted data:', input_gt.size(), target_gt.size(), mask_gt.size(),
              # 'gen:', input_gen.size(), target_gen.size(), mask_gen.size(), scores_gen.size())

        # For the first 5 captions per image (gt) compute LM
        _, output_gt, stats_gt = self.crit_gt(input_gt, target_gt, mask_gt)

        # For the rest of the captions: importance sampling
        _, output_gen, stats_gen = self.crit_augmented(input_gen, target_gen, mask_gen, scores_gen)
        # TODO check if must combine with ml augmented as well
        stats = {}
        if stats_gen:
            for k in stats_gen:
                stats['gen_'+k] = stats_gen[k]
        if stats_gt:
            for k in stats_gt:
                stats['gt_'+k] = stats_gt[k]
        return output_gt, self.beta * output_gen + (1 - self.beta) * output_gt, stats


class PairsLanguageModelCriterion(nn.Module):
    def __init__(self, opt):
        super(PairsLanguageModelCriterion, self).__init__()
        self.seq_per_img = opt.seq_per_img

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        #  print "target:", target
        #  print "mask:", mask
        # duplicate
        num_img = input.size(0) // self.seq_per_img
        input_per_image = input.chunk(num_img)
        input = torch.cat([t.repeat(self.seq_per_img, 1, 1) for t in input_per_image], dim=0)
        target = torch.unsqueeze(target, 0)
        target = target.permute(1, 0, 2)
        target = target.repeat(1, self.seq_per_img, 1)
        target = target.resize(target.size(0) * target.size(1), target.size(2))
        mask = mask[:, :input.size(1)]
        mask = torch.unsqueeze(mask, 0)
        mask = mask.permute(1, 0, 2)
        mask = mask.repeat(1, self.seq_per_img, 1)
        mask = mask.resize(mask.size(0) * mask.size(1), mask.size(2))
        #  print "target:", target
        #  print "mask:", mask
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output, output


