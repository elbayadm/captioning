import random
import math
from collections import OrderedDict
import pickle
import numpy as np
import torch
from torch.autograd import Variable
from .utils import hamming_distrib, to_contiguous


def init_sampler(select, opt):
    # sv = opt.sent_sampler.lower()
    if select == 'greedy':
        sampler = GreedySampler()
    elif select == 'hamming':
        sampler = HammingSampler(opt)
    elif select == 'tfifd':
        sampler = TFIDFSampler(opt)
    else:
        raise ValueError('Unknonw sampler %s' % select)
    return sampler


class GreedySampler(object):
    """
    sampling from p_\theta greedily
    """
    def __init__(self):
        self.version = 'greedy p_theta'

    def sample(self, logp, labels):
        # greedy decoding q=p_\theta
        batch_size = logp.size(0)
        seq_length = logp.size(1)
        vocab_size = logp.size(2)
        # TODO add sampling
        sampled = torch.max(logp, dim=2)[1].squeeze().cpu().data.numpy()
        # get p_\theta(\tilde y| x, y*)
        # Flatten
        sampled_flat = sampled.reshape((-1, 1))
        logp_sampled_greedy = to_contiguous(logp).view(-1, vocab_size).cpu().data.numpy()
        logp_sampled_greedy = np.take(logp_sampled_greedy, sampled_flat)
        logp_sampled_greedy = logp_sampled_greedy.reshape(batch_size, seq_length).mean(axis=1)
        cand_probs = np.exp(logp_sampled_greedy)
        stats = {"qpt_mean": np.mean(logp_sampled_greedy),
                 "qpt_std": np.std(logp_sampled_greedy)}

        sampled = np.hstack((np.zeros((batch_size, 1)), sampled))  # pad <BOS>
        sampled = Variable(torch.from_numpy(sampled)).cuda().type_as(labels)
        return sampled, cand_probs, stats


class HammingSampler(object):
    """
    Sample a hamming distance and alter the truth
    """
    def __init__(self, opt):
        self.limited = opt.limited_vocab_sub
        self.seq_per_img = opt.seq_per_img
        self.vocab_size = opt.vocab_size
        if opt.stratify_reward:
            # sampler = r
            self.tau = opt.tau_sent
            self.prefix = 'rhamm'
        else:
            # sampler = q
            self.tau = opt.tau_sent_q
            self.prefix = 'qhamm'
        self.version = 'Hamming (Vpool=%d, tau=%.2f)' % (self.limited, self.tau)

    def sample(self, logp, labels):
        """
        Sample ~y given y*
        return ~y and r(~y|y*)
        """
        batch_size = labels.size(0)
        seq_length = labels.size(1)
        # get batch vocab size
        refs = labels.cpu().data.numpy()
        if self.limited == 1:  # In-batch vocabulary substitution
            batch_vocab = np.delete(np.unique(refs), 0)
            lv = len(batch_vocab)
        elif self.limited == 2:  # In-image vocabulary substitution
            num_img = batch_size // self.seq_per_img
            refs_per_image = np.split(refs, num_img)
            im_vocab = [np.delete(np.unique(chunk), 0) for chunk in refs_per_image]
            del refs_per_image
            lv = np.max([len(chunk) for chunk in im_vocab])
        else:  # Full vocabulary substitution
            lv = self.vocab_size
        distrib, Z = hamming_distrib(seq_length, lv, self.tau)
        # Sample a distance i.e. a reward
        select = np.random.choice(a=np.arange(seq_length + 1),
                                  p=distrib)
        score = math.exp(-select / self.tau) / Z
        stats = {"%s_mean" % self.prefix: score,
                 "%s_std" % self.prefix: 0}

        # Format preds by changing d=select tokens at random
        preds = refs
        # choose tokens to replace
        change_index = np.random.randint(seq_length, size=(batch_size, select))
        rows = np.arange(batch_size).reshape(-1, 1).repeat(select, axis=1)
        # select substitutes
        if self.limited == 1:
            select_index = np.random.choice(batch_vocab, size=(batch_size, select))
        elif self.limited == 2:
            select_index = np.vstack([np.random.choice(chunk,
                                                       size=(self.seq_per_img, select))
                                      for chunk in im_vocab])
        else:
            select_index = np.random.randint(low=4, high=self.vocab_size, size=(batch_size, select))
        preds[rows, change_index] = select_index
        preds_matrix = np.hstack((np.zeros((batch_size, 1)), preds))  # padd <BOS>
        preds_matrix = Variable(torch.from_numpy(preds_matrix)).cuda().type_as(labels)
        return preds_matrix, np.ones(batch_size) * score, stats


class TFIDFSampler(object):
    """
    Alter an n-gram
    """
    def __init__(self, opt):
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
        self.version = 'TFIDF, n=%d, rare=%d, tau=%.2e' % (self.n, self.select_rare, self.tau)

    def sample(self, logp, labels):
        batch_size = labels.size(0)
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
            change_index = np.zeros((batch_size, 1), dtype=np.int32)
            for i in range(batch_size):
                p = np.array([self.ngrams.get(tuple(refs[i, j:j+ng]), 1)
                              for j in range(seq_length - ng)])
                p = 1/p
                p /= np.sum(p)
                change_index[i] = np.random.choice(seq_length - ng,
                                                   p=p,
                                                   size=1)
        else:
            change_index = np.random.randint(seq_length - ng, size=(batch_size, 1))
        change_index = np.hstack((change_index + k for k in range(ng)))
        rows = np.arange(batch_size).reshape(-1, 1).repeat(ng, axis=1)
        # select substitutes from occuring n-grams in the training set:
        if self.select_rare:
            picked = np.random.choice(np.arange(len(self.ngrams)),
                                      p=list(self.ngrams.values()),
                                      size=(batch_size,))
            picked_ngrams = [list(self.ngrams)[k] for k in picked]
        else:
            picked_ngrams = random.sample(self.ngrams, batch_size)
        preds[rows, change_index] = picked_ngrams
        preds_matrix = np.hstack((np.zeros((batch_size, 1)), preds))  # padd <BOS>
        preds_matrix = Variable(torch.from_numpy(preds_matrix)).cuda().type_as(labels)
        return preds_matrix, np.ones(batch_size) * score, stats


