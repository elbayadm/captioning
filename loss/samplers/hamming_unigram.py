"""
Stratified sampler for the reward associated to
the negative hamming distance
Sampling w.r.t the unigram distribution
"""
import math
import numpy as np
from scipy.misc import comb
import pickle
import torch
from torch.autograd import Variable
from utils import pl

def normalize_reward(distrib):
    """
    Normalize so that each row sum to one
    """
    if isinstance(distrib, np.ndarray):
        l1 = distrib.sum(axis=1)
        distrib /= l1.reshape(len(distrib), 1)
        return distrib
    else:
        sums = torch.sum(distrib, dim=1).unsqueeze(1)
        return distrib / sums.repeat(1, distrib.size(1))


def hamming_distrib_soft(m, v, tau):
    x = [np.log(comb(m, d, exact=False)) + d * np.log(v) -
         d/tau * np.log(v) - d/tau for d in range(m + 1)]
    x = np.array(x)
    p = np.exp(x)
    p /= np.sum(p)
    return p


def hamming_distrib(m, v, tau):
    x = [comb(m, d, exact=False) * (v-1)**d * math.exp(-d/tau) for d in range(m+1)]
    x = np.absolute(np.array(x))  # FIXME negative values occuring !!
    Z = np.sum(x)
    return x/Z, Z


def hamming_Z(m, v, tau):
    pd = hamming_distrib(m, v, tau)
    popul = v ** m
    Z = np.sum(pd * popul * np.exp(-np.arange(m+1)/tau))
    return np.clip(Z, a_max=1e30, a_min=1)


class HammingUnigramSampler(object):
    """
    Sample a hamming distance and alter the truth wrt to words similarity
    """
    def __init__(self, opt):
        self.seq_per_img = opt.seq_per_img
        self.vocab_size = opt.vocab_size
        if opt.stratify_reward:
            # sampler = r
            self.tau = opt.tau_sent
            self.prefix = 'rhamm_sim'
        else:
            # sampler = q
            self.tau = opt.tau_sent_q
            self.prefix = 'qhamm_sim'
        # substitution options
        self.limited = opt.limited_vocab_sub
        self.tau_word = opt.tau_word
        self.unigram_disrtib = pl('data/coco/unigram_coco.pkl')[0]  # FIXME define in opts and resave in 1D
        self.version = 'Hamming-Unigram (Vpool=%d, tau=%.2f)' % (self.limited, self.tau)

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

        # Format preds by changing d=select tokens
        preds = refs
        # choose tokens to replace
        change_index = np.random.randint(seq_length, size=(batch_size, select))
        rows = np.arange(batch_size).reshape(-1, 1).repeat(select, axis=1)
        # select substitutes
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

        # Select subs:
        if self.limited == 1:
            select_index = np.random.choice(batch_vocab,
                                            size=(batch_size, select),
                                            p=self.unigram_disrtib[batch_vocab])
        elif self.limited == 2:
            select_index = np.vstack([np.random.choice(chunk,
                                                       size=(self.seq_per_img, select),
                                                       p=self.unigram_disrtib[chunk])
                                      for chunk in im_vocab])
        else:
            select_index = np.random.choice(self.vocab_size,
                                            size=(batch_size, select),
                                            p=self.unigram_disrtib)

        preds[rows, change_index] = select_index
        preds_matrix = np.hstack((np.zeros((batch_size, 1)), preds))  # padd <BOS>
        preds_matrix = Variable(torch.from_numpy(preds_matrix)).cuda().type_as(labels)
        return preds_matrix, np.ones(batch_size) * score, stats



