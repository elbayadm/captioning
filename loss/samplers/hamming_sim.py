"""
Stratified sampler for the reward associated to
the negative hamming distance
"""
import math
import numpy as np
from scipy.misc import comb
import pickle
import torch
from torch.autograd import Variable


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


class HammingSimSampler(object):
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
        M = pickle.load(open(opt.similarity_matrix, 'rb'), encoding='iso-8859-1')
        if opt.rare_tfidf:
            IDF = pickle.load(open('data/coco/idf_coco_01.pkl', 'rb'))
            M -= self.tau_word * opt.rare_tfidf * IDF
        M = M.astype(np.float32)
        n, d = M.shape
        print('Sim matrix:', n, 'x', d, ' V=', opt.vocab_size)
        assert n == d and n == opt.vocab_size, 'Similarity matrix has incompatible shape'
        self.words_distribs = M
        self.version = 'Hamming-Sim (Vpool=%d, tau=%.2f)' % (self.limited, self.tau)

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
        if self.limited == 1:
            # get distributions of selected tokens and limit to the batch_vocab
            indices = preds[rows, change_index].flatten()
            p = self.words_distribs[indices][:, batch_vocab]
            if self.tau_word:
                p = np.exp(p/self.tau_word)
            p = normalize_reward(p)
            select_index = np.zeros((batch_size, select))
            for i in range(batch_size):
                for ss in range(select):
                    select_index[i, ss] = np.random.choice(batch_vocab, p=p[i * select + ss])

        elif self.limited == 2:
            # get distributions of selected tokens and limit to the batch_vocab
            indices = preds[rows, change_index]
            p = [self.words_distribs[indices[i]][:, im_vocab[i // self.seq_per_img]] for i in range(batch_size)]
            # print("distance:", select, "indices:", indices[0], "p item shape:", p[0].shape, "im vocab:", im_vocab[0].shape)
            if self.tau_word:
                p = [np.exp(pp/self.tau_word) for pp in p]
            p = [normalize_reward(pp) for pp in p]
            select_index = np.zeros((batch_size, select))
            for i in range(batch_size):
                for ss in range(select):
                    select_index[i, ss] = np.random.choice(im_vocab[i // self.seq_per_img], p=p[i][ss])
        else:
            select_index = np.random.randint(low=4, high=self.vocab_size, size=(batch_size, select))
        preds[rows, change_index] = select_index
        preds_matrix = np.hstack((np.zeros((batch_size, 1)), preds))  # padd <BOS>
        preds_matrix = Variable(torch.from_numpy(preds_matrix)).cuda().type_as(labels)
        return preds_matrix, np.ones(batch_size) * score, stats



