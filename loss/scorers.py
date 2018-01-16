import numpy as np
import pickle
from .utils import decode_sequence
from .cider_scorer import CiderScorer
from .metrics import sentence_bleu


def init_scorer(method, opt, vocab):
    if method == 'constant':
        scorer = AllIsGoodScorer()
    elif method == 'cider':
        scorer = CiderRewardScorer(opt, vocab)
    elif 'bleu' in method:
        scorer = BleuRewardScorer(opt, vocab)
    else:
        raise ValueError('Unknown reward %s' % method)
    return scorer


class AllIsGoodScorer(object):
    """
    constant scores
    """

    def __init__(self):
        self.version = "constant"

    def get_scores(self, preds, target):
        return np.ones(target.size(0))


class CiderRewardScorer(object):
    """
    Evaluate CIDEr scores of given sentences wrt gt
    TODO : write scorer without decoding using only indices
    """

    def __init__(self, opt, vocab):
        self.vocab = vocab
        self.seq_per_img = opt.seq_per_img
        self.clip_reward = opt.clip_reward
        self.tau_sent = opt.tau_sent
        doc_frequency = pickle.load(open(opt.cider_df, 'rb'),
                                    encoding="iso-8859-1")
        if isinstance(doc_frequency, dict):
            self.doc_frequency = doc_frequency['freq']
            self.doc_frequency_len = doc_frequency['length']
        else:
            self.doc_frequency = doc_frequency
            self.doc_frequency_len = 40504
        self.version = 'CIDEr (tau=%.2f)' % self.tau_sent

    def get_scores(self, preds, target):
        # The reward loss:
        cider_scorer = CiderScorer(n=4, sigma=6)
        # Go to sentence space to compute scores:
        # FIXME test if numpy, variable or tensor
        hypo = decode_sequence(self.vocab, preds.data)  # candidate
        refs = decode_sequence(self.vocab, target.data)  # references
        for e, h in enumerate(hypo):
            ix_start = e // self.seq_per_img * self.seq_per_img
            ix_end = ix_start + 5  # self.seq_per_img
            cider_scorer += (h, refs[ix_start: ix_end])
            # print('hypo:', h)
            # print('refs:', refs[ix_start: ix_end])

        (score, scores) = cider_scorer.compute_score(df_mode=self.doc_frequency,
                                                     df_len=self.doc_frequency_len)
        # scale scores:
        scores = np.array(scores)
        rstats = {"rcider_raw_mean": np.mean(scores),
                  "rcider_raw_std": np.std(scores)}
        # if self.clip_reward:
        scores = np.clip(scores, 0, 1) - 1
        # Process scores:
        if self.tau_sent:
            scores = np.exp(scores / self.tau_sent)
        if not np.sum(scores):
            print('All scores == 0')
            scores += 1
        rstats["rcider_mean"] = np.mean(scores)
        rstats['rcider_std'] = np.std(scores)
        return scores, rstats


class BleuRewardScorer(object):
    """
    Evaluate Bleu scores of given sentences wrt gt
    """
    def __init__(self, opt, vocab):
        self.version = 'bleu'
        self.vocab = vocab
        self.bleu_order = int(opt.sent_reward[-1])  # FIXME check opts
        self.seq_per_img = opt.seq_per_img
        self.clip_reward = opt.clip_reward
        self.tau_sent = opt.tau_sent

    def get_scores(self, preds, target):
        scores = []
        # Go to sentence space to compute scores:
        hypo = decode_sequence(self.vocab, preds)  # candidate
        refs = decode_sequence(self.vocab, target.data)  # references
        for e, h in enumerate(hypo):
            ix_start = e // self.seq_per_img * self.seq_per_img
            ix_end = ix_start + 5  # self.seq_per_img
            scores.append(sentence_bleu(h, ' '.join(refs[ix_start: ix_end]),
                                        order=self.bleu_order))
        # scale scores:
        scores = np.array(scores)
        if self.clip_reward:
            scores = np.clip(scores, 0, 1) - 1
        # Process scores:
        if self.tau_sent:
            scores = np.exp(scores / self.tau_sent)
        if not np.sum(scores):
            scores += 1

        return scores



