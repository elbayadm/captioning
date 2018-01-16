import sys
import pickle
import math
import numpy as np
from collections import Counter
from scipy.spatial.distance import hamming
import torch
import torch.nn as nn
from torch.autograd import Variable
from .utils import to_contiguous, decode_sequence, get_ml_loss

sys.path.append("coco-caption")
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.bleu.bleu_scorer import BleuScorer


def group_similarity(u, refs):
    sims = []
    for v in refs:
        sims.append(1 + np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return np.mean(sims)


def sentence_bleu(hypothesis, reference, smoothing=True, order=4, **kwargs):
    """
    Compute sentence-level BLEU score between a translation hypothesis and a reference.

    :param hypothesis: list of tokens or token ids
    :param reference: list of tokens or token ids
    :param smoothing: apply smoothing (recommended, especially for short sequences)
    :param order: count n-grams up to this value of n.
    :param kwargs: additional (unused) parameters
    :return: BLEU score (float)
    """
    log_score = 0

    if len(hypothesis) == 0:
        return 0

    for i in range(order):
        hyp_ngrams = Counter(zip(*[hypothesis[j:] for j in range(i + 1)]))
        ref_ngrams = Counter(zip(*[reference[j:] for j in range(i + 1)]))

        numerator = sum(min(count, ref_ngrams[bigram]) for bigram, count in hyp_ngrams.items())
        denominator = sum(hyp_ngrams.values())

        if smoothing:
            numerator += 1
            denominator += 1

        score = numerator / denominator

        if score == 0:
            log_score += float('-inf')
        else:
            log_score += math.log(score) / order

    bp = min(1, math.exp(1 - len(reference) / len(hypothesis)))

    return math.exp(log_score) * bp


class SentSmoothCriterion(nn.Module):  # the correct one
    """
    Apply sentence level loss smoothing with importance sampling q=p_\theta
    """
    def __init__(self, opt):
        nn.Module.__init__(self)
        self.logger = opt.logger
        self.seq_per_img = opt.seq_per_img
        self.version = opt.loss_version
        self.clip_scores = opt.clip_scores
        self.penalize_confidence = opt.penalize_confidence  #FIXME
        self.alpha = opt.alpha_sent
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
        ml_output = get_ml_loss(input, target, mask, penalize=self.penalize_confidence)
        preds = torch.max(input, dim=2)[1].squeeze().cpu().data
        sent_scores = self.get_scores(preds, target)

        # scale scores:
        sent_scores = np.array(sent_scores)
        stats = {"sent_mean": np.mean(sent_scores),
                 "sent_std": np.std(sent_scores)}
        preds = Variable(preds[:, :seq_length]).cuda()
        # Flatten
        preds = to_contiguous(preds).view(-1, 1)
        input = to_contiguous(input).view(-1, input.size(2))
        flat_mask = to_contiguous(mask).view(-1, 1)
        logprob = input.gather(1, preds) * flat_mask
        logprob = logprob.view(batch_size, seq_length)
        logprob = torch.sum(logprob, dim=1).unsqueeze(1) / seq_length
        importance = Variable(torch.from_numpy(sent_scores).view(-1, 1)).cuda().float()
        importance = importance / torch.exp(logprob).float()
        if self.sentence_version == 2:
            output = torch.sum(importance * torch.log(importance)) / batch_size
        elif self.sentence_version == 3:
            output = - torch.sum(importance * logprob) / batch_size
        return ml_output, output, stats


class AllIsGoodCriterion(SentSmoothCriterion):
    def __init__(self, opt, vocab):
        SentSmoothCriterion.__init__(self, opt)

    def get_scores(self, preds, target):
        return np.ones(target.size(0))


class CiderRewardCriterion(SentSmoothCriterion):
    def __init__(self, opt, vocab):
        SentSmoothCriterion.__init__(self, opt)
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


class BleuRewardCriterion(SentSmoothCriterion):
    def __init__(self, opt, vocab):
        SentSmoothCriterion.__init__(self, opt)
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


class InfersentRewardCriterion(SentSmoothCriterion):
    def __init__(self, opt, vocab):
        SentSmoothCriterion.__init__(self, opt)
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


