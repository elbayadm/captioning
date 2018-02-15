import sys
import codecs
import os
import operator
from math import exp, log

def fetch_data(cand, ref):
    """ Store each reference and candidate sentences as a list """
    references = []
    if '.txt' in ref:
        reference_file = codecs.open(ref, 'r', 'utf-8')
        references.append(reference_file.readlines())
    else:
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference_file = codecs.open(os.path.join(root, f), 'r', 'utf-8')
                references.append(reference_file.readlines())
    candidate_file = codecs.open(cand, 'r', 'utf-8')
    candidate = candidate_file.readlines()
    return candidate, references


def count_ngram(candidate, references, n):
    # Calculate precision for each sentence
    ref_counts = []
    ref_lengths = []
    # Build dictionary of ngram counts
    for words in references:
        words = words.strip().split()
        ngram_d = {}
        # words = ref_sentence.strip().split()
        ref_lengths.append(len(words))
        limits = len(words) - n + 1
        # loop through the sentance consider the ngram length
        for i in range(limits):
            ngram = ' '.join(words[i:i+n]).lower()
            if ngram in ngram_d.keys():
                ngram_d[ngram] += 1
            else:
                ngram_d[ngram] = 1
        ref_counts.append(ngram_d)
    # candidate
    cand_dict = {}
    words = candidate.strip().split()
    limits = len(words) - n + 1
    for i in range(0, limits):
        ngram = ' '.join(words[i:i + n]).lower()
        if ngram in cand_dict:
            cand_dict[ngram] += 1
        else:
            cand_dict[ngram] = 1

    clipped_count = clip_count(cand_dict, ref_counts)
    r = best_length_match(ref_lengths, len(words))
    c = len(words)
    pr = float(clipped_count + 1) / (limits + 1)  # smoothing
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = exp(1-(float(r)/c))
    return bp


def bleu_multi(candidate, references):
    precisions = []
    for i in range(4):
        pr, bp = count_ngram(candidate, references, i+1)
        precisions.append(pr)

    bleu = exp(sum([log(pr)/(r+1) for r, pr in enumerate(precisions)]))
    return bleu * bp

