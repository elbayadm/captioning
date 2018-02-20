import os, sys
import json
import random
from collections import Counter, OrderedDict
import numpy as np
from scipy.misc import comb
from math import log, exp, sqrt
from sklearn.manifold import TSNE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch

A = json.load(open('../data/coco/captions_train2014.json', 'r'))['annotations']
caps = {}
for a in A:
    if a['image_id'] in caps:
        caps[a['image_id']].append(a['caption'][:-1].lower().split())
    else:
        caps[a['image_id']] = [a['caption'][:-1].lower().split()]
del A

ids = list(caps)

GLOVE_PATH = '../../InferSent/dataset/GloVe/glove.840B.300d.txt'
sys.path.insert(0, '../../InferSent/encoder/')
model = torch.load('../../InferSent/encoder/infersent.allnli.pickle',
                   map_location=lambda storage, loc: storage)
model.set_glove_path(GLOVE_PATH)
model.build_vocab_k_words(K=100000)


def fold(sentence, maxlen=5, bf=False):
    if isinstance(sentence, str):
        sentence = sentence.split()
    folds = len(sentence) // maxlen
    sent = r""
    for f in range(folds+1):
        if bf:
            sent += '\\bf '
        if len(sentence) > maxlen * (f+1):
            sent += ' '.join(sentence[maxlen*f: maxlen*(f+1)]) + '\n'
        else:
            sent += ' '.join(sentence[maxlen*f:])
    return sent


def highlight(sentence, indices):
#     print('Highlighting ', sentence , '@', indices)
    sent = sentence.copy()
    for ind in indices:
        sent[ind] = '\\underline{%s}' % sent[ind]
    return sent


def embed_sent(sentences):
    embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=False)
    return embeddings


def alter(sentence, vocab, tau):
    m = len(sentence)
    p = distrib(len(sentence), len(vocab), tau)
    distance = np.random.choice(range(m+1), p=p)
    sampled = sentence.copy()
    m = len(sampled)
    indices = list(range(m))
#     print('indices:', indices)
    random.shuffle(indices)
    for k in indices[:distance]:
        sampled[k] = random.choice(vocab)
    return sampled, indices[:distance]

def distrib(m, v, tau=0.7):
    x = [comb(m, d, exact=False) * (v-1)**d / v**m * exp(-d/tau) for d in range(m+1)]
    x = np.array(x)
    x/= np.sum(x)
    return x

def hamming(s1, s2):
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def sentence_bleu(hypothesis, reference, smoothing=True, order=4, **kwargs):
    log_score = 0
    if len(hypothesis) == 0:
        return 0
    for i in range(order):
        hyp_ngrams = Counter(zip(*[hypothesis[j:] for j in range(i + 1)]))
        ref_ngrams = Counter(zip(*[reference[j:] for j in range(i + 1)]))
        numerator = sum(min(count, ref_ngrams[bigram])
                        for bigram, count in hyp_ngrams.items())
        denominator = sum(hyp_ngrams.values())
        if smoothing:
            numerator += 1
            denominator += 1
        score = numerator / denominator
        if score == 0:
            log_score += float('-inf')
        else:
            log_score += log(score) / order
    bp = min(1, exp(1 - len(reference) / len(hypothesis)))
    return exp(log_score) * bp


def alter_and_score(imid, d=2, NS=1, normalize=1, tauh=.2, V='refs', verbose=0):
    sampled = OrderedDict()
    Vsub = []
    for c in caps[imid]:
        Vsub += c
    for c in caps[imid]:
        k = ' '.join(c)
        for ii in range(NS):
            if not ii:
                sampled[k] = []
            s, a = alter(c, Vsub, tau=tauh)
            sampled[k].append({'sample': s, "alter": a})
    # Embed:
    sentences = []
    for c in sampled:
        sentences.append(c)
        for s in sampled[c]:
            sentences.append(' '.join(s["sample"]))
        
    if verbose:
        for s in sentences:
            print(s)
    embeddings = embed_sent(sentences)
    X2 = TSNE(n_components=2, verbose=0).fit_transform(embeddings)
    for c in sampled:
        for e, _ in enumerate(sampled[c]):
            sampled[c][e]['bleu'] = sentence_bleu(sampled[c][e]['sample'], c.split())
            sampled[c][e]['hamming'] = hamming(sampled[c][e]['sample'], c.split())        
    return X2, sampled
