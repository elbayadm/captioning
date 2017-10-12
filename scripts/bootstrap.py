import json
import numpy as np
import sys
import torch
from six.moves import cPickle as pickle
sys.path.append('coco-caption')
# sys.path.append('../InferSent/encoder')

from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.bleu.bleu_scorer import BleuScorer

GLOVE_PATH = '../InferSent/dataset/GloVe/glove.840B.300d.txt'
infersent = torch.load('../InferSent/infersent.allnli.pickle')
infersent.set_glove_path(GLOVE_PATH)
infersent.build_vocab_k_words(K=100000)


def multi_cosine(u, refs):
    sims = []
    for v in refs:
        sims.append(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return np.mean(sims)


def infer_cosine_gp(hyps, refs):
    """
    """
    lh = len(hyps)
    lr = len(refs)
    codes = infersent.encode(hyps + refs)
    chyps = codes[:lh,]
    crefs = codes[lh:,]
    output = []
    for i in range(lr):
        against = np.vstack((crefs[:i], crefs[i+1:]))
        output.append(multi_cosine(crefs[i], against))
    for i in range(lh):
        output.append(multi_cosine(chyps[i], crefs))
    return output


def infer_cosine(s1, s2):
    if type(s2) == list:
        # print('Encoding:', [s1] + s2)
        codes = infersent.encode([s1] + s2)
    else:
        codes = infersent.encode([s1] + [s2])
    u = codes[0]
    refs = codes[1:]
    # compute average sim:
    sims = []
    for v in refs:
        sims.append(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return np.mean(sims)

print(infer_cosine_gp(['The cat plays', 'The cat is here'], ['The cat is playing', 'The cat is laying down', 'The cat is annoying']))
# sys.exit()


def c_softmax(scores1, scores2):
    """
    Merge scores and softmax
    """
    S = np.array(scores1 + scores2)
    S = np.exp(S)
    S /= np.sum(S)
    S = np.log(S)
    return list(S[:len(scores1)]), list(S[len(scores1):])

# Load InferSent pretrained model:

# bs1 = json.load(open('captions/bootstrap_bs1.json', 'r'))
bs1 = json.load(open('captions/bootstrap_bs1.json', 'r'))

Caps = {}
for item in bs1:
    if item['image_id'] not in Caps:
        Caps[item['image_id']] = {'gen': {'sents': [],
                                          'scores': []},
                                  'gt': {'sents': [],
                                          'scores': []}
                                 }
    Caps[item['image_id']][item['source']]['sents'].append(item['caption'])
    Caps[item['image_id']][item['source']]['scores'].append(float(item['score'])/len(item['caption'].split()))

# Stats:
gens = []
gts = []
for k in Caps:
    lgen = len(Caps[k]['gen']['sents'])
    if lgen > 5:
        idx = np.argsort(- np.array(Caps[k]['gen']['scores']))[:5]
        print('Selecting:', idx)
        sents = [Caps[k]['gen']['sents'][i] for i in idx]
        scores = [Caps[k]['gen']['scores'][i] for i in idx]
        print('Sents:', sents)
        print('Scores:', scores)
        Caps[k]['gen']['sents'] = sents
        Caps[k]['gen']['scores'] = scores
    gens.append(len(Caps[k]['gen']['sents']))
    lgt = len(Caps[k]['gt']['sents'])
    if lgt > 5:
        idx = np.argsort(- np.array(Caps[k]['gt']['scores']))[:5]
        print('Selecting:', idx)
        sents = [Caps[k]['gt']['sents'][i] for i in idx]
        scores = [Caps[k]['gt']['scores'][i] for i in idx]
        print('Sents:', sents)
        print('Scores:', scores)
        Caps[k]['gt']['sents'] = sents
        Caps[k]['gt']['scores'] = scores
    gts.append(len(Caps[k]['gt']['sents']))
    # Average the scores on the sampled + gt (reduced support)
    sum_scores = sum(Caps[k]['gt']['scores']) + sum(Caps[k]['gen']['scores'])
    print('Sum of scores:', sum_scores)
    Caps[k]['gt']['scores'],  Caps[k]['gen']['scores'] = c_softmax(Caps[k]['gt']['scores'],  Caps[k]['gen']['scores'])
    print(sum(np.exp(Caps[k]['gt']['scores'])), sum(np.exp(Caps[k]['gen']['scores'])))
    print(Caps[k])

print('Gen:', np.unique(np.array(gens)))
print('Gt:', np.unique(np.array(gts)))


keys = np.array(list(Caps))
batches = np.array_split(keys, 1000)
print(len(batches))
for batch in batches:
    cider_scorer = CiderScorer(n=4, sigma=6)
    bleu4 = BleuScorer(n=4)
    infer = []
    for k in batch:
        refs = Caps[k]['gt']['sents']
        for e, ref in enumerate(refs):
            _refs = refs.copy()
            _refs.pop(e)
            cider_scorer += (ref, _refs)
            bleu4 += (ref, _refs)
        for c in Caps[k]['gen']['sents']:
            cider_scorer += (c, refs)
            bleu4 += (c, refs)
        infer += infer_cosine_gp(Caps[k]['gen']['sents'], refs)


    (cd, cds) = cider_scorer.compute_score()
    (bl, bls) = bleu4.compute_score(option='closest', verbose=1)
    infer = np.array(infer)
    bls = bls[-1]
    index = 0
    for k in batch:
        cgen = len(Caps[k]['gen']['sents'])
        cgt = len(Caps[k]['gt']['sents'])
        assert(cgt == 5)
        # print('Generated:', cgen)
        assert(cgen == 5)
        Caps[k]['gt']['cider'] = cds[index: index + cgt]
        Caps[k]['gen']['cider'] = cds[index + cgt : index + cgt + cgen]
        Caps[k]['gt']['bleu4'] = bls[index: index + cgt]
        Caps[k]['gen']['bleu4'] = bls[index + cgt : index + cgt + cgen]
        Caps[k]['gt']['infersent'] = infer[index: index + cgt]
        Caps[k]['gen']['infersent'] = infer[index + cgt : index + cgt + cgen]
        print('UPDATE:', Caps[k])

        index += (cgen + cgt)
###### from math import exp
Caps_gen = []
for k in Caps:
    entry = {'captions': Caps[k]['gt']['sents'] + Caps[k]['gen']['sents'],
             'scores':  Caps[k]['gt']['scores'] + Caps[k]['gen']['scores'],
             'cider':  Caps[k]['gt']['cider'].tolist() + Caps[k]['gen']['cider'].tolist(),
             'infersent':  Caps[k]['gt']['infersent'].tolist() + Caps[k]['gen']['infersent'].tolist(),
             'bleu4':  list(Caps[k]['gt']['bleu4']) + list(Caps[k]['gen']['bleu4']),
             'id': k}
    print(entry)
    Caps_gen.append(entry)
pickle.dump(Caps_gen, open('data/coco/captions_bootstrap_baseline_genconf15.pkl', 'wb'))

