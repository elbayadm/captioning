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
# infersent = torch.load('../InferSent/infersent.allnli.pickle')
infersent = torch.load('../InferSent/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
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
# bs1 = json.load(open('captions/bootstrap_20samples_baseline.json', 'r'))
bs1 = json.load(open('data/coco/generated_captions_confusion_bootstrap.json', 'r'))
print('length of bs1:', len(bs1))
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
max_gt = 5
max_gen = 10
for k in Caps:
    lgen = len(Caps[k]['gen']['sents'])
    if lgen > max_gen:
        idx = np.argsort(- np.array(Caps[k]['gen']['scores']))[:max_gen]
        print('Selecting:', idx)
        sents = [Caps[k]['gen']['sents'][i] for i in idx]
        scores = [Caps[k]['gen']['scores'][i] for i in idx]
        print('Sents:', sents)
        print('Scores:', scores)
        Caps[k]['gen']['sents'] = sents
        Caps[k]['gen']['scores'] = scores
    elif lgen < max_gen:
        print(Caps[k])
        raise ValueError('too few generated caps only %d (id: %d)' % (lgen, k))
    gens.append(len(Caps[k]['gen']['sents']))
    lgt = len(Caps[k]['gt']['sents'])
    # print('Number of gt caps:', lgt)
    if lgt > max_gt:
        idx = np.argsort(- np.array(Caps[k]['gt']['scores']))[:max_gt]
        print('Selecting:', idx)
        sents = [Caps[k]['gt']['sents'][i] for i in idx]
        scores = [Caps[k]['gt']['scores'][i] for i in idx]
        print('Sents:', sents)
        print('Scores:', scores)
        Caps[k]['gt']['sents'] = sents
        Caps[k]['gt']['scores'] = scores
    elif lgt < max_gt:
        raise ValueError('Too few captions than expected')
    gts.append(len(Caps[k]['gt']['sents']))
    # Average the scores on the sampled + gt (reduced support)
    sum_scores = sum(Caps[k]['gt']['scores']) + sum(Caps[k]['gen']['scores'])
    Caps[k]['gt']['scores'],  Caps[k]['gen']['scores'] = c_softmax(Caps[k]['gt']['scores'],  Caps[k]['gen']['scores'])
    print("Mass distribution:", "gt:", sum(np.exp(Caps[k]['gt']['scores'])), "gen:", sum(np.exp(Caps[k]['gen']['scores'])))
    # print(Caps[k])

print('Gen:', np.unique(np.array(gens)))
print('Gt:', np.unique(np.array(gts)))


keys = np.array(list(Caps))
batches = np.array_split(keys, 1000)
print("Processing in %d batches" % len(batches))
cnt = 0
for batch in batches:
    cnt += 1
    cider_scorer = CiderScorer(n=4, sigma=6)
    bleu4 = BleuScorer(n=4)
    infer = []
    print('batch indices:', batch)
    for k in batch:
        # print('all caps:', Caps[k])
        refs = Caps[k]['gt']['sents']
        print("Refs:", refs)
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
    (bl, bls) = bleu4.compute_score(option='closest', verbose=0)
    # print('Bleu multi:', np.mean(bl), [np.mean(n) for n in bls])
    infer = np.array(infer)
    bls4 = bls[3]
    bls3 = bls[2]
    bls2 = bls[1]

    # print('Mean BLeu4:', np.mean(bls))
    index = 0
    for k in batch:
        cgen = len(Caps[k]['gen']['sents'])
        cgt = len(Caps[k]['gt']['sents'])
        assert(cgt == max_gt)
        # print('Generated:', cgen)
        assert(cgen == max_gen)
        Caps[k]['gt']['cider'] = cds[index: index + cgt]
        Caps[k]['gen']['cider'] = cds[index + cgt : index + cgt + cgen]
        Caps[k]['gt']['bleu4'] = bls4[index: index + cgt]
        Caps[k]['gen']['bleu4'] = bls4[index + cgt : index + cgt + cgen]

        Caps[k]['gt']['bleu3'] = bls3[index: index + cgt]
        Caps[k]['gen']['bleu3'] = bls3[index + cgt : index + cgt + cgen]

        Caps[k]['gt']['bleu2'] = bls2[index: index + cgt]
        Caps[k]['gen']['bleu2'] = bls2[index + cgt : index + cgt + cgen]

        Caps[k]['gt']['infersent'] = infer[index: index + cgt]
        Caps[k]['gen']['infersent'] = infer[index + cgt : index + cgt + cgen]
        # print('UPDATE:', Caps[k])

        index += (cgen + cgt)
    if not cnt % 5:
        print("Processed %d batches" % cnt)
###### from math import exp
Caps_gen = []
for k in Caps:
    entry = {'captions': Caps[k]['gt']['sents'] + Caps[k]['gen']['sents'],
             'scores':  Caps[k]['gt']['scores'] + Caps[k]['gen']['scores'],
             'cider':  Caps[k]['gt']['cider'].tolist() + Caps[k]['gen']['cider'].tolist(),
             'infersent':  Caps[k]['gt']['infersent'].tolist() + Caps[k]['gen']['infersent'].tolist(),
             'bleu4':  list(Caps[k]['gt']['bleu4']) + list(Caps[k]['gen']['bleu4']),
             'bleu3':  list(Caps[k]['gt']['bleu3']) + list(Caps[k]['gen']['bleu3']),
             'bleu2':  list(Caps[k]['gt']['bleu2']) + list(Caps[k]['gen']['bleu2']),
             'id': k}
    # print(entry)
    Caps_gen.append(entry)
pickle.dump(Caps_gen, open('data/coco/captions_bootstrap_confusion_s15.pkl', 'wb'))

