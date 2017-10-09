import json
import numpy as np
import sys
sys.path.append('coco-caption')
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.bleu.bleu_scorer import BleuScorer


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
    Caps[item['image_id']][item['source']]['scores'].append(float(item['score']))

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

print('Gen:', np.unique(np.array(gens)))
print('Gt:', np.unique(np.array(gts)))


keys = np.array(list(Caps))
batches = np.array_split(keys, 100)
print(len(batches))
for batch in batches:
    cider_scorer = CiderScorer(n=4, sigma=6)
    bleu4 = BleuScorer(n=4)
    for k in batch:
        refs = Caps[k]['gt']['sents']
        for ref in refs:
            cider_scorer += (ref, refs)
            bleu4 += (ref, refs)
        for c in Caps[k]['gen']['sents']:
            cider_scorer += (c, refs)
            bleu4 += (c, refs)
    (cd, cds) = cider_scorer.compute_score()
    (bl, bls) = bleu4.compute_score()
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
        index += (cgen + cgt)
print(Caps[batches[0][0]])
###### from math import exp
Caps_gen = []
for k in Caps:
    entry = {'captions': Caps[k]['gt']['sents'] + Caps[k]['gen']['sents'],
             'scores':  Caps[k]['gt']['scores'] + Caps[k]['gen']['scores'],
             'cider':  list(Caps[k]['gt']['cider']) + list(Caps[k]['gen']['cider']),
             'bleu4':  list(Caps[k]['gt']['bleu4']) + list(Caps[k]['gen']['bleu4']),
             'id': k}
    Caps_gen.append(entry)
json.dump(Caps_gen, open('data/coco/captions_bootstrap_baseline_s5.json', 'w'))

