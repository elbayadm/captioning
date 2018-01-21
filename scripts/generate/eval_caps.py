import json
import sys
sys.path.append("coco-caption")
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.bleu.bleu_scorer import BleuScorer

def normalize_scores(d):
    scores = d['scores']
    d['scores'] = list(scores / sum(scores))
    return d


def eval_caps(suffix):
    #Load generated captions:
    COCO = json.load(open('data/coco/dataset_coco.json', 'r'))
    COCO = COCO['images']
    GEN = json.load(open('data/coco/generated_captions_%s.json' % suffix))
    GEN_dict = {}
    for g in GEN:
        GEN_dict[g['id']] = g['sampled']
    COCO_dict = {}
    for c in COCO:
        COCO_dict[c["cocoid"]] = [" ".join(cc['tokens']) for cc in c['sentences']]
    print "Read coco (%d) and gen(%d)" % (len(COCO_dict), len(GEN_dict))
    D = []
    e = 0
    K = GEN_dict.keys()
    while e + 1 < len(GEN_dict):
        #  if e > 50:
        #      break
        k = K[e]
        j = K[e+1]
        e += 2
        cider_scorer = CiderScorer(n=4, sigma=6)
        hypos = GEN_dict[k]
        refs = [str(x) for x in COCO_dict[k]]
        for hypo in refs + hypos:
            cider_scorer += (hypo, refs)
        len_k = len(hypos + refs)
        d_k = {"id": k,
                "captions": refs + hypos}
        hypos = GEN_dict[j]
        refs = [str(x) for x in COCO_dict[j]]
        for hypo in refs + hypos:
            cider_scorer += (hypo, refs)
        (score, scores) = cider_scorer.compute_score()
        d_j = {"id": j,
               "captions": refs + hypos,
               "scores": list(scores[len_k:])}
        d_k['scores'] = list(scores[:len_k])
        d_j = normalize_scores(d_j)
        d_k = normalize_scores(d_k)
        #  print k, j
        D.append(d_k)
        D.append(d_j)
    if e < len(GEN_dict):
        k = K[e]
        j = K[e-1]
        cider_scorer = CiderScorer(n=4, sigma=6)
        hypos = GEN_dict[k]
        refs = [str(x) for x in COCO_dict[k]]
        for hypo in refs + hypos:
            cider_scorer += (hypo, refs)
        len_k = len(hypos + refs)
        d_k = {"id": k,
                "captions": refs + hypos}
        hypos = GEN_dict[j]
        refs = [str(x) for x in COCO_dict[j]]
        for hypo in refs + hypos:
            cider_scorer += (hypo, refs)
        #  print cider_scorer.ctest, cider_scorer.crefs
        (score, scores) = cider_scorer.compute_score()
        d_k['scores'] = list(scores[:len_k])
        d_k = normalize_scores(d_k)
        D.append(d_k)
    json.dump(D, open('data/coco/generated_captions_%s_scored_distrib.json' % suffix, 'w'))

if __name__ == "__main__":
    eval_caps(suffix=sys.argv[1])


