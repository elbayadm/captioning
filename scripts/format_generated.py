import json
import glob
import os.path as osp
import sys
sys.path.append("coco-caption")
from pycocoevalcap.cider.cider_scorer import CiderScorer



def format_conf_gen():
    # Array of dicts with key 'id' for cocoid and 'sampled' array of generated
    # captions.
    # Match imid to cocoid
    COCO = json.load(open('data/coco/dataset_coco.json', 'r'))
    COCO = COCO['images']
    Match = {}
    for datum in COCO:
        #  print datum
        Match[datum['imgid']] = datum['cocoid']
    D = []
    genfiles = glob.glob('tmpcaps/gen_srilm_viterbi_caps*.txt')
    for genfile in genfiles:
        d = {}
        print osp.basename(genfile).split('.')[0].split('_')[-1]
        imid = int(osp.basename(genfile).split('.')[0].split('_')[-1][4:])
        print "imid:", imid
        d['id'] = Match[imid]
        d['sampled'] = []
        with open(genfile, 'r') as f:
            for line in f.readlines():
                sent = ' '.join(line.strip().split(' ')[1:])
                print  "%d : %s" % (d['id'], sent)
                d['sampled'].append(sent)
        D.append(d)
    json.dump(D, open('data/coco/generated_captions_viterbi_confusion.json', 'w'))


def score_gen():
    #Load generated captions:
    COCO = json.load(open('data/coco/dataset_coco.json', 'r'))
    COCO = COCO['images']
    GEN = json.load(open('data/coco/generated_captions_confusion.json'))
    GEN_dict = {}
    for g in GEN:
        GEN_dict[g['id']] = g['sampled']
    COCO_dict = {}
    for c in COCO:
        COCO_dict[c["cocoid"]] = [cc['raw'] for cc in c['sentences']]
    D = []
    e = 0
    while e + 1 < len(GEN_dict):
        #  if e > 50:
        #      break
        k = GEN_dict.keys()[e]
        j = GEN_dict.keys()[e+1]
        e += 2
        cider_scorer = CiderScorer(n=4, sigma=6)
        hypos = GEN_dict[k]
        refs = [str(x) for x in COCO_dict[k]]
        for hypo in hypos:
            cider_scorer += (hypo, refs)
        len_k = len(hypos)
        d_k = {"id": k,
                "sampled": hypos}
        hypos = GEN_dict[j]
        refs = [str(x) for x in COCO_dict[j]]
        for hypo in hypos:
            cider_scorer += (hypo, refs)
        #  print cider_scorer.ctest, cider_scorer.crefs
        (score, scores) = cider_scorer.compute_score()
        print "Scores:", scores
        d_j = {"id": j,
               "sampled": hypos,
               "scores": list(scores[len_k:])}
        d_k['scores'] = list(scores[:len_k])

        D.append(d_k)
        D.append(d_j)
    if e < len(GEN_dict):
        k = GEN_dict.keys()[e]
        j = GEN_dict.keys()[e-1]
        cider_scorer = CiderScorer(n=4, sigma=6)
        hypos = GEN_dict[k]
        refs = [str(x) for x in COCO_dict[k]]
        for hypo in hypos:
            cider_scorer += (hypo, refs)
        len_k = len(hypos)
        d_k = {"id": k,
                "sampled": hypos}
        hypos = GEN_dict[j]
        refs = [str(x) for x in COCO_dict[j]]
        for hypo in hypos:
            cider_scorer += (hypo, refs)
        #  print cider_scorer.ctest, cider_scorer.crefs
        (score, scores) = cider_scorer.compute_score()
        print "Scores:", scores
        d_k['scores'] = list(scores[:len_k])
        D.append(d_k)

    json.dump(D, open('data/coco/generated_captions_confusion_scored.json', 'w'))


if __name__ == "__main__":
    score_gen()

