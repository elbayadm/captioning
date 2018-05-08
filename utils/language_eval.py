import sys
import os
import time
from collections import Counter
import random
import string
import json
import glob


def captions_creativity(caps, minfreq):
    """
    Stats on generated n-grams unseen in the training corpus
    """
    print('Minfreq == %d' % minfreq)
    creativity = {}
    tmp = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
    os.makedirs('tmp', exist_ok=True)
    tmp  = 'tmp/' + tmp
    with open('%s_tmp.txt' % tmp, 'w') as f:
        for cp in caps:
            f.write('%s\n' % cp)
    # count grams in tmp.txt:
    print('Running ngram-count')
    os.system("ngram-count -text %s_tmp.txt -write3 %s_tmpcount3.txt -write4 %s_tmpcount4.txt -write5 %s_tmpcount5.txt -order 5 -no-eos -no-sos" %
              (tmp, tmp, tmp, tmp))
    os.system('ngram-count -text data/coco/train_captions_reduced_freq%d.txt -intersect %s_tmpcount4.txt -write4 %s_intersect_count4_tmp.txt -order 4 -no-sos -no-eos' %
              (minfreq, tmp, tmp))
    os.system('ngram-count -text data/coco/train_captions_reduced_freq%d.txt -intersect %s_tmpcount3.txt -write3 %s_intersect_count3_tmp.txt -no-sos -no-eos' %
              (minfreq, tmp, tmp))
    os.system('ngram-count -text data/coco/train_captions_reduced_freq%d.txt -intersect %s_tmpcount5.txt -write5 %s_intersect_count5_tmp.txt -no-sos -no-eos -order 5' %
              (minfreq, tmp, tmp))
    unseen_4g = []
    unseen_3g = []
    unseen_5g = []
    counts = {}
    with open('%s_intersect_count5_tmp.txt' % tmp, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            counts[line[0]] = int(line[-1])
    creative_5g = 0
    for k, v in list(counts.items()):
        if not v:
            creative_5g += 1
            unseen_5g.append(k)
            # print('Unseen 5g:', k)
    total_5g = 1 + sum([1 for line in open('%s_tmpcount5.txt' % tmp, 'r')])
    #------------------------------------------------------------------------------------------
    counts = {}
    with open('%s_intersect_count4_tmp.txt' % tmp, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            counts[line[0]] = int(line[-1])
    creative_4g = 0
    for k, v in list(counts.items()):
        if not v:
            creative_4g += 1
            unseen_4g.append(k)
            # print('Unseen 4g:', k)
    total_4g = 1 + sum([1 for line in open('%s_tmpcount4.txt' % tmp, 'r')])
    #------------------------------------------------------------------------------------------
    counts = {}
    with open('%s_intersect_count3_tmp.txt' % tmp, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            counts[line[0]] = int(line[-1])
    creative_3g = 0
    for k, v in list(counts.items()):
        if not v:
            creative_3g += 1
            unseen_3g.append(k)
            # print('Unseen 3g:', k)
    total_3g = 1 + sum([1 for line in open('%s_tmpcount3.txt' % tmp, 'r')])
    #------------------------------------------------------------------------------------------
    tmpfiles = glob.glob('%s_*' % tmp)
    for f in tmpfiles:
        os.remove(f)
    #  os.remove('%s_tmpcount3.txt')
    #  os.remove('tmpcount4.txt')
    #  os.remove('tmpcount5.txt')
    #  os.remove('intersect_count3_tmp.txt')
    #  os.remove('intersect_count4_tmp.txt')
    #  os.remove('intersect_count5_tmp.txt')
    creativity = {'creative 3grams': float(creative_3g) / total_3g,
                  'creative 4grams': float(creative_4g) / total_4g,
                  'creative 5grams': float(creative_5g) / total_5g,
                  'unseen 3grams': random.sample(unseen_3g, min(10, len(unseen_3g))),
                  'unseen 4grams': random.sample(unseen_4g, min(10, len(unseen_4g))),
                  'unseen 5grams': random.sample(unseen_5g, min(10, len(unseen_5g)))}
    return creativity


def vocab_use(caps):
    """
    Count used words
    """
    TOK = Counter()
    for cap in caps:
        TOK.update(cap.split())
    return len(TOK)

def language_lm_eval(refs, cands):
    """
    Measure BLEU4 score
    """
    from nltk.translate.bleu_score import corpus_bleu
    refs = [[ref.split()] for ref in refs]
    cands = [cand.split() for cand in cands]
    B4 = corpus_bleu(refs, cands)
    return {'Bleu4': B4}

def language_eval(dataset, preds, logger,
                  all_metrics=False,
                  single_metrics=False,
                  get_creativity=True):
    """
    Measure language performance:
        BLEU_1:4, ROUGE_L, CIDER, #FIXME METEOR, SPICE;
        DIVERSITY and CREATIVITY, LENGHT
    """
    if 'coco' in dataset:
        sys.path.append("coco-caption")
        annFile = 'coco-caption/annotations/captions_val2014.json'
    else:
        sys.path.append("coco-caption")
        annFile = 'data/flickr30k/flickr30k_val_annotations.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    logger.warn('Loading reference captions..')
    coco = COCO(annFile)
    logger.warn('Reference captions loaded!')
    valids = coco.getImgIds()

    remove_tmp = False
    # filter results to only those in MSCOCO validation set (will be about a third)
    if isinstance(preds, str):
        assert(preds.endswith('.json'))
        resFile = preds
        load_preds = json.load(open(preds))
        preds_filt = [p for p in load_preds if p['image_id'] in valids]
    else:
        random.seed(time.time())
        tmp_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
        preds_filt = [p for p in preds if p['image_id'] in valids]
        logger.warn('using %d/%d predictions' % (len(preds_filt), len(preds)))
        json.dump(preds_filt, open(tmp_name + '.json', 'w'))  # serialize to temporary json file. Sigh, COCO API...
        resFile = tmp_name + '.json'
        remove_tmp = True
    logger.warn('Loading model captions')
    cocoRes = coco.loadRes(resFile)
    logger.warn('Model captions loaded')
    cocoEval = COCOEvalCap(coco, cocoRes, all_metrics)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    logger.warn('Starting evaluation...')
    cocoEval.evaluate()

    if remove_tmp:
        # delete the temp file
        os.remove(tmp_name + '.json')

    # create output dictionary
    out = {}
    for metric, score in list(cocoEval.eval.items()):
        out[metric] = score

    caps = [p['caption'] for p in preds_filt]
    # get score per sample:
    preds = None
    if single_metrics:
        preds = cocoEval.ImgToEval

    out['vocab_use'] = vocab_use(caps)
    unseen = None
    if get_creativity:
        cr = captions_creativity(caps, 5)
        out['creative 3grams'] = cr['creative 3grams']
        out['creative 4grams'] = cr['creative 4grams']
        out['creative 5grams'] = cr['creative 5grams']
        unseen = {"3g": cr['unseen 3grams'],
                  "4g": cr['unseen 4grams'],
                  "5g": cr['unseen 5grams']}
    return out, preds, unseen


