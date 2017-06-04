from __future__ import absolute_import
from __future__ import division
#  from __future__ import print_function
# setup gpu
try:
    import os
    import subprocess
    gpu_id = subprocess.check_output('gpu_getIDs.sh', shell=True)
    print "Gpu%s" % gpu_id
except:
    print "Failed to get gpu_id (setting gpu_id to 0)"
    gpu_id = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
import json
from json import encoder
import string
import time
import os
import sys
import glob
import misc.utils as utils
from collections import Counter
import cPickle as pickle

_OKGREEN = '\033[92m'
_WARNING = '\033[93m'
_FAIL = '\033[91m'
_ENDC = '\033[0m'


def captions_creativity(caps, minfreq):
    """
    Stats on generated n-grams unseen in the training corpus
    """
    print 'Minfreq == %d' % minfreq
    creativity = {}
    tmp = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
    with open('%s_tmp.txt' % tmp, 'w') as f:
        for cp in caps:
            f.write('%s\n' % cp)
    # count grams in tmp.txt:
    print 'Running ngram-count'
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
    for k, v in counts.items():
        if not v:
            creative_5g += 1
            unseen_5g.append(k)
            print 'Unseen 5g:', k
    total_5g = 1 + sum([1 for line in open('%s_tmpcount5.txt' % tmp, 'r')])
    #------------------------------------------------------------------------------------------
    counts = {}
    with open('%s_intersect_count4_tmp.txt' % tmp, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            counts[line[0]] = int(line[-1])
    creative_4g = 0
    for k, v in counts.items():
        if not v:
            creative_4g += 1
            unseen_4g.append(k)
            print 'Unseen 4g:', k
    total_4g = 1 + sum([1 for line in open('%s_tmpcount4.txt' % tmp, 'r')])
    #------------------------------------------------------------------------------------------
    counts = {}
    with open('%s_intersect_count3_tmp.txt' % tmp, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            counts[line[0]] = int(line[-1])
    creative_3g = 0
    for k, v in counts.items():
        if not v:
            creative_3g += 1
            unseen_3g.append(k)
            print 'Unseen 3g:', k
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


def language_eval(dataset, preds):
    """
    Measure language performance:
        BLEU_1:4, ROUGE_L, CIDER, #FIXME METEOR, SPICE;
        DIVERSITY and CREATIVITY, LENGHT
    """
    import sys
    if 'coco' in dataset:
        sys.path.append("coco-caption")
        annFile = 'coco-caption/annotations/captions_val2014.json'
    else:
        sys.path.append("f30k-caption")
        annFile = 'f30k-caption/annotations/dataset_flickr30k.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    random.seed(time.time())
    tmp_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(tmp_name + '.json', 'w')) # serialize to temporary json file. Sigh, COCO API...
    resFile = tmp_name+'.json'
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # delete the temp file
    os.remove(tmp_name + '.json')

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    caps = [p['caption'] for p in preds_filt]
    out['vcab_use'] = vocab_use(caps)
    cr = captions_creativity(caps, 5)
    out['creative 3grams'] = cr['creative 3grams']
    out['creative 4grams'] = cr['creative 4grams']
    out['creative 5grams'] = cr['creative 5grams']
    unseen = {"3g" : cr['unseen 3grams'],
              "4g" : cr['unseen 4grams'],
              "5g" : cr['unseen 5grams']}

    return out, unseen

def generate_caps(encoder, decoder, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'train')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    beam_size = 1
    logger = eval_kwargs.get('logger')
    lm_model = eval_kwargs.get('lm_model')
    vocab_size = eval_kwargs.get('vocab_size')
    sample_max = eval_kwargs.get('sample_max')
    temperature = eval_kwargs.get('temperature')
    tries = eval_kwargs.get('tries', 5)

    print 'Using sample_max = %d  ||  temperature %.2f' % (sample_max, temperature)

    # Make sure in the evaluation mode
    encoder.eval()
    decoder.eval()
    logger.warn('Generating captions for the full training set')
    loader.reset_iterator(split)
    n = 0
    blobs = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        # forward the model to get loss
        infos = data['infos']
        ids = [inf['id'] for inf in infos]
        assert len(ids) == 1, "Batch size larger than 1"
        tmp = [data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        labels, masks = tmp
        tr = 0
        gt = utils.decode_sequence(loader.get_vocab(), labels[:,1:].data)
        blob_batch = { "id": ids[0], "gt": gt, "sampled": []}
        for igt in gt:
            print _OKGREEN + igt + _ENDC

        while tr < tries:
            #  z_mu, z_var, codes = encoder(labels)
            codes = encoder.sample(labels)
            seq, _ = decoder.sample(codes, {'beam_size': beam_size,
                                            "vocab_size": vocab_size,
                                            "sample_max": sample_max,
                                            "temperature": temperature})
            sents = utils.decode_sequence(loader.get_vocab(), seq)
            for isent in sents:
                print _WARNING + isent + _ENDC
            blob_batch['sampled'] += sents
            tr += 1
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if data['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
        blobs.append(blob_batch)
        print "Blob batch:", blob_batch
    json.dump(blobs, open('data/coco/generated_captions.json', 'w'))
    encoder.train()
    decoder.train()
    return 1

def eval_lm_split(encoder, decoder, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    beam_size = 1
    logger = eval_kwargs.get('logger')
    lm_model = eval_kwargs.get('lm_model')
    vocab_size = eval_kwargs.get('vocab_size')
    sample_max = eval_kwargs.get('sample_max')
    temperature = eval_kwargs.get('temperature')
    print 'Using sample_max = %d  ||  temperature %.2f' % (sample_max, temperature)

    # Make sure in the evaluation mode
    encoder.eval()
    decoder.eval()
    logger.warn('Evaluating %d val images' % val_images_use)
    loader.reset_iterator(split)
    n = 0
    loss_sum = 0
    loss_evals = 0
    predictions = []
    CODES = []
    R_CODES = []
    MU = []
    VAR = []
    SENTS = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        # forward the model to get loss
        tmp = [data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        labels, masks = tmp
        if lm_model == "rnn":
            codes = encoder(labels)
        elif lm_model == "rnn_vae":
            z_mu, z_var, codes = encoder(labels)
            r_codes = encoder.sample(labels)
            #  r_codes2 = encoder.sample(labels)
            #  kld_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
            #  train_kld_loss = kld_loss.data[0]
        #  if len(CODES):
        #      CODES = np.vstack((CODES, codes.cpu().data.numpy()))
        #      R_CODES = np.vstack((R_CODES, r_codes.cpu().data.numpy()))
        #      MU = np.vstack((MU, z_mu.cpu().data.numpy()))
        #      VAR = np.vstack((VAR, z_var.cpu().data.numpy()))
        #  else:
        #      CODES = codes.cpu().data.numpy()
        #      R_CODES = r_codes.cpu().data.numpy()
        #      MU = z_mu.cpu().data.numpy()
        #      VAR = z_var.cpu().data.numpy()
        gt = utils.decode_sequence(loader.get_vocab(), labels[:,1:].data)
        SENTS += gt
        seq, _ = decoder.sample(codes, {'beam_size': beam_size, "vocab_size": vocab_size, "sample_max": sample_max, "temperature": temperature})
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        try:
            r_seq, _ = decoder.sample(r_codes, {'beam_size': beam_size, "vocab_size": vocab_size})
            r_sents = utils.decode_sequence(loader.get_vocab(), r_seq)
        except:
            pass
        ###############################################################################################
        #  r_seq2, _ = decoder.sample(r_codes2, {'beam_size': beam_size, "vocab_size": vocab_size})
        #  r_sents2 = utils.decode_sequence(loader.get_vocab(), r_seq2)
        ##############################################################################################
        try:
            for co, de, r_de in zip(gt, sents, r_sents):
                print co, _OKGREEN, "\n>>", de, _ENDC, _WARNING, "\n>>", r_de, _ENDC
        except:
            for co, de in zip(gt, sents):
                print co, _OKGREEN, "\n>>", de, _ENDC

        loss = crit(decoder(codes, labels), labels[:,1:], masks[:,1:])[0].data[0]
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        if data['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
    lang_stats = None
    unseen_grams = None
    # Switch back to training mode
    encoder.train()
    decoder.train()
    #  print "Saving the codes: sents: %d and codes: %d (%s)" % (len(SENTS), len(CODES), str(CODES[0].shape))
    #  pickle.dump({'sentences': SENTS, 'codes': CODES, 'r_codes': R_CODES, 'mu': MU, 'var': VAR}, open('save/textLM_vae/codes.pkl', 'w'))
    #  print "Done"
    return loss_sum/loss_evals


def eval_split(cnn_model, model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    logger = eval_kwargs.get('logger')
    caption_model = eval_kwargs.get('caption_model')
    vae_weight = eval_kwargs.get('vae_weight')
    vocab_size = eval_kwargs.get('vocb_size')

    print "Eval %s" % caption_model


    # Make sure in the evaluation mode
    cnn_model.eval()
    model.eval()
    logger.warn('Evaluating %d val images' % val_images_use)

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    real_loss_sum = 0
    loss_evals = 0
    predictions = []
    Feats = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        # forward the model to get loss
        tmp = [data['images'], data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        images, labels, masks = tmp
        att_feats, fc_feats = cnn_model.forward(images)
        #  Feats.append(fc_feats.cpu().data.numpy())
        _att_feats = att_feats
        _fc_feats = fc_feats
        att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), loader.seq_per_img,) + att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) * loader.seq_per_img,) + att_feats.size()[1:]))
        fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), loader.seq_per_img,) + fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) * loader.seq_per_img,) + fc_feats.size()[1:]))
        if caption_model == "show_tell_vae":
            preds, recon_loss, kld_loss = model(fc_feats, att_feats, labels)
            real_loss, loss = crit(preds, labels[:, 1:], masks[:, 1:])
            real_loss = real_loss.data[0]
            loss = loss.data[0]
            loss += vae_weight * (recon_loss.data[0] + kld_loss.data[0])
            #  print "Incrementing loss" , loss
        else:
            real_loss, loss = crit(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:])
            real_loss = real_loss.data[0]
            loss = loss.data[0]
        loss_sum = loss_sum + loss
        real_loss_sum += real_loss
        loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        fc_feats, att_feats = _fc_feats, _att_feats
        seq, _ = model.sample(fc_feats, att_feats, {'beam_size': beam_size, "vocab_size": vocab_size})
        #set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        #  seq2, _ = model.sample(fc_feats, att_feats, {'beam_size': beam_size, "vocab_size": vocab_size})
        #set_trace()
        #  sents2 = utils.decode_sequence(loader.get_vocab(), seq2)

        for k, sent in enumerate(sents):
            #  print _OKGREEN, '%d >> %s\n %s %8s %s' % (data['infos'][k]['id'], sent, _WARNING, '>>', sents2[k]), _ENDC
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)
            #  logger.debug('image %s: %s' %(entry['image_id'], entry['caption']))
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        #  logger.warn('ix1 = %d - ix0 = %d' % (ix1, ix0))
        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        for i in range(n - ix1):
            predictions.pop()
        #  logger.debug('validation loss ... %d/%d (%f)' %(ix0 - 1, ix1, loss))
        if data['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
    lang_stats = None
    unseen_grams = None
    if lang_eval == 1:
        lang_stats, unseen_grams = language_eval(dataset, predictions)
    # Switch back to training mode
    model.train()
    #  pickle.dump(Feats, open('cnn_features.pkl', 'w'))
    return real_loss_sum/loss_evals, loss_sum/loss_evals, predictions, lang_stats, unseen_grams


# Evaluation fun(ction)
def eval_eval(cnn_model, model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', -1)
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    batch_size = eval_kwargs.get('batch_size', 1)

    # Make sure in the evaluation mode
    cnn_model.eval()
    model.eval()
    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []

    while True:
        # fetch a batch of data
        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        #evaluate loss if we have the labels
        loss = 0

        # Get the image features first
        tmp = [data['images'], data.get('labels', np.zeros(1)), data.get('masks', np.zeros(1))]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        images, labels, masks = tmp

        att_feats, fc_feats = cnn_model.forward(images)
        _att_feats = att_feats
        _fc_feats = fc_feats

        # forward the model to get loss
        if data.get('labels', None) is not None:
            att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), loader.seq_per_img,) + att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) * loader.seq_per_img,) + att_feats.size()[1:]))
            fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), loader.seq_per_img,) + fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) * loader.seq_per_img,) + fc_feats.size()[1:]))

            loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:]).data[0]
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        fc_feats, att_feats = _fc_feats, _att_feats
        # forward the model to also get generated samples for each image
        seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)

        #set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)
            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs['dump_path'] == 1:
                entry['file_name'] = data['infos'][k]['file_path']
                table.insert(predictions, entry)
            if eval_kwargs['dump_images'] == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    unseen_grams = None
    if lang_eval == 1:
        lang_stats, unseen_grams = language_eval(dataset, predictions)
    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats, unseen_grams
