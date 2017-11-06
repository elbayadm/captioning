# setup gpu
try:
    import os
    import subprocess
    gpu_id = subprocess.check_output('gpu_getIDs.sh', shell=True)
    print("Gpu%s" % gpu_id)
except:
    print("Failed to get gpu_id (setting gpu_id to 0)")
    gpu_id = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import os
import os.path as osp
import sys
from six.moves import cPickle as pickle
import copy
import opts
import models
from dataloader import *
import misc.utils as utils
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.tensorboard.plugins import projector
from misc.lm import MultiVAE_LM_encoder, VAE_LM_encoder, LM_encoder, LM_decoder

import json
from dataloaderraw import *
import argparse
import misc.utils as utils
from opts import create_logger


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
    sample_max = eval_kwargs.get('sample_max', 1)
    temperature = eval_kwargs.get('temperature', 1.0)
    print('Using sample_max = %d  ||  temperature %.2f' % (sample_max, temperature))

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
    gen_SENTS = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        # forward the model to get loss
        tmp = [data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        labels, masks = tmp
        if lm_model == "rnn":
            codes = encoder(labels)
        elif lm_model == "rnn_multi_vae":
            z_mu, z_var, codes = encoder(labels)
            r_codes = encoder.sample(labels)
            r_codes1 = encoder.sample_group(labels)
            r_codes2 = encoder.sample_group(labels)
        elif lm_model == 'rnn_vae':
            z_mu, z_var, codes = encoder(labels)
            r_codes = encoder.sample(labels)
            r_codes2 = encoder.sample(labels)
        gt = utils.decode_sequence(loader.get_vocab(), labels[:,1:].data)
        SENTS += gt
        seq, _ = decoder.sample(codes, {'beam_size': beam_size, "vocab_size": vocab_size, "sample_max": sample_max, "temperature": temperature})
        loss = crit(decoder(codes, labels), labels[:,1:], masks[:,1:])[0].data[0]
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        gen_SENTS += sents
        try:
            r_seq, _ = decoder.sample(r_codes, {'beam_size': beam_size, "vocab_size": vocab_size})
            r_loss = crit(decoder(r_codes, labels), labels[:,1:], masks[:,1:])[0].data[0]
            r_sents = utils.decode_sequence(loader.get_vocab(), r_seq)

            r_seq1, _ = decoder.sample(r_codes1, {'beam_size': beam_size, "vocab_size": vocab_size})
            r_loss1 = crit(decoder(r_codes1, labels), labels[:,1:], masks[:,1:])[0].data[0]
            r_sents1 = utils.decode_sequence(loader.get_vocab(), r_seq1)

            r_seq2, _ = decoder.sample(r_codes2, {'beam_size': beam_size, "vocab_size": vocab_size})
            r_loss2 = crit(decoder(r_codes2, labels), labels[:,1:], masks[:,1:])[0].data[0]
            r_sents2 = utils.decode_sequence(loader.get_vocab(), r_seq2)

            k = 0
            for co, de, r_de, r_de1, r_de2 in zip(gt, sents, r_sents, r_sents1,  r_sents2):
                print(" %d) source:" % data['infos'][k]['id'], co, _OKGREEN, "\n>> (group, z)", de, _ENDC, _WARNING, "\n>> (single, rand z)", r_de, "\n>> (group, rand z)", r_de1, "\n>> (group, rand z)", r_de2, _ENDC)
            print("Loss (group, z): %.2f, single, rand z: %.2f, group, rand z: %.2f, group, rand z: %.2f" % (loss, r_loss, r_loss1, r_loss2))

        except:
            for co, de in zip(gt, sents):
                print("source:", co, _OKGREEN, "\n>> (determ.)", de, _ENDC)

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
    if lang_eval == 1:
        lang_stats = language_lm_eval(SENTS, gen_SENTS)
    print(lang_stats)
    # Switch back to training mode
    encoder.train()
    decoder.train()
    return loss_sum/loss_evals, lang_stats


# Input arguments and options
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='',
                    help='model name')
parser.add_argument('--model_path', type=str, default='',
                    help='path to model to evaluate')
parser.add_argument('--cnn_model_path', type=str, default='',
                    help='path to cnn model to evaluate')
parser.add_argument('--infos_path', type=str, default='',
                    help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=1,
                     help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                    help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=1,
                    help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                    help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                    help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                    help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=2,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='',
                     help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='',
                    help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_h5', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='',
                    help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test',
                    help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='',
                    help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--tries', type=int, default=1,
                     help='number ofsampling ties')

opt = parser.parse_args()
opt.infos_path = 'save/textLM/%s/infos.pkl' % opt.model
opt.model_path = 'save/textLM/%s/model.pth' % opt.model
# Load infos
with open(opt.infos_path) as f:
    infos = pickle.load(f)

saved_model_opt = infos['opt']

# override and collect parameters
if len(opt.input_h5) == 0:
    opt.input_h5 = infos['opt'].input_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
for k in list(vars(infos['opt']).keys()):
    if k not in ignore:
        if k in vars(opt):
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistenti (' + str(vars(opt)[k]) + ' vs. ' + str(vars(infos['opt'])[k])+ ')'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

opt.logger = create_logger('./tmplog_eval.log')
vocab = infos['vocab'] # ix -> word mapping

# Build the captioning model
# Build the captioning model
try:
    if saved_model_opt.lm_model == "rnn_vae":
        opt.logger.warn('Injecting VAE block in the encoder-decoder model')
        encoder = VAE_LM_encoder(opt)
    elif saved_model_opt.lm_model == "rnn_multi_vae":
        opt.logger.warn('Injecting VAE block in the encoder-decoder model')
        encoder = MultiVAE_LM_encoder(opt)
except:
    saved_model_opt.lm_model = "rnn"
    opt.logger.warn('Using Basic RNN encoder-decoder')
    encoder = LM_encoder(opt)

decoder = LM_decoder(opt)
encoder.cuda()
decoder.cuda()
model = nn.Sequential(encoder,
                        decoder)

model.load_state_dict(torch.load(opt.model_path))
model.cuda()
model.eval()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    if 'coco' in opt.input_h5:
        loader = DataLoader(opt)
    else:
        loader = textDataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size})
    loader.ix_to_word = infos['vocab']
#  opt.val_images_use = 5
opt.lm_model = saved_model_opt.lm_model
opt.output_file = 'data/coco/generated_captions_%s_%d.json' % (opt.model, int(10 * opt.temperature))
#  opt.tries = 1
crit = utils.LanguageModelCriterion()
#  loss = eval_utils.eval_lm_split(encoder, decoder, crit, loader, vars(opt))
#  print "Loss:", loss
print(opt)
eval_lm_split(encoder, decoder, crit, loader, vars(opt))

