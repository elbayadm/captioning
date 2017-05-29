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
import eval_utils
import misc.utils as utils
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.tensorboard.plugins import projector
from misc.lm import VAE_LM_encoder, LM_encoder, LM_decoder
import json
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
from opts import create_logger


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
parser.add_argument('--id', type=str, default='evalscript',
                     help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')

opt = parser.parse_args()
opt.infos_path = 'save/%s/infos.pkl' % opt.model
opt.model_path = 'save/%s/model.pth' % opt.model
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
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistenti (' + str(vars(opt)[k]) + ' vs. ' + str(vars(infos['opt'])[k])+ ')'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

opt.logger = create_logger('./tmplog_eval.log')
vocab = infos['vocab'] # ix -> word mapping

# Build the captioning model
# Build the captioning model
if saved_model_opt.lm_model == "rnn":
    opt.logger.warn('Using Basic RNN encoder-decoder')
    encoder = LM_encoder(opt)
elif saved_model_opt.lm_model == "rnn_vae":
    opt.logger.warn('Injecting VAE block in the encoder-decoder model')
    encoder = VAE_LM_encoder(opt)
else:
    raise ValueError('Unknown LM model %s' % opt.lm_model)
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
opt.val_images_use = 200
crit = utils.LanguageModelCriterion()
loss = eval_utils.eval_lm_split(encoder, decoder, crit, loader, vars(opt))
print "Loss:", loss

