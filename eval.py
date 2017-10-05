

#  from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle as pickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
from misc.ssd import build_ssd

NUM_THREADS = 2 #int(os.environ['OMP_NUM_THREADS'])

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model_path', type=str, default='',
                    help='path to model to evaluate')
parser.add_argument('--cnn_model_path', type=str, default='',
                    help='path to cnn model to evaluate')
parser.add_argument('--infos_path', type=str, default='',
                    help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
                    help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                    help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0,
                    help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                    help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--output_json', type=str,
                    help='json path')
parser.add_argument('--dump_path', type=int, default=0,
                    help='Write image paths along with predictions into vis json? (1=yes,0=no)')
# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                    help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=0.5,
                    help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='',
                    help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_list', type=str,
                    help='List of image from folder')
parser.add_argument('--max_images', type=int, default=-1,
                    help='If not -1 limit the number of evaluated images')

parser.add_argument('--image_root', type=str, default='data/coco/images',
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
parser.add_argument('--model', type=str, default='',
                    help='shortcut')

opt = parser.parse_args()

if len(opt.model_path) == 0:
    opt.model_path = "save/%s/model.pth" % opt.model
    opt.infos_path = "save/%s/infos.pkl" % opt.model

# Load infos
with open(opt.infos_path, 'rb') as f:
    print('Opening %s' % opt.infos_path)
    infos = pickle.load(f, encoding="iso-8859-1")

# override and collect parameters
if len(opt.input_h5) == 0:
    opt.input_h5 = infos['opt'].input_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
#Check if new features in opt:
if "raml_loss" not in opt:
    opt.raml_loss = 0
if "less_confident" not in opt:
    opt.less_confident = 0
if "scheduled_sampling_strategy" not in opt:
    opt.scheduled_sampling_strategy = "step"
if "scheduled_sampling_vocab" not in opt:
    opt.scheduled_sampling_vocab = 0


ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
for k in list(vars(infos['opt']).keys()):
    if k not in ignore:
        if k in vars(opt):
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping
opt.use_feature_maps = infos['opt'].use_feature_maps
opt.cnn_model = infos['opt'].cnn_model
opt.logger = opts.create_logger('./tmp_eval.log')
opt.start_from = "save/" + opt.model
opt.rnn_bias = 0
try:
    opt.use_glove = infos['opt'].use_glove
except:
    opt.use_glove = 0
try:
    opt.use_synonyms = infos['opt'].use_synonyms
except:
    opt.use_synonyms = 0
# Setup the model
if opt.use_feature_maps:
    ###############################################################################################################
    print('using single CNN branch with feature maps as regions embeddings')
    # Build CNN model for single branch use
    if opt.cnn_model.startswith('resnet'):
        cnn_model = utils.ResNetModel(opt)
    elif opt.cnn_model.startswith('vgg'):
        cnn_model = utils.VggNetModel(opt)
    else:
        print('Unknown model %s' % opt.cnn_model)
        sys.exit(1)
    ################################################################################################################
else:
    print('using SSD')
    cnn_model = build_ssd('train', 300, 21)

cnn_model.cuda()
cnn_model.eval()
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model_path))
model.cuda()
model.eval()
#  print('Parsed options:', opt)
crit = utils.LanguageModelCriterion(opt)

# Create the Data Loader instance
start = time.time()
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder,
                          'files_list': opt.image_list,
                          'coco_json': opt.coco_json,
                           'batch_size': opt.batch_size,
                          'max_images': opt.max_images})
  loader.ix_to_word = infos['vocab']


# Set sample options
print('Options:', opt.beam_size)
loss, split_predictions, lang_stats, _ = eval_utils.eval_eval(cnn_model, model, crit, loader, vars(opt))
#  eval_kwargs = {'split': 'val',
#                 'dataset': opt.input_json}
#  eval_kwargs.update(vars(infos['opt']))
#  eval_kwargs.update(vars(opt))
#  eval_kwargs['num_images'] = opt.max_images
#  eval_kwargs['beam_size'] = opt.beam_size
#  print("Evaluation beam size:", eval_kwargs['beam_size'])
#  predictions = eval_utils.eval_external(cnn_model, model, crit, loader, eval_kwargs)
print("Finished evaluation in ", (time.time() - start))
if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open(opt.output_json, 'w'))
