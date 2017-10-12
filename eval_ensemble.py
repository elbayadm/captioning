

#  from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle as pickle

import opts
import capmodels
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

NUM_THREADS = 2 #int(os.environ['OMP_NUM_THREADS'])

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--verbose', type=int, default=0,
                    help='code verbosity')
# Basic options
parser.add_argument('--batch_size', type=int, default=1,
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
parser.add_argument('--forbid_unk', type=int, default=1,
                    help='Forbid unk token generations.')

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
parser.add_argument('--model', nargs="+",
                    help='shortcut')

ens_opt = parser.parse_args()
models_paths = []
infos_paths = []
infos_list = []
cnn_models = []
rnn_models = []
for m in ens_opt.model:
    models_paths.append('save/%s/model.pth' % m)
    infos_path = "save/%s/infos.pkl" % m
    with open(infos_path, 'rb') as f:
        print('Opening %s' % infos_path)
        infos = pickle.load(f, encoding="iso-8859-1")
        infos_list.append(infos)

print("Parsed % infos files" % len(infos_list))
options = []
for e, infos in enumerate(infos_list):
    # override and collect parameters
    opt = argparse.Namespace(**vars(ens_opt))
    print('List of models:', ens_opt.model)
    opt.model = ens_opt.model[e]
    print("Current model:", opt.model)
    if len(opt.input_h5) == 0:
        opt.input_h5 = infos['opt'].input_h5
    if len(opt.input_json) == 0:
        opt.input_json = infos['opt'].input_json
    if opt.batch_size == 0:
        opt.batch_size = infos['opt'].batch_size
    #Check if new features in opt:
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
    opt.use_feature_maps = infos['opt'].use_feature_maps
    opt.cnn_model = infos['opt'].cnn_model
    opt.logger = opts.create_logger('./tmp_eval.log')
    opt.start_from = "save/" + opt.model
    opt.model_path = "save/" + opt.model + "/model.pth"

    opt.rnn_bias = 0
    try:
        opt.use_glove = infos['opt'].use_glove
    except:
        opt.use_glove = 0
    try:
        opt.use_synonyms = infos['opt'].use_synonyms
    except:
        opt.use_synonyms = 0
    options.append(opt)

vocab = infos_list[0]['vocab'] # ix -> word mapping
ens_opt.logger = opts.create_logger('./tmp_eval.log')

# Build CNN model for single branch use
for opt in options:
    if opt.cnn_model.startswith('resnet'):
        cnn_model = utils.ResNetModel(opt)
    elif opt.cnn_model.startswith('vgg'):
        cnn_model = utils.VggNetModel(opt)
    else:
        print('Unknown model %s' % opt.cnn_model)
        sys.exit(1)
    cnn_model.cuda()
    cnn_model.eval()
    print('Loading model with', opt)
    model = capmodels.setup(opt)
    model.load_state_dict(torch.load(opt.model_path))
    model.cuda()
    model.eval()
    cnn_models.append(cnn_model)
    rnn_models.append(model)

# Create the Data Loader instance
start = time.time()
external = False
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    external = True
    loader = DataLoaderRaw({'folder_path': opt.image_folder,
                              'files_list': opt.image_list,
                              'coco_json': opt.coco_json,
                               'batch_size': opt.batch_size,
                              'max_images': opt.max_images})
    loader.ix_to_word = infos['vocab']

if external:
    split_predictions = eval_utils.eval_external_ensemble(cnn_models, rnn_models, loader, vars(ens_opt))
else:
    split_predictions, lang_stats, _ = eval_utils.eval_ensemble(cnn_models, rnn_models, loader, vars(ens_opt))
print("Finished evaluation in ", (time.time() - start))
if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open(opt.output_json, 'w'))
