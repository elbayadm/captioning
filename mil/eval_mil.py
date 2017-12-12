import json
import time
import os
import numpy as np
from six.moves import cPickle as pickle
import opts
import models
from misc.mil import upsample_images
from dataloader import *
from dataloaderraw import *
import argparse
import misc.utils as utils
import torch
from misc.ssd import build_ssd
from misc.mil import VGG_MIL


def eval_mil_extended(cnn_model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')
    logger = eval_kwargs.get('logger')
    upsampling_size = eval_kwargs.get('upsampling_size')
    vocab_size = eval_kwargs.get('vocb_size')
    max_tokens = eval_kwargs.get('max_tokens', 16)

    # Make sure in the evaluation mode
    cnn_model.eval()
    logger.warn('Evaluating %d val images' % val_images_use)

    loader.reset_iterator(split)
    n = 0
    loss_sum = 0
    loss_evals = 0
    predictions = []
    seq_per_img = 5
    while True:
        data = loader.get_batch(split, seq_per_img=seq_per_img, batch_size=1)
        n = n + loader.batch_size
        # forward the model to get loss
        tmp = [data['images'], data['labels']]
        tmp[0] = upsample_images(tmp[0], upsampling_size)
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        images, labels = tmp
        #  print("Images:", images.size())
        regions, xf, probs = cnn_model.forward_extended(images)
        xf = xf.cpu().data.numpy()
        regions = regions.cpu().data.numpy()
        loss = crit(probs, labels[:, 1:])
        loss = loss.data[0]
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1
        # Pick the 16th most probable tokens as predictions:
        probs = probs.squeeze(0).cpu().data.numpy()
        indices = np.argpartition(-probs, max_tokens)[:max_tokens]
        print('Tokens', indices, indices.shape)
        sel = probs[indices]
        region_indices = np.argmax(xf[:, :, indices], axis=1)
        print("Matched regions:", region_indices, region_indices.shape)
        print('Region probas:', xf[0][region_indices, indices])
        region_codes = regions[0, region_indices]
        print('Region codes:', region_codes.shape)
        # sel, indices = torch.sort(probs, dim=1, descending=True)
        indices = np.expand_dims(indices, axis=0)
        sents = utils.decode_sequence(loader.get_vocab(), torch.from_numpy(indices[:, :max_tokens]))
        #  print("Output:", sents)
        for k in range(loader.batch_size):
            entry = {'image_id': data['infos'][k]['id'], 'words': sents[k], "probs": sel,
                     'regions': region_indices[0], 'region probs': xf[0][region_indices, indices][0],
                     'region codes': region_codes[0]}
            print(entry)
            predictions.append(entry)
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
    # Switch back to training mode
    cnn_model.train()
    #  pickle.dump(Feats, open('cnn_features.pkl', 'w'))
    return loss_sum/loss_evals, predictions


def eval_mil(cnn_model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')
    logger = eval_kwargs.get('logger')
    upsampling_size = eval_kwargs.get('upsampling_size')

    vocab_size = eval_kwargs.get('vocb_size')
    max_tokens = eval_kwargs.get('max_tokens', 16)

    # Make sure in the evaluation mode
    cnn_model.eval()
    logger.warn('Evaluating %d images' % val_images_use)

    loader.reset_iterator(split)
    n = 0
    loss_sum = 0
    loss_evals = 0
    predictions = []
    seq_per_img = 5
    while True:
        data = loader.get_batch(split, seq_per_img=seq_per_img, batch_size=1)
        n = n + loader.batch_size
        # forward the model to get loss
        tmp = [data['images'], data['labels']]
        tmp[0] = upsample_images(tmp[0], upsampling_size)
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        images, labels = tmp
        #  print("Images:", images.size())
        probs = cnn_model(images)
        loss = crit(probs, labels[:, 1:])
        loss = loss.data[0]
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1
        # Pick the 16th most probable tokens as predictions:
        # print("Probabilities:", probs)
        # probs = probs.squeeze(0).cpu().data.numpy()
        # print('Tokens', indices)
        # sel = probs[indices]
        # indices = np.expand_dims(indices, axis=0)
        sel, indices = torch.sort(probs, dim=1, descending=True)
        # print('highest probas', sel)
        # sents = utils.decode_sequence(loader.get_vocab(), torch.from_numpy(indices[:, :max_tokens]))
        sents = utils.decode_sequence(loader.get_vocab(), indices[:, :max_tokens].cpu().data)

        #  print("Output:", sents)
        for k in range(loader.batch_size):
            entry = {'image_id': data['infos'][k]['id'], 'words': sents[k]}
            predictions.append(entry)
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
    # Switch back to training mode
    cnn_model.train()
    #  pickle.dump(Feats, open('cnn_features.pkl', 'w'))
    return loss_sum/loss_evals, predictions




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
parser.add_argument('--split', type=str, default='val',
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
print('Loading infos file %s' % opt.infos_path)
infos = pickle.load(open(opt.infos_path, 'rb'), encoding='iso-8859-1')
# override and collect parameters
if len(opt.input_h5) == 0:
    opt.input_h5 = infos['opt'].input_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
#Check if new features in opt:
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
cnn_model = VGG_MIL(opt)
cnn_model.cuda()
cnn_model.eval()
crit = utils.MIL_crit(opt)
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
if not 'upsampling_size' in opt:
    opt.upsampling_size = 300
eval_kwargs = {'split': opt.split,
               'dataset': opt.input_json,
               'upsampling_size' : opt.upsampling_size}
eval_kwargs.update(vars(infos['opt']))
eval_kwargs.update(vars(opt))
eval_kwargs['num_images'] = opt.max_images
eval_kwargs['beam_size'] = opt.beam_size
print("Evaluation beam size:", eval_kwargs['beam_size'])
val_loss, predictions = eval_mil_extended(cnn_model, crit, loader, eval_kwargs)
print("Finished evaluation in ", (time.time() - start))
if opt.dump_json == 1:
    # dump the json
    pickle.dump(predictions, open(opt.output_json, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
