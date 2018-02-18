import os.path as osp
import copy
import glob
import json
import numpy as np
import time
import os
import pickle as pickle
import utils
from utils import opts
from loader import DataLoader, DataLoaderRaw
import models.setup as ms
from utils.language_eval import language_eval
from utils.logging import print_sampled
import argparse
import models.cnn as cnn
from models.ensemble import Ensemble
import torch
from torch.autograd import Variable


def short_path(path):
    return int(path.split('.')[0].split('_')[-1])


def eval_external_ensemble(ensemble, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', -1)
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    logger = eval_kwargs.get('logger')
    caption_model = eval_kwargs.get('caption_model')
    vocab_size = eval_kwargs.get('vocb_size')
    dump_path = eval_kwargs.get('dump_path')

    # Make sure in the evaluation mode
    for cnn_model  in ensemble.cnn_models:
        cnn_model.eval()

    for model in ensemble.models:
        model.eval()

    loader.reset_iterator(split)

    n = 0
    predictions = []
    Feats = []
    seq_per_img = 5
    while True:
        data = loader.get_batch(split, seq_per_img=seq_per_img)
        n = n + loader.batch_size

        # forward the model to get loss
        images = data['images']
        images = Variable(torch.from_numpy(images), volatile=True).cuda()

        att_feats_ens, fc_feats_ens = ensemble.get_feats(images)
        seq, probs = ensemble.sample(fc_feats_ens, att_feats_ens, eval_kwargs)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            spath = short_path(data['infos'][k]['file_path'])
            print_sampled(spath, sent)
            entry = {'image_id': spath, 'caption': sent}
            predictions.append(entry)
            #  logger.debug('image %s: %s' %(entry['image_id'], entry['caption']))
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        #  logger.warn('ix1 = %d - ix0 = %d' % (ix1, ix0))
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
        #  logger.debug('validation loss ... %d/%d (%f)' %(ix0 - 1, ix1, loss))
        if data['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
    #  pickle.dump(Feats, open('cnn_features.pkl', 'w'))
    return  predictions



def eval_ensemble(ens_model, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    seq_length = eval_kwargs.get('seq_length', 16)
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    batch_size = eval_kwargs.get('batch_size', 1)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    print('Evaluating ', val_images_use, ' images')

    # Make sure in the evaluation mode
    for cnn_model in ens_model.cnn_models:
        cnn_model.eval()

    for model in ens_model.models:
        model.eval()

    loader.reset_iterator(split)

    n = 0
    # loss_sum = 0
    # real_loss_sum = 0
    # loss_evals = 0
    predictions = []

    while True:
        # fetch a batch of data
        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        #evaluate loss if we have the labels
        # loss = 0

        # Get the image features first
        tmp = [data['images'], data.get('labels', np.zeros(1)), data.get('masks', np.zeros(1)), data['scores']]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        images, labels, masks, scores = tmp

        att_feats_ens = []
        fc_feats_ens = []
        for cnn_model in ens_model.cnn_models:
            att_feats, fc_feats = cnn_model.forward(images)
            att_feats_ens.append(att_feats)
            fc_feats_ens.append(fc_feats)
        # Eavluate the loss:
        # real_loss, loss = ens_model.step(data)
        # loss_sum = loss_sum + loss.data[0]
        # real_loss_sum += real_loss.data[0]
        # loss_evals = loss_evals + 1

        seq, probs = ens_model.sample(fc_feats_ens, att_feats_ens, eval_kwargs)
        sent_scores = probs.cpu().numpy().sum(axis=1)
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        for k, sent in enumerate(sents):
            # print('id:', data['infos'][k]['id'])
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'score': str(round(sent_scores[k], 4)), "source": 'gen'}
            predictions.append(entry)
            if verbose:
                print(('image %s (%s) %s' %(entry['image_id'], entry['score'], entry['caption'])))
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        if data['bounds']['wrapped']:
            break
        if n >= ix1:
            ens_model.logger.warn('Evaluated the required samples (%s)' % n)
            break
    lang_stats = None
    unseen_grams = None
    if lang_eval == 1:
        lang_stats, unseen_grams = language_eval(dataset, predictions, ens_opt.logger, get_creativity=False)
    # Switch back to training mode
    # model.train()
    return predictions, lang_stats


def eval(ens_opt):
    models_paths = []
    cnn_models = []
    rnn_models = []
    options = []
    # Reformat:
    for m in ens_opt.models:
        models_paths.append('save/%s/model-best.pth' % m)  # FIXME check that cnn-best is the one loaded
        infos_path = "save/%s/infos-best.pkl" % m
        with open(infos_path, 'rb') as f:
            print('Opening %s' % infos_path)
            infos = pickle.load(f, encoding="iso-8859-1")
        vocab = infos['vocab']
        iopt = infos['opt']
        # define single model options
        params = copy.copy(vars(ens_opt))
        params.update(vars(iopt))
        opt = argparse.Namespace(**params)
        opt.modelname = 'save/'+m
        opt.start_from_best = ens_opt.start_from_best
        opt.beam_size = ens_opt.beam_size
        opt.batch_size = ens_opt.batch_size
        opt.logger = ens_opt.logger
        if opt.start_from_best:
            flag = '-best'
            opt.logger.warn('Starting from the best saved model')
        else:
            flag = ''
        opt.cnn_start_from = osp.join(opt.modelname, 'model-cnn%s.pth' % flag)
        opt.infos_start_from = osp.join(opt.modelname, 'infos%s.pkl' % flag)
        opt.start_from = osp.join(opt.modelname, 'model%s.pth' % flag)
        opt.logger.warn('Starting from %s' % opt.start_from)

        # Load infos
        with open(opt.infos_start_from, 'rb') as f:
            print('Opening %s' % opt.infos_start_from)
            infos = pickle.load(f, encoding="iso-8859-1")
            infos['opt'].logger = None
        ignore = ["batch_size", "beam_size", "start_from",
                  'cnn_start_from', 'infos_start_from',
                  "start_from_best", "language_eval", "logger",
                  "val_images_use", 'input_data', "loss_version", "region_size",
                  "use_adaptive_pooling", "clip_reward",
                  "gpu_id", "max_epochs", "modelname", "config",
                  "sample_max", "temperature"]
        for k in list(vars(infos['opt']).keys()):
            if k not in ignore and "learning" not in k:
                if k in vars(opt):
                    assert vars(opt)[k] == vars(infos['opt'])[k], (k + ' option not consistent ' +
                                                                   str(vars(opt)[k]) + ' vs. ' + str(vars(infos['opt'])[k]))
                else:
                    vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

        opt.fliplr = 0
        opt.language_creativity = 0
        opt.seq_per_img = 5
        opt.bootstrap = 0
        opt.sample_cap = 0
        vocab = infos['vocab']  # ix -> word mapping
        # Build CNN model for single branch use
        if opt.cnn_model.startswith('resnet'):
            cnn_model = cnn.ResNetModel(opt)
        elif opt.cnn_model.startswith('vgg'):
            cnn_model = cnn.VggNetModel(opt)
        else:
            print('Unknown model %s' % opt.cnn_model)
            sys.exit(1)

        cnn_model.cuda()
        cnn_model.eval()
        model = ms.select_model(opt)
        model.load()
        model.cuda()
        model.eval()
        options.append(opt)
        cnn_models.append(cnn_model)
        rnn_models.append(model)

        # Create the Data Loader instance
    start = time.time()
    external = False
    if len(ens_opt.image_folder) == 0:
        loader = DataLoader(options[0])
    else:
        external = True
        loader = DataLoaderRaw({'folder_path': ens_opt.image_folder,
                                'files_list': ens_opt.image_list,
                                'batch_size': ens_opt.batch_size})
        loader.ix_to_word = vocab


    # Define the ensemble:
    ens_model = Ensemble(rnn_models, cnn_models, ens_opt)

    if external:
        preds = eval_external_ensemble(ens_model, loader, vars(ens_opt))
    else:
        preds, lang_stats = eval_ensemble(ens_model, loader, vars(ens_opt))
    print("Finished evaluation in ", (time.time() - start))
    if ens_opt.dump_json == 1:
        # dump the json
        json.dump(preds, open(ens_opt.output+".json", 'w'))


if __name__ == "__main__":
    ens_opt = opts.parse_ens_opt()
    ens_opt.models = [_[0] for _ in ens_opt.models]
    print('Models:', ens_opt.models)
    if not ens_opt.output:
        if not len(ens_opt.image_folder):
            evaldir = '%s/evaluations/%s' % (ens_opt.ensemblename, ens_opt.split)
        else:
            ens_opt.split = ens_opt.image_list.split('/')[-1].split('.')[0]
            print('Split :: ', ens_opt.split)
            evaldir = '%s/evaluations/server_%s' % (ens_opt.ensemblename, ens_opt.split)

        if not osp.exists(evaldir):
            os.makedirs(evaldir)
        ens_opt.output = '%s/bw%d' % (evaldir, ens_opt.beam_size)
    eval(ens_opt)

