import glob
import json
import numpy as np
import time
import os
from six.moves import cPickle as pickle
import opts
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import misc.decoder_utils as du
import misc.cnn as cnn
from misc.ensemble import Ensemble
import torch

def eval_finetuned(ens_opt):
    with open(ens_opt.ensemblename + '/infos.pkl', 'rb') as f:
        print('Opening %s' % ens_opt.infos_path)
        infos = pickle.load(f, encoding="iso-8859-1")
    models_paths = []
    infos_paths = []
    infos_list = []
    cnn_models = []
    rnn_models = []
    options = []
    print('Ensemble ref:', ens_opt.ensemblename)
    saved_models = glob.glob(ens_opt.ensemblename + '/model-best_*')  # Add best as option
    # get indices to loop
    indices = [int(f.split('_')[-1].split('.')[0]) for f in saved_models]
    print('Parsed indices:', indices)
    print("Saved models:", saved_models)
    ens_opt.model = []
    for ind, pth in zip(indices, saved_models):
        models_paths.append(pth)
        infos_path = ens_opt.ensemblename + '/infos_%d' % ind + ".pkl"  # FIXME add bestas option
        ens_opt.model.append(pth)
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
        ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval", "logger"]
        for k in list(vars(infos['opt']).keys()):
            if k not in ignore:
                if k in vars(opt):
                    assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
                else:
                    vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model
        opt.use_feature_maps = infos['opt'].use_feature_maps
        opt.cnn_model = infos['opt'].cnn_model
        opt.logger = ens_opt.logger
        opt.start_from = ens_opt.ensemblename  # Change according to where i'll be using this
        opt.model_path = opt.model

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

    # Build CNN model for single branch use
    for opt in options:
        if opt.cnn_model.startswith('resnet'):
            cnn_model = cnn.ResNetModel(opt)
        elif opt.cnn_model.startswith('vgg'):
            cnn_model = cnn.VggNetModel(opt)
        else:
            print('Unknown model %s' % opt.cnn_model)
            sys.exit(1)
        cnn_model.cuda()
        cnn_model.eval()
        print('Loading model with', opt)
        model = du.select_model(opt)
        # model.load_state_dict(torch.load(opt.model_path))
        model.load()
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


    # Define the ensemble:
    ens_model = Ensemble(rnn_models, cnn_models, ens_opt)

    if external:
        split_predictions = eval_utils.eval_external_ensemble(ens_model, loader, vars(ens_opt))
    else:
        split_predictions, lang_stats, _ = eval_utils.eval_ensemble(ens_model, loader, vars(ens_opt))
    print("Finished evaluation in ", (time.time() - start))
    if opt.dump_json == 1:
        # dump the json
        json.dump(split_predictions, open(opt.output_json, 'w'))



def eval(ens_opt):
    models_paths = []
    infos_paths = []
    infos_list = []
    cnn_models = []
    rnn_models = []
    options = []
    for m in ens_opt.model:
        models_paths.append('save/%s/model-best.pth' % m) #FIXME check that cnn-best is the one loaded
        infos_path = "save/%s/infos-best.pkl" % m
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
        ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval", "logger"]
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
        opt.model_path = "save/" + opt.model + "/model-best.pth"

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

    # Build CNN model for single branch use
    for opt in options:
        if opt.cnn_model.startswith('resnet'):
            cnn_model = cnn.ResNetModel(opt)
        elif opt.cnn_model.startswith('vgg'):
            cnn_model = cnn.VggNetModel(opt)
        else:
            print('Unknown model %s' % opt.cnn_model)
            sys.exit(1)
        cnn_model.cuda()
        cnn_model.eval()
        print('Loading model with', opt)
        model = du.select_model(opt)
        # model.load_state_dict(torch.load(opt.model_path))
        model.load()
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


    # Define the ensemble:
    ens_model = Ensemble(rnn_models, cnn_models, ens_opt)

    if external:
        split_predictions = eval_utils.eval_external_ensemble(ens_model, loader, vars(ens_opt))
    else:
        split_predictions, lang_stats, _ = eval_utils.eval_ensemble(ens_model, loader, vars(ens_opt))
    print("Finished evaluation in ", (time.time() - start))
    if opt.dump_json == 1:
        # dump the json
        json.dump(split_predictions, open(opt.output_json, 'w'))


if __name__ == "__main__":
    ens_opt = opts.parse_eval_opt()
    ens_opt.logger = opts.create_logger('./tmp_eval.log')
    # Evaluaing a finetuned ensemble
    if ens_opt.ensemblename:
        eval_finetuned(ens_opt)
    else:
        eval(ens_opt)

