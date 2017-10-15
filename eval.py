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
import torch
from misc.ssd import build_ssd


if __name__ == "__main__":
    opt = opts.parse_eval_opt()
    opt.model = opt.model[0]
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
    if "bootstrap_loss" not in opt:
        opt.bootstrap_loss = 0
    if "less_confident" not in opt:
        opt.less_confident = 0
    if "scheduled_sampling_strategy" not in opt:
        opt.scheduled_sampling_strategy = "step"
    if "scheduled_sampling_vocab" not in opt:
        opt.scheduled_sampling_vocab = 0


    ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval", "input_h5"]
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
    # opt.seq_per_img = 15
    if opt.use_feature_maps:
        print('using single CNN branch with feature maps as regions embeddings')
        # Build CNN model for single branch use
        if opt.cnn_model.startswith('resnet'):
            cnn_model = cnn.ResNetModel(opt)
        elif opt.cnn_model.startswith('vgg'):
            cnn_model = cnn.VggNetModel(opt)
        else:
            print('Unknown model %s' % opt.cnn_model)
            sys.exit(1)
    else:
        print('using SSD')
        cnn_model = build_ssd('train', 300, 21)

    cnn_model.cuda()
    cnn_model.eval()
    model = du.select_model(opt)
    model.load()
    # model.load_state_dict(torch.load(opt.model_path))
    model.cuda()
    model.eval()
    #  print('Parsed options:', opt)
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
    print('Seq per img:', loader.seq_per_img)
    model.define_loss(loader.get_vocab())
    loss, split_predictions, lang_stats, _ = eval_utils.eval_eval(cnn_model, model,
                                                                  model.crit,
                                                                  loader, vars(opt))
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
