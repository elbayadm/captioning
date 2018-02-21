import os
import subprocess
import os.path as osp
import copy
import glob
import json
import numpy as np
import time
import pickle as pickle
import argparse


def exec_cmd(command):
    # return stdout, stderr output of a command
    return subprocess.Popen(command, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE).communicate()


def get_gpu_memory(gpuid):
    # Get the current gpu usage.
    result, _ = exec_cmd('nvidia-smi -i %d --query-gpu=memory.free \
                         --format=csv,nounits,noheader' % int(gpuid))
    # Convert lines into a dictionary
    result = int(result.strip())
    return result


def main(ens_opt):
    # setup gpu
    try:
        gpu_id = int(subprocess.check_output('gpu_getIDs.sh', shell=True))
    except:
        print("Failed to get gpu_id (setting gpu_id to %d)" % ens_opt.gpu_id)
        gpu_id = str(ens_opt.gpu_id)
        # beware seg fault if tf after torch!!
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    ens_opt.logger.warn('GPU ID: %s | available memory: %dM' \
                        % (os.environ['CUDA_VISIBLE_DEVICES'], get_gpu_memory(gpu_id)))
    import tensorflow as tf
    import torch
    import models.setup as ms
    from models.ensemble import Ensemble, eval_ensemble, eval_external_ensemble
    import models.cnn as cnn
    from loader import DataLoader, DataLoaderRaw

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
    import opts
    ens_opt = opts.parse_ens_opt()
    main(ens_opt)

