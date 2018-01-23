import random
from math import exp
import numpy as np
import time
import os
import os.path as osp
import sys
import pickle
import json
import subprocess
import utils


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


if __name__ == "__main__":
    opt = utils.parse_eval_opt()
    # setup gpu
    try:
        gpu_id = int(subprocess.check_output('gpu_getIDs.sh', shell=True))
    except:
        print("Failed to get gpu_id (setting gpu_id to %d)" % opt.gpu_id)
        gpu_id = str(opt.gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import torch
    opt.logger.warn('GPU ID: %s | available memory: %dM' \
                    % (os.environ['CUDA_VISIBLE_DEVICES'], get_gpu_memory(gpu_id)))
    from loader import DataLoader, DataLoaderRaw
    import models.eval_utils as evald
    import models.cnn as cnn
    import models.setup as ms
    import utils.logging as lg

    # reproducibility:
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

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
              "use_adaptive_pooling", "clip_reward"]
    for k in list(vars(infos['opt']).keys()):
        if k not in ignore and "learning" not in k:
            if k in vars(opt):
                assert vars(opt)[k] == vars(infos['opt'])[k], (k + ' option not consistent ' +
                                                               str(vars(opt)[k]) + ' vs. ' + str(vars(infos['opt'])[k]))
            else:
                vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

    opt.fliplr = opt.fliplr_eval
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

    # if opt.beam_size > 1:
        # opt.sample_max = 1
    model.define_loss(loader.get_vocab())
    # opt.n_gen = 10
    # opt.score_ground_truth = True
    eval_kwargs = {'split': 'val',
                   'dataset': opt.input_data + '.json'}
    eval_kwargs.update(vars(opt))
    print('evaluation setting:')
    print('Split: %s\nBatch: %d\nBeam: %d\nArgMax: %d\nT: %.2e\nForbid UNK: %d' % (eval_kwargs['split'],
                                                                                   eval_kwargs['batch_size'],
                                                                                   eval_kwargs['beam_size'],
                                                                                   eval_kwargs['sample_max'],
                                                                                   eval_kwargs['temperature'],
                                                                                   eval_kwargs['forbid_unk']))
    ml_loss, loss, preds, perf = evald.eval_split(cnn_model,
                                                  model,
                                                  loader,
                                                  opt.logger,
                                                  eval_kwargs)

    print("Finished evaluation in ", (time.time() - start))
    print('ML loss:', ml_loss)
    print('Training loss:', loss)
    perf['loss'] = loss
    perf['ml_loss'] = ml_loss

    perf['params'] = eval_kwargs
    perf['params']['logger'] = None

    if not opt.output:
        evaldir = '%s/evaluations/%s' % (opt.modelname, opt.split)
        if not osp.exists(evaldir):
            os.makedirs(evaldir)
        opt.output = '%s/bw%d' % (evaldir, opt.beam_size)
        if opt.beam_size == 1:
            sampling = "_samplemax" if opt.sample_max else "_sample_temp_%.3f" % opt.temperature
            opt.output += sampling
        if opt.fliplr:
            opt.output += "_flip"

    pickle.dump(perf, open(opt.output + ".res", 'wb'))
    if opt.dump_json:
        json.dump(preds, open(opt.output + '.json', 'w'))
