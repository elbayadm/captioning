import sys
from shutil import copy2
import glob
import pickle
import os.path as osp

import torch
import torch.optim as optim

from .ShowTellModel import ShowTellModel
from .AdaptiveAttentionModel import AdaptiveAttentionModel
from .TopDownModel import TopDownModel
from .SelfCriticalModel import SelfCriticalModel


def select_model(opt):
    select = opt.caption_model
    if select == 'show_tell':
        model = ShowTellModel(opt)
    elif select == 'adaptive_attention':
        model = AdaptiveAttentionModel(opt)
    elif select == 'self_critical':
        model = SelfCriticalModel(opt)
    elif select == 'top_down':
        model = TopDownModel(opt)
    else:
        raise ValueError("Caption model not supported: {}".format(select))
    return model


def recover_infos(opt):
    infos = {}
    # Restart training (useful with oar idempotant)
    if opt.restart and osp.exists(osp.join(opt.modelname, 'model.pth')):
        opt.start_from_best = 0
        opt.logger.warning('Picking up where we left')
        opt.start_from = osp.join(opt.modelname, 'model.pth')
        opt.cnn_start_from = osp.join(opt.modelname, 'model-cnn.pth')
        opt.infos_start_from = osp.join(opt.modelname, 'infos.pkl')
        opt.optimizer_start_from = osp.join(opt.modelname, 'optimizer.pth')
        opt.cnn_optimizer_start_from = osp.join(opt.modelname, 'cnn-optimizer.pth')
        infos = pickle.load(open(opt.infos_start_from, 'rb'), encoding='iso-8859-1')

    elif opt.start_from is not None:
        # open old infos and check if models are compatible
        # start_from of the config file is a folder name
        opt.logger.warn('Starting from %s' % opt.start_from)
        if opt.start_from_best:
            flag = '-best'
            opt.logger.warn('Starting from the best saved model')
        else:
            flag = ''
        opt.cnn_start_from = osp.join(opt.start_from, 'model-cnn%s.pth' % flag)
        opt.infos_start_from = osp.join(opt.start_from, 'infos%s.pkl' % flag)
        if not opt.reset_optimizer:
            opt.optimizer_start_from = osp.join(opt.start_from, 'optimizer%s.pth' % flag)
            opt.cnn_optimizer_start_from = osp.join(opt.start_from, 'cnn-optimizer%s.pth' % flag)
        opt.start_from = osp.join(opt.start_from, 'model%s.pth' % flag)

        infos = pickle.load(open(opt.infos_start_from, 'rb'), encoding='iso-8859-1')
        saved_model_opt = infos['opt']
        need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
        for checkme in need_be_same:
            assert vars(saved_model_opt)[checkme] == vars(opt)[checkme],\
                    "Command line argument and saved model disagree on '%s' " % checkme

    # Recover iteration index
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    history = {}
    history['val_perf'] = infos.get('val_result_history', {})
    val_losses = []
    history['loss'] = infos.get('loss_history', {})
    history['lr'] = infos.get('lr_history', {})
    history['ss_prob'] = infos.get('ss_prob_history', {})
    history['scores_stats'] = infos.get('scores_stats', {})
    return iteration, epoch, opt, infos, history


def set_optimizer(opt, epoch, model, cnn_model):
    #  rmsprop | sgd | sgdmom | adagrad | adam
    ref = opt.optim
    if ref.lower() == 'adam':
        optim_func = optim.Adam
    elif ref.lower() == 'sgd':
        optim_func = optim.SGD
    elif ref.lower() == 'rmsprop':
        optim_func = optim.RMSprop
    elif ref.lower() == 'adagrad':
        optim_func = optim.Adagrad
    else:
        raise ValueError('Unknown optimizer % s' % ref)

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim_func(params,
                           lr=opt.learning_rate,
                           betas=(opt.optim_alpha, opt.optim_beta),
                           eps=opt.optim_epsilon,
                           weight_decay=opt.weight_decay)

    optimizers = [optimizer]
    if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
        cnn_params = [p for module in cnn_model.to_finetune for p in module.parameters()]
        cnn_optimizer = optim_func(cnn_params,
                                   lr=opt.learning_rate * opt.cnn_learning_rate,
                                   betas=(opt.optim_alpha, opt.optim_beta),
                                   eps=opt.optim_epsilon,
                                   weight_decay=opt.weight_decay)

        optimizers.append(cnn_optimizer)
        opt.logger.warn('Cnn finetuning flag ON')
        model.cnn_finetuning = 1

    # Load the optimizer
    if vars(opt).get('optimizer_start_from', None) is not None:
        if osp.isfile(opt.optimizer_start_from) and not opt.finetune_cnn_only:
            opt.logger.warn('Loading saved optimizer')
            optimizers[0].load_state_dict(torch.load(opt.optimizer_start_from))
    if vars(opt).get('cnn_optimizer_start_from', None) is not None:
        if osp.isfile(opt.cnn_optimizer_start_from):
            opt.logger.warn('Loading saved optimizer')
            optimizers[1].load_state_dict(torch.load(opt.cnn_optimizer_start_from))

    # Require grads
    for optimizer in optimizers:
        for p in optimizer.param_groups:
            if isinstance(p, dict):
                for pp in p['params']:
                    pp.requires_grad = True
    return optimizers


