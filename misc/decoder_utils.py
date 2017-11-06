import sys
from shutil import copy2
import glob
from six.moves import cPickle as pickle
import os.path as osp
import torch
import torch.optim as optim
from misc.ShowTellModel import ShowTellModel
from misc.ShowTellModel_RAML import ShowTellModel_RAML
from misc.ShowTellVAEModel import ShowTellVAEModel
from misc.ShowAttendTellModel import ShowAttendTellModel


def select_model(opt):
    select = opt.caption_model
    if select == 'show_tell':
        model = ShowTellModel(opt)
    elif select == 'show_tell_vae':
        model = ShowTellVAEModel(opt)
    elif select == 'show_tell_raml':
        model = ShowTellModel_RAML(opt)
    elif select == 'show_attend_tell':
        model = ShowAttendTellModel(opt)
    else:
        raise ValueError("Caption model not supported: {}".format(select))
    return model



def recover_ens_infos(opt):
    infos = {}
    # Restart training (useful with oar idempotant)
    if opt.restart and osp.exists(osp.join(opt.ensemblename, 'model_0.pth')): # Fix the saving names
        opt.logger.warning('Picking up where we left')
        opt.start_from = glob.glob(opt.ensemblename + '/model_*.pth')
        opt.logger.debug('Loading saved models: %s' % str(opt.start_from))
        opt.optimizer_start_from = opt.ensemblename + '/optimizer.pth'
        opt.cnn_start_from = glob.glob(opt.ensemblename + '/model-cnn_*.pth')
        opt.infos_start_from = glob.glob(opt.ensemblename + '/infos_*.pkl')
        infos = pickle.load(open(osp.join(opt.ensemblename, 'infos.pkl'), 'rb'), encoding='iso-8859-1')
    if 'cnn_start_from' not in vars(opt):
        opt.start_from = []
        opt.infos_start_from = []
        opt.cnn_start_from = []
        # Start from the top:
        if opt.start_from_best:
            # add best flag:
            flag = '-best'
        else:
            flag = ''
        opt.logger.debug('Starting from %s' % str(opt.model))
        for e, m in enumerate(opt.model):
            m = m[0]
            opt.start_from.append('save/%s/model%s.pth' % (m, flag))
            opt.infos_start_from.append("save/%s/infos%s.pkl" % (m, flag))
            opt.cnn_start_from.append('save/%s/model-cnn%s.pth' % (m, flag))
            copy2(opt.infos_start_from[-1], osp.join(opt.ensemblename, 'infos_%d.pkl' % e))

    # Recover iteration index
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    history = {}
    history['val_perf'] = infos.get('val_result_history', {})
    val_losses = []
    history['loss'] = infos.get('loss_history', {})
    history['lr'] = infos.get('lr_history', {})
    history['ss_prob'] = infos.get('ss_prob_history', {})
    return iteration, epoch, opt, infos, history


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
        opt.optimizer_start_from = osp.join(opt.start_from, 'optimizer%s.pth' % flag)
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
    # TODO; add sclaed lr for every chunk of cnn layers
    if isinstance(model, list):
        params = [{'params': m.parameters(), 'lr': opt.learning_rate}
                       for m in model]
    else:
        active_params = filter(lambda p: p.requires_grad, model.parameters())
        params = [{'params': active_params, 'lr': opt.learning_rate}]

    if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
        if isinstance(cnn_model, list):
            cnn_params = [{'params': module.parameters(),
                           'lr': opt.cnn_learning_rate * opt.learning_rate}
                          for cnnm in cnn_model
                          for module in cnnm.to_finetune]
        else:
            cnn_params = [{'params': module.parameters(),
                           'lr': opt.cnn_learning_rate * opt.learning_rate}
                          for module in cnn_model.to_finetune]
        params += cnn_params
    optimizer = optim_func(params,
                           lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # Load the optimizer
    if vars(opt).get('optimizer_start_from', None) is not None:
        if osp.isfile(opt.optimizer_start_from) and not opt.finetune_cnn_only:
            opt.logger.warn('Loading saved optimizer')
            try:
                optimizer.load_state_dict(torch.load(opt.optimizer_start_from))
            except:
                print("Starting with blank optimizer")
    # Require grads
    for p in optimizer.param_groups:
        if isinstance(p, dict):
            for pp in p['params']:
                pp.requires_grad = True
    return optimizer


