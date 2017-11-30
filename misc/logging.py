import copy
from six.moves import cPickle as pickle
import torch
import tensorflow as tf
import os.path as osp

_OKGREEN = '\033[92m'
_WARNING = '\033[93m'
_FAIL = '\033[91m'
_ENDC = '\033[0m'


def print_sampled(id, sent, score=None, warn=False):
    transition = ' >> ' if not score else " >> %.3f >> " % score
    color = _WARNING if warn else _OKGREEN
    if isinstance(id, int):
        id = "%06d" % id
    print(color, id, transition, sent, _ENDC)


def log_epoch(writer, iteration, opt,
              losses, stats, grad_norm,
              model):

    train_loss = losses['train_loss']
    train_ml_loss = losses['train_ml_loss']
    add_summary_value(writer, 'train_loss', train_loss, iteration)
    add_summary_value(writer, 'train_ml_loss', train_ml_loss, iteration)
    add_summary_value(writer, 'learning_rate', opt.current_lr, iteration)
    add_summary_value(writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
    try:
        add_summary_value(writer, 'alpha', model.crit.alpha, iteration)
    except:
        print('No alpha_word found')
        pass

    add_summary_value(writer, 'RNN_grad_norm', grad_norm[0], iteration)
    if stats:
        for k in stats:
            add_summary_value(writer, k, stats[k], iteration)
    try:
        add_summary_value(writer, 'CNN_grad_norm', grad_norm[1], iteration)
        add_summary_value(writer, 'CNN_learning_rate', opt.current_cnn_lr, iteration)

    except:
        pass
    try:
        train_kld_loss = losses['train_kld_loss']
        train_recon_loss = losses['train_recon_loss']
        add_summary_value(writer, 'kld_loss', train_kld_loss, iteration)
        add_summary_value(writer, 'recon_loss', train_recon_loss, iteration)
    except:
        pass
    writer.flush()

def save_model(model, cnn_model, optimizers, opt,
               iteration, epoch, loader, best_val_score,
               history, best_flag):
    checkpoint_path = osp.join(opt.modelname, 'model.pth')
    cnn_checkpoint_path = osp.join(opt.modelname, 'model-cnn.pth')
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
    opt.logger.info("model saved to {}".format(checkpoint_path))
    opt.logger.info("cnn model saved to {}".format(cnn_checkpoint_path))
    optimizer_path = osp.join(opt.modelname, 'optimizer.pth')
    torch.save(optimizers[0].state_dict(), optimizer_path)
    if len(optimizers) > 1:
        cnn_optimizer_path = osp.join(opt.modelname, 'cnn-optimizer.pth')
        torch.save(optimizers[1].state_dict(), cnn_optimizer_path)

    infos = {}
    # Dump miscalleous informations
    infos['iter'] = iteration
    infos['epoch'] = epoch
    infos['iterators'] = loader.iterators
    infos['best_val_score'] = best_val_score
    infos['opt'] = copy.copy(opt)
    infos['opt'].logger = None
    infos['val_result_history'] = history['val_perf']
    infos['loss_history'] = history['loss']
    infos['lr_history'] = history['lr']
    infos['ss_prob_history'] = history['ss_prob']
    infos['scores_stats'] = history['scores_stats']
    infos['vocab'] = loader.get_vocab()
    with open(osp.join(opt.modelname, 'infos.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    if best_flag:
        checkpoint_path = osp.join(opt.modelname, 'model-best.pth')
        cnn_checkpoint_path = osp.join(opt.modelname, 'model-cnn-best.pth')
        torch.save(model.state_dict(), checkpoint_path)
        torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
        opt.logger.info("model saved to {}".format(checkpoint_path))
        opt.logger.info("cnn model saved to {}".format(cnn_checkpoint_path))
        optimizer_path = osp.join(opt.modelname, 'optimizer-best.pth')
        torch.save(optimizers[0].state_dict(), optimizer_path)
        if len(optimizers) > 1:
            cnn_optimizer_path = osp.join(opt.modelname, 'cnn-optimizer-best.pth')
            torch.save(optimizers[1].state_dict(), cnn_optimizer_path)

        with open(osp.join(opt.modelname, 'infos-best.pkl'), 'wb') as f:
            pickle.dump(infos, f)


def stderr_epoch(epoch, iteration, opt, losses, grad_norm, ttt):
    train_loss = losses['train_loss']
    train_ml_loss = losses['train_ml_loss']
    message = "iter {} (epoch {}), train_ml_loss = {:.3f}, train_loss = {:.3f}, lr = {:.2e}, grad_norm = {:.3e}"\
              .format(iteration, epoch, train_ml_loss,  train_loss, opt.current_lr, grad_norm[0])

    try:
        message += ", cnn_lr = {:.2e}, cnn_grad_norm = {:.3e}".format(opt.current_cnn_lr, grad_norm[1])
    except:
        pass
    try:
        train_kld_loss = losses['train_kld_loss']
        train_recon_loss = losses['train_recon_loss']
        message += "\n{:>25s} = {:.3e}, kld loss = {:.3e}".format('recon loss', train_recon_loss, train_kld_loss)
    except:
        pass
    message += "\n{:>25s} = {:.3f}" \
                .format("Time/batch", ttt)
    opt.logger.debug(message)


def log_optimizer(opt, optimizers):
    for e, optimizer in enumerate(optimizers):
        opt.logger.debug('########### OPTIMIZER %d ###########' % e)
        for p in optimizer.param_groups:
            if isinstance(p, dict):
                print('LR:', p['lr'], )
                # for pp in p['params']:
                    # print(pp.size(), end=' ')
                # print('\n')
        opt.logger.debug('########### /OPTIMIZER %d ###########' % e)




def add_summary_value(writer, key, value, iteration, collections=None):
    """
    Add value to tensorflow events
    """
    _summary = tf.summary.scalar(name=key,
                                 tensor=tf.Variable(value),
                                 collections=collections)
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)




def save_ens_model(ens_model, optimizer, opt,
                   iteration, epoch, loader, best_val_score,
                   history, best_flag):

    for e, (cnn_model, model) in enumerate(zip(ens_model.cnn_models, ens_model.models)):
        checkpoint_path = osp.join(opt.ensemblename, 'model_%d.pth' % e)
        cnn_checkpoint_path = osp.join(opt.ensemblename, 'model-cnn_%d.pth' % e)
        torch.save(model.state_dict(), checkpoint_path)
        torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
        opt.logger.info("model saved to {}".format(checkpoint_path))
        opt.logger.info("cnn model saved to {}".format(cnn_checkpoint_path))
    optimizer_path = osp.join(opt.ensemblename, 'optimizer.pth')
    torch.save(optimizer.state_dict(), optimizer_path)
    infos = {}
    # Dump miscalleous informations
    infos['iter'] = iteration
    infos['epoch'] = epoch
    infos['iterators'] = loader.iterators
    infos['best_val_score'] = best_val_score
    infos['opt'] = copy.copy(opt)
    infos['opt'].logger = None
    infos['val_result_history'] = history['val_perf']
    infos['loss_history'] = history['loss']
    infos['lr_history'] = history['lr']
    infos['ss_prob_history'] = history['ss_prob']
    infos['vocab'] = loader.get_vocab()
    with open(osp.join(opt.ensemblename, 'infos.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    if best_flag:
        for e, (cnn_model, model) in enumerate(zip(ens_model.cnn_models, ens_model.models)):
            checkpoint_path = osp.join(opt.ensemblename, 'model-best_%d.pth' % e)
            cnn_checkpoint_path = osp.join(opt.ensemblename, 'model-cnn-best_%d.pth' % e)
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
            opt.logger.info("model saved to {}".format(checkpoint_path))
            opt.logger.info("cnn model saved to {}".format(cnn_checkpoint_path))
        optimizer_path = osp.join(opt.ensemblename, 'optimizer-best.pth')
        torch.save(optimizer.state_dict(), optimizer_path)
        with open(osp.join(opt.ensemblename, 'infos-best.pkl'), 'wb') as f:
            pickle.dump(infos, f)


