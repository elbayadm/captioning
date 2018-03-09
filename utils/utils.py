import os.path as osp
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def pl(path):
    return pickle.load(open(path, 'rb'),
                       encoding='iso-8859-1').astype(np.float32)


def short_path(path):
    basename, filename = osp.split(path)
    return osp.join(osp.basename(basename), filename)


def repackage(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return tuple(repackage(v) for v in h)


def covariance(X, Z):
    """
    Covariance matrix
    """
    n = X.size(0)
    assert Z.size(0) == n, "X and Z should have the same number of rows"
    xdim = X.size(1)
    Zdim = Z.size(1)
    Xbar = X - torch.mean(X, 0).repeat(X.size(0), 1)
    Zbar = Z - torch.mean(Z, 0).repeat(Z.size(0), 1)
    cov = torch.sum(torch.t(Xbar) * Zbar) / n



def decode_sequence(ix_to_word, seq):
    """
    Decode sequence into natural language
    Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
    """
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


def to_contiguous(tensor):
    """
    Convert tensor if not contiguous
    """
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()



def manage_lr(epoch, opt, val_losses):
    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
        opt.logger.error('Updating the lr')
        if opt.lr_strategy == "step":
            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            decay_factor = opt.learning_rate_decay_rate  ** frac
            opt.current_lr = opt.learning_rate * decay_factor
            opt.scale_lr = decay_factor
        elif opt.lr_strategy == "adaptive":
            opt.logger.error('Adaptive mode')
            print("val_losses:", val_losses)
            if len(val_losses) > 2:
                if val_losses[0] > val_losses[1]:
                    opt.lr_wait += 1
                    opt.logger.error('Waiting for more')
                if opt.lr_wait > opt.lr_patience:
                    opt.logger.error('You have plateaued, decreasing the lr')
                    # decrease lr:
                    opt.current_lr = opt.current_lr * opt.learning_rate_decay_rate
                    opt.scale_lr = opt.learning_rate_decay_factor
                    opt.lr_wait = 0
            else:
                opt.current_lr = opt.learning_rate
                opt.scale_lr = 1
    else:
        opt.current_lr = opt.learning_rate
        opt.scale_lr = 1
    return opt


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def scale_lr(optimizers, scale):
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group['lr'] *= scale


def clip_gradient(optimizers, max_norm, norm_type=2):
    max_norm = float(max_norm)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for optimizer in optimizers for group in optimizer.param_groups for p in group['params'])
    else:
        total_norm = 0.0
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for p in group['params']:
                    try:
                        param_norm = p.grad.data.norm(norm_type)
                        nn = param_norm ** norm_type
                        # print('norm:', nn, p.grad.size())
                        total_norm += nn
                        param_norm ** norm_type
                    except:
                        pass
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for p in group['params']:
                    try:
                        p.grad.data.mul_(clip_coef)
                    except:
                        pass
    return total_norm


def get_grad_norm(optimizer):
    grad_norm = 0
    for group in optimizer.param_groups:
        for param in group['params']:
            try:
                grad_norm += torch.norm(param.grad.data)
            except:
                #  print "Error with param of size", param.size()
                pass
    return grad_norm


