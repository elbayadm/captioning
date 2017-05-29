from __future__ import absolute_import
from __future__ import division
#  from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
#  import misc.resnet as resnet
import os
import os.path as osp
import torchvision.models as models
from torchvision.models.vgg import make_layers
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

model_configs = {
    'resnet50' : [3, 4, 6, 3],
    'resnet101' : [3, 4, 23, 3],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def repackage(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage(v) for v in h)


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(1).sqrt()+self.eps
        x /= norm.expand_as(x)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()




class ResNetModel(models.ResNet):
    """
    Wrapper for ResNet models
    """
    def __init__(self, opt):
        self.opt = opt
        spec = opt.cnn_model
        flag = False
        super(ResNetModel, self).__init__(models.resnet.Bottleneck, model_configs[spec])
        # Initialize the cnn weights:
        if vars(opt).get('start_from', None) is not None:
            flag = True
        else:
            opt.logger.debug('Setting ResNet weigths from the models zoo')
            self.load_state_dict(model_zoo.load_url(model_urls[spec]))
            #  opt.logger.debug('Setting ResNet weigths from ruotianluo model')
            #  self.load_state_dict(torch.load('/home/thoth/melbayad/scratch/.torch/models/ruotianluo_resnet50.pth'))
        #  self.avgpool = nn.Module()
        self.fc = nn.Module()
        self.norm2 = L2Norm(n_channels=2048, scale=True)
        if flag:
            opt.logger.debug('Setting CNN weigths from %s' % opt.start_from)
            self.load_state_dict(torch.load(osp.join(opt.start_from, 'model-cnn.pth')))

        self.to_finetune = self._modules.values()[5:]
        self.keep_asis = self._modules.values()[:5]
        #  print "RESNET:", self._modules

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        att_feats = x
        #  x = x.mean(2).mean(3).squeeze(2).squeeze(2)
        x = self.avgpool(x)
        if self.opt.norm_feat:
            x = self.norm2(x)
        x = x.view(x.size(0), -1)
        #  x = self.fc(x)
        # TODO: add norm2 to the fc_feat.
        return att_feats, x


class VggNetModel(models.VGG):
    """
    Wrapper for VGG models
    """
    def __init__(self, opt):
        spec = opt.cnn_model
        self.opt = opt
        flag = False
        super(VggNetModel, self).__init__(make_layers(model_configs[spec]))
        if vars(opt).get('start_from', None) is not None:
            flag = True # Reorder layers before loading
            #  self.load_state_dict(torch.load(osp.join(opt.start_from, 'model-cnn.pth')))
        else:
            opt.logger.debug('Setting VGG weigths from the models zoo')
            self.load_state_dict(model_zoo.load_url(model_urls[spec]))
        opt.logger.warn('Setting the fc feature as %s' % opt.cnn_fc_feat)
        if opt.cnn_fc_feat == 'fc7':
            self.keepdim_fc = 6
        elif opt.cnn_fc_feat == 'fc6':
            self.keepdim_fc = 3
        self.keepdim_att = 30
        #  print 'PRE:', self._modules
        # Reassemble:
        self.features1 = nn.Sequential(
            *self.features._modules.values()[:self.keepdim_att]
        )
        self.features2 = nn.Sequential(
            *self.features._modules.values()[self.keepdim_att:]
        )
        self.fc = nn.Sequential(
            *self.classifier._modules.values()[:self.keepdim_fc]
        )
        self.features = nn.Module()
        self.classifier = nn.Module()
        self.norm2 = L2Norm(n_channels=512, scale=True)
        if flag:
            self.load_state_dict(torch.load(osp.join(opt.start_from, 'model-cnn.pth')))
        self.to_finetune = self._modules.values()
        self.keep_asis = []
        #  print 'POST:', self._modules

    def forward(self, x):
        x = self.features1(x)
        #  print 'step1:', x.size()
        att_feats = x
        x = self.features2(x)
        #  print 'step2:', x.size()
        if self.opt.norm_feat:
            x = self.norm2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #  print "step2:", x.size()
        return att_feats, x


class LanguageModelCriterion(nn.Module):
    def __init__(self, use_syn):
        super(LanguageModelCriterion, self).__init__()
        self.use_syn = use_syn

    def forward(self, input, target, syntarget, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        if self.use_syn:
            syntarget = syntarget[:, :input.size(1)]
            syntarget = to_contiguous(syntarget).view(-1, 1)
            synoutput = - input.gather(1, syntarget) * mask
            synoutput = torch.sum(synoutput) / torch.sum(mask)

        return (1 - self.use_syn) * output + self.use_syn * synoutput



#  class LanguageModelCriterion(nn.Module):
#      def __init__(self):
#          super(LanguageModelCriterion, self).__init__()
#
#      def forward(self, input, target, mask):
#          # truncate to the same size
#          target = target[:, :input.size(1)]
#          mask =  mask[:, :input.size(1)]
#          input = to_contiguous(input).view(-1, input.size(2))
#          target = to_contiguous(target).view(-1, 1)
#          mask = to_contiguous(mask).view(-1, 1)
#          output = - input.gather(1, target) * mask
#          output = torch.sum(output) / torch.sum(mask)
#          return output
#

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, max_norm, norm_type=2):
    max_norm = float(max_norm)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for group in optimizer.param_groups for p in group['params'])
    else:
        total_norm = 0.0
        for group in optimizer.param_groups:
            for p in group['params']:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad.data.mul_(clip_coef)
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


