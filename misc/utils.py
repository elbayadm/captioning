from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path as osp
import sys
import collections
import math
from scipy.special import binom
import numpy as np
from scipy.spatial.distance import hamming

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F

#  import misc.resnet as resnet
import torchvision.models as models
from torchvision.models.vgg import make_layers

sys.path.append("coco-caption")
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.bleu.bleu_scorer import BleuScorer


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


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(1).sqrt()+self.eps
        x /= norm.expand_as(x)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


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

        self.to_finetune = self._modules.values() #less = 6, otherwise 5
        #  self.keep_asis = self._modules.values()[:6]
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


class MIL_crit(nn.Module):
    def __init__(self, opt):
        super(MIL_crit, self).__init__()
        self.opt = opt
        self.seq_per_img = opt.seq_per_img

    def forward(self, input, target):
        """
        input: prob(w\in Image): shape: #images, #vocab
        target: labels         : shape: #images * #seq_per_img, #seq_length
        mask:     ,,
        """
        # parse words in image from labels:
        num_img = input.size(0)
        labels_per_image = target.chunk(num_img)
        words_per_image = [np.unique(to_contiguous(t).data.cpu().numpy())
                           for t in labels_per_image]
        max_len_words = max([len(t) for t in words_per_image])
        words_per_image = [np.pad(t, (0, max_len_words - len(t)), 'constant')
                           for t in words_per_image]
        indices_words = Variable(torch.cat([torch.from_numpy(t).view(1, -1)
                                            for t in words_per_image], dim=0),
                                 requires_grad=False).cuda()
        #  print "probs:", input
        out = - torch.sum(input.gather(1, indices_words)) / num_img /max_len_words
        return out

class ResNet_MIL_corr(nn.Module):
    """
    Wrapper for ResNet with MIL
    """
    def __init__(self, opt):
        self.opt = opt
        super(ResNet_MIL_corr, self).__init__()
        self.resnet = ResNetModel(opt)
        #  self.pool_mil = nn.MaxPool2d(kernel_size=7, stride=0)
        self.classifier = nn.Linear(2048, opt.vocab_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        #  print "Modules:", self._modules

    def init_added_weights(self):
        initrange = 0.1
        self.classifier.bias.data.fill_(0)
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x0, _ = self.resnet(x)
        x0 = to_contiguous(x0.permute(0, 2, 3, 1))
        x = x0.view(x0.size(0) * x0.size(1) * x0.size(2), -1)
        x = self.classifier(x)
        x = self.softmax(self.sigmoid(x))
        x = x.view(x0.size(0), x0.size(1) * x0.size(2), -1)
        probs =  torch.log(1 - torch.prod(1 - x, dim=1))
        #  self.opt.logger.error('Final probs: %s' % str(probs))
        probs = probs.squeeze(1)
        return probs


class ResNet_MIL(nn.Module):
    """
    Wrapper for ResNet with MIL
    """
    def __init__(self, opt):
        self.opt = opt
        super(ResNet_MIL, self).__init__()
        self.resnet = ResNetModel(opt)
        #  self.sigmoid = torch.nn.Sigmoid()
        #  self.pool_mil = nn.MaxPool2d(kernel_size=7, stride=0)
        self.classifier = nn.Linear(49, opt.vocab_size)
        #  print "Modules:", self._modules

    def init_added_weights(self):
        initrange = 0.1
        self.classifier.bias.data.fill_(0)
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x0, _ = self.resnet(x)
        #  print "Resnet output:", x0.size()
        #  x = self.pool_mil(x0)
        x = x0.view(x0.size(0) * x0.size(1), -1)
        #  x = x0.resize(x0.size(0) * x0.size(1), -1)
        #  print 'Mil output:', x.size()
        #  x = x.squeeze(2).squeeze(2)
        x = self.classifier(x)
        #  print "Classified:", x.size()
        #  x = self.sigmoid(x)
        x = F.log_softmax(x)
        x = x.view(x0.size(0), x0.size(1), -1)
        #  print "Resized:", x.size()
        probs, cands = torch.max(x, 1)
        probs = probs.squeeze(1)
        #  print "vocab probs:", probs.size()
        return probs



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
        self.features1 = nn.Sequential(*self.features._modules.values()[:self.keepdim_att])
        self.features2 = nn.Sequential(*self.features._modules.values()[self.keepdim_att:])
        self.fc = nn.Sequential(*self.classifier._modules.values()[:self.keepdim_fc])
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


class SmoothLanguageModelCriterion(nn.Module):
    """
        Inputs:
            - Dist : similarity matrix d(y, y') for y,y' in vocab
            - version: between clipping below m or exponential temperature
                clip: params = {margin (m)}
                exp : params = {temperature(tau)}
            - isolate_gt : whether to include the gt in the smoother loss or not
            - alpha : weight of the smoothed weight when isolating gt
        Returns:
            - The original loss term (for reference)
            - The smoothed loss.
    """
    def __init__(self, Dist, loader_vocab, opt):
        super(SmoothLanguageModelCriterion, self).__init__()
        self.Dist = Dist
        self.margin = opt.raml_margin
        self.tau = opt.raml_tau
        self.version = opt.raml_version.lower()
        self.seq_per_img = opt.seq_per_img
        self.vocab_size = opt.vocab_size
        if self.version not in ['clip', 'exp', 'vocab',
                                'cider', 'cider-exp',  'bleu4',
                                'hamming', 'hamming-sample']:
            raise ValueError("Unknown smoothing version %s" % self.version)
        if 'cider' in self.version or 'bleu' in self.version:
            opt.logger.warn('RAML CIDER/Bleu: Storing the model vocab')
            self.loader_vocab = loader_vocab
        self.alpha = opt.raml_alpha
        self.isolate_gt = opt.raml_isolate
        self.normalize = opt.raml_normalize
        self.less_confident = opt.less_confident
        self.logger = opt.logger

    def forward(self, input, target, mask, pre_scores):
        # truncate to the same size
        input_ = input
        seq_length = input.size(1)
        target = target[:, :input.size(1)]
        target_ = target
        mask = mask[:, :input.size(1)]
        if self.less_confident:
            row_scores = pre_scores.repeat(1, input.size(1))
            #  gen_rows = np.arange(mask.size(0),)
            #  gen_rows = (gen_rows % self.seq_per_img) > 4
            #  gen_rows = torch.from_numpy(np.where(gen_rows)[0]).cuda()
            #  mask_ = mask
            #  mask_.data[gen_rows] = torch.mul(mask_.data[gen_rows], self.less_confident)
            mask_ = torch.mul(mask, row_scores)
            #  print "Mask (scaled)", mask_
        else:
            mask_ = mask
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        real_output = - input.gather(1, target) * mask_
        real_output = torch.sum(real_output) / torch.sum(mask_)

        #-------------------------------------------------------
        if self.alpha > 0:
            dist = self.Dist[target.squeeze().data]
            #  print "Dist:", dist
            if self.version == "exp":
                smooth_target = torch.exp(torch.mul(torch.add(dist, -1.), 1/self.tau))
            elif self.version == "clip":
                indices_up = dist.ge(self.margin)
                smooth_target = dist * indices_up.float()
            elif self.version == "vocab":
                num_img = target_.size(0) // self.seq_per_img
                vocab_per_image = target_.chunk(num_img)
                vocab_per_image = [np.unique(to_contiguous(t).data.cpu().numpy())
                                   for t in vocab_per_image]
                max_vocab = max([len(t) for t in vocab_per_image])
                vocab_per_image = [np.pad(t, (0, max_vocab - len(t)), 'constant')
                                   for t in vocab_per_image]
                indices_vocab = Variable(torch.cat([torch.from_numpy(t).\
                                         repeat(self.seq_per_img * seq_length, 1)
                                         for t in vocab_per_image], dim=0)).cuda()
                mask_ = mask_.repeat(1, indices_vocab.size(1))
                dist_vocab = dist.gather(1, indices_vocab)
                smooth_target = torch.exp(torch.mul(torch.add(dist_vocab, -1.),
                                                    1/self.tau))
                output = - input.gather(1, indices_vocab) * mask_ * smooth_target
                if self.isolate_gt:
                    indices_down = dist_vocab.lt(1.0)
                    smooth_target = smooth_target * indices_down.float()
                if torch.sum(smooth_target * mask_).data[0] > 0:
                    output = torch.sum(output) / torch.sum(smooth_target * mask_)
                else:
                    self.logger.warn("Smooth targets weights sum to 0")
                    output = torch.sum(output)
                return real_output, self.alpha * output + (1 - self.alpha) * real_output
            elif self.version == 'cider':
                cider_scorer = CiderScorer(n=4, sigma=6)
                preds = torch.max(input_, dim=2)[1].squeeze().cpu().data
                hypo = decode_sequence(self.loader_vocab, preds)  # candidate
                refs = decode_sequence(self.loader_vocab, target_.data)  # references
                num_img = target_.size(0) // self.seq_per_img
                for e, h in enumerate(hypo):
                    ix_start =  e // self.seq_per_img * self.seq_per_img
                    ix_end = ix_start + 5  # self.seq_per_img
                    cider_scorer += (h, refs[ix_start : ix_end])
                (score, scores) = cider_scorer.compute_score()
                self.logger.debug("CIDEr score: %s" %  str(scores))
                #  scores = np.maximum(1 - np.repeat(scores, seq_length), 0)
                scores = np.minimum(np.repeat(scores, seq_length), 1)
                smooth_target = Variable(torch.from_numpy(scores).view(-1, 1)).cuda().float()
                preds = Variable(preds[:, :input.size(1)]).cuda()
                preds = to_contiguous(preds).view(-1, 1)
                output = - input.gather(1, preds) * mask_ * smooth_target
                if torch.sum(smooth_target * mask_).data[0] > 0:
                    output = torch.sum(output) / torch.sum(smooth_target * mask_)
                else:
                    self.logger.warn("Smooth targets weights sum to 0")
                    output = torch.sum(output)
                return real_output, self.alpha * output + (1 - self.alpha) * real_output
            elif self.version == 'cider-exp':
                cider_scorer = CiderScorer(n=4, sigma=6)
                preds = torch.max(input_, dim=2)[1].squeeze().cpu().data
                hypo = decode_sequence(self.loader_vocab, preds)  # candidate
                refs = decode_sequence(self.loader_vocab, target_.data)  # references
                num_img = target_.size(0) // self.seq_per_img
                for e, h in enumerate(hypo):
                    ix_start = e // self.seq_per_img * self.seq_per_img
                    ix_end = ix_start + 5  # self.seq_per_img
                    cider_scorer += (h, refs[ix_start : ix_end])
                (score, scores) = cider_scorer.compute_score()
                self.logger.debug("CIDEr score: %s" %  str(scores))
                scores = np.exp(np.array(scores) / self.tau)
                scores = np.repeat(scores, seq_length)
                smooth_target = Variable(torch.from_numpy(scores).view(-1, 1)).cuda().float()
                preds = Variable(preds[:, :input.size(1)]).cuda()
                preds = to_contiguous(preds).view(-1, 1)
                output = - input.gather(1, preds) * mask_ * smooth_target
                if torch.sum(smooth_target * mask_).data[0] > 0:
                    output = torch.sum(output) / torch.sum(smooth_target * mask_)
                else:
                    self.logger.warn("Smooth targets weights sum to 0")
                    output = torch.sum(output)
                return real_output, self.alpha * output + (1 - self.alpha) * real_output
            elif self.version == 'bleu4':
                bleu_scorer = BleuScorer(n=4)
                preds = torch.max(input_, dim=2)[1].squeeze().cpu().data
                hypo = decode_sequence(self.loader_vocab, preds)  # candidate
                refs = decode_sequence(self.loader_vocab, target_.data)  # references
                num_img = target_.size(0) // self.seq_per_img
                for e, h in enumerate(hypo):
                    ix_start =  e // self.seq_per_img * self.seq_per_img
                    ix_end = ix_start + 5  # self.seq_per_img
                    bleu_scorer += (h, refs[ix_start : ix_end])
                (score, scores) = bleu_scorer.compute_score()
                scores = scores[-1]
                self.logger.debug("Bleu scores: %s" %  str(scores))
                #  scores = np.maximum(1 - np.repeat(scores, seq_length), 0)
                scores = np.minimum(np.repeat(scores, seq_length), 1)
                smooth_target = Variable(torch.from_numpy(scores).view(-1, 1)).cuda().float()
                preds = Variable(preds[:, :input.size(1)]).cuda()
                preds = to_contiguous(preds).view(-1, 1)
                output = - input.gather(1, preds) * mask_ * smooth_target
                if torch.sum(smooth_target * mask_).data[0] > 0:
                    output = torch.sum(output) / torch.sum(smooth_target * mask_)
                else:
                    self.logger.warn("Smooth targets weights sum to 0")
                    output = torch.sum(output)
                return real_output, self.alpha * output + (1 - self.alpha) * real_output

            elif self.version == 'hamming':
                preds = torch.max(input_, dim=2)[1].squeeze().cpu().data
                refs =  target_.cpu().data.numpy()
                num_img = target_.size(0) // self.seq_per_img
                # Hamming distances
                scores = np.array([hamming(u, v) for u, v in zip(preds.numpy(), refs)])
                #  scores = np.maximum(1 - np.repeat(scores, seq_length), 0)
                scores = np.exp(-1 * scores / self.tau)
                self.logger.debug("exp-neg Hamming distances: %s" %  str(scores))

                scores = np.repeat(scores, seq_length)
                smooth_target = Variable(torch.from_numpy(scores).view(-1, 1)).cuda().float()
                preds = Variable(preds[:, :input.size(1)]).cuda()
                preds = to_contiguous(preds).view(-1, 1)
                #  output = - input.gather(1, target) * mask * smooth_target
                output = - input.gather(1, preds) * mask_ * smooth_target

                if torch.sum(smooth_target * mask_).data[0] > 0:
                    output = torch.sum(output) / torch.sum(smooth_target * mask_)
                else:
                    self.logger.warn("Smooth targets weights sum to 0")
                    output = torch.sum(output)
                return real_output, self.alpha * output + (1 - self.alpha) * real_output

            elif self.version == 'hamming-sample':
                # Sample a distance:
                V = 30
                N = input_.size(0)
                distrib = [binom(seq_length, e) *
                           ((V-1) * math.exp(-1/self.tau))**(e-seq_length)
                           for e in range(seq_length+1)]
                select = np.random.choice(a=np.arange(seq_length+1),
                                          p=distrib/sum(distrib))
                score = math.exp(-select / self.tau)
                self.logger.debug("exp-neg Hamming distances (d=%d): %.2e" %
                                  (select, score))
                scores = np.ones((N, seq_length), dtype="float32") * score
                smooth_target = Variable(torch.from_numpy(scores).view(-1, 1)).cuda().float()
                refs =  target_.cpu().data.numpy()
                # Format preds by changing d=select tokens at random
                preds = refs
                change_index = np.random.randint(seq_length, size=(N, select))
                rows = np.arange(N).reshape(-1, 1).repeat(select, axis=1)
                select_index = np.random.randint(self.vocab_size, size=(N, select))
                preds[rows, change_index] = select_index
                preds = Variable(torch.from_numpy(preds)).cuda()
                preds = to_contiguous(preds).view(-1, 1)
                #  output = - input.gather(1, target) * mask * smooth_target
                output = - input.gather(1, preds) * mask_ * smooth_target

                if torch.sum(smooth_target * mask_).data[0] > 0:
                    output = torch.sum(output) / torch.sum(smooth_target * mask_)
                else:
                    self.logger.warn("Smooth targets weights sum to 0")
                    output = torch.sum(output)
                return real_output, self.alpha * output + (1 - self.alpha) * real_output

            # case exp & clip
            if self.isolate_gt:
                indices_down = dist.lt(1.0)
                smooth_target = smooth_target * indices_down.float()
            if self.normalize:
                # Make sur that each row of smoothtarget sum to 1:
                Z = torch.sum(smooth_target, 1).repeat(1, smooth_target.size(1))
                smooth_target = smooth_target / Z
            #  print "Smooth target:", smooth_target
            mask_ = mask_.repeat(1, dist.size(1))
            output = - input * smooth_target * mask_
            if torch.sum(smooth_target * mask_).data[0] > 0:
                output = torch.sum(output) / torch.sum(smooth_target * mask_)
            else:
                self.logger.warn("Smooth targets weights sum to 0")
                output = torch.sum(output)
            if self.isolate_gt:
                return real_output, self.alpha * output + (1 - self.alpha) * real_output
            else:
                return real_output, output
        else:
            return real_output, real_output


#  class LanguageModelCriterion(nn.Module):
#      def __init__(self, use_syn):
#          super(LanguageModelCriterion, self).__init__()
#          self.use_syn = use_syn
#
#      def forward(self, input, target, syntarget, mask):
#          # truncate to the same size
#          input_ = input
#          target = target[:, :input.size(1)]
#          mask =  mask[:, :input.size(1)]
#          input = to_contiguous(input).view(-1, input.size(2))
#          target = to_contiguous(target).view(-1, 1)
#          mask = to_contiguous(mask).view(-1, 1)
#          output = - input.gather(1, target) * mask
#          output = torch.sum(output) / torch.sum(mask)
#          if self.use_syn:
#              syntarget = syntarget[:, :input_.size(1)]
#              syntarget = to_contiguous(syntarget).view(-1, 1)
#              synoutput = - input.gather(1, syntarget) * mask
#              synoutput = torch.sum(synoutput) / torch.sum(mask)
#          else:
#              synoutput = 0
#          return (1 - self.use_syn) * output + self.use_syn * synoutput



class MultiLanguageModelCriterion(nn.Module):
    def __init__(self, seq_per_img=5):
        super(MultiLanguageModelCriterion, self).__init__()
        self.seq_per_img = seq_per_img

    def forward(self, input, target, mask):
        # truncate to the same size
        max_length = input.size(1)
        num_img = input.size(0) // self.seq_per_img
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        mask_ = mask
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        real_output = torch.sum(output) / torch.sum(mask)
        # ------------------------------------------------
        output = output.view(-1, max_length)
        sent_scores = output.sum(dim=1) / mask_.sum(dim=1)
        sent_scores_per_image = sent_scores.chunk(num_img)
        output = torch.sum(torch.cat([t.max() for t in sent_scores_per_image], dim=0))
        output = output / num_img
        return real_output, output


class LanguageModelCriterion(nn.Module):
    def __init__(self, opt):
        super(LanguageModelCriterion, self).__init__()
        self.less_confident = opt.less_confident
        self.seq_per_img = opt.seq_per_img

    def forward(self, input, target, mask, scores):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        if self.less_confident:
            row_scores = scores.repeat(1, input.size(1))
            #  gen_rows = np.arange(mask.size(0),)
            #  gen_rows = (gen_rows % self.seq_per_img) > 4
            #  gen_rows = torch.from_numpy(np.where(gen_rows)[0]).cuda()
            #  mask_ = mask
            #  mask_.data[gen_rows] = torch.mul(mask_.data[gen_rows], self.less_confident)
            mask_ = torch.mul(mask, row_scores)
        else:
            mask_ = mask
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        mask_ = to_contiguous(mask_).view(-1, 1)
        output = - input.gather(1, target) * mask_
        output = torch.sum(output) / torch.sum(mask_)
        return output, output


class PairsLanguageModelCriterion(nn.Module):
    def __init__(self, opt):
        super(PairsLanguageModelCriterion, self).__init__()
        self.seq_per_img = opt.seq_per_img

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        #  print "target:", target
        #  print "mask:", mask
        # duplicate
        num_img = input.size(0) // self.seq_per_img
        input_per_image = input.chunk(num_img)
        input = torch.cat([t.repeat(self.seq_per_img, 1, 1) for t in input_per_image], dim=0)
        target = torch.unsqueeze(target, 0)
        target = target.permute(1, 0, 2)
        target = target.repeat(1, self.seq_per_img, 1)
        target = target.resize(target.size(0) * target.size(1), target.size(2))
        mask = mask[:, :input.size(1)]
        mask = torch.unsqueeze(mask, 0)
        mask = mask.permute(1, 0, 2)
        mask = mask.repeat(1, self.seq_per_img, 1)
        mask = mask.resize(mask.size(0) * mask.size(1), mask.size(2))
        #  print "target:", target
        #  print "mask:", mask
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output, output


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def scale_lr(optimizer, scale):
    for group in optimizer.param_groups:
        group['lr'] *= scale


def clip_gradient(optimizer, max_norm, norm_type=2):
    max_norm = float(max_norm)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for group in optimizer.param_groups for p in group['params'])
    else:
        total_norm = 0.0
        for group in optimizer.param_groups:
            for p in group['params']:
                #  print p.size()
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


