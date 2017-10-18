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
from misc.utils import to_contiguous, decode_sequence


def group_similarity(u, refs):
    sims = []
    for v in refs:
        sims.append(1 + np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return np.mean(sims)



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
        self.tau_bis = opt.raml_tau_bis
        self.version = opt.raml_version.lower()
        self.seq_per_img = opt.seq_per_img
        self.vocab_size = opt.vocab_size
        if self.version not in ['clip', 'exp', 'vocab',
                                'cider', 'cider-exp',  'bleu4',
                                'glove-hamming', 'glove-cider', 'glove-cider-exp',
                                'hamming', 'hamming-sample', 'infersent']:
            raise ValueError("Unknown smoothing version %s" % self.version)
        if 'cider' in self.version or 'bleu' in self.version:
            opt.logger.warn('RAML CIDER/Bleu: Storing the model vocab')
            self.loader_vocab = loader_vocab
        if 'infersent' in self.version:
            # load infersent model
            opt.logger.info('Loading the infersent pretrained model')
            GLOVE_PATH = '../InferSent/dataset/GloVe/glove.840B.300d.txt'
            self.infersent = torch.load('../InferSent/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
            self.infersent.set_glove_path(GLOVE_PATH)
            self.infersent.build_vocab_k_words(K=100000)
            # Freeze infersent params:
            for p in self.infersent.parameters():
                p.requires_grad = False

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
                # print('Smooth target:', smooth_target)
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
            elif self.version == 'glove-cider':
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
                dist = self.Dist[preds.squeeze().data]
                smooth_target_wl = torch.exp(torch.mul(torch.add(dist, -1.), 1/self.tau))
                mask_wl = mask_.repeat(1, dist.size(1))
                output_wl = - input * smooth_target_wl
                output = - output_wl.gather(1, preds) * mask_ * smooth_target
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
            elif self.version == 'glove-cider-exp':
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
                scores = np.exp(np.array(scores) / self.tau_bis)
                scores = np.repeat(scores, seq_length)
                smooth_target = Variable(torch.from_numpy(scores).view(-1, 1)).cuda().float()
                preds = Variable(preds[:, :input.size(1)]).cuda()
                preds = to_contiguous(preds).view(-1, 1)
                dist = self.Dist[preds.squeeze().data]
                smooth_target_wl = torch.exp(torch.mul(torch.add(dist, -1.), 1/self.tau))
                mask_wl = mask_.repeat(1, dist.size(1))
                output_wl = - input * smooth_target_wl
                output = - output_wl.gather(1, preds) * mask_ * smooth_target
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

            elif self.version == 'infersent':
                preds = torch.max(input_, dim=2)[1].squeeze().cpu().data
                hypo = decode_sequence(self.loader_vocab, preds)  # candidate
                refs = decode_sequence(self.loader_vocab, target_.data)  # references
                num_img = target_.size(0) // self.seq_per_img
                scores = []
                lr = len(refs)
                codes = self.infersent.encode(refs + hypo)
                refs = codes[:lr]
                hypo = codes[lr:]
                for e, h in enumerate(hypo):
                    ix_start =  e // self.seq_per_img * self.seq_per_img
                    ix_end = ix_start + 5  # self.seq_per_img
                    scores.append(group_similarity(h, refs[ix_start : ix_end]))
                self.logger.debug("Infersent similairities: %s" %  str(scores))
                #  scores = np.maximum(1 - np.repeat(scores, seq_length), 0)
                scores = np.repeat(np.exp(np.array(scores)/ self.tau), seq_length)
                print('Scaling with', scores)
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

            elif self.version == 'hamming': # here sampling with p instead of the reward q
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

            elif self.version == 'glove-hamming': # here sampling with p instead of the reward q
                preds = torch.max(input_, dim=2)[1].squeeze().cpu().data
                refs =  target_.cpu().data.numpy()
                num_img = target_.size(0) // self.seq_per_img
                # Hamming distances
                scores = np.array([hamming(u, v) for u, v in zip(preds.numpy(), refs)])
                #  scores = np.maximum(1 - np.repeat(scores, seq_length), 0)
                scores = np.exp(-1 * scores / self.tau_bis)
                self.logger.debug("exp-neg Hamming distances: %s" %  str(scores))

                scores = np.repeat(scores, seq_length)
                smooth_target = Variable(torch.from_numpy(scores).view(-1, 1)).cuda().float()
                preds = Variable(preds[:, :input.size(1)]).cuda()
                preds = to_contiguous(preds).view(-1, 1)

                #  output = - input.gather(1, target) * mask * smooth_target
                dist = self.Dist[preds.squeeze().data]
                smooth_target_wl = torch.exp(torch.mul(torch.add(dist, -1.), 1/self.tau))
                output_wl = - input * smooth_target_wl
                print('Output scaled at word level', output_wl.size())
                output = - output_wl.gather(1, preds) * mask_ * smooth_target
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
            # Deprecated
            if self.normalize:
                # Make sur that each row of smoothtarget sum to 1:
                Z = torch.sum(smooth_target, 1).repeat(1, smooth_target.size(1))
                smooth_target = smooth_target / Z
            # // Deprecated
            mask_ = mask_.repeat(1, dist.size(1))
            output = - input * smooth_target * mask_
            if torch.sum(smooth_target * mask_).data[0] > 0:
                output = torch.sum(output) / torch.sum(smooth_target * mask_)
            else:
                self.logger.warn("Smooth targets weights sum to 0")
                output = torch.sum(output)
            return real_output, self.alpha * output + (1 - self.alpha) * real_output
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



class ImportanceLanguageModelCriterion(nn.Module):
    def __init__(self, opt):
        super(ImportanceLanguageModelCriterion, self).__init__()
        self.opt = opt
        self.alpha = opt.raml_alpha

    def forward(self, input, target, mask, sampling_ratios):
        # truncate to the same size
        ratios = sampling_ratios
        sampling_ratios = sampling_ratios.repeat(1, input.size(1))
        # self.opt.logger.debug('Importance scores shaped:', sampling_ratios)
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        # print('Updated mask:', mask_)
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        real_output = - input.gather(1, target) * mask
        real_output = torch.sum(real_output) / torch.sum(mask)
        output = - input.gather(1, target) * mask * sampling_ratios
        if torch.sum(mask).data[0] > 0 and torch.sum(ratios).data[0] > 0:
            # self.opt.logger.debug('output without avg %s' % str(torch.sum(output).data))
            output = torch.sum(output) / torch.sum(mask) / torch.sum(ratios)
            self.opt.logger.warn('Avergaging over the sampling scores and the seq length')
        else:
            self.opt.logger.warn("Smooth targets weights sum to 0")
            output = torch.sum(output)
            print('WARNING: Output loss without averaging:', output.data[0])
            output = real_output

        return real_output, self.alpha * output + (1 - self.alpha) * real_output


class ImportanceLanguageModelCriterion_v2(nn.Module):
    def __init__(self, opt):
        super(ImportanceLanguageModelCriterion_v2, self).__init__()
        self.opt = opt
        self.alpha = opt.raml_alpha
        self.seq_per_img = opt.seq_per_img

    def forward(self, input, target, mask, sampling_ratios):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        num_img = input.size(0) // self.seq_per_img
        input_per_image = input.chunk(num_img)
        mask_per_image = mask.chunk(num_img)
        target_per_image = target.chunk(num_img)
        ratios_per_image = sampling_ratios.chunk(num_img)
        input_gt = torch.cat([t[:5] for t in input_per_image], dim=0)
        target_gt = torch.cat([t[:5] for t in target_per_image], dim=0)
        mask_gt = torch.cat([t[:5] for t in mask_per_image],dim=0)

        input_gen = torch.cat([t[5:] for t in input_per_image], dim=0)
        target_gen = torch.cat([t[5:] for t in target_per_image], dim=0)
        mask_gen = torch.cat([t[5:] for t in mask_per_image], dim=0)
        ratios_gen = torch.cat([t[5:] for t in ratios_per_image], dim=0)
        # print('Ratios GEN:', ratios_gen)

        # For the first 5 captions per image (gt) compute LM
        input_gt = to_contiguous(input_gt).view(-1, input_gt.size(2))
        target_gt = to_contiguous(target_gt).view(-1, 1)
        mask_gt = to_contiguous(mask_gt).view(-1, 1)
        output_gt = - input_gt.gather(1, target_gt) * mask_gt
        output_gt = torch.sum(output_gt) / torch.sum(mask_gt)

        # For the rest of the captions: importance sampling
        # truncate to the same size
        sampling_ratios = ratios_gen.repeat(1, input.size(1))
        input_gen = to_contiguous(input_gen).view(-1, input_gen.size(2))
        target_gen = to_contiguous(target_gen).view(-1, 1)
        mask_gen = to_contiguous(mask_gen).view(-1, 1)
        output_gen = - input_gen.gather(1, target_gen) * mask_gen * sampling_ratios
        if torch.sum(mask_gen).data[0] > 0 and torch.sum(ratios_gen).data[0] > 0:
            # self.opt.logger.debug('output without avg %s' % str(torch.sum(output).data))
            output_gen = torch.sum(output_gen) / torch.sum(mask_gen) / torch.sum(ratios_gen)
            self.opt.logger.warn('Avergaging over the sampling scores and the seq length')
        else:
            self.opt.logger.warn("Smooth targets weights sum to 0")
            output_gen = torch.sum(output_gen)
            print('WARNING: Output loss without averaging:', output_gen.data[0])
        return output_gt, self.alpha * output_gen + (1 - self.alpha) * output_gt


class LanguageModelCriterion(nn.Module):
    def __init__(self, opt):
        super(LanguageModelCriterion, self).__init__()
        self.less_confident = opt.less_confident
        self.opt = opt
        self.seq_per_img = opt.seq_per_img

    def forward(self, input, target, mask, scores):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        if self.less_confident:
            row_scores = scores.repeat(1, input.size(1))
            # print('Scaling with:', scores)
            #  gen_rows = np.arange(mask.size(0),)
            #  gen_rows = (gen_rows % self.seq_per_img) > 4
            #  gen_rows = torch.from_numpy(np.where(gen_rows)[0]).cuda()
            #  mask_ = mask
            #  mask_.data[gen_rows] = torch.mul(mask_.data[gen_rows], self.less_confident)
            mask_ = torch.mul(mask, row_scores)
        else:
            mask_ = mask
        # print('Updated mask:', mask_)
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
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


class MIL_crit(nn.Module):
    def __init__(self, opt):
        super(MIL_crit, self).__init__()
        self.opt = opt
        self.seq_per_img = opt.seq_per_img

    def forward(self, input, target):
        """
        input: prob(w\in Image): shape: #images, #vocab
        target: labels         : shape: #images * #seq_per_img, #seq_length
        Beware batch_size = 1
        """
        # input = torch.log(input + 1e-30)
        # print("Probs:", input)
        # parse words in image from labels:
        num_img = input.size(0)
        # assert num_img == 1, "Batch size larger than 1"
        words_per_image = np.unique(to_contiguous(target).data.cpu().numpy())
        # print('Word in image:', words_per_image)
        indices_pos = Variable(torch.from_numpy(words_per_image).view(1, -1), requires_grad=False).cuda()
        indices_neg = Variable(torch.from_numpy(np.array([a for a in np.arange(self.opt.vocab_size) if a not in words_per_image])).view(1, -1), requires_grad=False).cuda()
        mask_pos = torch.gt(indices_pos, 0).float()
        mask_neg = torch.gt(indices_neg, 0).float()
        # print('Positives:', torch.sum(mask_pos), "Negatives:", torch.sum(mask_neg))
        log_pos = - torch.sum(torch.log(input + 1e-30).gather(1, indices_pos) * mask_pos)
        log_neg = - torch.sum(torch.log(1 - input + 1e-15).gather(1, indices_neg) * mask_neg)
        # print('Log_pos:', log_pos, 'log_neg:', log_neg)
        out = log_pos / torch.sum(mask_pos) + log_neg / torch.sum(mask_neg)
        # print("Final output:", out)
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
        self.opt.logger.error('Final probs: %s' % str(probs))
        probs = probs.squeeze(1)
        return probs


