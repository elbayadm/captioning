"""
Additional experimental losses
"""

import torch.nn as nn
import torch
from .utils import to_contiguous
from .cross_entropy import MLCriterion
from .word import WordSmoothCriterion


class MultiMLCriterion(nn.Module):
    def __init__(self, seq_per_img=5):
        super(MultiMLCriterion, self).__init__()
        self.seq_per_img = seq_per_img

    def forward(self, input, target, mask):
        # truncate to the same size
        max_length = input.size(1)
        num_img = input.size(0) // self.seq_per_img
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        mask_ = mask
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        real_output = torch.sum(output) / torch.sum(mask)
        output = output.view(-1, max_length)
        sent_scores = output.sum(dim=1) / mask_.sum(dim=1)
        sent_scores_per_image = sent_scores.chunk(num_img)
        output = torch.sum(torch.cat([t.max() for t in sent_scores_per_image], dim=0))
        output = output / num_img
        return real_output, output


class DataAugmentedCriterion(nn.Module):
    """
    Treat the augmented captions separately
    """
    def __init__(self, opt):
        super(DataAugmentedCriterion, self).__init__()
        self.opt = opt
        self.beta = opt.beta
        self.seq_per_img = opt.seq_per_img
        assert self.seq_per_img > 5, 'Captions per image is seq than 5'
        # The GT loss
        if opt.gt_loss_version == 'word':
            self.crit_gt = WordSmoothCriterion(opt)
        else:
            # The usual ML
            self.crit_gt = MLCriterion(opt)
            self.crit_gt.scale_loss = 0

        # The augmented loss
        if opt.augmented_loss_version == 'word':
            self.crit_augmented = WordSmoothCriterion(opt)
        else:
            # The usual ML
            self.crit_augmented = MLCriterion(opt)
            # Ensure loss scaling with the imprtance sampling ratios
            self.crit_augmented.scale_loss = 1

    def forward(self, input, target, mask, scores):
        seq_length = input.size(1)
        batch_size = input.size(0)
        # truncate
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        # Separate gold from augmented
        num_img = batch_size // self.seq_per_img
        input_per_image = input.chunk(num_img)
        mask_per_image = mask.chunk(num_img)
        target_per_image = target.chunk(num_img)
        scores_per_image = scores.chunk(num_img)

        input_gt = torch.cat([t[:5] for t in input_per_image], dim=0)
        target_gt = torch.cat([t[:5] for t in target_per_image], dim=0)
        mask_gt = torch.cat([t[:5] for t in mask_per_image], dim=0)

        input_gen = torch.cat([t[5:] for t in input_per_image], dim=0)
        target_gen = torch.cat([t[5:] for t in target_per_image], dim=0)
        mask_gen = torch.cat([t[5:] for t in mask_per_image], dim=0)
        scores_gen = torch.cat([t[5:] for t in scores_per_image], dim=0)

        # For the first 5 captions per image (gt) compute LM
        _, output_gt, stats_gt = self.crit_gt(input_gt, target_gt, mask_gt)

        # For the rest of the captions: importance sampling
        _, output_gen, stats_gen = self.crit_augmented(input_gen, target_gen, mask_gen, scores_gen)
        # TODO check if must combine with ml augmented as well
        stats = {}
        if stats_gen:
            for k in stats_gen:
                stats['gen_'+k] = stats_gen[k]
        if stats_gt:
            for k in stats_gt:
                stats['gt_'+k] = stats_gt[k]
        return output_gt, self.beta * output_gen + (1 - self.beta) * output_gt, stats


class PairsMLCriterion(nn.Module):
    def __init__(self, opt):
        super(PairsMLCriterion, self).__init__()
        self.seq_per_img = opt.seq_per_img

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
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


