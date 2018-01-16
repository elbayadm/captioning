import numpy as np
import torch
import torch.nn as nn
from .utils import to_contiguous


class MLCriterion(nn.Module):
    """
    The defaul cross entropy loss with the option
    of scaling the sentence loss
    """
    def __init__(self, opt):
        super().__init__()
        self.logger = opt.logger
        self.scale_loss = opt.scale_loss
        self.normalize_batch = opt.normalize_batch
        self.penalize_confidence = opt.penalize_confidence

    def log(self):
        self.logger.info('Default ML loss')

    def forward(self, logp, target, mask, scores=None):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        mask : the ground truth mask to ignore UNK tokens (N, seq_length)
        scores: scalars to scale the loss of each sentence (N, 1)
        """
        # truncate to the same size
        seq_length = logp.size(1)
        target = target[:, :seq_length]
        mask = mask[:, :seq_length]
        if self.scale_loss:
            row_scores = scores.unsqueeze(1).repeat(1, seq_length)
            mask = torch.mul(mask, row_scores)
        logp = to_contiguous(logp).view(-1, logp.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        if self.penalize_confidence:
            logp = logp.gather(1, target)
            neg_entropy = torch.sum(torch.exp(logp) * logp)
            output = torch.sum(-logp * mask) + self.penalize_confidence * neg_entropy
        else:
            output = - logp.gather(1, target) * mask
            output = torch.sum(output)
        if self.normalize_batch:
            output /= torch.sum(mask)
        return output, output, None

    def track(self, logp, target, mask, add_dirac=False):
        """
        logp : the decoder logits (N, seq_length, V)
        target : the ground truth labels (N, seq_length)
        mask : the ground truth mask to ignore UNK tokens (N, seq_length)
        scores: scalars to scale the loss of each sentence (N, 1)
        """
        # truncate to the same size
        N = logp.size(0)
        seq_length = logp.size(1)
        target = target[:, :seq_length].data.cpu().numpy()
        logp = torch.exp(logp).data.cpu().numpy()
        target_d = np.zeros_like(logp)
        rows = np.arange(N).reshape(-1, 1).repeat(seq_length, axis=1)
        cols = np.arange(seq_length).reshape(1, -1).repeat(N, axis=0)
        target_d[rows, cols, target] = 1
        return logp, target_d


