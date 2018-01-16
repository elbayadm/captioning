import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

from .lstm import AdaptiveAttentionLSTM, AdaptiveAttention
from .AttentionModel import AttentionModel


class AdaAttCore(nn.Module):
    def __init__(self, opt):
        super(AdaAttCore, self).__init__()
        self.lstm = AdaptiveAttentionLSTM(opt)
        self.attention = AdaptiveAttention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, step):
        top_h, sentinel, state = self.lstm(xt, fc_feats, state, step)
        atten_out = self.attention(top_h, sentinel, att_feats, p_att_feats)
        return atten_out, state


class AdaptiveAttentionModel(AttentionModel):

    def __init__(self, opt):
        AttentionModel.__init__(self, opt)

        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_feat_im))
        self.core = AdaAttCore(opt)
        opt.logger.warn('Adaptive attention : %s' % str(self._modules))

