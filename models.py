import os
import copy
import os.path as osp
import numpy as np
import misc.utils as utils
import torch

from misc.ShowTellModel import ShowTellModel
from misc.ShowTellVAEModel import ShowTellVAEModel

# from misc.AttentionModel import AttentionModel
from misc.ShowAttendTellModel import ShowAttendTellModel
# from misc.ShowAttendTellModel_new import ShowAttendTellModel_new
# from misc.TestAttentionModel import TestAttentionModel

def setup(opt):

    if opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    elif opt.caption_model == 'show_tell_vae':
        model = ShowTellVAEModel(opt)
    elif opt.caption_model == 'show_attend_tell':
        model = ShowAttendTellModel(opt)
    else:
        raise ValueError("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        assert osp.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        if opt.start_from_best:
            assert osp.isfile(osp.join(opt.start_from, "infos-best.pkl")), "infos-best.pkl file does not exist in path %s" % opt.start_from
            saved = torch.load(osp.join(opt.start_from, 'model-best.pth'))
            opt.logger.warn('Loading the model dict (best checkpoint) %s' % str(list(saved.keys())))
            model.load_state_dict(saved)
        else:
            assert osp.isfile(osp.join(opt.start_from, "infos.pkl")), "infos.pkl file does not exist in path %s" % opt.start_from
            saved = torch.load(osp.join(opt.start_from, 'model.pth'))
            opt.logger.warn('Loading the model dict (last checkpoint) %s' % str(list(saved.keys())))
            model.load_state_dict(saved)
    return model
