import os.path as osp
from six.moves import cPickle as pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import misc.loss as loss


class DecoderModel(nn.Module):
    def __init__(self, opt):
        super(DecoderModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.use_glove = opt.use_glove
        if self.use_glove:
            self.input_encoding_size = 300
        else:
            self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.num_regions = opt.num_regions  # Or the number of candidate regions
        self.att_feat_size = opt.att_feat_size
        self.opt = opt
        self.logger = opt.logger
        self.ss_prob = 0.0 # Schedule sampling probability
        self.ss_vocab = opt.scheduled_sampling_vocab

    def load(self):
        opt = self.opt
        if vars(opt).get('start_from', None) is not None:
            # check if all necessary files exist
            assert osp.isfile(opt.infos_start_from),\
                    "infos file %s does not exist" % opt.start_from
            saved = torch.load(opt.start_from)
            self.logger.warn('Loading the model dict (last checkpoint) %s'\
                             % str(list(saved.keys())))
            self.load_state_dict(saved)

    def define_loss(self, loader_vocab):
        opt = self.opt
        if opt.raml_loss:
            # D = np.eye(opt.vocab_size + 1, dtype="float32")
            # D = np.random.uniform(size=(opt.vocab_size + 1,
                                        # opt.vocab_size + 1)).astype(np.float32)
            # D = pickle.load(open('data/Glove/cocotalk_similarities_v2.pkl', 'rb'),
                            # encoding='iso-8859-1')
            D = pickle.load(open(opt.similarity_matrix, 'rb'), encoding='iso-8859-1')

            D = D.astype(np.float32)
            D = Variable(torch.from_numpy(D)).cuda()
            crit = loss.SmoothLanguageModelCriterion(Dist=D,
                                                      loader_vocab=loader_vocab,
                                                      opt=opt)
        elif opt.bootstrap_loss == 1:
            # Using importance sampling loss:
            crit = loss.ImportanceLanguageModelCriterion(opt)
        elif opt.bootstrap_loss == 2:
            # Using importance sampling loss:
            crit = loss.ImportanceLanguageModelCriterion_v2(opt)
        elif opt.combine_caps_losses:
            crit = loss.MultiLanguageModelCriterion(opt.seq_per_img)
        else:
            opt.logger.warn('Using baseline loss criterion')
            crit = loss.LanguageModelCriterion(opt)
        self.crit = crit

    def step(self, data, att_feats, fc_feats):
        opt = self.opt
        if opt.bootstrap_loss:
            if opt.bootstrap_version in ["cider", "cider-exp"]:
                tmp = [data['labels'], data['masks'], data['scores'], data['cider']]
                tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
                labels, masks, scores, s_scores= tmp

            elif opt.bootstrap_version in ["bleu4", "bleu4-exp"]:
                tmp = [data['labels'], data['masks'], data['scores'], data['bleu']]
                tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
                labels, masks, scores, s_scores = tmp
            elif opt.bootstrap_version in ["infersent", "infersent-exp"]:
                tmp = [data['labels'], data['masks'], data['scores'], data['infersent']]
                tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
                labels, masks, scores, s_scores = tmp
            else:
                raise ValueError('Unknown bootstrap distribution %s' % opt.bootstrap_version)
            if "exp" in opt.bootstrap_version:
                print('Original rewards:', torch.mean(s_scores))
                s_scores = torch.exp(torch.div(s_scores, opt.raml_tau))
                print('Tempering the reward:', torch.mean(s_scores))
            r_scores = torch.div(s_scores, torch.exp(scores))
            opt.logger.debug('Mean importance scores: %.3e' % torch.mean(r_scores).data[0])
        else:
            tmp = [data['labels'], data['masks'], data['scores']]
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            labels, masks, scores = tmp


        if opt.caption_model == "show_tell_vae":
                preds, recon_loss, kld_loss = self.forward(fc_feats, att_feats, labels)
                real_loss, loss = self.crit(preds, labels[:, 1:], masks[:, 1:])
                loss += opt.vae_weight * (recon_loss + opt.kld_weight * kld_loss)
                #FIXME add the scaling as parameter

        elif opt.caption_model == 'show_tell_raml':
            probs, reward = self.forward(fc_feats, att_feats, labels)
            raml_scores = reward * Variable(torch.ones(scores.size()))
            # raml_scores = Variable(torch.ones(scores.size()))
            print('Raml reward:', reward)
            real_loss, loss = self.crit(probs, labels[:, 1:], masks[:, 1:], raml_scores)
        else:
            if opt.bootstrap_loss:
                real_loss, loss = self.crit(self.forward(fc_feats, att_feats, labels),
                                            labels[:, 1:], masks[:, 1:], r_scores)
            else:
                real_loss, loss = self.crit(self.forward(fc_feats, att_feats, labels),
                                            labels[:, 1:], masks[:, 1:], scores)
        return real_loss, loss


