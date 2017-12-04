import os.path as osp
from six.moves import cPickle as pickle
import torch
import torch.nn as nn
from torch.autograd import Variable, gradcheck
import numpy as np
import misc.loss as loss


class DecoderModel(nn.Module):
    def __init__(self, opt):
        super(DecoderModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.logger = opt.logger
        self.input_encoding_size = opt.input_encoding_size
        if len(opt.init_decoder_W):
            # Load W intializer:
            self.W = pickle.load(open(opt.init_decoder_W, 'rb'),
                                 encoding="iso-8859-1")
            self.logger.info('Loading weights to initialize W')
            self.input_encoding_size = self.W.shape[1]
        else:
            self.W = None
        self.require_W_grad = not bool(opt.freeze_decoder_W)
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.num_regions = opt.num_regions  # Or the number of candidate regions
        self.att_feat_size = opt.att_feat_size
        self.opt = opt
        self.ss_prob = 0.0  # Schedule sampling probability
        self.ss_vocab = opt.scheduled_sampling_vocab
        self.cnn_finetuning = 0

    def load(self):
        opt = self.opt
        if vars(opt).get('start_from', None) is not None:
            # check if all necessary files exist
            assert osp.isfile(opt.infos_start_from),\
                    "infos file %s does not exist" % opt.start_from
            saved = torch.load(opt.start_from)
            for k in list(saved):
                if 'crit' in k:
                    self.logger.warn('Deleting key %s' % k)
                    del saved[k]
            self.logger.warn('Loading the model dict (last checkpoint) %s'\
                             % str(list(saved.keys())))
            self.load_state_dict(saved)

    def define_sum_loss(self, vocab):
        opt = self.opt
        if 'hamming' in opt.loss_version:
            crit_sent = loss.HammingRewardSampler(opt, vocab)
            crit_sent.log()
        elif 'tfidf' in opt.loss_version:
            crit_sent = loss.TFIDFRewardSampler(opt, vocab)
            crit_sent.log()
        else:
            raise ValueError('Loss function %s in sample_reward mode unknown' % (opt.loss_version))
        crit_word = loss.WordSmoothCriterion2(opt)
        crit_word.log()
        self.crit_word = crit_word
        self.crit_sent = crit_sent
        self.crit = self.crit_word

    def define_alter_loss(self, vocab):
        opt = self.opt
        if 'hamming' in opt.loss_version:
            crit_sent = loss.HammingRewardSampler(opt, vocab)
            crit_sent.log()
        elif 'tfidf' in opt.loss_version:
            crit_sent = loss.TFIDFRewardSampler(opt, vocab)
            crit_sent.log()
        else:
            raise ValueError('Loss function %s in sample_reward mode unknown' % (opt.loss_version))
        crit_word = loss.WordSmoothCriterion2(opt)
        crit_word.log()
        self.crit = crit_word
        self.crit_word = crit_word
        self.crit_sent = crit_sent

    def define_loss(self, vocab):
        opt = self.opt
        if opt.sum_loss:
            return self.define_sum_loss(vocab)
        if opt.alter_loss:
            return self.define_alter_loss(vocab)
        if opt.sample_cap:
            # Sampling from the captioning model itself
            if 'dummy' in opt.loss_version:
                crit = loss.AllIsGoodCriterion(opt, vocab)
            elif 'cider' in opt.loss_version:
                crit = loss.CiderRewardCriterion(opt, vocab)
            elif 'hamming' in opt.loss_version:
                crit = loss.HammingRewardCriterion(opt)
            elif 'infersent' in opt.loss_version:
                crit = loss.InfersentRewardCriterion(opt, vocab)
            elif 'bleu' in opt.loss_version:
                crit = loss.BleuRewardCriterion(opt, vocab)
            elif opt.loss_version == "word":
                crit = loss.WordSmoothCriterion(opt)
            elif opt.loss_version == "word2":
                crit = loss.WordSmoothCriterion2(opt)
            else:
                raise ValueError('Loss function %s in sample_cap mode unknown' % (opt.loss_version))

        elif opt.bootstrap:
            crit = loss.DataAugmentedCriterion(opt)
        # elif opt.combine_caps_losses:
            # crit = loss.MultiLanguageModelCriterion(opt.seq_per_img)
        elif opt.sample_reward:
            if 'hamming' in opt.loss_version:
                crit = loss.HammingRewardSampler(opt, vocab)
            elif 'tfidf' in opt.loss_version:
                crit = loss.TFIDFRewardSampler(opt, vocab)
            else:
                raise ValueError('Loss function %s in sample_reward mode unknown' % (opt.loss_version))
        else:
            # The defualt ML
            opt.logger.warn('Using baseline loss criterion')
            crit = loss.LanguageModelCriterion(opt)
        crit.log()
        self.crit = crit

    def step_alter(self, data, att_feats, fc_feats, batch, epoch):
        opt = self.opt
        mode = opt.alter_mode
        tmp = [data['labels'], data['masks'], data['scores']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        labels, masks, scores = tmp
        stats = None
        if mode == 'even-odd':
            pick_sent = batch % 2
        elif mode == 'even-odd-epoch':
            pick_sent = (batch + epoch) % 2  # same oddity
            print('bacth : %d, epoch : %d, picked sentenece: %d' % (batch, epoch, pick_sent))
        elif mode == 'epoch':
            pick_sent = epoch % 2
        else:
            raise ValueError('Unknown alterning mode %s' % mode)
        if pick_sent:
            ml_loss, raml_loss, stats = self.crit_sent(self, fc_feats, att_feats, labels, masks[:, 1:], scores)
            print('sent loss:', raml_loss.data[0])
            alpha = self.crit_sent.alpha
        else:
            logprobs = self.forward(fc_feats, att_feats, labels)
            ml_loss, raml_loss, stats = self.crit_word(logprobs,
                                                       labels[:, 1:],
                                                       masks[:, 1:],
                                                       scores)
            alpha = self.crit_word.alpha
            raml_loss *= opt.gamma
            print('word loss (scaled):', raml_loss.data[0])
        alt_loss = alpha * raml_loss + (1 - alpha) * ml_loss
        return ml_loss, alt_loss, stats

    def step_sum(self, data, att_feats, fc_feats):
        opt = self.opt
        tmp = [data['labels'], data['masks'], data['scores']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        labels, masks, scores = tmp
        stats = None
        ml_loss_, loss_sent, stats = self.crit_sent(self, fc_feats, att_feats, labels, masks[:, 1:], scores)
        logprobs = self.forward(fc_feats, att_feats, labels)
        ml_loss, loss_word, stats_ = self.crit_word(logprobs,
                                                   labels[:, 1:],
                                                   masks[:, 1:],
                                                   scores)
        stats.update(stats_)
        print('word loss:', loss_word.data[0])
        print('sent loss:', loss_sent.data[0])
        raml_loss = opt.gamma * loss_sent + (1 - opt.gamma) * loss_word
        print('raml loss:', raml_loss.data[0])
        assert self.crit_sent.alpha == self.crit_word.alpha, "When summing the losses, there should be a single alpha"
        alpha = self.crit_sent.alpha
        sum_loss = alpha * raml_loss + (1 - alpha) * ml_loss
        return ml_loss, sum_loss, stats


    def step(self, data, att_feats, fc_feats, batch=None, epoch=None, train=True):
        opt = self.opt
        if opt.alter_loss and train:
            return self.step_alter(data, att_feats, fc_feats, batch, epoch)
        if opt.sum_loss:
            return self.step_sum(data, att_feats, fc_feats)
        if opt.bootstrap:
            assert opt.bootstrap_score in ['cider', 'bleu2', 'bleu3', 'bleu4', 'infersent']
            tmp = [data['labels'], data['masks'], data['scores'], data[opt.bootstrap_score]]
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            labels, masks, scores, s_scores = tmp
            # Exponentiate the scores
            if opt.tau_sent:
                print('Original mean reward:', torch.mean(s_scores).data[0])
                s_scores = torch.exp(torch.div(s_scores, opt.tau_sent))
                print('Tempering the reward (new mean):', torch.mean(s_scores).data[0])
            scores = torch.div(s_scores, torch.exp(scores))
            opt.logger.debug('Mean importance scores: %.3e' % torch.mean(scores).data[0])
        else:
            tmp = [data['labels'], data['masks'], data['scores']]
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            labels, masks, scores = tmp

        # if opt.caption_model == "show_tell_vae":
                # preds, recon_loss, kld_loss = self.forward(fc_feats, att_feats, labels)
                # real_loss, loss = self.crit(preds, labels[:, 1:], masks[:, 1:])
                # loss += opt.vae_weight * (recon_loss + opt.kld_weight * kld_loss)
                # #FIXME add the scaling as parameter

        stats = None
        if opt.caption_model == "show_attend_tell":
            seq = labels
            msk = masks
        else:
            seq = labels[:, 1:]
            msk = masks[:, 1:]
        # FIXME Deprecated
        if opt.caption_model == 'show_tell_raml':
            probs, reward = self.forward(fc_feats, att_feats, labels)
            raml_scores = reward * Variable(torch.ones(scores.size()))
            # raml_scores = Variable(torch.ones(scores.size()))
            print('Raml reward:', reward)
            ml_loss, raml_loss = self.crit(probs, seq, msk, raml_scores)
        else:
            if self.opt.sample_reward:
                ml_loss, raml_loss, stats = self.crit(self, fc_feats, att_feats, labels, msk, scores)
            else:
                logprobs = self.forward(fc_feats, att_feats, labels)
                # print('Model forward output:', logprobs.size(), seq.size(), msk.size(), 'lab pure:', labels.size())
                ml_loss, raml_loss, stats = self.crit(logprobs,
                                                      seq,
                                                      msk,
                                                      scores)
        # print('raml loss:', raml_loss.data[0])
        if self.opt.sample_reward or self.opt.sample_cap:
            c_loss = self.crit.alpha * raml_loss + (1 - self.crit.alpha) * ml_loss
        else:
            c_loss = ml_loss
        return ml_loss, c_loss, stats


