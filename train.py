from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import exp

# setup gpu
try:
    import os
    import subprocess
    gpu_id = int(subprocess.check_output('gpu_getIDs.sh', shell=True))
    print("GPU:", gpu_id)
except:
    print("Failed to get gpu_id (setting gpu_id to 0)")
    gpu_id = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import os
import os.path as osp
import sys
from six.moves import cPickle as pickle
import opts
from dataloader import *
import eval_utils
import misc.utils as utils
import misc.cnn as cnn
import misc.decoder_utils as du
import misc.logging as lg
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.tensorboard.plugins import projector


def train(opt):
    """
    main training loop
    """
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    opt.lr_wait = 0

    tf_summary_writer = tf.summary.FileWriter(opt.eventname)
    iteration, epoch, opt, infos, history = du.recover_infos(opt)
    opt.logger.warn('Starting from iteration %d (epoch %d)' % (iteration, epoch))
    # Recover data iterator and best perf
    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    # FIXME - temporary
    if opt.use_feature_maps:
        opt.logger.warn('using single CNN branch with feature maps as regions embeddings')
        # Build CNN model for single branch use
        if opt.cnn_model.startswith('resnet'):
            cnn_model = cnn.ResNetModel(opt)
        elif opt.cnn_model.startswith('vgg'):
            cnn_model = cnn.VggNetModel(opt)
        else:
            opt.logger.error('Unknown model %s' % opt.cnn_model)
            sys.exit(1)
    else:
        # Deprecated
        opt.logger.warn('using SSD')
        # // Deprecated
    cnn_model.cuda()
    # Build the captioning model
    opt.logger.error('-----------------------------SETUP')
    model = du.select_model(opt)
    # model.define_loss(loader.get_vocab())
    model.load()
    opt.logger.error('-----------------------------/SETUP')
    model.cuda()
    update_lr_flag = True
    # Assure in training mode
    model.train()
    model.define_loss(loader.get_vocab())
    optimizer = du.set_optimizer(opt, epoch,
                                 model, cnn_model)
    lg.log_optimizer(opt, optimizer)
    # Main loop
    # To save before training:
    iteration -= 1
    val_losses = []
    while True:
        if update_lr_flag:
            # Assign the learning rate
            opt = utils.manage_lr(epoch, opt, val_losses)
            utils.scale_lr(optimizer, opt.scale_lr) # set the decayed rate
            lg.log_optimizer(opt, optimizer)
            # Assign the scheduled sampling prob
            if opt.scheduled_sampling_strategy == "step":
                if epoch >= opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob
                    opt.logger.warn('ss_prob= %.2e' % model.ss_prob )
            if opt.raml_loss and opt.raml_alpha_strategy == "step":
                # Update ncrit's alpha:
                opt.logger.warn('Updating the loss scaling param alpha')
                frac = epoch // opt.raml_alpha_increase_every
                new_alpha = min(opt.raml_alpha_increase_factor  * frac, 1)
                model.crit.alpha = new_alpha
                opt.logger.warn('New alpha %.3e' % new_alpha)
            update_lr_flag = False

        if opt.scheduled_sampling_strategy == "sigmoid":
            if epoch >= opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                opt.logger.warn("setting up the ss_prob")
                opt.ss_prob = 1 - opt.scheduled_sampling_speed / (opt.scheduled_sampling_speed + exp(iteration / opt.scheduled_sampling_speed))
                model.ss_prob = opt.ss_prob
                opt.logger.warn("ss_prob =  %.3e" % model.ss_prob)
        if opt.raml_loss and opt.raml_alpha_strategy == "sigmoid":
            # Update crit's alpha:
            opt.logger.warn('Updating the loss scaling param alpha')
            new_alpha = 1 - opt.raml_alpha_speed / (opt.raml_alpha_speed + exp(iteration / opt.raml_alpha_speed))
            new_alpha = min(new_alpha, 1)
            model.crit.alpha = new_alpha
            opt.logger.warn('New alpha %.3e' % new_alpha)
        # if opt.raml_loss:
            # opt.logger.error('Sanity check alpha = %.3e' % model.crit.alpha)

        # Load data from train split (0)
        data = loader.get_batch('train')
        torch.cuda.synchronize()
        start = time.time()
        images = data['images']
        images = Variable(torch.from_numpy(images), requires_grad=False).cuda()
        att_feats, fc_feats = cnn_model.forward_caps(images, opt.seq_per_img)
        real_loss, loss = model.step(data, att_feats, fc_feats)
        optimizer.zero_grad()
        # // Move
        loss.backward()
        grad_norm = []
        grad_norm.append(utils.clip_gradient(optimizer, opt.grad_clip))
        optimizer.step()
        train_loss = loss.data[0]
        if np.isnan(train_loss):
            sys.exit('Loss is nan')
        train_real_loss = real_loss.data[0]
        try:
            train_kld_loss = kld_loss.data[0]
            train_recon_loss = recon_loss.data[0]
        except:
            pass
        #  grad_norm = [utils.get_grad_norm(optimizer)]
        torch.cuda.synchronize()
        end = time.time()
        losses = {'train_loss': train_loss,
                  'train_real_loss': train_real_loss}

        lg.stderr_epoch(epoch, iteration, opt, losses, grad_norm, end-start)
        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True
        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            lg.log_epoch(tf_summary_writer, iteration, opt,
                         losses, grad_norm,
                         model.ss_prob)
            history['loss'][iteration] = losses['train_loss']
            history['lr'][iteration] = opt.current_lr
            history['ss_prob'][iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                           'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            (real_val_loss, val_loss,
             predictions, lang_stats, unseen_grams) = eval_utils.eval_split(cnn_model, model,
                                                                            model.crit, loader,
                                                                            eval_kwargs)

            # Write validation result into summary
            lg.add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
            lg.add_summary_value(tf_summary_writer, 'real validation loss', real_val_loss, iteration)

            for k, v in lang_stats.items():
                lg.add_summary_value(tf_summary_writer, k, v, iteration)
            tf_summary_writer.flush()
            history['val_perf'][iteration] = {'loss': val_loss,
                                              'lang_stats': lang_stats,
                                              'predictions': predictions}
            val_losses.insert(0, val_loss)
            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss
            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            lg.save_model(model, cnn_model, optimizer, opt,
                          iteration, epoch, loader, best_val_score,
                          history, best_flag)
        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

if __name__ == "__main__":
    opt = opts.parse_opt()
    train(opt)
