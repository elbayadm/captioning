import random
from math import exp
import numpy as np
import time
import os
import os.path as osp
import sys
import pickle
import subprocess


def exec_cmd(command):
    # return stdout, stderr output of a command
    return subprocess.Popen(command, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE).communicate()


def get_gpu_memory(gpuid):
    # Get the current gpu usage.
    result, _ = exec_cmd('nvidia-smi -i %d --query-gpu=memory.free \
                         --format=csv,nounits,noheader' % int(gpuid))
    # Convert lines into a dictionary
    result = int(result.strip())
    return result


def train(opt):
    """
    main training loop
    """
    # setup gpu
    try:
        gpu_id = int(subprocess.check_output('gpu_getIDs.sh', shell=True))
    except:
        print("Failed to get gpu_id (setting gpu_id to %d)" % opt.gpu_id)
        gpu_id = str(opt.gpu_id)
        # beware seg fault if tf after torch!!
    import tensorflow as tf
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import torch
    opt.logger.warn('GPU ID: %s | available memory: %dM' \
                    % (os.environ['CUDA_VISIBLE_DEVICES'], get_gpu_memory(gpu_id)))
    import torch.nn as nn
    from torch.autograd import Variable
    import torch.optim as optim
    from loader import DataLoader
    import models.eval_utils as evald
    import models.cnn as cnn
    import models.setup as ms
    import utils
    import utils.logging as lg

    # reproducibility:
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size + 1
    opt.seq_length = loader.seq_length
    opt.lr_wait = 0

    tf_summary_writer = tf.summary.FileWriter(opt.eventname)
    iteration, epoch, opt, infos, history = ms.recover_infos(opt)
    if opt.shift_epoch:
        opt.logger.warn('Resetting epoch count (%d -> 0)' % epoch)
        epoch = 0
    opt.logger.warn('Starting from iteration %d (epoch %d)' % (iteration, epoch))
    # Recover data iterator and best perf
    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    opt.logger.warn('using single CNN branch with feature maps as regions embeddings')
    # Build CNN model for single branch use
    if opt.cnn_model.startswith('resnet'):
        cnn_model = cnn.ResNetModel(opt)
    elif opt.cnn_model.startswith('vgg'):
        cnn_model = cnn.VggNetModel(opt)
    else:
        opt.logger.error('Unknown model %s' % opt.cnn_model)
        sys.exit(1)
    try:
        cnn_model.cuda()
    except:
        cnn_model.cuda(gpu_id)
    # Build the captioning model
    opt.logger.error('-----------------------------SETUP')
    model = ms.select_model(opt)
    # model.define_loss(loader.get_vocab())
    model.load()
    opt.logger.error('-----------------------------/SETUP')
    model.cuda()
    update_lr_flag = True
    # Assure in training mode
    model.train()
    cnn_model.eval()
    model.define_loss(loader.get_vocab())
    optimizers = ms.set_optimizer(opt, epoch,
                                  model, cnn_model)
    lg.log_optimizer(opt, optimizers)
    # Main loop
    # To save before training:
    # iteration -= 1
    val_losses = []
    while True:
        if update_lr_flag:
            # Assign the learning rate
            opt = utils.manage_lr(epoch, opt, val_losses)
            utils.scale_lr(optimizers, opt.scale_lr) # set the decayed rate
            lg.log_optimizer(opt, optimizers)
            # Assign the scheduled sampling prob
            if opt.scheduled_sampling_strategy == "step":
                if epoch >= opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob
                    opt.logger.warn('ss_prob= %.2e' % model.ss_prob )
            if opt.loss_version in ['word', 'seq'] and opt.alpha_strategy == "step":
                if epoch >= opt.alpha_increase_start:
                    # Update ncrit's alpha:
                    opt.logger.warn('Updating alpha')
                    frac = (epoch - opt.alpha_increase_start) // opt.alpha_increase_every
                    new_alpha = min(opt.alpha_increase_factor  * frac, opt.alpha_max)
                    model.crit.alpha = new_alpha
                    opt.logger.warn('New alpha %.3e' % new_alpha)
            update_lr_flag = False

        if opt.scheduled_sampling_strategy == "sigmoid":
            if epoch >= opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                opt.logger.warn("setting up the ss_prob")
                opt.ss_prob = 1 - opt.scheduled_sampling_speed / (opt.scheduled_sampling_speed + exp(iteration / opt.scheduled_sampling_speed))
                model.ss_prob = opt.ss_prob
                opt.logger.warn("ss_prob =  %.3e" % model.ss_prob)
        if opt.loss_version in ['word', 'seq'] and opt.alpha_strategy == "sigmoid":
            # Update crit's alpha:
            opt.logger.warn('Updating the loss scaling param alpha')
            new_alpha = 1 - opt.alpha_speed / (opt.alpha_speed + exp(iteration / opt.alpha_speed))
            new_alpha = min(new_alpha, opt.alpha_max)
            model.crit.alpha = new_alpha
            opt.logger.warn('New alpha %.3e' % new_alpha)

        # Load data from train split (0)
        data = loader.get_batch('train')
        torch.cuda.synchronize()
        start = time.time()
        images = data['images']
        images = Variable(torch.from_numpy(images), requires_grad=False).cuda()
        att_feats, fc_feats = cnn_model.forward_caps(images, opt.seq_per_img)
        ml_loss, loss, stats = model.step(data, att_feats, fc_feats,
                                          iteration, epoch)
        for optimizer in optimizers:
            optimizer.zero_grad()
        # // Move
        loss.backward()
        grad_norm = []
        grad_norm.append(utils.clip_gradient(optimizers, opt.grad_clip))
        for optimizer in optimizers:
            optimizer.step()
        train_loss = loss.data[0]
        if np.isnan(train_loss):
            sys.exit('Loss is nan')
        train_ml_loss = ml_loss.data[0]
        try:
            train_kld_loss = kld_loss.data[0]
            train_recon_loss = recon_loss.data[0]
        except:
            pass
        #  grad_norm = [utils.get_grad_norm(optimizer)]
        torch.cuda.synchronize()
        end = time.time()
        losses = {'train_loss': train_loss,
                  'train_ml_loss': train_ml_loss}

        lg.stderr_epoch(epoch, iteration, opt, losses, grad_norm, end-start)
        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True
        # Write the training loss summary
        if iteration % opt.losses_log_every == 0:
            lg.log_epoch(tf_summary_writer, iteration, opt,
                         losses, stats, grad_norm,
                         model)
            history['loss'][iteration] = losses['train_loss']
            history['lr'][iteration] = opt.current_lr
            history['ss_prob'][iteration] = model.ss_prob
            history['scores_stats'][iteration] = stats

        # make evaluation on validation set, and save model
        if iteration % opt.save_checkpoint_every == 0:
            # eval model
            eval_kwargs = {'split': 'val',
                           'dataset': opt.input_data + '.json'}
            eval_kwargs.update(vars(opt))
            eval_kwargs['batch_size'] = 5  #FIXME
            # print("eval kwargs: ", eval_kwargs)
            (val_ml_loss, val_loss,
             predictions, lang_stats) = evald.eval_split(cnn_model,
                                                         model,
                                                         loader,
                                                         opt.logger,
                                                         eval_kwargs)
            # Write validation result into summary
            lg.add_summary_value(tf_summary_writer, 'validation_loss',
                                 val_loss, iteration)
            lg.add_summary_value(tf_summary_writer, 'validation_ML_loss',
                                 val_ml_loss, iteration)

            for k, v in lang_stats.items():
                lg.add_summary_value(tf_summary_writer, k, v, iteration)
            tf_summary_writer.flush()
            history['val_perf'][iteration] = {'loss': val_loss,
                                              'ml_loss': val_ml_loss,
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
            lg.save_model(model, cnn_model, optimizers, opt,
                          iteration, epoch, loader, best_val_score,
                          history, best_flag)
        # Stop if reaching max epochs
        if epoch > opt.max_epochs and opt.max_epochs != -1:
            opt.logger.info('Max epochs reached')
            break

if __name__ == "__main__":
    from opts import parse_opt
    opt = parse_opt()
    train(opt)
