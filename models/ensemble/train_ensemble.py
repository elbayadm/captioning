import json
import numpy as np
import time
import os
from six.moves import cPickle as pickle
import opts
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import misc.decoder_utils as du
import misc.logging as lg
import misc.cnn as cnn
from misc.ensemble import Ensemble
import torch
import tensorflow as tf


if __name__ == "__main__":
    ens_opt = opts.parse_ens_opt()
    print('Ensembling:', ens_opt.model)
    iteration, epoch, ens_opt, ens_infos, history = du.recover_ens_infos(ens_opt)
    ens_opt.logger = opts.create_logger('%s/train.log' % ens_opt.ensemblename)
    options = []
    tf_summary_writer = tf.summary.FileWriter(ens_opt.eventname)
    if ens_opt.load_best_score == 1:
        best_val_score = ens_infos.get('best_val_score', None)

    rnn_models = []
    cnn_models = []
    for (cnn_path, rnn_path, infos_path) in zip(ens_opt.cnn_start_from, ens_opt.start_from, ens_opt.infos_start_from):
        with open(infos_path, 'rb') as f:
            print('Opening %s' % infos_path)
            infos = pickle.load(f, encoding="iso-8859-1")
        # override and collect parameters
        opt = argparse.Namespace(**vars(ens_opt))
        for k in list(vars(infos['opt']).keys()):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model
        # Override:
        opt.input_h5 = ens_opt.input_h5
        opt.input_json = ens_opt.input_json
        opt.batch_size = ens_opt.batch_size
        opt.seq_per_img = ens_opt.seq_per_img
        print('Setting batch size to', opt.batch_size)
        # Check if new features in opt:
        if "less_confident" not in opt:
            opt.less_confident = 0
        if "scheduled_sampling_strategy" not in opt:
            opt.scheduled_sampling_strategy = "step"
        if "scheduled_sampling_vocab" not in opt:
            opt.scheduled_sampling_vocab = 0
        opt.use_feature_maps = infos['opt'].use_feature_maps
        opt.cnn_model = infos['opt'].cnn_model
        opt.logger = ens_opt.logger
        opt.start_from = rnn_path
        opt.cnn_start_from = cnn_path
        opt.infos_start_from = infos_path
        opt.rnn_bias = 0
        try:
            opt.use_glove = infos['opt'].use_glove
        except:
            opt.use_glove = 0
        try:
            opt.use_synonyms = infos['opt'].use_synonyms
        except:
            opt.use_synonyms = 0
        options.append(opt)
    loader = DataLoader(opt)

    # Build CNN model for single branch use
    for opt in options:
        if opt.cnn_model.startswith('resnet'):
            cnn_model = cnn.ResNetModel(opt)
        elif opt.cnn_model.startswith('vgg'):
            cnn_model = cnn.VggNetModel(opt)
        else:
            print('Unknown model %s' % opt.cnn_model)
            sys.exit(1)
        cnn_model.cuda()
        print('Loading model with', opt)
        model = du.select_model(opt)
        # model.load_state_dict(torch.load(opt.model_path))
        model.load()
        model.cuda()
        cnn_models.append(cnn_model)
        rnn_models.append(model)

    # Define the ensemble:
    ens_model = Ensemble(rnn_models, cnn_models, ens_opt)
    ens_model.define_loss(loader.get_vocab())
    # Create the Data Loader instance
    start = time.time()
    # Main loop
    # To save before training:
    iteration -= 1
    val_losses = []
    update_lr_flag = True
    # Define ensemble optimizer
    optimizer = du.set_optimizer(ens_opt, epoch, rnn_models, cnn_models)
    while True:
        if update_lr_flag:
            # Assign the learning rate
            ens_opt = utils.manage_lr(epoch, ens_opt, val_losses)
            utils.scale_lr(optimizer, ens_opt.scale_lr) # set the decayed rate
            lg.log_optimizer(ens_opt, optimizer)
            update_lr_flag = False
        # Load data from train split (0)
        data = loader.get_batch('train')
        torch.cuda.synchronize()
        start = time.time()
        # Forward the ensemble
        real_loss, loss = ens_model.step(data)
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

        lg.stderr_epoch(epoch, iteration, ens_opt, losses, grad_norm, end-start)
        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True
        # Write the training loss summary
        if (iteration % ens_opt.losses_log_every == 0):
            lg.log_epoch(tf_summary_writer, iteration, ens_opt,
                         losses, grad_norm,
                         model.ss_prob)
            history['loss'][iteration] = losses['train_loss']
            history['lr'][iteration] = ens_opt.current_lr
            history['ss_prob'][iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % ens_opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                           'dataset': ens_opt.input_json}
            eval_kwargs.update(vars(ens_opt))
            (real_val_loss, val_loss,
             predictions, lang_stats, unseen_grams) = eval_utils.eval_ensemble(ens_model,
                                                                               loader,
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
            if ens_opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss
            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            lg.save_ens_model(ens_model, optimizer, ens_opt,
                              iteration, epoch, loader, best_val_score,
                              history, best_flag)
        # Stop if reaching max epochs
        if epoch >= ens_opt.max_epochs and ens_opt.max_epochs != -1:
            break

