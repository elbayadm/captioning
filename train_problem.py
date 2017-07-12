# Use tensorboard





# setup gpu
try:
    import os
    import subprocess
    gpu_id = subprocess.check_output('gpu_getIDs.sh', shell=True)
    print("Gpu%s" % gpu_id)
except:
    print("Failed to get gpu_id (setting gpu_id to 0)")
    gpu_id = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

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
import copy
import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.ssd import build_ssd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.tensorboard.plugins import projector
from math import exp

def log_optimizer(opt, optimizer):
    opt.logger.debug('####################################')
    print("Optimized params shapes:")
    for p in optimizer.param_groups:
        if isinstance(p, dict):
            opt.logger.error('Dict: %s' % list(p.keys()))
            print('LR:', p['lr'])
            for pp in p['params']:
                print(pp.size(), end=' ')
            print('\n')
    opt.logger.debug('####################################')


def manage_lr(epoch, opt, val_losses):
    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
        opt.logger.error('Updating the lr')
        if opt.lr_strategy == "step":
            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            decay_factor = opt.learning_rate_decay_rate  ** frac
            opt.current_lr = opt.learning_rate * decay_factor
            opt.scale_lr = decay_factor
        elif opt.lr_strategy == "adaptive":
            opt.logger.error('Adaptive mode')
            print("val_losses:", val_losses)
            if len(val_losses) > 2:
                if val_losses[0] > val_losses[1]:
                    opt.lr_wait += 1
                    opt.logger.error('Waiting for more')
                if opt.lr_wait > opt.lr_patience:
                    opt.logger.error('You have plateaued, decreasing the lr')
                    # decrease lr:
                    opt.current_lr = opt.current_lr * opt.learning_rate_decay_rate
                    opt.scale_lr = opt.learning_rate_decay_factor
                    opt.lr_wait = 0
            else:
                opt.current_lr = opt.learning_rate
                opt.scale_lr = 1

    else:
        opt.current_lr = opt.learning_rate
        opt.scale_lr = 1
    return opt



def get_optimizer(ref):
    #  rmsprop | sgd | sgdmom | adagrad | adam
    if ref.lower() == 'adam':
        return optim.Adam
    elif ref.lower() == 'sgd':
        return optim.SGD
    elif ref.lower() == 'rmsprop':
        return optim.RMSprop
    elif ref.lower() == 'adagrad':
        return optim.Adagrad
    else:
        raise ValueError('Unknown optimizer % s' % ref)



def add_summary_value(writer, key, value, iteration, collections=None):
    """
    Add value to tensorflow events
    """
    _summary = tf.summary.scalar(name=key,
                                 tensor=tf.Variable(value),
                                 collections=collections)
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def train(opt):
    """
    main training loop
    """
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    opt.lr_wait = 0

    tf_summary_writer = tf.summary.FileWriter(opt.modelname)
    infos = {}
    #  opt.logger.warn('\n' + '\n'.join(['%24s : %s' % (str(k),str(v)) for k, v in vars(opt).iteritems()]))
    # Restart training (useful with oar idempotant)
    if opt.restart and osp.exists(osp.join(opt.modelname, 'model.pth')):
        opt.start_from_best = 0
        opt.logger.warning('Picking up where we left')
        opt.start_from = opt.modelname

    if opt.start_from is not None:
        # open old infos and check if models are compatible
        opt.logger.warn('Starting from %s' % opt.start_from)
        if opt.start_from_best:
            opt.logger.warn('Starting from the best saved checkpoint (infos)')
            f = open(osp.join(opt.start_from, 'infos-best.pkl'))
        else:
            opt.logger.warn('Starting from the last saved checkpoint (infos)')
            f = open(osp.join(opt.start_from, 'infos.pkl'))
        infos = pickle.load(f)
        saved_model_opt = infos['opt']
        need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
        for checkme in need_be_same:
            assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        f.close()
    # Recover iteration index
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    opt.logger.warn('Starting from iteration %d (epoch %d)' % (iteration, epoch))
    # Recover training histtory
    val_result_history = infos.get('val_result_history', {})
    val_losses = []
    loss_history = infos.get('loss_history', {})
    lr_history = infos.get('lr_history', {})
    ss_prob_history = infos.get('ss_prob_history', {})
    # Recover data iterator and best perf
    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)


    # FIXME - temporary
    if opt.use_feature_maps:
        ###############################################################################################################
        opt.logger.warn('using single CNN branch with feature maps as regions embeddings')
        # Build CNN model for single branch use
        if opt.cnn_model.startswith('resnet'):
            cnn_model = utils.ResNetModel(opt)
        elif opt.cnn_model.startswith('vgg'):
            cnn_model = utils.VggNetModel(opt)
        else:
            opt.logger.error('Unknown model %s' % opt.cnn_model)
            sys.exit(1)
        ################################################################################################################
    else:
        opt.logger.warn('using SSD')
        cnn_model = build_ssd('train', 300, 21)

    cnn_model.cuda()

    # Build the captioning model
    opt.logger.error('-----------------------------SETUP')
    model = models.setup(opt)
    opt.logger.error('-----------------------------/SETUP')
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()
    if opt.raml_loss:
        #  D = np.eye(opt.vocab_size + 1, dtype="float32")
        #  D = np.random.uniform(size=(opt.vocab_size + 1, opt.vocab_size + 1)).astype(np.float32)
        D = pickle.load(open('data/Glove/cocotalk_similarities_v2.pkl', 'r'))
        D = D.astype(np.float32)
        D = Variable(torch.from_numpy(D)).cuda()
        crit = utils.SmoothLanguageModelCriterion(D, opt)
    elif opt.combine_caps_losses:
        crit = utils.MultiLanguageModelCriterion(opt.seq_per_img)
    else:
        crit = utils.LanguageModelCriterion()
    optim_func = get_optimizer(opt.optim)
    if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
        cnn_params = [{'params': module.parameters(), 'lr': opt.cnn_learning_rate * opt.learning_rate} for module in cnn_model.to_finetune]
        main_params = [{'params': model.parameters(), 'lr': opt.learning_rate}]
        optimizer = optim_func(cnn_params + main_params,
                               lr=opt.learning_rate, weight_decay=opt.weight_decay)
    else:
        optimizer = optim_func(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)



    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        if osp.isfile(osp.join(opt.start_from, 'optimizer.pth')) and not opt.finetune_cnn_only:
            opt.logger.warn('Loading saved optimizer')
            if opt.start_from_best:
                try:
                    optimizer.load_state_dict(torch.load(osp.join(opt.start_from, 'optimizer-best.pth')))
                except:
                    print("Starting with blank optimizer")
            else:
                try:
                    optimizer.load_state_dict(torch.load(osp.join(opt.start_from, 'optimizer.pth')))
                except:
                    print("The laoded optimizer doesn't have the same parms :> starting a clean optimizer")
    # Require grads
    for p in optimizer.param_groups:
        if isinstance(p, dict):
            for pp in p['params']:
                pp.requires_grad = True

    log_optimizer(opt, optimizer)
    # Main loop
    # To save before training:
    iteration -= 1
    while True:
        if update_lr_flag:
            # Assign the learning rate
            opt = manage_lr(epoch, opt, val_losses)
            #  utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            utils.scale_lr(optimizer, opt.scale_lr) # set the decayed rate
            log_optimizer(opt, optimizer)

            # Assign the scheduled sampling prob
            #  if opt.scheduled_sampling_strategy == "step":
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                opt.logger.warn('ss_prob = %.3e' % opt.ss_prob )
                model.ss_prob = opt.ss_prob
            # Update the training stage of cnn
            #  if opt.finetune_cnn_after == -1 or epoch < opt.finetune_cnn_after:
            #      for p in cnn_model.parameters():
            #          p.requires_grad = False
            #      cnn_model.eval()
            #  else:
            #      for p in cnn_model.parameters():
            #          p.requires_grad = True
            #      # Fix the first few layers:
            #      for module in cnn_model.keep_asis:
            #          for p in module.parameters():
            #              p.requires_grad = False
            cnn_model.train()
            opt.current_cnn_lr = opt.cnn_learning_rate * opt.current_lr
            update_lr_flag = False
        # Update scheduled sampling proba (iter instead of epoch based)
        #  if opt.scheduled_sampling_strategy == "sigmoid":
        #      if epoch >= opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
        #          opt.logger.warn("setting up the ss_prob")
        #          opt.ss_prob = 1 - opt.scheduled_sampling_speed / (opt.scheduled_sampling_speed + exp(iteration / opt.scheduled_sampling_speed))
        #          #  frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
        #          #  opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
        #          model.ss_prob = opt.ss_prob
        #          opt.logger.warn("ss_prob =  %.3e" % model.ss_prob)

        torch.cuda.synchronize()
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        torch.cuda.synchronize()
        #  opt.logger.info('Read data: %.3e' % (time.time() - start))
        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['images'], data['labels'], data['masks']]

        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        images, labels, masks = tmp

        if opt.use_feature_maps:
            ################################################## Att_feats and fc_feats from the same branch with att_feats as feature maps.
            att_feats, fc_feats = cnn_model(images)
            #  print "Fc_feats:", fc_feats.size()
            #  print "Att_feats:", att_feats.size()
            # Duplicate for caps per image
            att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), opt.seq_per_img,) +
                                                        att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) * opt.seq_per_img,) + att_feats.size()[1:]))
            fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), opt.seq_per_img,) +
                                                    fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) * opt.seq_per_img,) + fc_feats.size()[1:]))
            ###########################################################################
        else:
            conf, loc, priorbox = cnn_model.forward(images)
            print("conf:", conf.size())
            print("loc:", loc.size())
            print("priorbox", priorbox.size())

        optimizer.zero_grad()
        if opt.caption_model == "show_tell_vae":
            preds, recon_loss, kld_loss = model(fc_feats, att_feats, labels)
            real_loss, loss = crit(preds, labels[:, 1:], masks[:, 1:])
            loss += opt.vae_weight * (recon_loss + opt.kld_weight * kld_loss)  #FIXME add the scaling as parameter
        else:
            real_loss, loss = crit(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:])
        loss.backward()
        grad_norm = []
        grad_norm.append(utils.clip_gradient(optimizer, opt.grad_clip))
        optimizer.step()
        train_loss = loss.data[0]
        train_real_loss = real_loss.data[0]
        try:
            train_kld_loss = kld_loss.data[0]
            train_recon_loss = recon_loss.data[0]
        except:
            pass
        #  grad_norm = [utils.get_grad_norm(optimizer)]
        torch.cuda.synchronize()
        end = time.time()
        message = "iter {} (epoch {}), train_real_loss = {:.3f}, train_loss = {:.3f}, lr = {:.2e}, grad_norm = {:.3e}"\
                   .format(iteration, epoch, train_real_loss,  train_loss, opt.current_lr, grad_norm[0])

        try:
            message += ", cnn_lr = {:.2e}, cnn_grad_norm = {:.3e}".format(opt.current_cnn_lr, grad_norm[1])
        except:
            pass
        try:
            message += "\n{:>25s} = {:.3e}, kld loss = {:.3e}".format('recon loss', train_recon_loss, train_kld_loss)
        except:
            pass
        message += "\n{:>25s} = {:.3f}" \
                    .format("Time/batch", end - start)
        opt.logger.debug(message)
        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tf_summary_writer, 'train_real_loss', train_real_loss, iteration)
            add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            add_summary_value(tf_summary_writer, 'RNN_grad_norm', grad_norm[0], iteration)
            try:
                add_summary_value(tf_summary_writer, 'CNN_grad_norm', grad_norm[1], iteration)
                add_summary_value(tf_summary_writer, 'CNN_learning_rate', opt.current_cnn_lr, iteration)

            except:
                pass
            try:
                add_summary_value(tf_summary_writer, 'kld_loss', train_kld_loss, iteration)
                add_summary_value(tf_summary_writer, 'recon_loss', train_recon_loss, iteration)
            except:
                pass

            tf_summary_writer.flush()

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                           'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            real_val_loss, val_loss, predictions, lang_stats, unseen_grams = eval_utils.eval_split(cnn_model, model, crit, loader, eval_kwargs)

            # Write validation result into summary
            add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
            add_summary_value(tf_summary_writer, 'real validation loss', real_val_loss, iteration)

            for k, v in lang_stats.items():
                add_summary_value(tf_summary_writer, k, v, iteration)
            #  for k, v in unseen_grams.iteritems():
            #      add_summary_text(tf_summary_writer, k, v, iteration)
            #  add_summary_embedding(tf_summary_writer)
            tf_summary_writer.flush()
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}
            val_losses.insert(0, val_loss)
            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = osp.join(opt.modelname, 'model.pth')
                cnn_checkpoint_path = osp.join(opt.modelname, 'model-cnn.pth')
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                opt.logger.info("model saved to {}".format(checkpoint_path))
                opt.logger.info("cnn model saved to {}".format(cnn_checkpoint_path))
                optimizer_path = osp.join(opt.modelname, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['best_val_score'] = best_val_score
                infos['opt'] = copy.copy(opt)
                infos['opt'].logger = None
                infos['val_result_history'] = val_result_history
                infos['loss_history'] = loss_history
                infos['lr_history'] = lr_history
                infos['ss_prob_history'] = ss_prob_history
                infos['vocab'] = loader.get_vocab()
                with open(osp.join(opt.modelname, 'infos.pkl'), 'wb') as f:
                    pickle.dump(infos, f)

                if best_flag:
                    checkpoint_path = osp.join(opt.modelname, 'model-best.pth')
                    cnn_checkpoint_path = osp.join(opt.modelname, 'model-cnn-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                    opt.logger.info("model saved to {}".format(checkpoint_path))
                    opt.logger.info("cnn model saved to {}".format(cnn_checkpoint_path))
                    optimizer_path = osp.join(opt.modelname, 'optimizer-best.pth')
                    torch.save(optimizer.state_dict(), optimizer_path)
                    with open(osp.join(opt.modelname, 'infos-best.pkl'), 'wb') as f:
                        pickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
