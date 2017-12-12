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
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.tensorboard.plugins import projector
from misc.lm import MultiVAE_LM_encoder, VAE_LM_encoder, LM_encoder, LM_decoder

def get_optimizer(ref):
    #  rmsprop | sgd | sgdmom | adagrad | adam
    if ref.lower() == 'adam':
        return optim.Adam
    elif ref.lower == 'sgd':
        return optim.SGD
    elif ref.lower() == 'rmsprop':
        return optim.RMSprop
    elif ref.lower() == 'adagrad':
        return optim.Adagrad
    else:
        raise ValueError('Unknown optimizer % s' % ref)


#  def add_summary_text(write, key, value, iteration):
#      text_tensor = tf.convert_to_tensor(value, dtypes.string)
#      summary = tf.summary.text(key, text_tensor)
#      writer.add_summary(summary, iteration)


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
    if 'coco' in opt.input_json:
        opt.logger.error('coco loader')
        # MS-COCO
        loader = DataLoader(opt)
    else:
        # Pure text
        loader = textDataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tf_summary_writer = tf.summary.FileWriter(opt.modelname)
    infos = {}
    #  opt.logger.warn('\n' + '\n'.join(['%24s : %s' % (str(k),str(v)) for k, v in vars(opt).iteritems()]))
    # Restart training (useful with oar idempotant)
    if opt.restart and osp.exists(osp.join(opt.modelname, 'model.pth')):
        opt.logger.warning('Picking up where we left')
        opt.start_from = opt.modelname

    if opt.start_from is not None:
        # open old infos and check if models are compatible
        assert osp.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert osp.isfile(osp.join(opt.start_from,"infos.pkl")), "infos.pkl file does not exist in path %s" % opt.start_from
        with open(osp.join(opt.start_from, 'infos.pkl')) as f:
            infos = pickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        opt.logger.warn('Starting from %s' % opt.start_from)

    # Recover iteration index
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    opt.logger.warn('Starting from iteration %d (epoch %d)' % (iteration, epoch))
    # Recover training histtory
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    lr_history = infos.get('lr_history', {})
    ss_prob_history = infos.get('ss_prob_history', {})
    # Recover data iterator and best perf
    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    # Build the captioning model
    if opt.lm_model == "rnn":
        opt.logger.warn('Using Basic RNN encoder-decoder')
        encoder = LM_encoder(opt)
    elif opt.lm_model == "rnn_vae":
        opt.logger.warn('Injecting VAE block in the encoder-decoder model')
        encoder = VAE_LM_encoder(opt)
    elif opt.lm_model == "rnn_multi_vae":
        opt.logger.warn('Injecting multiVAE block in the encoder-decoder model')
        encoder = MultiVAE_LM_encoder(opt)
    else:
        raise ValueError('Unknown LM model %s' % opt.lm_model)
    decoder = LM_decoder(opt)
    encoder.cuda()
    decoder.cuda()
    model = nn.Sequential(encoder,
                          decoder)
    if opt.start_from is not None:
        # check if all necessary files exist
        assert osp.isfile(osp.join(opt.start_from,"model.pth")), "model.pth file does not exist in path %s" % opt.start_from
        model.load_state_dict(torch.load(osp.join(opt.start_from, 'model.pth')))

    update_lr_flag = True
    # Assure in training mode
    model.train()
    if opt.match_pairs:
        crit = utils.PairsLanguageModelCriterion(opt)
    else:
        crit = utils.LanguageModelCriterion()
    optim_func = get_optimizer(opt.optim)
    optimizer = optim_func(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    print("Model params shapes:")
    for p in optimizer.param_groups:
        if isinstance(p, dict):
            print('Dict:', list(p.keys()))
            print('LR:', p['lr'])
            for pp in p['params']:
                print(pp.size())

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        if osp.isfile(osp.join(opt.start_from, 'optimizer.pth')):
            opt.logger.warn('Loading saved optimizer')
            optimizer.load_state_dict(torch.load(osp.join(opt.start_from, 'optimizer.pth')))

    # Main loop
    while True:
        if update_lr_flag:
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                decoder.ss_prob = opt.ss_prob
            update_lr_flag = False

        torch.cuda.synchronize()
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        torch.cuda.synchronize()
        #  opt.logger.info('Read data: %.3e' % (time.time() - start))
        torch.cuda.synchronize()
        start = time.time()
        tmp = [data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        labels, masks = tmp
        optimizer.zero_grad()
        if opt.lm_model == "rnn":
            codes = encoder(labels)
        else:
            z_mu, z_var, codes = encoder(labels)
            kld_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
            train_kld_loss = kld_loss.data[0]
        loss = crit(decoder(codes, labels), labels[:,1:], masks[:,1:])[0]
        if "vae" in opt.lm_model and opt.kld_weight:
            loss += opt.kld_weight *  kld_loss
        loss.backward()
        grad_norm = utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.data[0]
        torch.cuda.synchronize()
        end = time.time()
        try:
            opt.logger.info("iter {} (epoch {}), train_loss = {:.3f}, kld_loss = {:.3f},  grad = {:.3e}, time/batch = {:.3f}" \
                            .format(iteration, epoch, train_loss, train_kld_loss, grad_norm, end - start))
        except:
            opt.logger.info("iter {} (epoch {}), train_loss = {:.3f}, grad = {:.3e}, time/batch = {:.3f}" \
                            .format(iteration, epoch, train_loss, grad_norm, end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', decoder.ss_prob, iteration)
            add_summary_value(tf_summary_writer, 'RNN_grad_norm', grad_norm, iteration)
            try:
                add_summary_value(tf_summary_writer, 'kld_loss', train_kld_loss, iteration)
            except:
                pass

            tf_summary_writer.flush()

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = decoder.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                           'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, lang_stats = eval_utils.eval_lm_split(encoder, decoder, crit, loader, eval_kwargs)

            #  Write validation result into summary
            add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
            for k, v in lang_stats.items():
                add_summary_value(tf_summary_writer, k, v, iteration)
            tf_summary_writer.flush()
            val_result_history[iteration] = {'loss': val_loss}
            current_score = - val_loss
            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = osp.join(opt.modelname, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                opt.logger.info("model saved to {}".format(checkpoint_path))
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
                    torch.save(model.state_dict(), checkpoint_path)
                    opt.logger.info("model saved to {}".format(checkpoint_path))
                    with open(osp.join(opt.modelname, 'infos-best.pkl'), 'wb') as f:
                        pickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
