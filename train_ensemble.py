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
import copy
import opts
import capmodels
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.ssd import build_ssd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.tensorboard.plugins import projector


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

def train_ensemble(ens_opt):
    """
    main training loop
    """
    models_paths = []
    infos_paths = []
    infos_list = []
    cnn_models = []
    rnn_models = []
    for m in ens_opt.model:
        models_paths.append('save/%s/model.pth' % m)
        infos_path = "save/%s/infos.pkl" % m
        with open(infos_path, 'rb') as f:
            print('Opening %s' % infos_path)
            infos = pickle.load(f, encoding="iso-8859-1")
            infos_list.append(infos)

    print("Parsed % infos files" % len(infos_list))
    options = []
    for e, infos in enumerate(infos_list):
        # override and collect parameters
        opt = argparse.Namespace(**vars(ens_opt))
        print('List of models:', ens_opt.model)
        opt.model = ens_opt.model[e]
        print("Current model:", opt.model)
        if len(opt.input_h5) == 0:
            opt.input_h5 = infos['opt'].input_h5
        if len(opt.input_json) == 0:
            opt.input_json = infos['opt'].input_json
        if opt.batch_size == 0:
            opt.batch_size = infos['opt'].batch_size
        #Check if new features in opt:
        if "less_confident" not in opt:
            opt.less_confident = 0
        if "scheduled_sampling_strategy" not in opt:
            opt.scheduled_sampling_strategy = "step"
        if "scheduled_sampling_vocab" not in opt:
            opt.scheduled_sampling_vocab = 0
        ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
        for k in list(vars(infos['opt']).keys()):
            if k not in ignore:
                if k in vars(opt):
                    assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
                else:
                    vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model
        opt.use_feature_maps = infos['opt'].use_feature_maps
        opt.cnn_model = infos['opt'].cnn_model
        opt.logger = opts.create_logger('./tmp_eval.log')
        opt.start_from = "save/" + opt.model
        opt.model_path = "save/" + opt.model + "/model.pth"

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

    vocab = infos_list[0]['vocab'] # ix -> word mapping
    ens_opt.logger = opts.create_logger('%s/train.log' % ens_opt.ensemble_dir)
    # Build CNN model for single branch use
    for opt in options:
        if opt.cnn_model.startswith('resnet'):
            cnn_model = utils.ResNetModel(opt)
        elif opt.cnn_model.startswith('vgg'):
            cnn_model = utils.VggNetModel(opt)
        else:
            print('Unknown model %s' % opt.cnn_model)
            sys.exit(1)
        cnn_model.cuda()
        # cnn_model.eval()
        print('Loading model with', opt)
        model = capmodels.setup(opt)
        model.load_state_dict(torch.load(opt.model_path))
        model.cuda()
        # model.eval()
        cnn_models.append(cnn_model)
        rnn_models.append(model)
    # Create the Data Loader instance
    start = time.time()












    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    opt.lr_wait = 0

    tf_summary_writer = tf.summary.FileWriter(opt.modelname)
    infos = {}
    #  opt.logger.warn('\n' + '\n'.join(['%24s : %s' % (str(k),str(v)) for k, v in vars(opt).iteritems()]))
    # Restart training (useful with oar idempotant)
    if opt.restart and osp.exists(osp.join(opt.modelname, 'model.pth')):
        opt.tart_from_best = 0
        opt.logger.warning('Picking up where we left')
        opt.start_from = opt.modelname

    if opt.start_from is not None:
        # open old infos and check if models are compatible
        opt.logger.warn('Starting from %s' % opt.start_from)
        if opt.start_from_best:
            opt.logger.warn('Starting from the best saved checkpoint (infos)')
            f = open(osp.join(opt.start_from, 'infos-best.pkl'), 'rb')
        else:
            opt.logger.warn('Starting from the last saved checkpoint (infos)')
            f = open(osp.join(opt.start_from, 'infos.pkl'), 'rb')
        infos = pickle.load(f, encoding='iso-8859-1')
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
    model = capmodels.setup(opt)
    opt.logger.error('-----------------------------/SETUP')
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()
    if opt.raml_loss:
        #  D = np.eye(opt.vocab_size + 1, dtype="float32")
        #  D = np.random.uniform(size=(opt.vocab_size + 1, opt.vocab_size + 1)).astype(np.float32)
        # D = pickle.load(open('data/Glove/cocotalk_similarities_v2.pkl', 'rb'), encoding='iso-8859-1')
        D = pickle.load(open(opt.similarity_matrix, 'rb'), encoding='iso-8859-1')

        D = D.astype(np.float32)
        D = Variable(torch.from_numpy(D)).cuda()
        crit = utils.SmoothLanguageModelCriterion(Dist=D,
                                                  loader_vocab=loader.get_vocab(),
                                                  opt=opt)
    elif opt.bootstrap_loss:
        # Using importance sampling loss:
        crit = utils.ImportanceLanguageModelCriterion(opt)

    elif opt.combine_caps_losses:
        crit = utils.MultiLanguageModelCriterion(opt.seq_per_img)
    else:
        opt.logger.warn('Using baseline loss criterion')
        crit = utils.LanguageModelCriterion(opt)
    optim_func = get_optimizer(opt.optim)
    # TODO; add sclaed lr for every chunk of cnn layers
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
            if opt.scheduled_sampling_strategy == "step":
                if epoch >= opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob
                    opt.logger.warn('ss_prob= %.2e' % model.ss_prob )
            # CNN finetuning
            # -------------------------------------------------------FIXME
            #  cnn_model.train()
            #  opt.current_cnn_lr = opt.cnn_learning_rate * opt.current_lr
            #-------------------------------------------------------------
            if opt.raml_loss and opt.raml_alpha_strategy == "step":
                # Update crit's alpha:
                opt.logger.warn('Updating the loss scaling param alpha')
                frac = epoch // opt.raml_alpha_increase_every
                new_alpha = min(opt.raml_alpha_increase_factor  * frac, 1)
                crit.alpha = new_alpha
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
            crit.alpha = new_alpha
            opt.logger.warn('New alpha %.3e' % new_alpha)
        if opt.raml_loss:
            opt.logger.error('Sanity check alpha = %.3e' % crit.alpha)

        # Load data from train split (0)
        data = loader.get_batch('train')
        torch.cuda.synchronize()
        start = time.time()
        if opt.bootstrap_loss:
            if opt.bootstrap_version in ["cider", "cider-exp"]:
                tmp = [data['images'], data['labels'], data['masks'], data['scores'], data['cider']]
                tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
                images, labels, masks, scores, s_scores= tmp

            elif opt.bootstrap_version in ["bleu4", "bleu4-exp"]:
                tmp = [data['images'], data['labels'], data['masks'], data['scores'], data['bleu']]
                tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
                images, labels, masks, scores, s_scores = tmp
            elif opt.bootstrap_version in ["infersent", "infersent-exp"]:
                tmp = [data['images'], data['labels'], data['masks'], data['scores'], data['infersent']]
                tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
                images, labels, masks, scores, s_scores = tmp
            else:
                raise ValueError('Unknown bootstrap distribution %s' % opt.bootstrap_version)
            if "exp" in opt.bootstrap_version:
                print('Original rewards:', torch.mean(s_scores))
                s_scores = torch.exp(torch.div(s_scores, opt.raml_tau))
                print('Tempering the reward:', torch.mean(s_scores))
            r_scores = torch.div(s_scores, torch.exp(scores))

            print('Importance scores:', torch.mean(r_scores))
        else:
            tmp = [data['images'], data['labels'], data['masks'], data['scores']]
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            images, labels, masks, scores = tmp

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
        elif opt.caption_model == 'show_tell_raml':
            probs, reward = model(fc_feats, att_feats, labels)
            raml_scores = reward * Variable(torch.ones(scores.size()))
            # raml_scores = Variable(torch.ones(scores.size()))
            print('Raml reward:', reward)
            real_loss, loss = crit(probs, labels[:, 1:], masks[:, 1:], raml_scores)
        else:
            if opt.bootstrap_loss:
                real_loss, loss = crit(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:], r_scores)
            else:
                real_loss, loss = crit(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:], scores)
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

if __name__ == "__main__":
    # Input arguments and options
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--verbose', type=int, default=0,
                        help='code verbosity')
    # Basic options
    parser.add_argument('--batch_size', type=int, default=1,
                        help='if > 0 then overrule, otherwise load from checkpoint.')
    parser.add_argument('--language_eval', type=int, default=1,
                        help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    # Sampling options
    parser.add_argument('--sample_max', type=int, default=1,
                        help='1 = sample argmax words. 0 = sample from distributions.')
    parser.add_argument('--forbid_unk', type=int, default=1,
                        help='Forbid unk token generations.')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    # For evaluation on MSCOCO images from some split:
    parser.add_argument('--input_h5', type=str, default='',
                        help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_json', type=str, default='',
                        help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
    parser.add_argument('--model', nargs="+",
                        help='shortcut')
    parser.add_argument('--ensemble_dir', type=str,
                        default='save/ensemble',
                        help='shortcut')


    #TODO:  Add option for epoch eval