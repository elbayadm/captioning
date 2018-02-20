import json
import numpy as np
import torch
from torch.autograd import Variable
import utils
from utils.language_eval import language_eval
import pickle as pickle
from utils.logging import print_sampled


def short_path(path):
    return int(path.split('.')[0].split('_')[-1])


def generate_caps(encoder, decoder, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    split = eval_kwargs.get('split', 'train')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    beam_size = 1
    logger = eval_kwargs.get('logger')
    lm_model = eval_kwargs.get('lm_model')
    vocab_size = eval_kwargs.get('vocab_size')
    sample_max = eval_kwargs.get('sample_max')
    temperature = eval_kwargs.get('temperature')
    tries = eval_kwargs.get('tries', 5)
    sample_limited_vocab = eval_kwargs.get('sample_limited_vocab', 0)
    output_file = eval_kwargs.get('output_file')

    print('Using sample_max = %d  ||  temperature %.2f' % (sample_max, temperature))

    # Make sure in the evaluation mode
    encoder.eval()
    decoder.eval()
    logger.warn('Generating captions for the full training set')
    loader.reset_iterator(split)
    n = 0
    blobs = []
    SENTS = []
    gen_SENTS = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        # forward the model to get loss
        #  if n > 100:
        #      break
        infos = data['infos']
        ids = [inf['id'] for inf in infos]
        assert len(ids) == 1, "Batch size larger than 1"
        tmp = [data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        labels, masks = tmp
        tr = 0
        gt = utils.decode_sequence(loader.get_vocab(), labels[:,1:].data)
        SENTS += gt
        blob_batch = { "id": ids[0], "gt": gt, "sampled": []}
        for igt in gt:
            print_sampled(ids[0], gt)

        while tr < tries:
            #  z_mu, z_var, codes = encoder(labels)
            if lm_model == "rnn_vae":
                codes = encoder.sample(labels)
            elif lm_model == "rnn_multi_vae":
                codes = encoder.sample_group(labels)
                #  scodes = encoder.sample(labels)
            else:
                codes = encoder(labels)
            if sample_limited_vocab:
                sample_vocab = np.unique(labels[:, 1:].cpu().data.numpy())
                print("sample_vocab:", sample_vocab.tolist())
                seq, _ = decoder.sample_ltd(codes, sample_vocab, {'beam_size': beam_size,
                                                                  "vocab_size": vocab_size,
                                                                  "sample_max": sample_max,
                                                                  "temperature": temperature})
            else:
                seq, _ = decoder.sample(codes, {'beam_size': beam_size,
                                                 "vocab_size": vocab_size,
                                                 "sample_max": sample_max,
                                                 "temperature": temperature})

            sents = utils.decode_sequence(loader.get_vocab(), seq)
            #  ssents = utils.decode_sequence(loader.get_vocab(), sseq)
            gen_SENTS += sents
            #  gen_SENTS += ssents
            for isent in sents:
                print_sampled(0, isent, warn=True)
            #  print '--------------------(SINGLE)------------------------'
            #  for isent in ssents:
            #      print _WARNING + isent + _ENDC
            print('----------------------------------------------------')

            blob_batch['sampled'] += sents
            #  blob_batch['sampled'] += ssents
            tr += 1
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if data['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
        blobs.append(blob_batch)
        #  print "Blob batch:", blob_batch
    json.dump(blobs, open(output_file, 'w'))
    if lang_eval:
        lang_stats = language_lm_eval(SENTS, gen_SENTS)
        print(lang_stats)
    encoder.train()
    decoder.train()
    return 1

def track_rnn(cnn_model, model, loader, logger, eval_kwargs={}):
    split = eval_kwargs.get('split', 'val')
    val_images_use = eval_kwargs.get('val_images_use', -1)
    add_dirac = eval_kwargs.get('add_dirac', 0)
    seq_per_img = eval_kwargs.get('seq_per_img')
    # Make sure to be in the evaluation mode
    cnn_model.eval()
    model.eval()
    logger.warn('Evaluating the %s split (%d)' % (split,
                                                  val_images_use))
    loader.reset_iterator(split)
    n = 0
    rew = []
    logp = []
    sampl = []
    logp_sampl = []
    sids = []
    while True:
        data = loader.get_batch(split, batch_size=5, seq_per_img=seq_per_img)
        n = n + loader.batch_size
        images = data['images']
        sids.append(data['infos'])
        images = Variable(torch.from_numpy(images), requires_grad=False).cuda()
        att_feats, fc_feats = cnn_model.forward_caps(images, seq_per_img)
        logprobs, rewards, sampled, logprobs_sampled = model.step_track(data,
                                                                        att_feats,
                                                                        fc_feats,
                                                                        add_dirac)
        rew.append(rewards)
        logp.append(logprobs)
        sampl.append(sampled)
        logp_sampl.append(logprobs_sampled)
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        if data['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
    return sids, logp, rew, sampl, logp_sampl


def track_rnn_decode(cnn_model, model, loader, logger, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', False)
    split = eval_kwargs.get('split', 'val')
    val_images_use = eval_kwargs.get('val_images_use', -1)
    seq_per_img = eval_kwargs.get('seq_per_img')
    # Make sure to be in the evaluation mode
    cnn_model.eval()
    model.eval()
    logger.warn('Evaluating the %s split (%d)' % (split,
                                                  val_images_use))
    assert eval_kwargs.get('beam_size', 1) == 1
    loader.reset_iterator(split)
    n = 0
    rew = []
    logp = []
    sampl = []
    logp_sampl = []
    sids = []
    attention = []
    while True:
        data = loader.get_batch(split, batch_size=5, seq_per_img=seq_per_img)
        n = n + loader.batch_size
        sids.append(data['infos'])
        images = data['images']
        images = Variable(torch.from_numpy(images), requires_grad=False).cuda()
        att_feats, fc_feats, att_unique, fc_unique = cnn_model.forward_caps(images,
                                                                            seq_per_img,
                                                                            return_unique=True)
        seq, probs, alphas = model.sample(fc_unique, att_unique, True, eval_kwargs)
        logp.append(probs.cpu().numpy())
        # print('logp:', logp[-1])
        sampl.append(seq.cpu().numpy())
        attention.append(alphas.cpu().numpy())
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        if data['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
    return sids, logp, sampl, attention


def eval_split(cnn_model, model, loader, logger, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', False)
    dataset = eval_kwargs.get('dataset', 'coco')
    split = eval_kwargs.get('split', 'val')
    val_images_use = eval_kwargs.get('val_images_use', -1)
    lang_eval = eval_kwargs.get('language_eval', 1)
    language_creativity = eval_kwargs.get('language_creativity', 1)
    all_metrics = eval_kwargs.get('all_metrics', 0)
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_max = eval_kwargs.get('sample_max', 1)
    temperature = eval_kwargs.get('temperature', 0.5)
    forbid_unk = eval_kwargs.get('forbid_unk', 1)
    batch_size = eval_kwargs.get('batch_size', 1)
    seq_per_img = eval_kwargs.get('seq_per_img')
    region_size = model.region_size
    # Make sure to be in the evaluation mode
    cnn_model.eval()
    model.eval()
    logger.warn('Evaluating the %s split (%d)' % (split,
                                                  val_images_use))
    loader.reset_iterator(split)
    n = 0
    loss_sum = 0
    ml_loss_sum = 0
    loss_evals = 0
    predictions = []
    while True:
        data = loader.get_batch(split, batch_size=batch_size, seq_per_img=seq_per_img)
        n = n + loader.batch_size
        images = data['images']
        images = Variable(torch.from_numpy(images), requires_grad=False).cuda()
        att_feats, fc_feats, att_unique, fc_unique = cnn_model.forward_caps(images,
                                                                            seq_per_img,
                                                                            return_unique=True)
        ml_loss, loss, stats = model.step(data, att_feats, fc_feats, train=False)
        # print('Scores : ', stats)
        ml_loss_sum += ml_loss.data[0]
        loss_sum += loss.data[0]
        loss_evals = loss_evals + 1
        # TODO Only leave one feature for each image, in case duplicate sample
        seq, probs = model.sample(fc_unique, att_unique,
                                  opt={'beam_size': beam_size,
                                       "forbid_unk": forbid_unk,
                                       "sample_max": sample_max,
                                       "temperature": temperature})
        sent_scores = probs.cpu().numpy().sum(axis=1)
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        for k, sent in enumerate(sents):
            if loader.flip:
                entry = {'image_id': data['infos'][k // 2]['id'],
                         'caption': sent, 'score': sent_scores[k]}
                if not k % 2:
                    unflipped = entry
                else:
                    if entry['score'] > unflipped['score']:
                        del entry['score']
                        predictions.append(entry)
                    else:
                        del unflipped['score']
                        predictions.append(unflipped)
            else:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                predictions.append(entry)
            print_sampled(entry['image_id'], entry['caption'])
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        for i in range(n - ix1):
            predictions.pop()
        if data['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
    lang_stats = None
    if lang_eval:
        lang_stats, _ = language_eval(dataset, predictions, logger,
                                      all_metrics,
                                      language_creativity)
    # Back to training:
    model.train()
    if model.cnn_finetuning:
        logger.warn('Finetuning cnn ON, filtering the BN layers')
        cnn_model.train()
        cnn_model.filter_bn()
    return ml_loss_sum/loss_evals, loss_sum/loss_evals, predictions, lang_stats


def eval_external(cnn_model, model, loader, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', -1)
    split = eval_kwargs.get('split', 'val')
    # serves no purpose except to have the same signature for get_batch
    beam_size = eval_kwargs.get('beam_size', 1)
    logger = eval_kwargs.get('logger')
    caption_model = eval_kwargs.get('caption_model')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_max = eval_kwargs.get('sample_max', 1)
    temperature = eval_kwargs.get('temperature', 0.5)
    forbid_unk = eval_kwargs.get('forbid_unk', 1)

    print("Eval %s" % caption_model)

    # Make sure in the evaluation mode
    cnn_model.eval()
    model.eval()
    loader.reset_iterator(split)

    n = 0
    predictions = []
    seq_per_img = 1
    while True:
        data = loader.get_batch(split, seq_per_img=seq_per_img)
        n = n + loader.batch_size
        # forward the model to get loss
        images = data['images']
        images = Variable(torch.from_numpy(images), volatile=True).cuda()
        att_feats, fc_feats, att_unique, fc_unique = cnn_model.forward_caps(images,
                                                                            seq_per_img,
                                                                            return_unique=True)
        seq, _ = model.sample(fc_feats, att_feats,
                              {'beam_size': beam_size,
                               'forbid_unk': forbid_unk,
                               "sample_max": sample_max,
                               "temperature": temperature}
                              )
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            spath = short_path(data['infos'][k]['file_path'])
            entry = {'image_id': spath, 'caption': sent}
            print_sampled(spath, sent)
            predictions.append(entry)
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
        if data['bounds']['wrapped']:
            break
        if n >= ix1:
            logger.warn('Evaluated the required samples (%s)' % n)
            break
    # Switch back to training mode
    model.train()
    return predictions


def eval_multiple(cnn_model, model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    score_ground_truth = eval_kwargs.get('score_ground_truth', False)
    n_gen = eval_kwargs.get('n_gen', 5)
    num_images = eval_kwargs.get('num_images', -1)
    seq_length = eval_kwargs.get('seq_length', 16)
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    batch_size = eval_kwargs.get('batch_size', 1)

    # Make sure in the evaluation mode
    cnn_model.eval()
    model.eval()
    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []

    while True:
        # fetch a batch of data
        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        #evaluate loss if we have the labels
        loss = 0

        # Get the image features first
        tmp = [data['images'], data.get('labels', np.zeros(1)), data.get('masks', np.zeros(1)), data['scores']]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        images, labels, masks, scores = tmp

        att_feats, fc_feats = cnn_model.forward(images)
        _att_feats = att_feats
        _fc_feats = fc_feats

        # forward the model to get loss
        if data.get('labels', None) is not None:
            att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), loader.seq_per_img,) + att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) * loader.seq_per_img,) + att_feats.size()[1:]))
            fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), loader.seq_per_img,) + fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) * loader.seq_per_img,) + fc_feats.size()[1:]))

            input = model(fc_feats, att_feats, labels)
            probs = input
            N = input.size(0)
            mask = masks[:,1:]
            target = labels[:, 1:]
            target = target[:, :input.size(1)]
            mask = mask[:, :input.size(1)]
            input = utils.to_contiguous(input).view(-1, input.size(2))
            target = utils.to_contiguous(target).view(-1, 1)
            mask = mask[:, :input.size(1)]
            mask = utils.to_contiguous(mask).view(-1, 1)
            output = input.gather(1, target) * mask
            output = output.cpu().data.numpy()
            # sum over seq_length
            gt_scores = [np.sum(output[seq_length * i: seq_length * (i+1)]) for i in np.arange(N)]
            gt_sents =  utils.decode_sequence(loader.get_vocab(), labels[:,1:].data)
            real_loss, loss = crit(probs, labels[:,1:], masks[:,1:], scores)
            loss_sum = loss_sum + loss.data[0]
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        fc_feats, att_feats = _fc_feats, _att_feats
        # forward the model to also get generated samples for each image
        for _ in range(n_gen):
            seq, probs = model.sample(fc_feats, att_feats, eval_kwargs)
            sent_scores = probs.cpu().numpy().sum(axis=1)
            #set_trace()
            sents = utils.decode_sequence(loader.get_vocab(), seq)
            print('Gen:', len(sents), len(sent_scores))
            for k, sent in enumerate(sents):
                # print('id:', data['infos'][k]['id'])
                if loader.flip:
                    entry = {'image_id': data['infos'][k // 2]['id'], 'caption': sent, 'score': str(round(sent_scores[k], 4)), "source": 'gen'}
                    if not k % 2:
                        unflipped = entry
                    else:
                        # compare the new entry to unflipped and keep the best candidate
                        # print('Comparing:', entry, ' to ', unflipped)
                        if float(entry['score']) > float(unflipped['score']):
                            predictions.append(entry)
                            # print('picking:', entry)
                        else:
                            predictions.append(unflipped)
                            # print('picking:', unflipped)
                else:
                    entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'score': str(round(sent_scores[k], 4)), "source": 'gen'}
                    predictions.append(entry)
                if verbose:
                    # print(entry)
                    print(('%s >>  %s' %(entry['image_id'], entry['caption'])))
        if score_ground_truth:
            print('Gt:', len(gt_sents), len(gt_scores))
            for k, sent in enumerate(gt_sents):
                if loader.flip:
                    entry = {'image_id': data['infos'][k // (loader.seq_per_img * 2)]['id'], 'caption': sent, 'score': str(round(gt_scores[k], 4)), "source": 'gt'}
                else:
                    entry = {'image_id': data['infos'][k // loader.seq_per_img]['id'], 'caption': sent, 'score': str(round(gt_scores[k], 4)), "source": 'gt'}
                predictions.append(entry)
                if verbose:
                    print(('image %s (GT : %s) %s' %(entry['image_id'], entry['score'], entry['caption'])))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss.data[0]))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    unseen_grams = None
    if lang_eval == 1:
        lang_stats, unseen_grams = language_eval(dataset, predictions, logger=None)  # FIXME
    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats, unseen_grams
