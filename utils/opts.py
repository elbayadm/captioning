import configargparse
import argparse
import os
import os.path as osp
import logging
from .colorstreamhandler import ColorStreamHandler


def add_vae_params(parser):
    # VAE model
    parser.add('--z_size', type=int,
               default=100, help='VAE/CVAE latent variable size')
    parser.add('--z_interm_size', type=int,
               default=1000, help='intermediary layer between the input and the latent')
    parser.add('--vae_nonlin', type=str,
               default="sigmoid", help="Non-linearity applied to the input of the cvae")
    parser.add('--vae_weight', type=float,
               default=0.1, help="weight of the vae loss (recon + kld)")
    parser.add('--kld_weight', type=float,
               default=0, help="weight of the kld loss")
    return parser


def add_eval_params(parser):
    # Evaluation and Checkpointing
    # Sampling options
    parser.add('--sample_max', type=int, default=1,
               help='1 = sample argmax words. 0 = sample from distributions.')
    parser.add('--forbid_unk', type=int, default=1,
               help='Forbid unk token generations.')
    parser.add('--beam_size', type=int, default=3,
               help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    parser.add('--temperature', type=float, default=0.5,
               help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
    parser.add('--val_images_us_', type=int,
               default=-1, help='how many images to use when evaluating (-1 = all)')
    parser.add('--save_checkpoint_every', type=int,
               default=4000, help='how often to save a model checkpoint (in iterations)?')
    parser.add('--language_creativity', type=int,
               default=1, help='Evaluate creativity scores')
    parser.add('--language_eval', type=int,
               default=1, help='Evaluate performance scores (CIDEr, Bleu...)')
    parser.add('--losses_log_every', type=int,
               default=25, help='How often do we snapshot')
    parser.add('--load_best_score', type=int,
               default=1, help='Do we load previous best score when resuming training.')
    parser.add('--add_dirac', type=int,
               default=0, help='add dirac to the saved reward track_distrib.py')
    parser.add('--save_stats', type=int,
               default=0, help='if> 0,save rewards and ptheta for x first captions')

    return parser


def add_loss_params(parser):
    # Deprecated
    # Importance sampling:
    parser.add('--bootstrap', type=int,
               default=0, help='use bootstrap/importance sampling loss.')
    parser.add('--bootstrap_score', type=str,
               default="cider", help='Version of Bootstrap loss')

    parser.add('--gt_loss_version', type=str,
               default="ml", help='Separate loss for the gold caps')
    parser.add('--augmented_loss_version', type=str,
               default="ml", help='Separate loss for the augmented caps')
    # //Deprecated

    # Combining lossess
    parser.add('--alter_loss', type=int, default=0, help='Alter between losses at every iteration')
    parser.add('--alter_mode', type=str, default='even-odd', help='How to altern between losses: even-odd, even-odd-epoch, epoch')
    parser.add('--sum_loss', type=int, default=0, help='Sum two different losses')
    parser.add('--beta', type=float,
               default=0.1, help='Scalar used to weight the losses')
    parser.add('--gamma', type=float,
               default=.33, help='Scalar used to weight the losses')
    # Deprecated
    parser.add('--combine_loss', type=int,
               default=0, help='combine WL with SL')

    # Loss smoothing
    parser.add('--loss_version', type=str,
               default="ml", help='The loss version:\
               ml: cross entropy,\
               word: word smoothing,\
               seq: sentence smoothing')
    # Generic loss parameters:
    parser.add('--normalize_batch', type=int,
               default=1, help='whether to normalize the batch loss via the mask')
    parser.add('--penalize_confidence', type=float,
               default=0, help='if neq 0, penalize the confidence by subsiding this * H(p) to the loss')
    parser.add('--scale_loss', type=float,
               default=0, help='if neq 0, each sentence loss will be scaled by a pre-computed score (cf dataloader)')

    # loss_version == word params
    parser.add('--similarity_matrix', type=str,
               default='data/Glove/cocotalk_similarities_v2.pkl',
               help='path to the pre-computed similarity matrix between the vocab words')
    parser.add('--use_cooc', type=int, default=0,
               help='Use cooccurrences matrix instead of glove similarities')
    parser.add('--margin_sim', type=float,
               default=0, help='if neq 0 clip the similarities below this')
    parser.add('--limited_vocab_sim', type=int,
               default=0, help='whether or not to restrain to a subset of similarities\
               0 : the full vocabulary,\
               1 : the 5 captions vocabulary')
    parser.add('--rare_tfidf', type=int,
               default=0, help='increase the similarity of rare words')
    parser.add('--alpha_word', type=float,
               default=0.9, help='Scalar used to weigh the word loss\
               the final loss = alpha * word + (1-alpha) ml')
    parser.add('--tau_word', type=float,
               default=0.005, help='Temperature applied to the words similarities')

    # loss_version == seq params
    parser.add('--lazy_rnn', type=int,
               default=0, help='lazy estimation of the sampled sentences logp')
    parser.add('--mc_samples', type=int,
               default=1, help='Number of MC samples')
    parser.add('--reward', type=str, default='hamming',
               help='rewards at the seuqence level,\
               options: hamming, bleu1:4, cider, tfidf')
    parser.add('--stratify_reward', type=int,
               default=1, help='sample the reward itself, only possible with reward=Hamming, tfidf')
    parser.add('--importance_sampler', type=str,
               default="greedy", help='the method used to sample candidate sequences,\
               options: greedy (the captioning model itself),\
               hamming: startified sampling of haming')

    parser.add('--alpha_sent', type=float,
               default=0.4, help='Scalar used to weight the losses')
    parser.add('--tau_sent', type=float,
               default=0, help='Temperature applied to the sentences scores (r)')
    parser.add('--tau_sent_q', type=float,
               default=0.3, help='Temperature applied to the sentences scores (q) if relevant')

    parser.add('--clip_reward', type=int,
               default=0, help='Clip and scale the sentence scores')

    # CIDEr specific
    parser.add('--cider_df', type=str,
               default='data/coco-train-df.p', help='path to dataset n-grams frequency')

    # Hamming specific
    parser.add('--limited_vocab_sub', type=int,
               default=1, help='Hamming vocab pool, options:\
               0: the full vocab \
               1: in-batch,\
               2: captions of the image')
    # TFIDF specific
    parser.add('--sub_idf', type=int,
               default=0, help='Substitute commnon ngrams')
    parser.add('--ngram_length', type=int,
               default=2, help='ngram length to substitute')

    # Alpha scheme:
    parser.add('--alpha_increase_every', type=int,
               default=2, help='step width')
    parser.add('--alpha_increase_factor', type=float,
               default=0.1, help='increase factor when step')
    parser.add('--alpha_max', type=float,
               default=0.9, help='increase factor when step')
    parser.add('--alpha_increase_start', type=int,
               default=1, help='increase factor when step')
    parser.add('--alpha_speed', type=float,
               default=20000, help='alpha decreasing speed')
    parser.add('--alpha_strategy', type=str,
               default="constant", help='Increase strategy')

    return parser


def add_scheduled_sampling(parser):
    # Scheduled sampling parameters:
    parser.add('--scheduled_sampling_start', type=int,
               default=-1, help='at what iteration to start decay gt probability')
    parser.add('--scheduled_sampling_vocab', type=int,
               default=0, help='if 1 limits sampling to the gt vocab')
    parser.add('--scheduled_sampling_speed', type=int,
               default=100, help='ss speed')
    parser.add('--scheduled_sampling_increase_every', type=int,
               default=5, help='every how many iterations thereafter to gt probability')
    parser.add('--scheduled_sampling_increase_prob', type=float,
               default=0.05, help='How much to update the prob')
    parser.add('--scheduled_sampling_max_prob', type=float,
               default=0.25, help='Maximum scheduled sampling prob.')
    parser.add('--scheduled_sampling_strategy', type=str,
               default="step", help='the decay schedule')
    parser.add('--match_pairs', type=int, #FIXME
               default=0, help='match senetences pairs')
    return parser


def add_optim_params(parser):
    # Optimization: General
    parser.add('--restart', type=int,
               default=1, help='0: to override the existing model or 1: to pick up the training')

    parser.add('--max_epochs', type=int,
               default=20, help='number of epochs')
    parser.add('--grad_clip', type=float,
               default=0.1, help='clip gradients at this value')
    parser.add('--finetune_cnn_after', type=int,
               default=-1, help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add('--finetune_cnn_only', type=int,
               default=0, help="if 1, finetune the cnn only.")
    parser.add('--finetune_cnn_slice', type=int,
               default=0, help='modules from which start finetuning')

    ## RNN optimizer
    parser.add('--optim', type=str,
               default='adam', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add('--optim_alpha', type=float,
               default=0.8, help='alpha for adam')
    parser.add('--optim_beta', type=float,
               default=0.999, help='beta used for adam')
    parser.add('--optim_epsilon', type=float,
               default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add('--weight_decay', type=float,
               default=0, help='main optimizer weight decay')

    ## LR and its scheme
    parser.add('--learning_rate', type=float,
               default=4e-4, help='learning rate')
    parser.add('--learning_rate_decay_start', type=int,
               default=5, help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add('--lr_patience', type=int,
               default=2, help='Epochs after overfitting before decreasing the lr')
    parser.add('--lr_strategy', type=str,
               default="step", help="between step (automatic decrease) or adaptive to the val loss")
    parser.add('--learning_rate_decay_every', type=int,
               default=3, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add('--learning_rate_decay_rate', type=float,
               default=0.6, help='every how many iterations thereafter to drop LR?(in epoch)')

    ## CNN optimizer
    parser.add('--cnn_optim', type=str,
               default='adam', help='optimization to use for CNN')
    parser.add('--cnn_optim_alpha', type=float,
               default=0.8, help='alpha for momentum of CNN')
    parser.add('--cnn_optim_beta', type=float,
               default=0.999, help='beta for momentum of CNN')
    parser.add('--cnn_weight_decay', type=float,
               default=0, help='L2 weight decay just for the CNN')

    ## CNN LR
    parser.add('--cnn_learning_rate', type=float,
               default=1., help='learning rate for the CNN = factor * learning_rate')
    return parser

def add_generic(parser):
    parser.add('-c', '--config', is_config_file=True, help='Config file path')
    parser.add('--modelname', type=str,
               default='model1', help='directory to store checkpointed models')

    parser.add('--caption_model', type=str,
               default="show_tell", help='show_tell, show_attend_tell, attention, test_att, show_attend_tell_new')
    parser.add('--seed', type=int, default=1, help="seed for all randomizer")
    parser.add('--verbose', type=int, default=0,
               help='code verbosity')

    # Running settings
    # Gpu id if required (LIG servers)
    parser.add('--gpu_id', type=int, default=0)
    parser.add('--input_data', type=str,
               default='data/coco/cocotalk',
               help='data filename, extension h5 & json will be needed')
    parser.add('--train_only', type=int,
               default=1, help='if true then use 80k, else use 110k')
    parser.add('--upsampling_size', type=int,
               default=300, help='upsampling size for MIL')
    parser.add('--batch_size', type=int,
               default=10, help='minibatch size')
    parser.add('--seq_per_img', type=int,
               default=5, help='number of captions to sample for each image during training')
    parser.add('--fliplr', type=int, default=0,
               help="Whether or not to add flipped image when generating the batch")
    # CNN model parameters:
    parser.add('--cnn_model', type=str,
               default='resnet50', help='CNN branch')
    parser.add('--cnn_fc_feat', type=str,
               default='fc7', help='CNN branch')
    parser.add('--cnn_att_feat', type=str,
               default='pool5', help='CNN branch')
    parser.add('--cnn_weight', type=str,
               default='', help='path to CNN tf model. Note this MUST be a resnet right now.')
    parser.add('--pretrained_cnn', type=bool,
               default=True, help='Wheter or not to load cnn weights pretrained on imagenet')

    # Decoder parameters:
    parser.add('--fc_feat_size', type=int,
               default=2048, help='2048 for resnet, 4096 for vgg')
    parser.add('--norm_feat', type=int,
               default=1, help='whether or not to normalize (N2) the last cnn feature prior to the lienar layers')
    parser.add('--rnn_size', type=int,
               default=512, help='size of the rnn in number of hidden nodes in each layer')
    parser.add('--rnn_bias', type=int,
               default=0, help='if 1 add rnn bias')

    parser.add('--num_layers', type=int,
               default=1, help='number of layers in the RNN')
    parser.add('--rnn_type', type=str,
               default='lstm', help='rnn, gru, or lstm')
    parser.add('--input_encoding_size', type=int,
               default=512, help='the encoding size of each token in the vocabulary, and the image.')
    parser.add('--init_decoder_W', type=str,
               default="", help='Path to intialize the decoder words embeddings, default random')
    parser.add('--freeze_decoder_W', type=int,
               default=0, help='Freeze the deocder W')

    parser.add('--drop_x_lm', type=float,
               default=0.5, help='strength of dropout in the Language Model RNN input')
    parser.add('--drop_prob_lm', type=float,
               default=0.5, help='strength of dropout in the Language Model RNN')
    parser.add('--drop_feat_im', type=float,
               default=0., help='strength of dropout in the Language Model RNN')
    parser.add('--drop_sentinel', type=float,
               default=0.5, help='strength of dropout in the Language Model RNN')


    # Special for atention
    parser.add('--attend_mode', type=str,
               default='concat', help='Concat, product: how to combine the hidden state with the region embedding to predict the weighing scales')
    parser.add('--region_size', type=int,
               default=14, help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add('--att_feat_size', type=int,
               default=2048, help='2048 for resnet, 512 for vgg')
    parser.add_argument('--att_hid_size', type=int, default=512,
                        help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add('--use_adaptive_pooling', type=int,
               default=1, help='get a pooled region of size region_size')
    parser.add('--use_maxout', type=int,
               default=0, help='use maxout')
    parser.add('--add_fc_img', type=int,
               default=1, help='add image feature to x_t every step')

    return parser


def create_logger(log_file=None, debug=True):
    """
    Initialize global logger and return it.
    :param log_file: log to this file, besides console output
    :return: created logger
    """
    logging.basicConfig(level=5,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    logging.root.handlers = []
    if debug:
        chosen_level = 5
    else:
        chosen_level = logging.INFO
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S')
    if log_file is not None:
        log_dir = osp.dirname(log_file)
        if log_dir:
            if not osp.exists(log_dir):
                os.makedirs(log_dir)
        # cerate file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(chosen_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # Colored stream handler
    sh = ColorStreamHandler()
    sh.setLevel(chosen_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger



def parse_opt():
    parser = configargparse.ArgParser()
    # When restarting or finetuning
    parser.add('--start_from_best', type=int, default=0,
               help="Whether to start from the best saved model (1) or the from the last checkpoint (0)")
    # Model to finetune (if restart, the same variable is used to refer to the model itself)
    parser.add('--start_from', type=str,
               default=None, help="The directory of the initialization model, must contain model.pth (resp model-best.pth) \
               the optimizer and the pickled infos")
    parser.add('--reset_optimizer', type=int, default=0,
               help="whether or not to start with a clean optimizer")
    parser.add('--shift_epoch', type=int, default=0,
               help="Start from epoch 0")
    parser = add_generic(parser)
    parser = add_loss_params(parser)
    parser = add_optim_params(parser)
    parser = add_eval_params(parser)
    parser = add_scheduled_sampling(parser)

    # UNTESTED #TODO
    # Use loss combining among the captions of the same image
    # parser.add('--combine_caps_losses', type=int,
               # default=0, help='combine the loss of the captions relative to a single image.')

    args = parser.parse_args()
    # mkdir the model save directory
    args.eventname = 'events/' + args.modelname
    args.modelname = 'save/' + args.modelname
    # Make sure the dirs exist:
    if not os.path.exists(args.eventname):
        os.makedirs(args.eventname)
    if not os.path.exists(args.modelname):
        os.makedirs(args.modelname)
    # Create the logger
    args.logger = create_logger('%s/train.log' % args.modelname)
    return args


def parse_eval_opt():
    """
    Evaluation parameters
    """
    # Input arguments and options
    parser = configargparse.ArgParser()
    parser.add('--start_from_best', type=int, default=1,
               help="Whether to start from the best saved model (1) or the from the last checkpoint (0)")
    parser.add('--start_from', type=str,
               default=None, help="The directory of the initialization model, must contain model.pth (resp model-best.pth) \
               the optimizer and the pickled infos")
    # Basic options
    parser = add_generic(parser)
    parser = add_eval_params(parser)
    parser = add_scheduled_sampling(parser)
    parser = add_optim_params(parser)
    parser = add_loss_params(parser)
    parser.add('--num_images', type=int, default=-1,
               help='how many images to use when periodically evaluating the loss? (-1 = all)')
    parser.add('--dump_images', type=int, default=0,
               help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
    parser.add('--dump_json', type=int, default=1,
               help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add('--output', type=str,
               help='results file name')
    parser.add('--dump_path', type=int, default=0,
               help='Write image paths along with predictions into vis json? (1=yes,0=no)')
    # For evaluation on a folder of images:
    parser.add('--image_folder', type=str, default='',
               help='If this is nonempty then will predict on the images in this folder path')
    parser.add('--image_list', type=str,
               help='List of image from folder')
    parser.add('--max_images', type=int, default=-1,
               help='If not -1 limit the number of evaluated images')
    parser.add('--image_root', type=str, default='data/coco/images',
               help='In case the image paths have to be preprended with a root path to an image folder')
    # For evaluation on MSCOCO images from some split:
    parser.add('--input_h5', type=str, default='',
               help='path to the h5file containing the preprocessed dataset')
    parser.add('--input_json', type=str, default='',
               help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
    parser.add('--split', type=str, default='val',
               help='if running on MSCOCO images, which split to use: val|test|train')
    # misc
    parser.add('--fliplr_eval', type=int, default=0,
               help='add flipped image')
    args = parser.parse_args()
    args.modelname = 'save/' + args.modelname
    args.logger = create_logger('%s/eval.log' % args.modelname)
    return args


def parse_ens_opt():
    parser = configargparse.ArgParser()
    parser.add('-c', '--config', is_config_file=True, help='Config file path')
    # Data input settings
    parser.add('--model', nargs="+", action="append",
                        help='Model(s) to evaluate')

