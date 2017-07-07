import configargparse
import os
import os.path as osp
import logging
from colorstreamhandler import ColorStreamHandler


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
            print 'Parsed log dir', log_dir
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
    parser.add('-c', '--config', is_config_file=True, help='Config file path')
    # Data input settings
    parser.add('--input_json', type=str,
               default='data/coco/cocotalk.json',
               help='path to the json file containing additional info and vocab')
    parser.add('--input_h5', type=str,
               default='data/coco/cocotalk.h5',
               help='path to the h5file containing the preprocessed dataset')
    parser.add('--use_feature_maps', type=int,
               default=1, help='use feature maps as context vectors otherwise (0) run with SSD')
    parser.add('--smoothing_version', type=int,
               default=1, help="between v=1, zeroing similarities below a margin or v=2 applying an rbf kernel")

    parser.add('--cnn_model', type=str,
               default='resnet50', help='CNN branch')
    parser.add('--cnn_fc_feat', type=str,
               default='fc7', help='CNN branch')
    parser.add('--cnn_att_feat', type=str,
               default='pool5', help='CNN branch')
    parser.add('--attend_mode', type=str,
               default='concat', help='Concat, product: how to combine the hidden state with the region embedding to predict the weighing scales')
    parser.add('--cnn_weight', type=str,
               default='', help='path to CNN tf model. Note this MUST be a resnet right now.')
    parser.add('--pretrained_cnn', type=bool,
               default=True, help='Wheter or not to load cnn weights pretrained on imagenet')
    parser.add('--restart', type=int,
               default=1, help='0: to override the existing model or 1: to pick up the training')
    parser.add('--start_from_best', type=int, default=0,
               help="Whether to start from the best saved model (1) or the from the last checkpoint (0)")

    parser.add('--start_from', type=str,
               default=None,
               help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)

    # Model settings
    parser.add('--caption_model', type=str,
               default="show_tell", help='show_tell, show_attend_tell, attention, test_att, show_attend_tell_new')
    parser.add('--lm_model', type=str,
               default="rnn", help='rnn, rnn_vae')

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
    parser.add('--use_glove', type=int,
               default=0, help='whether or not to use glove embeddings.')
    parser.add('--less_confident', type=float,
               default=0, help='if neq 0, be less confident in the added captions (used to scale down the loss)')

    # RAML Loss params:
    parser.add('--raml_loss', type=int,
               default=0, help='use smooth loss via similar words.')
    parser.add('--raml_version', type=str,
               default="exp", help='Version of RAML loss between (clip) and (exp)')
    parser.add('--raml_tau', type=float,
               default=0.8, help='Temperature for the rbf kernel')
    parser.add('--raml_margin', type=float,
               default=0.9, help='clipping margin for the similarities')
    parser.add('--raml_isolate', type=int,
                default=0, help='Whether to treat the gt separately or not')
    parser.add('--raml_alpha', type=float,
               default=0.9, help='Weight accorded to the gt is isolated')
    parser.add('--raml_alpha_increase_every', type=int,
               default=2, help='step width')
    parser.add('--raml_alpha_increase_factor', type=float,
               default=0.1, help='increase factor when step')
    parser.add('--raml_alpha_speed', type=float,
               default=20000, help='alpha decreasing speed')
    parser.add('--raml_alpha_strategy', type=str,
               default="constant", help='Increase strategy')
    parser.add('--raml_normalize', type=float,
               default=0, help='Apply tempered softmax (exp version)')
    #--------------------//RAML
    parser.add('--combine_caps_losses', type=int,
               default=0, help='combine the loss of the captions relative to a single image.')
    parser.add('--num_regions', type=int,
               default=512, help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add('--fc_feat_size', type=int,
               default=2048, help='2048 for resnet, 4096 for vgg')
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
    parser.add('--att_feat_size', type=int,
               default=2048, help='2048 for resnet, 512 for vgg')
    parser.add('--norm_feat', type=int,
               default=1, help='whether or not to normalize (N2) the last cnn feature prior to the lienar layers')

    # Optimization: General
    parser.add('--max_epochs', type=int,
               default=-1, help='number of epochs')
    parser.add('--batch_size', type=int,
               default=25, help='minibatch size')
    parser.add('--grad_clip', type=float,
               default=0.1, help='clip gradients at this value')
    parser.add('--drop_x_lm', type=float,
               default=0.5, help='strength of dropout in the Language Model RNN input')
    parser.add('--drop_prob_lm', type=float,
               default=0.5, help='strength of dropout in the Language Model RNN')
    parser.add('--finetune_cnn_after', type=int,
               default=-1, help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add('--finetune_cnn_only', type=int,
               default=0, help="if 1, finetune the cnn only.")
    parser.add('--seq_per_img', type=int,
               default=5, help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add('--beam_size', type=int,
               default=1, help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    #Optimization: for the Language Model
    parser.add('--optim', type=str,
               default='adam', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add('--learning_rate', type=float,
               default=4e-4, help='learning rate')
    parser.add('--learning_rate_decay_start', type=int,
               default=-1, help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add('--lr_patience', type=int,
               default=2, help='Epochs after overfitting before decreasing the lr')
    parser.add('--lr_strategy', type=str,
               default="step", help="between step (automatic decrease) or adaptive to the val loss")
    parser.add('--learning_rate_decay_every', type=int,
               default=3, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add('--learning_rate_decay_rate', type=float,
               default=0.8, help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add('--optim_alpha', type=float,
               default=0.8, help='alpha for adam')
    parser.add('--optim_beta', type=float,
               default=0.999, help='beta used for adam')
    parser.add('--optim_epsilon', type=float,
               default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add('--weight_decay', type=float,
               default=0, help='main optimizer weight decay')
    #Optimization: for the CNN
    parser.add('--cnn_optim', type=str,
               default='adam', help='optimization to use for CNN')
    parser.add('--cnn_optim_alpha', type=float,
               default=0.8, help='alpha for momentum of CNN')
    parser.add('--cnn_optim_beta', type=float,
               default=0.999, help='beta for momentum of CNN')
    parser.add('--cnn_learning_rate', type=float,
               default=0.1, help='learning rate for the CNN = factor * learning_rate')
    parser.add('--cnn_weight_decay', type=float,
               default=0, help='L2 weight decay just for the CNN')
    parser.add('--scheduled_sampling_strategy', type=str,
               default="step", help='the decay schedule')
    parser.add('--match_pairs', type=int,
               default=0, help='match senetences pairs')

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


    # Evaluation/Checkpointing
    parser.add('--val_images_use', type=int,
               default=-1, help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add('--save_checkpoint_every', type=int,
               default=4000, help='how often to save a model checkpoint (in iterations)?')
    parser.add('--modelname', type=str,
               default='model1', help='directory to store checkpointed models')
    parser.add('--language_eval', type=int,
               default=1, help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add('--forbid_unk', type=int,
               default=1, help='Do not generate UNK tokens when evaluating')

    parser.add('--losses_log_every', type=int,
               default=25, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add('--load_best_score', type=int,
               default=1, help='Do we load previous best score when resuming training.')

    # misc
    parser.add('--id', type=str,
               default='', help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add('--train_only', type=int,
               default=1, help='if true then use 80k, else use 110k')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "load_best_score should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "train_only should be 0 or 1"
    assert args.norm_feat == 0 or args.norm_feat == 1, "norm_feat should be 0 or 1"
    # mkdir the model save directory
    args.modelname = 'save/' + args.modelname
    args.logger = create_logger('%s/train.log' % args.modelname)
    return args
