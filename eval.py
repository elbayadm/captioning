import json
import time
import os.path as osp
from six.moves import cPickle as pickle
import opts
from dataloader import DataLoader
from dataloaderraw import DataLoaderRaw
import eval_utils
import misc.decoder_utils as du
import misc.cnn as cnn


if __name__ == "__main__":
    opt = opts.parse_eval_opt()
    if opt.start_from_best:
        flag = '-best'
        opt.logger.warn('Starting from the best saved model')
    else:
        flag = ''
    opt.cnn_start_from = osp.join(opt.modelname, 'model-cnn%s.pth' % flag)
    opt.infos_start_from = osp.join(opt.modelname, 'infos%s.pkl' % flag)
    opt.start_from = osp.join(opt.modelname, 'model%s.pth' % flag)
    opt.logger.warn('Starting from %s' % opt.start_from)

    # Load infos
    with open(opt.infos_start_from, 'rb') as f:
        print('Opening %s' % opt.infos_start_from)
        infos = pickle.load(f, encoding="iso-8859-1")
        infos['opt'].logger = None
    ignore = ["batch_size", "beam_size", "start_from",
              'cnn_start_from', 'infos_start_from',
              "start_from_best", "language_eval", "logger",
              "val_images_use", 'input_data']
    for k in list(vars(infos['opt']).keys()):
        if k not in ignore:
            if k in vars(opt):
                assert vars(opt)[k] == vars(infos['opt'])[k], (k + ' option not consistent ' +
                                                               str(vars(opt)[k]) + ' vs. ' + str(vars(infos['opt'])[k]))
            else:
                vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

    opt.fliplr = opt.fliplr_eval
    opt.language_creativity = 0
    opt.seq_per_img = 5
    opt.bootstrap = 0
    opt.sample_cap = 0
    vocab = infos['vocab'] # ix -> word mapping
    # Build CNN model for single branch use
    if opt.cnn_model.startswith('resnet'):
        cnn_model = cnn.ResNetModel(opt)
    elif opt.cnn_model.startswith('vgg'):
        cnn_model = cnn.VggNetModel(opt)
    else:
        print('Unknown model %s' % opt.cnn_model)
        sys.exit(1)

    cnn_model.cuda()
    cnn_model.eval()
    model = du.select_model(opt)
    model.load()
    model.cuda()
    model.eval()
    # Create the Data Loader instance
    start = time.time()
    if len(opt.image_folder) == 0:
        loader = DataLoader(opt)
    else:
        loader = DataLoaderRaw({'folder_path': opt.image_folder,
                                'files_list': opt.image_list,
                                'coco_json': opt.coco_json,
                                'batch_size': opt.batch_size,
                                'max_images': opt.max_images})
    loader.ix_to_word = infos['vocab']

    # Set sample options
    print('Seq per img:', loader.seq_per_img)
    print('Flipping the images: ', opt.fliplr)
    print('Beam width: ', opt.beam_size)
    if opt.beam_size > 1:
        opt.sample_max = 1
    model.define_loss(loader.get_vocab())
    # opt.n_gen = 10
    # opt.score_ground_truth = True
    ml_loss, loss, preds, perf = eval_utils.eval_split(cnn_model,
                                                       model,
                                                       model.crit,
                                                       loader,
                                                       opt.logger,
                                                       vars(opt))
    print("Finished evaluation in ", (time.time() - start))
    print('ML loss:', ml_loss)
    print('Training loss:', loss)
    perf['loss'] = loss
    perf['ml_loss'] = ml_loss
    if not opt.output:
        sampling = "sample_max" if opt.sample_max else "sample_temp_%.3f" % opt.temperature
        opt.output = '%s/%s_bw%d_%s_flip%d' % (opt.modelname,
                                               opt.split,
                                               opt.beam_size,
                                               sampling,
                                               opt.fliplr)

    pickle.dump(perf, open(opt.output + ".res", 'wb'))
    if opt.dump_json:
        json.dump(preds, open(opt.output + '.json', 'w'))
