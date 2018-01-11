import sys
import json
import time
import os.path as osp
from six.moves import cPickle as pickle
import numpy as np
from scipy.stats import entropy
sys.path.append('.')
import opts
from dataloader import DataLoader
from dataloaderraw import DataLoaderRaw
from eval_utils import track_rnn
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
    del infos
    opt.vocab_size = loader.vocab_size + 1
    opt.seq_length = loader.seq_length

    model = du.select_model(opt)
    model.load()
    model.cuda()
    model.eval()
    model.define_loss(loader.get_vocab())
    # opt.n_gen = 10
    # opt.score_ground_truth = True
    eval_kwargs = {'split': 'val',
                   'dataset': opt.input_data + '.json'}
    eval_kwargs.update(vars(opt))
    eval_kwargs['beam_size'] = 1
    # eval_kwargs['val_images_use'] = -1
    eval_kwargs['add_dirac'] = opt.add_dirac
    rewards, probs = track_rnn(cnn_model,
                               model,
                               loader,
                               opt.logger,
                               eval_kwargs)
    # Evaluate etropy & kl div:
    kl = np.mean([entropy(rewards[batch][n, c, :], probs[batch][n, c, :]) for batch in range(len(probs)) for n in range(len(probs[batch])) for c in range(len(probs[batch][n]))])
    enp = np.mean([entropy(probs[batch][n, c, :]) for batch in range(len(probs)) for n in range(len(probs[batch])) for c in range(len(probs[batch][n]))])
    enr = np.mean([entropy(rewards[batch][n, c, :]) for batch in range(len(probs)) for n in range(len(probs[batch])) for c in range(len(probs[batch][n]))])
    print('Average KL: %.2e & average entropy(p) :%.2e & Average entropy(r): %.2e' % (kl, enp, enr))
    if opt.save_stats:
        output = 'Results/%s_track' % opt.modelname.split('/')[-1]
        if opt.add_dirac:
            output += '_dirac'
        pickle.dump({"probas": probs[:opt.save_stats],
                     "rewards": rewards[:opt.save_stats]},
                    open(output+'.tr', 'wb'))

