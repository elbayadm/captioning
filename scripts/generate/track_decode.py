import os
import sys
import subprocess
import time
import os.path as osp
import pickle
import numpy as np
from scipy.stats import entropy
sys.path.append('.')
from utils import opts
from loader import DataLoader, DataLoaderRaw
from models.eval_utils import track_rnn_decode
import models.setup as ms
import models.cnn as cnn


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


if __name__ == "__main__":
    opt = opts.parse_eval_opt()
    # setup gpu
    try:
        gpu_id = int(subprocess.check_output('gpu_getIDs.sh', shell=True))
    except:
        print("Failed to get gpu_id (setting gpu_id to %d)" % opt.gpu_id)
        gpu_id = str(opt.gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    opt.logger.warn('GPU ID: %s | available memory: %dM' \
                    % (os.environ['CUDA_VISIBLE_DEVICES'], get_gpu_memory(gpu_id)))

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
    print('Split:', opt.split)
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

    model = ms.select_model(opt)
    model.load()
    model.cuda()
    model.eval()
    model.define_loss(loader.get_vocab())
    # opt.n_gen = 10
    # opt.score_ground_truth = True
    eval_kwargs = vars(opt)
    eval_kwargs['beam_size'] = 1
    eval_kwargs['val_images_use'] = -1
    opt.save_stats = 50
    # eval_kwargs['val_images_use'] = -1
    ids, probs, sampled, attention = track_rnn_decode(cnn_model,
                                                      model,
                                                      loader,
                                                      opt.logger,
                                                      eval_kwargs)
    # Evaluate etropy & kl div:
    stats = {}
    enp = np.mean([entropy(probs[batch][n, c, :])
                   for batch in range(len(probs))
                   for n in range(len(probs[batch]))
                   for c in range(len(probs[batch][n]))])
    opt.logger.warn('Model average entropy: %.2f' % enp)
    stats['H(p)'] = enp
    output = 'results/%s_track_decode' % opt.modelname.split('/')[-1]
    if opt.save_stats:
        opt.logger.info('Saving results up to %d samples' % opt.save_stats)
        RES = {"ids": ids[:opt.save_stats],
               "probas": probs[:opt.save_stats],
               "sampled": sampled[:opt.save_stats],
               "attention": attention[:opt.save_stats]}
        pickle.dump(RES, open(output+'.tr', 'wb'))
    opt.logger.info('Dumping the entropies and kl divs')
    pickle.dump(stats, open(output+'.entropy', 'wb'))


