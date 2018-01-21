import sys
import os.path as osp
import socket
import glob
import operator
import pickle
import argparse
from math import exp
from prettytable import PrettyTable

FIELDS = ['Beam', 'Temperature', 'CIDEr', 'Bleu4', 'Perplexity']


def get_results(model, split='val'):
    model_dir = model
    # Read training results:
    if osp.exists('%s/infos.pkl' % model_dir):
        infos = pickle.load(open('%s/infos.pkl' % model_dir, 'rb'))
        tr_res = infos['val_result_history']
        iters = list(tr_res)
        cids = [tr_res[it]['lang_stats']['CIDEr'] for it in iters]
        best_iter = iters[cids.index(max(cids))]
        last_iter = max(iters)
        tr_res = tr_res[best_iter]
        out = tr_res['lang_stats']
        out['best/last'] = '%dk / %dk' % (best_iter/1000, last_iter/1000)
        # print('best.last:', out['best/last'])
        try:
            out['ml_loss'] = tr_res['ml_loss']
        except:
            print('%s: ml loss set to 0' % model)
            out["ml_loss"] = 0
        # Defaults
        params = {'beams_size': 1, 'sample_max': 1, 'temperature': 0.5, 'flip': 0}
        params.update(vars(infos['opt']))
        compiled = [[params, out]]
    else:
        print('infos not found in %s' % model_dir)
        compiled = []

    # Read post-results
    # val_results = sorted(glob.glob('%s/val_*.res' % model_dir))
    # for res in val_results:
        # eval_params = res.split('/')[-1].split('_')
        # beam = int(eval_params[1][2:])
        # sample_max = 1 if eval_params[3] == "max" else 0
        # temp = 0
        # if not sample_max:
            # temp = float(eval_params[4])
            # index = 5
        # else:
            # index = 4
        # flip = int(eval_params[index][4])
        # eval_params = {'beam_size': beam, 'sample_max': sample_max,
                       # 'temperature': temp, 'flip': flip}
        # if params:
            # params.update(eval_params)
        # else:
            # params = eval_params
        # # Load results:
        # out = pickle.load(open(res, 'rb'))
        # compiled.append([params, out])
    return compiled


def gather_results(model):
    outputs = get_results(model)
    tab = PrettyTable()
    tab.field_names = ['Sorter'] + FIELDS
    for (p, res) in outputs:
        print('params:', len(p))
        print('res:', res)
        tab.add_row([p['beam_size'] + p['temperature'],
                     p['beam_size'], p['temperature'],
                     round(res['CIDEr'] * 100, 2), round(res['Bleu_4'] * 100, 2),
                     round(exp(res['ml_loss']), 2)])
    return tab

def get_wl(params):
    if 'alpha_word' in params:
        alpha = params['alpha_word']
    else:
        alpha = params['alpha']
    if 'cooc' in params.get('similarity_matrix', ''):
        G = "Coocurences"
    elif 'train_coco' in params.get('similarity_matrix', ''):
        G = "Glove-Coco"
    else:
        G = "Glove-Wiki"
    if params.get('rare_tfidf', 0):
        G += ' xIDF'
    if params.get('word_add_entropy', 0):
        G += ' +H'
    if params.get('exact_dkl', 0):
        G += ' +ExDKL'
    modelparams = ' Word Level, Sim=%s, $\\tau=%.2f$, $\\alpha=%.1f$' % (G, params['tau_word'], alpha)
    return modelparams


def parse_name_clean(params):
    modelparams = ""
    # get the model parameters
    # modelparams = ' base lr: %.1e decay: %d, Adam(%.1f,%.3f), batch: %d, seq: %d' % (params['learning_rate'],
                                                                                     # params['learning_rate_decay_start'],
                                                                                     # params['optim_alpha'],
                                                                                     # params['optim_beta'],
                                                                                     # params['batch_size'],
                                                                                     # params["seq_length"])
    if params['caption_model'] == "adaptive_attention":
        if params.get('use_maxout', 0):
            modelparams += "Maxout"
        if not params.get('add_fc_img', 1):
            modelparams += "xt = wt"
    aug = 0
    if 'gen' in params.get('input_data'):
        modelparams += "Augment (%d)" % params['seq_per_img']
        aug = 1
    if not aug:
        if not params['seq_per_img'] == 5:
            modelparams += "Repeat (%d)" % params['seq_per_img']

    if params.get('init_decoder_W', ""):
        # print('Initializing the word emebddings')
        # print('Freezing:', params.get('freeze_decoder_W', 0))
        modelparams += "W(init): %s" % params['init_decoder_W'].split('/')[-1].split('.')[0]
        if params.get('freeze_decoder_W', 0):
            modelparams += ' frozen'

    # Get the loss:
    if "sample_cap" in params:
        loss_version = parse_loss_old(params)
    else:
        loss_version = parse_loss(params)
    # finetuning:
    if not modelparams:
        # print('Couldnt name ', params['modelname'])
        modelparams = 'Default'
    return modelparams, loss_version


def parse_loss(params):
    combine_loss = params.get('combine_loss', 0)
    loss_version = params['loss_version'].lower()
    if loss_version == "ml":
        return 'ML'
    elif loss_version == "word":
        return get_wl(params)
    elif loss_version == "seq":
        reward = params['reward']
        if reward == "tfidf":
            reward = 'tfidf n=%d, rare=%d' % (params['ngram_length'],
                                              params['rare_tfidf'])
        elif reward == 'hamming':
            reward = 'hamming Vpool=%d' % (params['limited_vocab_sub'])
        elif reward == 'cider':
            clip = params.get('clip_reward', 1)
            if not clip:
                clip = 1
            reward = 'CIDEr (clip=%.1f)' % clip
        reward += ' $\\tau=%.2f$' % params['tau_sent']

        if params['stratify_reward']:
            loss_version = 'Stratify r=(%s), MC=%d, \\alpha=%.1f$' % (reward,
                                                                      params['mc_samples'],
                                                                      params['alpha_sent'])

        else:
            sampler = params['importance_sampler']
            tau_q = params.get('tau_sent_q', params['tau_sent'])
            if sampler == "tfidf":
                sampler = 'tfidf n=%d, rare=%d $\\tau=%.2f$' % (params['ngram_length'],
                                                                params['rare_tfidf'],
                                                                tau_q)
            elif sampler == 'hamming':
                sampler = 'hamming Vpool=%d $\\tau=%.2f$' % (params['limited_vocab_sub'],
                                                             tau_q)
            elif sampler == 'greedy':
                sampler = '$p_\\theta$'

            extra = params.get('lazy_rnn', 0) * " (LAZY)"
            loss_version = 'Importance r=(%s), q=(%s), MC=%d, \\alpha=%.1f$ %s' % (reward,
                                                                                   sampler,
                                                                                   params['mc_samples'],
                                                                                   params['alpha_sent'],
                                                                                   extra)
        return loss_version


def parse_loss_old(params):
    sample_cap = params.get('sample_cap', 0)
    sample_reward = params.get('sample_reward', 0)
    alter_loss = params.get('alter_loss', 0)
    sum_loss = params.get('sum_loss', 0)
    combine_loss = params.get('combine_loss', 0)
    multi = alter_loss + sum_loss + combine_loss
    loss_version = params['loss_version']
    if "alpha" in params:
        alpha = (params['alpha'], params['alpha'])
    else:
        alpha = (params['alpha_sent'], params['alpha_word'])
    tau_sent = params['tau_sent']
    tau_word = params['tau_word']
    rare = params.get('rare_tfidf', 0)
    sub = params.get('sub_idf', 0)
    if 'tfidf' in loss_version:
        loss_version = "TFIDF n=%d, idf_select=%d, idf_sub=%d" % (params.get('ngram_length', 0), rare, sub)
    elif 'hamming' in loss_version:
        loss_version = "hamming Vpool=%d" % params.get('limited_vocab_sub', 1)
    if not multi:
        if loss_version == "word2":
            loss_version = get_wl(params)
        elif sample_cap:
            if loss_version == "dummy":
                loss_version = "constant"
            ver = params.get('sentence_loss_version', 1)
            loss_version = ' SampleP, r=%s V%d, $\\tau=%.2f, \\alpha=%.1f$' % (loss_version, ver, tau_sent, alpha[0])
        elif sample_reward:
            mc = params.get('mc_samples', 1)
            loss_version = ' Stratify r=(%s, $\\tau=%.2f), MC=%d, \\alpha=%.1f$' % (loss_version, tau_sent, mc,  alpha[0])
        else:
            # print('Model: %s - assuming baseline loss' % params['modelparams'])
            # modelparams = " ".join(params['modelparams'].split('_'))
            loss_version = " ML"
        if params.get('penalize_confidence', 0):
            loss_version += " Peanlize: %.2f" % params['penalize_confidence']

    else:
        wl = get_wl(params)
        if alter_loss:
            loss_version = " Alternating losses, %s  w/ Stratify, r=%s $\\tau=%.2f$, $\\alpha=%.1f$, $\\gamma=%.1f$ (mode:%s)" \
                           % (wl, loss_version, tau_sent, alpha[0], params.get('gamma', 0), params.get('alter_mode', 'iter'))
        elif sum_loss:
            loss_version = " Sum losses, %s w/ Stratify, r=%s $\\tau=%.2f$, $\\alpha=%.1f$, $\\gamma=%.1f$" \
                           % (wl, loss_version, tau_sent, alpha[0], params.get('gamma', 0))
        elif combine_loss:
            loss_version = " Combining losses, %s w/ Stratify, r=%s $\\tau=%.2f$, $\\alpha=%.1f$" \
                            % (wl, loss_version, tau_sent, alpha[0])
    return loss_version



def parse_name(model):
    rewards = ['hamming', 'cider', 'bleu2', 'bleu3', 'bleu4']
    if '/' in model:
        model = model.split('/')[-1]
        # parse parameters from name:
        if model.startswith("word2"):
            modelparams = ['WL v2']
            if 'tsent' in model:
                modelparams.append('& SL')
        elif model.startswith('word'):
            modelparams = ['WL']
            if 'tsent' in model:
                modelparams.append('& SL')
        elif model.startswith("sample"):
            modelparams = ['Reward sampling']
        elif model.split('_')[0] in rewards:
            modelparams = ['SL']
        elif model.split('_')[0] in [r+'2' for r in rewards]:
            modelparams = ['SL v2']
        else:
            modelparams = []

        chunks = model.split('_')
        for c in chunks:
            if c in rewards:
                modelparams.append(c)
            elif c in [r+'2' for r in rewards]:
                modelparams.append(c[:-1])
            elif 'tword' in c:
                tau = c[5:]
                if tau.startswith('0'):
                    div = tau.count('0') * (len(tau) - tau.count('0'))
                    tau = float(tau)/(10 ** div)
                else:
                    try:
                        tau = float(tau)
                    except:
                        print('Other case:', tau)
                        tau = 0
                modelparams.append('$\\tau(w) = %.2f$' % tau)
            elif 'tsent' in c:
                tau = c[5:]
                if tau.startswith('0'):
                    div = tau.count('0') * (len(tau) - tau.count('0'))
                    tau = float(tau)/(10 ** div)
                else:
                    try:
                        tau = float(tau)
                    except:
                        print('Other case:', tau)
                        tau = 0

                modelparams.append('$\\tau(s) = %.2f$' % tau)
            elif c.startswith('a'):
                modelparams.append('$\\alpha = %.2f$' % (float(c[1:])/10))
            elif c in ['word', 'word2', 'sample', 'sample2']:
                continue
            else:
                # print('Unknown', c)
                modelparams.append(c)

    if modelparams:
        return ' '.join(modelparams)
    else:
        return model

def highlight(score, tresh):
    if score >= tresh:
        return '<b> %.2f </b>' % score
    else:
        return '%.2f' % score

def crawl_results(filter='', exc=None):
    models = sorted(glob.glob('save/*%s*' % filter))
    if exc:
        # Exclude models containg exc:
        models = [model for model in models if not sum([e in model for e in exc])]
    print("Found:", models)
    fields = ["Model", "CNN", "params", 'loss', 'weights', 'Beam', 'CIDEr', 'Bleu4', 'Perplexity', 'best/last']
    recap = {}
    tab = PrettyTable()
    tab.field_names = fields
    dump = []
    for model in models:
        outputs = get_results(model)
        if len(outputs):
            modelparams, loss_version = parse_name_clean(outputs[0][0])
            dump.append(outputs)
            for (p, res) in outputs:
                finetuning = p.get('finetune_cnn_after', -1)
                if finetuning > -1:
                    finetuning = "RNN + %.1f x cnn(%d:)" % (p.get('cnn_learning_rate'),
                                                            p.get('finetune_cnn_slice', -1))
                else:
                    finetuning = "RNN"

                cid = float(res['CIDEr'] * 100)
                try:
                    recap[p['alpha']] = cid
                except:
                    recap[p['alpha_sent']] = cid
                    recap[p['alpha_word']] = cid
                bl = float(res['Bleu_4'] * 100)
                try:
                    perpl = float(exp(res['ml_loss']))
                except:
                    perpl = 1.
                tab.add_row([p['caption_model'],
                             p['cnn_model'],
                             modelparams,
                             loss_version,
                             finetuning,
                             p['beam_size'],
                             cid, bl, perpl, res['best/last']])
    return tab, dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # tab = gather_results("save/baseline")
    # print(tab.get_string(fields=FIELDS, sortby="Sorter"))
    # print(tab.get_html_string(fields=FIELDS, sortby="Sorter"))
    parser.add_argument('--filter', '-f', type=str, default='', help='kewyord to include')
    parser.add_argument('--exclude', '-e', nargs="+", help='keyword to exculdeh')
    parser.add_argument('--save', '-s', action='store_true', help="save results")
    args = parser.parse_args()

    save = args.save
    filter = args.filter
    exc = args.exclude
    print('filter:', filter, 'exclude:', exc, 'saving:', save)
    tab, dump = crawl_results(filter, exc)
    print(tab.get_string(sortby='CIDEr', reversesort=True))
    filename = "results/res%s_%s" % (filter, socket.gethostname())
    if save:
        with open(filename+'.html', 'w') as f:
            ss = tab.get_html_string(sortby="CIDEr", reversesort=True)
            # print(ss)
            f.write(ss)
        pickle.dump(dump, open(filename+".res", 'wb'))

