import sys
import os.path as osp
import socket
import glob
import operator
import pickle
import argparse
from math import exp
from prettytable import PrettyTable
from tabulate import tabulate
from html import escape

FIELDS = ["Model", "CNN", "params", 'loss', 'weights', 'Beam', 'CIDEr', 'Bleu4', 'Perplexity', 'best/last']

def correct(word):
    """
    Printable names for key options
    """
    if word == "show_tell":
        return 'Show \\& Tell'
    elif word == "resnet50":
        return "ResNet-50"
    elif word == "resnet152":
        return "ResNet-152"
    elif "cnn" in word:
        return "RNN + CNN"
    else:
        return word


def get_latex(ptab, **kwargs):
    """
    Print prettytable into latex table
    """
    options = ptab._get_options(kwargs)
    lines = []
    rows = ptab._get_rows(options)
    formatted_rows = ptab._format_rows(rows, options)
    aligns = []
    fields = []
    for field in ptab._field_names:
        if options["fields"] and field in options["fields"]:
            aligns.append(ptab._align[field])
            fields.append(field)
    lines = ['|' + '|'.join(['%s' % a for a in aligns]) + '|']
    lines.append('\hline')
    lines.append(' & '.join(fields) + '\\\\')
    lines.append('\hline')
    for row in formatted_rows:
        line = []
        for field, datum in zip(ptab._field_names, row):
            if field in fields:
                line.append(correct(datum))
        lines.append(' & '.join(line) + '\\\\')
    lines.append('\hline')
    return lines


def get_results(model, split='val'):
    model_dir = model
    if split == "val":
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

    elif split == "test":
        # Read post-results
        results = sorted(glob.glob('%s/evaluations/test/*.res' % model_dir))
        params = {}
        # if osp.exists('%s/infos.pkl' % model_dir):
            # infos = pickle.load(open('%s/infos.pkl' % model_dir, 'rb'))
            # params = vars(infos['opt'])
            # del infos
        # else:
            # params = {}
        compiled = []
        for res in results:
            out = pickle.load(open(res, 'rb'))
            params.update(out['params'])
            print('modelname:', model_dir)
            print('finetuning:', params.get('finetune_cnn_after', "Missing"))
            del out['params']
            out['best/last'] = "--"
            compiled.append([params, out])
    else:
        raise ValueError('Unknown split %s' % split)

    return compiled


def get_wl(params):
    """
    Word loss settings
    """
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
        G += ' xIDF(%.1f)' % params.get('rare_tfidf')
    if params.get('word_add_entropy', 0):
        G += ' +H'
    if params.get('exact_dkl', 0):
        G += ' +ExDKL'
    modelparams = ' Word, Sim=%s, $\\tau=%.2f$, $\\alpha=%.1f$' % (G, params['tau_word'], alpha)
    return modelparams


def parse_name_clean(params):
    modelparams = []
    if not params['batch_size'] == 10:
        modelparams.append("Batch=%d" % params['batch_size'])
    if not params['seq_length'] == 16:
        modelparams.append('SEQ=%d' % params["seq_length"])
    if not params['learning_rate_decay_start'] == 5:
        modelparams.append('DecayStart=%d' % params['learning_rate_decay_start'])
    # Special for adaptive attention
    if params['caption_model'] == "adaptive_attention":
        if params.get('use_maxout', 0):
            modelparams.append("Maxout")
        if not params.get('add_fc_img', 1):
            modelparams.append("xt = wt")
    aug = 0
    if 'gen' in params.get('input_data'):
        modelparams.append("Augment (%d)" % params['seq_per_img'])
        aug = 1
    if not aug:
        if not params['seq_per_img'] == 5:
            modelparams.append("Repeat (%d)" % params['seq_per_img'])

    if params.get('init_decoder_W', ""):
        modelparams.append("W_Decoder %s" % params['init_decoder_W'].split('/')[-1].split('.')[0])
        if params.get('freeze_decoder_W', 0):
            modelparams.append('frozen')

    # Get the loss:
    if "stratify_reward" in params:
        loss_version = parse_loss(params)
    else:
        loss_version = parse_loss_old(params)
    if len(modelparams):
        modelparams = ' '.join(modelparams)
    else:
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
            reward = 'TFIDF, n=%d, rare=%d' % (params['ngram_length'],
                                               params['rare_tfidf'])
        elif reward == 'hamming':
            reward = 'Hamming, Vpool=%d' % (params['limited_vocab_sub'])
        reward += ', $\\tau=%.2f$' % params['tau_sent']

        if params['stratify_reward']:
            loss_version = 'Stratify r=(%s), \\alpha=%.1f$' % (reward,
                                                               params['alpha_sent'])

        else:
            sampler = params['importance_sampler']
            tau_q = params.get('tau_sent_q', params['tau_sent'])
            if sampler == "tfidf":
                sampler = 'TFIDF, n=%d, rare=%d $\\tau=%.2f$' % (params['ngram_length'],
                                                                 params['rare_tfidf'],
                                                                 tau_q)
            elif sampler == 'hamming':
                sampler = 'Hamming, Vpool=%d $\\tau=%.2f$' % (params['limited_vocab_sub'],
                                                              tau_q)
            elif sampler == 'greedy':
                sampler = '$p_\\theta$'

            extra = params.get('lazy_rnn', 0) * " (LAZY)"
            loss_version = 'Importance r=(%s), q=(%s),\\alpha=%.1f$ %s' % (reward,
                                                                           sampler,
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
        loss_version = "TFIDF, n=%d, idf_select=%d, idf_sub=%d" % (params.get('ngram_length', 0), rare, sub)
    elif 'hamming' in loss_version:
        loss_version = "Hamming, Vpool=%d" % params.get('limited_vocab_sub', 1)
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
            loss_version = ' Stratify r=(%s, $\\tau=%.2f), \\alpha=%.1f$' % (loss_version, tau_sent,  alpha[0])
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


def crawl_results(filter='', exc=None, split="val", save_pkl=False):
    models = sorted(glob.glob('save/*%s*' % filter))
    if exc:
        # Exclude models containg exc:
        models = [model for model in models if not sum([e in model for e in exc])]
    # print("Found:", models)
    recap = {}
    tab = PrettyTable()
    tab.field_names = FIELDS
    dump = []
    for model in models:
        outputs = get_results(model, split)
        if len(outputs):
            modelparams, loss_version = parse_name_clean(outputs[0][0])
            if save_pkl:
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
                row = [p['caption_model'],
                       p['cnn_model'],
                       modelparams,
                       loss_version,
                       finetuning,
                       p['beam_size'],
                       cid, bl, perpl, res['best/last']]
                tab.add_row(row)
    return tab, dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', '-f', type=str, default='', help='kewyord to include')
    parser.add_argument('--exclude', '-e', nargs="+", help='keyword to exculdeh')
    parser.add_argument('--tex', '-t', action='store_true', help="save results into latex table")
    parser.add_argument('--html', action='store_true', help="save results into html")
    parser.add_argument('--pkl', action='store_true', help="save results into pkl")
    parser.add_argument('--split', type=str, default="val", help="split on which to report")

    args = parser.parse_args()
    split = args.split
    save_latex = args.tex
    save_html = args.html
    save_pkl = args.pkl

    filter = args.filter
    exc = args.exclude
    print('filter:', filter, 'exclude:', exc)
    tab, dump = crawl_results(filter, exc, split, save_pkl)
    print(tab.get_string(sortby='CIDEr', reversesort=True))
    filename = "results/%s_res%s_%s" % (split, filter, socket.gethostname())
    if save_html:
        with open(filename+'.html', 'w') as f:
            ss = tab.get_html_string(sortby="CIDEr", reversesort=True)
            f.write(ss)
    if save_latex:
        with open(filename+'.tex', 'w') as f:
            tex = get_latex(tab, sortby="CIDEr", reversesort=True, fields=FIELDS[:-2])
            f.write("\n".join(tex))
    if save_pkl:
        pickle.dump(dump, open(filename+".res", 'wb'))

