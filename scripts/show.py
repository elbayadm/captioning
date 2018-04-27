import os.path as osp
import socket
import glob
import pickle
import argparse
from collections import OrderedDict
from math import exp
from prettytable import PrettyTable
from html import escape
import numpy as np
import scipy.stats as st
from parse_options import *


PERF = ['Bleu_1', 'Bleu_4', 'ROUGE_L', 'SPICE', 'METEOR', 'CIDEr']
FIELDS = ["Model", "CNN", "params", 'loss', 'weights', 'Beam',
          'CIDEr', 'Bleu4', 'Perplexity', 'best/last']

PAPER_FIELDS = ["Model", 'Init',
                'Loss', 'Reward', 'Sampling', 'Beam'] + \
               [perf + '_ph1' for perf in PERF] + \
               ['Perplexity_ph1'] + \
               [perf + '_ph2' for perf in PERF] + \
               ['CI CIDEr'] + \
               ['Perplexity_ph2']

PAPER_FIELDS_FULL = ['Model', 'Init', 'Loss', 'Reward', 'Sampling', "Beam",
                     'Bleu_4_ph1', 'CIDEr_ph1',
                     'Bleu_1_ph2', 'Bleu_4_ph2',
                     'ROUGE_L_ph2', 'SPICE_ph2',
                     'METEOR_ph2', 'CIDEr_ph2',
                     # 'CI CIDEr',
                     ]

PAPER_FIELDS_SHORT = ['Loss', 'Reward', 'Sampling', "Beam",
                      # 'Bleu_1_ph2',
                      'Bleu_4_ph2',
                      'METEOR_ph2',
                      'CIDEr_ph2',
                      # 'CI CIDEr'
                      ]



def is_required(model, fltr, exclude):
    for fl in fltr:
        if fl not in model:
            return 0
    for exc in exclude:
        if exc in model:
            return 0
    return 1


def correct(word):
    """
    Printable names for key options
    """
    if word == "show_tell":
        return 'Show \\& Tell'
    elif word == 'top_down':
        return "Top-down"
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
    lines.append('\midrule')
    lines.append(' & '.join(fields) + '\\\\')
    lines.append('\midrule')
    for row in formatted_rows:
        line = []
        for field, datum in zip(ptab._field_names, row):
            if field in fields:
                line.append(correct(datum))
        lines.append(' & '.join(line) + '\\\\')
    lines.append('\midrule')
    return lines


def get_perf(res, get_cid=False):
    formatted_res = OrderedDict()
    for k in PERF:
        if k in res:
            formatted_res[k] = float(res[k] * 100)
        else:
            formatted_res[k] = 0
    out = list(formatted_res.values())
    if get_cid:
        out.append(res['CI_cider'])
    out.append(float(exp(res['ml_loss'])))
    return out


def get_results(model, split='val', verbose=False, get_cid=False):
    model_dir = model
    if split == "val":
        # Read training results:
        if osp.exists('%s/infos.pkl' % model_dir):
            print('Reading the model infos (%s)' % model)
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
                if verbose:
                    print('%s: ml loss set to 0' % model)
                out["ml_loss"] = 0
            # Defaults
            params = {'beam_size': 1, 'sample_max': 1, 'temperature': 0.5, 'flip': 0}
            params.update(vars(infos['opt']))
            compiled = [[params, out]]
        else:
            if verbose:
                print('infos not found in %s' % model_dir)
            compiled = []

    elif split == "test":
        # Read post-results
        results = sorted(glob.glob('%s/evaluations/test/*.res' % model_dir))
        compiled = []
        for res in results:
            out = pickle.load(open(res, 'rb'))
            out['best/last'] = "--"
            if get_cid:
                val = pickle.load(open('%s/infos.pkl' % model, 'rb'))['val_result_history']
                cid = [100 * v['lang_stats']['CIDEr'] for v in val.values()][-8:]
                ci = st.t.interval(0.95, len(cid)-1, loc=np.mean(cid), scale=st.sem(cid))
                out['CI_cider'] = ci[1] - ci[0]
            compiled.append([out['params'], out])
    else:
        raise ValueError('Unknown split %s' % split)

    return compiled


def crawl_results_paper(fltr=[], exclude=[], split="test", verbose=False, reset=False, beam=-1):
    """
    if beam == -1 report all evaluated beam widths
    """
    models = glob.glob('save/*')
    models = [model for model in models if is_required(model, fltr, exclude)]
    tab = PrettyTable()
    tab.field_names = PAPER_FIELDS
    fn_prefix = 'fncnn6_'
    if reset:
        fn_prefix += 'reset_'
    fn_models = [model for model in models if fn_prefix in model]
    models = list(set(models) - set(fn_models))
    for model in models:
        if "fncnn" in model:
            continue
        outputs = get_results(model, split, verbose)
        if len(outputs):
            if verbose:
                print(model.split('/')[-1])
            fn_model = "save/" + fn_prefix + model.split('/')[-1]
            if fn_model in fn_models:
                if verbose:
                    print('finetuned model exists')
                fn_outputs = get_results(fn_model, split, verbose, get_cid=True)
            else:
                fn_outputs = []

            outputs_dict = {}
            for params, res in outputs:
                outputs_dict[params['beam_size']] = [params, res]
            if fn_outputs:
                for params, res in fn_outputs:
                    if params['beam_size'] in outputs_dict:
                        outputs_dict[params['beam_size']].append(res)
                    else:
                        outputs_dict[params['beam_size']] = [params, None, res]
            for beam_size, results in outputs_dict.items():
                if len(results) == 3:
                    params, res, fn_res = results
                else:
                    params, res = results
                    fn_res = None
                row = nameit(params)
                if res:
                    perf = get_perf(res)
                else:
                    perf = [0] * (len(PERF) + 1)
                row += perf
                if fn_res:
                    row += get_perf(fn_res, get_cid=True)
                else:
                    row += (len(PERF) + 2) * [0]

                if beam == -1:
                    tab.add_row(row)
                else:
                    if beam == beam_size:
                        tab.add_row(row)
    return tab



def crawl_results(fltr='', exclude=None, split="val", save_pkl=False, verbose=False):
    models = glob.glob('save/*')
    models = [model for model in models if is_required(model, fltr, exclude)]
    print('Found models:', models)
    recap = {}
    tab = PrettyTable()
    tab.field_names = FIELDS
    dump = []
    for model in models:
        outputs = get_results(model, split, verbose)
        if len(outputs):
            if verbose:
                print(model.split('/')[-1])
            modelparams, loss_version = parse_name_clean(outputs[0][0])
            if save_pkl:
                dump.append(outputs)
            for (p, res) in outputs:
                finetuning = p.get('finetune_cnn_after', -1)
                if finetuning > -1:
                    finetuning = "RNN + %.1f x cnn(%d:)" % (p.get('cnn_learning_rate'),
                                                            p.get('finetune_cnn_slice', -1))
                    if p.get('reset_optimizer', 0):
                        finetuning += ', RESET'
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
    parser.add_argument('--filter', '-f', nargs="+", help='keyword to include')
    parser.add_argument('--exclude', '-e', nargs="+", help='keyword to exculde')
    # parser.add_argument('--tex', '-t', action='store_true', help="save results into latex table")
    parser.add_argument('--paper', '-p', action='store_true', help="run paper mode")
    parser.add_argument('--reset', '-r', action='store_true', help="run paper mode and get finetuned with optimizer reset")
    parser.add_argument('--html', action='store_true', help="save results into html")
    parser.add_argument('--pkl', action='store_true', help="save results into pkl")
    parser.add_argument('--split', type=str, default="val", help="split on which to report")
    parser.add_argument('--sort', type=str, default="CIDEr_ph2", help="criteria by which to order the terminal printed table")
    parser.add_argument('--verbose', '-v', action="store_true", help="script verbosity")
    parser.add_argument('--abridged', '-a', action="store_true", help="script verbosity")
    parser.add_argument('--beam', '-b', type=int, default=3, help="beam reported")


    args = parser.parse_args()
    print('Arguments:', vars(args))
    split = args.split
    verbose = args.verbose
    fltr = args.filter
    if not fltr:
        fltr = []
    exc = args.exclude
    if not exc:
        exc = []
    if args.verbose:
        print(vars(args))
    fltr_concat = "_".join(fltr)
    if not fltr_concat:
        fltr_concat = ''
    else:
        fltr_concat = '_' + fltr_concat
    if args.reset:
        fltr_concat += '_reset'
    if args.paper:
        print('Setting up split=test')
        split = 'test'
        filename = "results/%s%s" % (split, fltr_concat)  # socket.gethostname())
        SELECT = PAPER_FIELDS_FULL
        if args.abridged:
            SELECT = PAPER_FIELDS_SHORT
            filename += '_abr'
        if not args.beam == -1:
            filename += '_bw%d' % args.beam
            PAPER_FIELDS_FULL.remove('Beam')
            PAPER_FIELDS_SHORT.remove('Beam')
        tab = crawl_results_paper(fltr, exc, split, verbose, args.reset, args.beam)
        print(tab.get_string(sortby=args.sort, reversesort=True, fields=SELECT))
        print('saving latex table in %s.tex' % filename)
        with open(filename+'.tex', 'w') as f:
            tex = get_latex(tab, sortby=args.sort,
                            reversesort=False, fields=SELECT)
            f.write("\n".join(tex))
    else:
        tab, dump = crawl_results(fltr, exc, split,
                                  args.pkl, verbose)
        print(tab.get_string(sortby='CIDEr', reversesort=True))
        filename = "results/%s%s_%s" % (split, fltr_concat, socket.gethostname())
        if args.pkl:
            pickle.dump(dump, open(filename+".res", 'wb'))
        if args.html:
            with open(filename+'.html', 'w') as f:
                ss = tab.get_html_string(sortby="CIDEr", reversesort=True)
                f.write(ss)

