import os.path as osp
import socket
import glob
import pickle
import argparse
from collections import OrderedDict
from math import exp
from prettytable import PrettyTable
from html import escape
from parse import *

FIELDS = ["Model", "CNN", "params", 'loss', 'weights', 'Beam',
          'CIDEr', 'Bleu4', 'Perplexity', 'best/last']

PAPER_FIELDS = ["Model", "CNN", 'Loss', 'Beam',
                'Bleu1_ph1', 'Bleu4_ph1',
                'ROUGE-L_ph1', 'CIDEr-D_ph1', 'SPICE_ph1',
                'Perplexity_ph1',
                'Bleu1_ph2', 'Bleu4_ph2',
                'ROUGE-L_ph2', 'CIDEr-D_ph2', 'SPICE_ph2',
                'Perplexity_ph2']

PAPER_FIELDS_SELECT = ["Model", 'Loss',
                       'Bleu4_ph1',
                       'CIDEr-D_ph1',  # 'SPICE_ph1',
                       # 'Perplexity_ph1',
                       'Bleu1_ph2', 'Bleu4_ph2',
                       'ROUGE-L_ph2', 'CIDEr-D_ph2', 'SPICE_ph2',
                       # 'Perplexity_ph2'
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


def get_perf(res):
    formatted_res = OrderedDict()
    for k in ['Bleu_1', 'Bleu_4', 'ROUGE_L', 'CIDEr', 'SPICE']:
        if k in res:
            formatted_res[k] = float(res[k] * 100)
        else:
            formatted_res[k] = 0
    out = list(formatted_res.values())
    out.append(float(exp(res['ml_loss'])))
    return out


def get_results(model, split='val', verbose=False):
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
                if verbose:
                    print('%s: ml loss set to 0' % model)
                out["ml_loss"] = 0
            # Defaults
            params = {'beams_size': 1, 'sample_max': 1, 'temperature': 0.5, 'flip': 0}
            params.update(vars(infos['opt']))
            compiled = [[params, out]]
        else:
            if verbose:
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
            del out['params']
            out['best/last'] = "--"
            compiled.append([params, out])
    else:
        raise ValueError('Unknown split %s' % split)

    return compiled


def crawl_results_paper(fltr=[], exclude=[], split="test", verbose=False, reset=False):
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
            params, res = outputs[0]
            loss_version = parse_name_short(params)
            fn_model = "save/" + fn_prefix + model.split('/')[-1]
            if fn_model in fn_models:
                if verbose:
                    print('finetuned model exists')
                fn_outputs = get_results(fn_model, split, verbose)
            else:
                fn_outputs = []
            if len(fn_outputs):
                fn_res = fn_outputs[0][1]
            perf = get_perf(res)
            row = [params['caption_model'],
                   params['cnn_model'],
                   loss_version,
                   params["beam_size"]]
            row += perf
            if len(fn_outputs):
                row += get_perf(fn_res)
            else:
                row += 6 * [0]
            tab.add_row(row)
    return tab



def crawl_results(fltr='', exclude=None, split="val", save_pkl=False, verbose=False):
    models = glob.glob('save/*')
    models = [model for model in models if is_required(model, fltr, exclude)]
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
    parser.add_argument('--sort', type=str, default="CIDEr-D_ph1", help="criteria by which to order the terminal printed table")
    parser.add_argument('--verbose', '-v', action="store_true", help="script verbosity")
    args = parser.parse_args()

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
        filename = "results/%s%s_%s" % (split, fltr_concat, socket.gethostname())
        tab = crawl_results_paper(fltr, exc, split, verbose, args.reset)
        print(tab.get_string(sortby=args.sort, reversesort=True, fields=PAPER_FIELDS_SELECT))
        print('saving latex table in %s.tex' % filename)
        with open(filename+'.tex', 'w') as f:
            tex = get_latex(tab, sortby="Loss",
                            reversesort=False, fields=PAPER_FIELDS_SELECT)
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

