import sys
import os.path as osp
import glob
import operator
import pickle
from math import exp
from prettytable import PrettyTable

FIELDS = ['Beam', 'Temperature', 'CIDEr', 'Bleu4', 'Spice', 'Perplexity']


def get_results(model, split='val'):
    model_dir = model
    # Read training results:
    if osp.exists('%s/infos-best.pkl' % model_dir):
        infos = pickle.load(open('%s/infos-best.pkl' % model_dir, 'rb'))
        tr_res = infos['val_result_history']
        tr_res = tr_res[max(list(tr_res))]
        out = tr_res['lang_stats']
        out['ml_loss'] = tr_res['loss']
        # Defaults
        params = {'beams_size': 1, 'sample_max': 1, 'temperature': 0.5, 'flip': 0}
        params.update(vars(infos['opt']))
        compiled = [[params, out]]
    else:
        compiled = []

    # Read post-results
    val_results = sorted(glob.glob('%s/val_*.res' % model_dir))
    for res in val_results:
        params = res.split('/')[-1].split('_')
        beam = int(params[1][2:])
        sample_max = 1 if params[3] == "max" else 0
        temp = 0
        if not sample_max:
            temp = float(params[4])
            index = 5
        else:
            index = 4
        flip = int(params[index][4])
        params = {'beam_size': beam, 'sample_max': sample_max,
                 'temperature': temp, 'flip': flip}
        # Load results:
        out = pickle.load(open(res, 'rb'))
        compiled.append([params, out])
    return compiled


def gather_results(model):
    outputs = get_results(model)
    tab = PrettyTable()
    tab.field_names = ['Sorter'] + FIELDS
    for (p, res) in outputs:
        tab.add_row([p['beam_size'] + p['temperature'],
                     p['beam_size'], p['temperature'],
                     round(res['CIDEr'] * 100, 2), round(res['Bleu_4'] * 100, 2),
                     round(res['SPICE'] * 100, 2), round(exp(res['ml_loss']), 2)])
    return tab


def parse_name(model):
    rewards = ['hamming', 'cider', 'bleu2', 'bleu3', 'bleu4']
    if '/' in model:
        model = model.split('/')[-1]
        # parse parameters from name:
        if model.startswith("word2"):
            modelname = ['WL v2']
        elif model.startswith('word'):
            modelname = ['WL']
            if 'tsent' in model:
                modelname.append('& SL')
        elif model.startswith("sample"):
            modelname = ['Reward sampling']
        elif model.split('_')[0] in rewards:
            modelname = ['SL']
        elif model.split('_')[0] in [r+'2' for r in rewards]:
            modelname = ['SL v2']
        else:
            modelname = []

        chunks = model.split('_')
        for c in chunks:
            if c in rewards:
                modelname.append(c)
            elif c in [r+'2' for r in rewards]:
                modelname.append(c[:-1])
            elif 'tword' in c:
                tau = c[5:]
                if len(tau) == 4:
                    tau = float(tau)/1000
                elif len(tau) == 2:
                    tau = float(tau)/10
                else:
                    print('Other case:', tau)
                modelname.append('$\\tau(w) = %.3f$' % tau)
            elif 'tsent' in c:
                modelname.append('$\\tau(s) = %.2f$' % (float(c[5:])/10))
            elif c.startswith('a'):
                modelname.append('$\\alpha = %.2f$' % (float(c[1:])/10))
            elif c in ['word', 'word2', 'sample']:
                continue
            else:
                # print('Unknown', c)
                modelname.append(c)

    if modelname:
        return ' '.join(modelname)
    else:
        return model

def highlight(score, tresh):
    if score >= tresh:
        return '<b> %.2f </b>' % score
    else:
        return '%.2f' % score

def crawl_results(filter=''):
    models = sorted(glob.glob('save/*%s*' % filter))
    # print("Found:", models)
    fields = ["Model", 'Beam', 'CIDEr', 'Bleu4', 'Spice', 'Perplexity']
    tab = PrettyTable()
    tab.field_names = fields
    for model in models:
        modelname = parse_name(model)
        outputs = get_results(model)
        for (p, res) in outputs:
            if p['beam_size'] > 1:
                cid = float(res['CIDEr'] * 100)
                bl = float(res['Bleu_4'] * 100)
                sp = float(res['SPICE'] * 100)
                perpl = float(exp(res['ml_loss']))
                tab.add_row([modelname,
                             p['beam_size'],
                             cid, bl, sp, perpl])
    return tab


if __name__ == "__main__":

    # tab = gather_results("save/baseline")
    # print(tab.get_string(fields=FIELDS, sortby="Sorter"))
    # print(tab.get_html_string(fields=FIELDS, sortby="Sorter"))

    if len(sys.argv) > 1:
        filter = sys.argv[1]
    else:
        filter= ''
    tab = crawl_results(filter)
    print(tab.get_string(sortby='CIDEr', reversesort=True))
    with open('res%s.html' % filter, 'w') as f:
        ss = tab.get_html_string(sortby="CIDEr", reversesort=True)
        # print(ss)
        f.write(ss)

