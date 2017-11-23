import sys
import os.path as osp
import socket
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
        compiled = []

    # Read post-results
    val_results = sorted(glob.glob('%s/val_*.res' % model_dir))
    for res in val_results:
        eval_params = res.split('/')[-1].split('_')
        beam = int(eval_params[1][2:])
        sample_max = 1 if eval_params[3] == "max" else 0
        temp = 0
        if not sample_max:
            temp = float(eval_params[4])
            index = 5
        else:
            index = 4
        flip = int(eval_params[index][4])
        eval_params = {'beam_size': beam, 'sample_max': sample_max,
                       'temperature': temp, 'flip': flip}
        if params:
            params.update(eval_params)
        else:
            params = eval_params
        # Load results:
        out = pickle.load(open(res, 'rb'))
        compiled.append([params, out])
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
                     round(res['SPICE'] * 100, 2), round(exp(res['ml_loss']), 2)])
    return tab


def parse_name_clean(params):
    sample_cap = params.get('sample_cap', 0)
    sample_reward = params.get('sample_reward', 0)
    alter_loss = params.get('alter_loss', 0)

    loss_version = params['loss_version']
    alpha = params['alpha']
    tau_sent = params['tau_sent']
    tau_word = params['tau_word']
    rare = params.get('rare_tfidf', 0)
    simi = params['similarity_matrix']
    if 'tfidf' in loss_version:
        loss_version += " n=%d, idf=%d" % (params.get('ngram_length', 0), rare)
    if loss_version == "word2":
        if 'train_coco' in simi:
            G = "Coco"
        else:
            G = "Glove-Wiki"
        if rare:
            G += ' xIDF'
        if params.get('word_add_entropy', 0):
            G += ' +H'
        if params.get('exact_dkl', 0):
            G += ' +ExDKL'
        modelname = 'Word Level, Sim=%s, $\\tau=%.2f$, $\\alpha=%.1f$' % (G, tau_word, alpha)

    elif sample_cap:
        if loss_version == "dummy":
            loss_version = "constant"
        ver = params.get('sentence_loss_version', 1)
        modelname = 'SampleP, r=%s V%d, $\\tau=%.2f$, $\\alpha=%.1f$' % (loss_version, ver, tau_sent, alpha)
    elif sample_reward:
        modelname = 'SampleR, r=%s, $\\tau=%.2f$, $\\alpha=%.1f$' % (loss_version, tau_sent, alpha)
    elif alter_loss:
        modelname = "Alternating losses, WL $\\tau=%.2f$ w/ SampleR, r=%s $\\tau=%.2f$, $\\alpha=%.1f$" \
                     % (tau_word, loss_version, tau_sent, alpha)
    else:
        modelname = ""
        print('Couldnt name ', params['modelname'])
    return modelname



def parse_name(model):
    rewards = ['hamming', 'cider', 'bleu2', 'bleu3', 'bleu4']
    if '/' in model:
        model = model.split('/')[-1]
        # parse parameters from name:
        if model.startswith("word2"):
            modelname = ['WL v2']
            if 'tsent' in model:
                modelname.append('& SL')
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
                if tau.startswith('0'):
                    div = tau.count('0') * (len(tau) - tau.count('0'))
                    tau = float(tau)/(10 ** div)
                else:
                    try:
                        tau = float(tau)
                    except:
                        print('Other case:', tau)
                        tau = 0
                modelname.append('$\\tau(w) = %.2f$' % tau)
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

                modelname.append('$\\tau(s) = %.2f$' % tau)
            elif c.startswith('a'):
                modelname.append('$\\alpha = %.2f$' % (float(c[1:])/10))
            elif c in ['word', 'word2', 'sample', 'sample2']:
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

def crawl_results(filter='', exc=None):
    models = sorted(glob.glob('save/*%s*' % filter))
    if exc:
        # Exclude models containg exc:
        models = [model for model in models if exc not in model]
    # print("Found:", models)
    fields = ["Model", 'Beam', 'CIDEr', 'Bleu4', 'Spice', 'Perplexity']
    recap = {}
    tab = PrettyTable()
    tab.field_names = fields
    dump = []
    for model in models:
        outputs = get_results(model)
        if len(outputs):
            modelname = parse_name_clean(outputs[0][0])
            dump.append(outputs)
            for (p, res) in outputs:
                if p['beam_size'] > 1:
                    cid = float(res['CIDEr'] * 100)
                    recap[p['alpha']] = cid
                    bl = float(res['Bleu_4'] * 100)
                    sp = float(res['SPICE'] * 100)
                    try:
                        perpl = float(exp(res['ml_loss']))
                    except:
                        perpl = 1.
                    tab.add_row([modelname,
                                 p['beam_size'],
                                 cid, bl, sp, perpl])
    return tab, dump


if __name__ == "__main__":

    # tab = gather_results("save/baseline")
    # print(tab.get_string(fields=FIELDS, sortby="Sorter"))
    # print(tab.get_html_string(fields=FIELDS, sortby="Sorter"))

    if len(sys.argv) > 1:
        filter = sys.argv[1]
        if len(sys.argv) > 2:
            exc = sys.argv[2]
        else:
            exc = None
    else:
        filter= ''
    tab, dump = crawl_results(filter, exc)
    print(tab.get_string(sortby='CIDEr', reversesort=True))
    filename = "Results/res%s_%s" % (filter, socket.gethostname())
    with open(filename+'.html', 'w') as f:
        ss = tab.get_html_string(sortby="CIDEr", reversesort=True)
        # print(ss)
        f.write(ss)
    pickle.dump(dump, open(filename+".res", 'wb'))

