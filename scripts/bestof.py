"""
Get best performances of given model
"""
import pickle
import json
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Model directory in save')
    args = parser.parse_args()
    args = vars(args) # convert to ordinary dict
    # Load results history:
    print("Parsing the results of:", args['model'])
    infos = pickle.load(open('%s/infos-best.pkl' % args["model"], 'rb'), encoding='\"ISO-8859-1')
    val = infos["val_result_history"]
    iters = list(val.keys())
    ciders = [val[it]['lang_stats']["CIDEr"] for it in iters]
    b = np.argmax(np.array(ciders))
    b = iters[b]
    table = "<tr>\n<td>  </td><td>RNN only</td>\n<td>9487</td>\n"
    ftable = "\n</tr>"
    score = "<td>%.2f</td>"
    good_score = "<td><b>%.2f</b></td>"
    B4 = val[b]['lang_stats']['Bleu_4'] * 100
    C = val[b]['lang_stats']['CIDEr'] * 100
    S = val[b]['lang_stats']['SPICE'] * 100
    C3 = val[b]['lang_stats']['creative 3grams'] * 100
    C4 = val[b]['lang_stats']['creative 4grams'] * 100
    C5 = val[b]['lang_stats']['creative 5grams'] * 100
    V = val[b]['lang_stats']['vcab_use']

    b4 = good_score % B4 if B4 > 28.2 else score % B4
    c = good_score % C if C > 86 else score % C
    s = good_score % S if S > 16.5 else score % S
    out = table + b4 + c + s + '\n' + score % C3 + score % C4 + score % C5 + '\n' + "<td>%d</td>" % V + ftable
    print(out)
