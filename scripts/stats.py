import pickle
import json
import glob
import os.path as osp


L = {}
models = glob.glob('save/*')
for model in models:
    res = '%s/evaluations/test/bw3.json' % model
    if osp.exists(res):
        caps = json.load(open(res, 'r'))
        lengths = [len(cap['caption'].split()) for cap in caps]
        L[model] = lengths
pickle.dump(L, open('results/lengths.p', 'wb'))


