import pickle
import glob
import os.path as osp

RES = {}
models = glob.glob('save/*')
print('Initially:', len(models))
models = [m for m in models if 'fncnn6_reset' in m]
print('Found models:', models)

for model in models:
    print('Model:', model)
    if osp.exists('%s/infos.pkl' % model):
        val = pickle.load(open('%s/infos.pkl' % model, 'rb'))['val_result_history']
        lang_stats = [v['lang_stats'] for v in val.values()]
        RES[model.split('/')[-1]] = lang_stats
pickle.dump(RES, open('results/overall_ci.stats', 'wb'))




