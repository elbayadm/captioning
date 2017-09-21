import json
import numpy as np
import scipy.io as sio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import re


# meta = json.load(open('data/coco/dataset_coco.json', 'r'))
# meta = meta['images']
# print('Meta lenght:', len(meta))
# print(meta[0])
# F = sio.loadmat('data/coco/vgg_feats.mat')
# F = F['feats']
# Words = {}
# for im in meta:
    # for sent in im['sentences']:
        # for tok in sent['tokens']:
            # if tok in Words:
                # Words[tok].add(sent['imgid'])
            # else:
                # Words[tok] = set([sent['imgid']])
# print('Parsed words:', len(Words))
# Embed = {}
# for w in Words:
    # Embed[w] = np.mean(F[:, list(Words[w])], axis=1)
# print('Embeddings:', len(Embed))
# V = np.array(list(Embed.values()))
# print(V.shape)
# tsne = TSNE(n_components=2, random_state=0, verbose=2)
# X_tsne = tsne.fit_transform(V)
# print('Tsne embeddings:', X_tsne.shape)
# words = list(Embed)
# pickle.dump({"TSNE": X_tsne, "WORDS": words}, open('x2.pkl', 'wb'))


data = pickle.load(open('x2.pkl', 'rb'), encoding='iso-8859-1')
X_tsne = data['TSNE']
words = data['WORDS']
fig = plt.figure()
for i in range(len(words)):
    txt = re.escape(words[i])
    plt.text(X_tsne[i, 0], X_tsne[i, 1], txt, fontsize=9)
    print('Word:', txt)
plt.draw()
fig.savefig("visual_embedding.png", bbox_inches='tight')

