import json
import numpy as np
import scipy.io as sio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import re
from collections import Counter
import scipy
from nltk.corpus import stopwords


def get_pairwise_distances(F):
    eps = 1e-6
    print("Features shape:", F.shape, len(F))
    for i in range(len(F)):
        if not np.sum(F[i] ** 2):
            print('%d) norm(f) = 0' % i)
            F[i] = eps + F[i]
    Ds = scipy.spatial.distance.pdist(F, metric='cosine')
    Ds = scipy.spatial.distance.squareform(Ds)
    As = np.diag(Ds)
    print("(scipy) sum:", np.sum(As), "min:", np.min(Ds), np.min(As), "max:", np.max(Ds), np.max(As))
    Ds = 1 - Ds / 2# map to [0,1]
    print(Ds.shape, np.min(Ds), np.max(Ds), np.diag(Ds))
    # pickle.dump(Ds, open('data/cocotalk_similarities_visual_image.pkl', 'wb'),
                # protocol=pickle.HIGHEST_PROTOCOL)
    return Ds

CORRECT = {"surboard": "surfboard", "surfboarder": "surfboard", "surfboarders": "surfboard",
           "skatebaord": "skateboard", "skatboard": "skateboard",
           "frizbe": "frisbee", "fribee": "frisbee", "firsbee": "frisbee",
           "girafee": "giraffe", "griaffe": "giraffe",
           "hyrdant": "hydrant", "hyrdrant": "hydrant", "firehydrant": "fire-hydrant",
           "graffitid": "graffitied",
           "parasailer": "parasailers",
           "deckered": "decker",
           "stnading": "standing",
           "motorcyclers": "motorcyclists",
           "including:" : "including",
           "courch": "church",
           "skiies": "skiis",
           "brocclie": "brocoli",
           "frumpled": "frumple"
           }


def get_full_image_embedding():
    meta = json.load(open('data/coco/dataset_coco.json', 'r'))
    meta = meta['images']
    print('Meta lenght:', len(meta))
    print(meta[0])
    F = sio.loadmat('data/coco/vgg_feats.mat')
    F = F['feats']
    Words = {}
    cnt = Counter()
    for im in meta:
        for sent in im['sentences']:
            # print(sent['tokens'])
            cnt.update(sent['tokens'])
            for tok in sent['tokens']:
                if tok in Words:
                    Words[tok].add(sent['imgid'])
                else:
                    Words[tok] = set([sent['imgid']])
    print('Parsed words:', len(Words))
    print('Count:', cnt)
    Embed = {}
    for w in Words:
        Embed[w] = np.mean(F[:, list(Words[w])], axis=1)
    print('Embeddings:', len(Embed))
    pickle.dump(Embed, open('data/embeddings/full_image.pkl', 'wb'))



def compare_embeddings():
    # meta = json.load(open('data/coco/dataset_coco.json', 'r'))
    # meta = meta['images']
    # print('Meta lenght:', len(meta))
    # print(meta[0])
    # cnt = Counter()
    # for im in meta:
        # for sent in im['sentences']:
            # cnt.update(sent['tokens'])
    # stops = stopwords.words('english')
    # keep_words = [w for w in cnt if cnt[w] > 370 and w not in stops]

    regions = pickle.load(open('data/embeddings/regions.pkl', 'rb'))
    keep_words = list(regions)
    V = np.array([regions[id] for id in keep_words])
    cos_regions = get_pairwise_distances(V)
    print('Similiarity regions:', cos_regions.shape)
    # print('Keeping : ', keep_words)
    full_image = pickle.load(open('data/embeddings/full_image.pkl', 'rb'))
    V = np.array([full_image[id] for id in keep_words])
    cos_full = get_pairwise_distances(V)
    print('Similiarity full image:', cos_full.shape)
    # regions = pickle.load(open('data/embeddings/regions.pkl', 'rb'))
    # print('Regions keys:', regions.keys())
    # V = np.array([regions[id] for id in keep_words])
    # cos_regions = get_pairwise_distances(V)
    print('Similiarity regions:', cos_regions.shape)
    glove = pickle.load(open('data/embeddings/glove.pkl', 'rb'),
                        encoding='iso-8859-1')
    for id in glove:
        if id in CORRECT:
            glove[CORRECT[id]] = glove[id]
    V = np.array([glove[id] for id in keep_words])
    cos_glove = get_pairwise_distances(V)
    print('Similiarity glove:', cos_glove.shape)
    plot_sim(cos_glove, keep_words, 'glove')
    plot_sim(cos_regions, keep_words, 'regions')
    plot_sim(cos_full, keep_words, 'full_image')
    print('Delta(regions, full images)=', np.sum((cos_regions - cos_full) ** 2) / len(keep_words) ** 2)
    print('Delta(regions, glove)=', np.sum((cos_regions - cos_glove) ** 2) / len(keep_words) ** 2)
    print('Delta(glove, full images)=', np.sum((cos_glove - cos_full) ** 2) / len(keep_words) ** 2)

def plot_sim(F, labels, title):
    # plotting the matrices
    fig, ax = plt.subplots(figsize=(100, 100))
    cax = ax.matshow(F, interpolation='nearest')
    ax.grid(True)
    plt.title('%s similarities' % title)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    plt.yticks(range(len(labels)), labels, fontsize=6)
    fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5,
                             0.6, 0.7, .75,.8,.85,.90,.95,1])
    plt.savefig('data/embeddings/%s_sim.png' % title, bbox_inches='tight')


def reduce_tsne():
    tsne = TSNE(n_components=2, random_state=0, verbose=2)
    X_tsne = tsne.fit_transform(V)
    print('Tsne embeddings:', X_tsne.shape)
    words = list(Embed)
    pickle.dump({"TSNE": X_tsne, "WORDS": words}, open('x2.pkl', 'wb'))


def plot_tsne():
    data = pickle.load(open('x2.pkl', 'rb'), encoding='iso-8859-1')
    X_tsne = data['TSNE']
    words = data['WORDS']
    fig = plt.figure()
    for i in range(len(words)):
        if cnt[words[i]] > 20:
            txt = re.escape(words[i])
            plt.text(X_tsne[i, 0], X_tsne[i, 1], txt, fontsize=9)
            print('Word:', txt)
    plt.draw()
    fig.savefig("visual_embedding.png", bbox_inches='tight')



compare_embeddings()
