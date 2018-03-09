"""
Build the similarities matrix used in token-level smoothing
from a given embedding dictionary
"""


import sys
from os.path import expanduser
import argparse
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, '.')
from utils import pl, pd

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
           "including:": "including",
           "courch": "church",
           "skiies": "skiis",
           "brocclie": "brocoli",
           "frumpled": "frumple"
           }


def build_glove_dict(glove_txt):
    Glove = {}
    with open(glove_txt, 'r') as f:
        for line in f:
            code = line.strip().split()
            word = code[0]
            print("parsed word:", word)
            g = np.array(code[1:], dtype="float32")
            Glove[word] = g
    return Glove


def get_pairwise_distances(G):
    eps = 1e-6
    print("G shape:", G.shape, len(G))
    for i in range(len(G)):
        if not np.sum(G[i] ** 2):
            print('%d) norm(g) = 0' % i)
            G[i] = eps + G[i]
    Ds = pdist(G, metric='cosine')
    Ds = squareform(Ds)
    As = np.diag(Ds)
    print("(scipy) sum:", np.sum(As),
          "min:", np.min(Ds), np.min(As),
          "max:", np.max(Ds), np.max(As))
    Ds = 1 - Ds / 2  # map to [0,1]  # FIXME useless
    print(Ds.shape, np.min(Ds), np.max(Ds), np.diag(Ds))
    return Ds


def prepare_embeddings_dict(ixtow, source, output):
    """
    From a large dictionary of embeddings get that of the coco vocab in order
    inputs:
        ixtow : index to word dictionnary of the vocab
        source: dict of the glove vectors
        output: dumping path
    """
    dim = source[list(source)[0]].shape[0]
    print('Embedding dimension : %d' % dim)
    G = np.zeros((len(ixtow) + 1, dim), dtype="float32")
    print("EOS norm:", np.sum(G[0] ** 2))
    for i in range(1, len(ixtow) + 1):
        word = ixtow[str(i)]
        if word.lower() in source:
            G[i] = source[word.lower()]
            if not np.sum(G[i] ** 2):
                raise ValueError("Norm of the embedding null > token %d | word %s" % (i, word))
        else:
            try:
                if CORRECT[word.lower()] in source:
                    print("Correcting %s into %s" % (word.lower(), CORRECT[word.lower()]))
                    word = CORRECT[word.lower()]
                    G[i] = source[word]
                    if not np.sum(G[i] ** 2):
                        raise ValueError("Norm of the embedding null > token %d | word %s" % (i, word))
            except:
                print("Missing word %s in the given embeddings" % word)
    pd(G, output)
    return G


def get_synonyms(source, ixtow):
    """
    inputs:
        source: dict of glove embeddings
    """
    from sklearn.neighbors import KDTree
    kdt = KDTree(source, leaf_size=30, metric='euclidean')
    D, N = kdt.query(source, k=4, return_distance=True)
    NN = {}
    ixtow['0'] = 'eos'
    for i in range(len(D)):
        q = ixtow[str(i)]
        tmp = {}
        for dist, nbr in zip(D[i], N[i]):
            if dist > 0:
                # print "neighbor:", ixtow[str(nbr)], dist
                tmp[nbr] = {"word": ixtow[str(nbr)], "dist": dist}
        NN[i] = {"word": q, "neighbors": tmp}
    pd(NN, 'data/Glove/nearest_neighbors.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove_txt', type=str,
                        default="%s/work/GloVe-1.2/coco/vectors.w15.d512.txt" % expanduser('~'),
                        help='Path to the txt file of the word embeddings')
    parser.add_argument('--save_glove_dict', type=str,
                        help='Path to dump the glove dict',
                        default='data/Glove/glove_w15d512_coco.dict')
    parser.add_argument('--glove_dict', type=str,
                        default='',
                        help='Alternatively give a dictionnary of the coco words embeddings')
    parser.add_argument('--coco_json', type=str,
                        default='data/coco/cocotalk.json',
                        help='path of the preprocessing json to retrieve ix_to_word')
    parser.add_argument('--save_sim', type=str,
                        default='data/coco/glove_w15d512_coco_cocotalk.sim',
                        help='Path to dump the similaritiy matrix')
    parser.add_argument('--coco_stats', type=str,
                        default='data/coco/cocotalk.stats',
                        help='Path to load coco statistics')
    parser.add_argument('--save_rarity', type=str,
                        default='data/coco/promote_rare_bis.matrix',
                        help='Path to dump the _rarity_ matrix')


    args = parser.parse_args()
    # if len(args.glove_dict):
        # Glove = pl('data/coco/glove_w15d512_coco_cocotalk.dict')
    # else:
        # Glove = build_glove_dict(args.glove_txt)
        # if len(args.save_glove_dict):
            # # save for any eventual ulterior usage
            # pd(Glove, args.save_glove_dict)

    ixtow = json.load(open(args.coco_json, "r"))['ix_to_word']
    # print("Preparing Glove embeddings matrix")
    # coco_gloves = prepare_embeddings_dict(ixtow,
                                          # Glove,
                                          # output='data/coco/glove_w15d512_coco_cocotalk.embed')
    # print("Preparing similarities matrix")
    # sim = get_pairwise_distances(coco_gloves)
    # print('Saiving the similarity matrix into ', args.save_sim)
    # pd(sim, args.save_sim)

    # Rarity matrix:
    print(ixtow['1'], ixtow['2'], ixtow['9487'])
    stats = pl(args.coco_stats)
    counts = stats['counts']
    total_sentences = sum(list(stats['lengths'].values()))
    total_unk = sum([counts[w] for w in stats['bad words']])
    freq = np.array([total_sentences] +
                    [counts[ixtow[str(i)]] for i in range(1, len(ixtow))] +  # UNK is not referenced
                    [total_unk])
    print('Frequencies:', freq.shape,
          'min:', np.min(freq), 'max:', np.max(freq),
          "eos:", freq[0], "unk:", freq[-1])
    F = freq.reshape(1, -1)
    F1 = np.dot(np.transpose(1/F), F)
    F2 = np.dot(np.transpose(F), 1/F)
    FF = np.minimum(F1, F2)
    print('FF:', FF.shape,
          'min:', np.min(FF), 'max:', np.max(FF))
    pd(FF.astype(np.float32), args.save_rarity)


