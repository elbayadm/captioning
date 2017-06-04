import cPickle as pickle
import numpy as np
import json
from sklearn.neighbors import KDTree
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import scipy
import matplotlib.pyplot as plt

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

"""
Prepare embedding matrix of shape ( 1 + len(vocab), embedding_size = 300)
"""

def get_pairwise_distances():
    G = pickle.load(open('data/Glove/cocotalk_glove_v2.pkl', 'r'))
    eps = 1e-6
    print "G shape:", G.shape, len(G)
    #  G[-1] = np.mean(G)
    for i in range(len(G)):
        if not np.sum(G[i] ** 2):
            print '%d) norm(g) = 0' % i
            G[i] = eps + G[i]
    Ds = scipy.spatial.distance.pdist(G, metric='cosine')
    Ds = scipy.spatial.distance.squareform(Ds)
    As = np.diag(Ds)
    print "(scipy) sum:", np.sum(As), "min:", np.min(Ds), np.min(As), "max:", np.max(Ds), np.max(As)
    Ds = 1 - Ds / 2# map to [0,1]
    print Ds.shape, np.min(Ds), np.max(Ds), np.diag(Ds)
    pickle.dump(Ds, open('data/Glove/cocotalk_similarities_v2.pkl', 'w'),
                protocol=pickle.HIGHEST_PROTOCOL)

def prepare_glove(ixtow, source):
    """
    inputs:
        ixtow : index to word dictionnary of the vocab
        source: dict of the glove vectors
    """
    G = np.zeros((len(ixtow) + 1, 300), dtype="float32")
    G[0] = source['eos']
    print "EOS norm:", np.sum(G[0] ** 2)
    for i in range(1, len(ixtow) + 1):
        word = ixtow[str(i)]
        #  print "word:", word
        if word.lower() in source:
            G[i] = source[word.lower()]
            if not np.sum(G[i] ** 2):
                raise ValueError("Norm of glove embedding null token %d| word %s" % (i, word) )
        else:
            try:
                if CORRECT[word.lower()] in source:
                    print "Correcting %s into %s" % (word.lower(), CORRECT[word.lower()])
                    word = CORRECT[word.lower()]
                    G[i] = source[word]
                    if not np.sum(G[i] ** 2):
                        raise ValueError("Norm of glove embedding null token %d| word %s" % (i, word) )
            except:
                print "Missing word %s in Glove" % word
    pickle.dump(G, open('data/Glove/cocotalk_glove_v2.pkl', 'w'), protocol=pickle.HIGHEST_PROTOCOL)


def get_synonyms(source, ixtow):
    """
    inputs:
        source: dict of glove embeddings
    """
    kdt = KDTree(source, leaf_size=30, metric='euclidean')
    D, N = kdt.query(source, k=4, return_distance=True)
    NN = {}
    ixtow['0'] = 'eos'
    for i in range(len(D)):
        q =  ixtow[str(i)]
        print "word query:", q
        print "nearest neighbors:"
        tmp = {}
        for dist, nbr in zip(D[i], N[i]):
            if dist > 0:
                print "neighbor:", ixtow[str(nbr)], dist
                tmp[nbr] = {"word": ixtow[str(nbr)], "dist": dist}
        NN[i] = {"word": q, "neighbors": tmp}
    pickle.dump(NN, open('data/Glove/nearest_neighbors.pkl', 'w'))



if __name__== '__main__':
    #  Glove = {}
    #  with open('data/Glove/glove.42B.300d.txt', 'r') as f:
    #      for line in f:
    #          code = line.strip().split()
    #          word = code[0]
    #          print "parsed word:", word
    #          g = np.array(code[1:], dtype="float32")
    #          assert g.shape == (300,)
    #          Glove[word] = g
    #  pickle.dump(Glove, open('data/Glove/glove_dict.pkl', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
    #  Glove = pickle.load(open('data/Glove/glove_dict.pkl', 'r'))
    #  data = json.load(open('data/coco/cocotalk.json', "r"))
    #  ixtow = data['ix_to_word']
    #  print "Preparing Glove embeddings matrix"
    #  prepare_glove(ixtow, Glove)
    #  data = json.load(open('data/coco/cocotalk.json', "r"))
    #  ixtow = data['ix_to_word']

    #  gloves = pickle.load(open('data/Glove/cocotalk_glove.pkl', 'r'))
    #  get_synonyms(gloves, ixtow)
    #  print "Preparing similarities matrix"
    #  get_pairwise_distances()
    Sim = pickle.load(open("data/Glove/cocotalk_similarities_v2.pkl"))
    plt.figure()
    plt.matshow(Sim)
    plt.colorbar()
    plt.savefig('data/Glove/cocotalk_similarites_v2.pdf', bbox_inches='tight')
    vector = Sim.flatten()
    plt.figure()
    plt.hist(vector, 100)
    plt.yscale('log', nonposy='clip')
    plt.savefig('data/Glove/cocotalk_similarities_v2_hist.pdf', bbox_inches="tight")
