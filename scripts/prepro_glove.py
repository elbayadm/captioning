import cPickle as pickle
import numpy as np
import json
from sklearn.neighbors import KDTree
from collections import OrderedDict
"""
Prepare embedding matrix of shape ( 1 + len(vocab), embedding_size = 300)
"""


def prepare_glove(ixtow, source):
    """
    inputs:
        ixtow : index to word dictionnary of the vocab
        source: dict of the glove vectors
    """
    G = np.zeros((len(ixtow) + 1, 300), dtype="float32")
    G[0] = source['eos']
    for i in range(1, len(ixtow) + 1):
        word = ixtow[str(i)]
        print "word:", word
        if word in source:
            G[i] = source[word]
        else:
            "Missing word in Glove"
    pickle.dump(G, open('data/Glove/cocotalk_glove.pkl', 'w'), protocol=pickle.HIGHEST_PROTOCOL)


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
    #  prepare_glove(ixtow, Glove)
    data = json.load(open('data/coco/cocotalk.json', "r"))
    ixtow = data['ix_to_word']

    gloves = pickle.load(open('data/Glove/cocotalk_glove.pkl', 'r'))
    get_synonyms(gloves, ixtow)
