#  from graphviz import Digraph
import torch
from torch.autograd import Variable
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
sns.set(context='poster',
        style='whitegrid',
        palette='deep',
        font='sans-serif',
        font_scale=1,
        color_codes=False,
        rc=None)
from sklearn.manifold import TSNE


def plot_codes(source="save/textLM_vae/codes.pkl"):
    F = pickle.load(open(source, 'r'))
    #  X = [f for f in F['codes']]
    X = F['codes']
    #  X = np.array(X)
    print "X.shape:", X.shape
    #  S = [s for s in F['sentences']]
    S = F['sentences']
    print "Read %d sentences" % len(S)
    plot_tsne(X, S, "save/textLM_vae/code.png")



def plot_tsne_words(model):
    words = pickle.load(open('save/%s/infos.pkl' % model, 'r'))
    print words.keys()
    words = words['vocab']
    words['0'] = 'eos'
    pth = torch.load('save/%s/model.pth' % model)
    W = pth['embed.weight'].cpu().numpy()
    tsne_model = TSNE(n_components=2, random_state=0, verbose=2)
    X2d = tsne_model.fit_transform(W)
    fig = plt.figure()
    for i in range(len(words)):
        plt.text(X2d[i, 0], X2d[i, 1], words[str(i)], fontsize=9)
    plt.draw()
    fig.savefig("save/%s/words.png" % model, bbox_inches='tight')

def plot_tsne_codes(X, Refs, figname):
    """
    tedlt
    """
    tsne_model = TSNE(n_components=2, random_state=0, verbose=2)
    X2d = tsne_model.fit_transform(X)
    fig = plt.figure()
    count = 0
    #  while (5 * count) < len(Refs):
    while count < 5:
        xy = X2d[count * 5: (count + 1) * 5]
        plt.plot(xy[:, 0], xy[:, 1], '.')
        count += 1
        #  plt.text(xy[0], xy[1], sent, fontsize=9)
    plt.draw()
    fig.savefig(figname, bbox_inches='tight')




def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map[id(u)], size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


if __name__ == "__main__":
    #  plot_tsne_words('baseline_resnet50_glove')
    plot_tsne_words('baseline_resnet50_glove_syn')
    plot_tsne_words('baseline_resnet50_r300')
