import json
import os
import os.path as osp
import networkx as nx
import random
import string
import matplotlib.pyplot as plt
from operator import mul
from math import log
import numpy as np


data = json.load(open('data/coco/dataset_coco.json', 'r'))
# filepath, sentids, flename,imgid,cocoid, split, sentences (array of dicts (tokens
# raw imgid sentid)
data = data['images']
for img in data:
    capsfile = 'tmpcaps/caps%d.txt' % img['imgid']
    if not osp.exists(capsfile):
        caps = [" ".join(c['tokens']) for c in img['sentences']]
        with open(capsfile, 'w') as f:
            for cap in caps:
                f.write('-40 1 %d %s\n' % (len(cap.split()), cap))
                print "%d >> %s" % (img['imgid'], cap)
    meshfile = 'tmpcaps/caps%d.mesh' % img['imgid']
    if not osp.exists(meshfile):
        print "Generating mesh file"
        os.system('nbest-lattice -use-mesh -nbest %s -write %s' % (capsfile, meshfile))
    genfile = 'tmpcaps/gen_caps%d.txt' % img['imgid']
    if not osp.exists(genfile):
        print "Reading %s" % meshfile
        gen = nx.DiGraph()
        #Read the lattice and generate all possible combinations:
        with open(meshfile, 'r') as f:
            lines = f.readlines()
            # skip the first 3:
            lines = lines[3:]
            leaves = []
            for align, l in enumerate(lines):
                l = l.split()
                new_leaves= []
                for i in range(1, len(l) / 2):
                    tok = l[2 * i]
                    if tok in gen.nodes():
                        tok = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))  + "_" + tok
                    p = - log(float(l[2 * i + 1]))
                    if not align:
                        gen.add_edge('start', tok, weight=p)
                    else:
                        edges = [(leaf, tok, p) for leaf in leaves]
                        gen.add_weighted_edges_from(edges)
                    new_leaves.append(tok)
                leaves = new_leaves
            assert len(leaves)
            #  assert nx.is_tree(gen)
            gen.add_weighted_edges_from([(leaf, "end", 1) for leaf in leaves])
        # Draw graph:
        #  pos = nx.spring_layout(gen)
        #  plt.figure()
        #  nx.draw(gen, pos, cmap=plt.get_cmap('jet'), with_labels=True)
        #  plt.savefig('tmpcaps/gen_caps%d.png' % img['imgid'], bbox_inches='tight')
        #get all paths:
        discard = ['start', 'end', '*DELETE*']
        with open(genfile, 'w') as f:
            c = []
            c_scores = []
            short_paths = [path for path in nx.all_shortest_paths(gen, source="start", target="end", weight="weight")]
            print "Found %d short paths" % len(short_paths)
            for path in random.sample(short_paths, min(20, len(short_paths))):
                scores = [gen.get_edge_data(path[i],path[i+1])['weight'] for i in range(len(path) - 1)]
                score = sum(scores)
                path  = [p.split('_')[1] if '_' in p else p for p in path]
                #  print path
                sent = ' '.join([p for p in path if p not in discard])
                print "candidate: %s (%.3f)" % (sent, score)
                c.append(sent)
                c_scores.append(score)
            #  c_scores = np.array(c_scores)
            #  ind = np.argpartition(c_scores, -40)[-40:]
            for i in range(len(c_scores)):
                f.write('%.3f %s' % (c_scores[i], c[i]) + '\n')




