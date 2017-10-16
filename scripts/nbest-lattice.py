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
                print("%d >> %s" % (img['imgid'], cap))
    meshfile = 'caps%d.mesh' % img['imgid']
    if not osp.exists('tmpcaps/' + meshfile):
        print("Generating mesh file")
        os.system('nbest-lattice -use-mesh -nbest %s -write tmpcaps/%s' % (capsfile, meshfile))
    genfile = 'tmpcaps/gen_srilm_caps%d.txt' % img['imgid']
    if not osp.exists(genfile):
        # os.system("lattice-tool -read-mesh -in-lattice tmpcaps/%s -nbest-decode 25 -nbest-viterbi -out-nbest-dir tmpcaps" % (meshfile))
        os.system("lattice-tool -read-mesh -in-lattice tmpcaps/%s -nbest-decode 25  -out-nbest-dir tmpcaps/gen -out-nbest-dir-xscore1 tmpcaps/scores" % (meshfile))
        os.system('gunzip tmpcaps/gen/%s.gz' % meshfile)
        os.system('gunzip tmpcaps/scores/%s.gz' % meshfile)
        with open(genfile, 'w') as f:
            sc = open('tmpcaps/scores/%s' % meshfile, 'r')
            g = open('tmpcaps/gen/%s' % meshfile, 'r')
            for l1, l2 in zip(g, sc):
                line = l1.split()
                line = " ".join(line[4:-1])
                score = float(l2.strip())
                print(score, ">> ", line)
                f.write("%f %s\n" % (score, line))
        # os.system('rm tmpcaps/tmp%d.mesh' % img['imgid'])
