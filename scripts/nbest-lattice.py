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
    genfile = 'tmpcaps/gen_srilm_viterbi_caps%d.txt' % img['imgid']
    if not osp.exists(genfile):
        os.system("lattice-tool -read-mesh -in-lattice %s -nbest-decode 25 -nbest-viterbi -out-nbest-dir tmpcaps" % (meshfile))
        os.system('mv %s.gz tmpcaps/tmp%d.mesh.gz' % (meshfile, img['imgid']))
        os.system('gunzip tmpcaps/tmp%d.mesh.gz' % img['imgid'])
        with open(genfile, 'w') as f:
            with open('tmpcaps/tmp%d.mesh' % img['imgid'], 'r') as s:
                for line in s:
                    line = line.split()
                    line = " ".join(line[4:-1])
                    print ">> ", line
                    f.write("0 %s\n" % line)
        os.system('rm tmpcaps/tmp%d.mesh' % img['imgid'])
