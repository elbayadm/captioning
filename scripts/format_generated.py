import json
import glob
import os.path as osp
# Array of dicts with key 'id' for cocoid and 'sampled' array of generated
# captions.
# Match imid to cocoid
COCO = json.load(open('data/coco/dataset_coco.json', 'r'))
COCO = COCO['images']
Match = {}
for datum in COCO:
    #  print datum
    Match[datum['imgid']] = datum['cocoid']
D = []
genfiles = glob.glob('tmpcaps/gen_srilm_viterbi_caps*.txt')
for genfile in genfiles:
    d = {}
    imid = int(osp.basename(genfile).split('.')[0][8:])
    d['id'] = Match[imid]
    d['sampled'] = []
    with open(genfile, 'r') as f:
        for line in f.readlines():
            sent = ' '.join(line.strip().split(' ')[1:])
            print  "%d : %s" % (d['id'], sent)
            d['sampled'].append(sent)
    D.append(d)
json.dump(D, open('data/coco/generated_captions_viterbi_confusion.json', 'w'))


