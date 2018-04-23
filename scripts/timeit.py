import os.path as osp
import re
import sys
from glob import glob
from prettytable import PrettyTable
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

"""
GTX 1080              : 8112/8114   8,873 GFLOPS
GTX 1080 ti           : 11172      11,340 GFLOPS
Titan X               : 12207/12287 6,691 GFLOPS
Titax x pascal        : 12189      10,974 GFLOPS
Titx xp               : 12189      12,150 GFLOPS
Tesla p1000           : 16276      10,609 GFLOPS
"""

MEM_MATCH = {'8112': "gtx1080",
             '8114': "gtx1080",
             '10765': "gtx1080 ti",
             '11162': "gtx1080 ti",
             '11171': "gtx1080 ti",
             '11172': "gtx1080 ti",
             '11636': "titan x",
             '11796': "titan x",
             '12205': "titan x",
             '12207': "titan x",
             '12287': "titan x",
             '12179': "titan x pascal/xp",
             '12189': "titan x pascal/xp",
             '16276': "p100"
             }

selected = [
    "baseline_showtell152",
    "fncnn6_reset_baseline_showtell152",
    "word_coco_tword015_a07_showtell152",
    "fncnn6_reset_word_coco_tword015_a07_showtell152",
    "word_coco_tword009_idf10_a07_showtell152",
    "fncnn6_reset_word_coco_tword009_idf10_a07_showtell152",
    "strat_rhamming_pool0_tsent01_a04_showtell152",
    "fncnn6_reset_strat_rhamming_pool0_tsent01_a04_showtell152",
    "strat_lazy_rhamming_pool0_tsent01_a04_showtell152",
    "fncnn6_reset_strat_lazy_rhamming_pool0_tsent01_a04_showtell152",
    "strat_rhamming_pool1_tsent017_a04_showtell152",
    "fncnn6_reset_strat_rhamming_pool1_tsent017_a04_showtell152",
    "strat_lazy_rhamming_pool1_tsent017_a04_showtell152",
    "fncnn6_reset_strat_lazy_rhamming_pool1_tsent017_a04_showtell152",
    "strat_rhamming_pool2_tsent03_a05_showtell152",
    "fncnn6_reset_strat_rhamming_pool2_tsent03_a05_showtell152",
    "strat_lazy_rhamming_pool2_tsent03_a05_showtell152",
    "fncnn6_reset_strat_lazy_rhamming_pool2_tsent03_a05_showtell152",
    "importance_qhamming_pool0_tsent01_rcider_tsent05_a04_showtell152",
    "fncnn6_reset_importance_qhamming_pool0_tsent01_rcider_tsent05_a04_showtell152",
    "importance_lazy_qhamming_pool0_tsent01_rcider_tsent05_a04_showtell152",
    "fncnn6_reset_importance_lazy_qhamming_pool0_tsent01_rcider_tsent05_a04_showtell152",
    "importance_qhamming_limited1_tsent017_rcider_tsent05_a04_showtell152",
    "fncnn6_reset_importance_qhamming_limited1_tsent017_rcider_tsent05_a04_showtell152",
    "importance_lazy_qhamming_limited1_tsent017_rcider_tsent05_a04_showtell152",
    "fncnn6_reset_importance_lazy_qhamming_limited1_tsent017_rcider_tsent05_a04_showtell152",
    "importance_qhamming_limited2_tsent03_rcider_tsent05_a05_showtell152",
    "fncnn6_reset_importance_qhamming_limited2_tsent03_rcider_tsent05_a05_showtell152",
    "importance_lazy_qhamming_limited2_tsent03_rcider_tsent05_a05_showtell152",
    "fncnn6_reset_importance_lazy_qhamming_limited2_tsent03_rcider_tsent05_a05_showtell152",
    "combine_strat_rhamming_pool1_tsent017_a04_word_coco_tword009_idf10_a07_showtell152",
    "fncnn6_reset_combine_strat_rhamming_pool1_tsent017_a04_word_coco_tword009_idf10_a07_showtell152",
]

topd = [
    "topdown_resnet152_msc",
    "fncnn6_reset_topdown_resnet152_msc",
]

th_steps = 100
tab = PrettyTable()
fields = ['model', 'mem', 'time/batch']
tab.field_names = fields
ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

if len(sys.argv) > 1:
    models = [sys.argv[1]]
else:
    models = glob('save/*time')
    models = [m.split('/')[-1] for m in models]
    print('models:', models)
for model in models:
    print(RED, 'Model:', model, NC)
    # logfile = "save/%s/stderr" % model
    logfile = "save/%s/train.log" % model
    # if not osp.exists(logfile):
        # logfile = "save/%s/train.log" % model
    if not osp.exists(logfile):
        continue
    timings = []
    memv = 0
    for line in open(logfile, 'r'):
        # line = ansi_escape.sub('', line)
        # if re.search("KILL", line):
            # print(model, line)
        if re.search("GPU", line):
            try:
                memv = int(line.strip().split()[-1][:-1])
            except:
                memv = 0
            try:
                mem = MEM_MATCH[str(memv)]
            except:
                mem = "Unk"
            if len(timings) > th_steps:  # and memv > 8000:
                avgt = sum(timings)/len(timings)
                print("Average time per batch: %f(s)" % avgt)
                tab.add_row([model, "%s (%dM)" % (mem, memv), avgt])
            print("Resetting counter")
            print(line.strip())
            timings = []
        if re.search("Time/batch", line):
            try:
                timings.append(float(line.strip().split()[-1]))
            except:
                pass
    if len(timings) > th_steps:  # and memv > 8000:
        avgt = sum(timings)/len(timings)
        print("Average time per batch: %.4f(s)" % avgt)
        tab.add_row([model, "%s (%dM)" % (mem, memv), avgt])
results = tab.get_string(fields=fields)
print(results)
with open("timings", 'w') as f:
    f.writelines(results)

