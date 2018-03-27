import re
import sys

selected = ["topdown_resnet152_msc",
            "fncnn6_reset_topdown_resnet152_msc",]

if len(sys.argv) > 1:
    models = [sys.argv[1]]
else:
    models = selected
for model in models:
    print('Model:', model)
    logfile = "save/%s/train.log" % model
    timings = []
    for line in open(logfile, 'r'):
        if re.search("GPU", line):
            if len(timings):
                print("Average time per batch: %f(s)" % (sum(timings)/len(timings)))
            print("Resetting counter")
            print(line.strip())
            timings = []
        if re.search("Time/batch", line):
            timings.append(float(line.strip().split()[-1]))
    print("Average time per batch: %f(s)" % (sum(timings)/len(timings)))
