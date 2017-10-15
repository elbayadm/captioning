import os
import os.path as osp
import glob
import shutil


dirs = os.listdir('save/')
for dir in dirs:
    events = glob.glob('save/%s/events.*' % dir)
    if events:
        if not osp.exists('events/%s' % dir):
            os.makedirs('events/%s' % dir)
        print('Found ', events, ' in directory ', dir)
        for e in events:
            dst = 'events/' + '/'.join(e.split('/')[1:])
            print(e, '>>>', dst)
            shutil.copy2(e, dst)

