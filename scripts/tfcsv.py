from tensorboard.backend.event_processing import event_accumulator as tfea
import numpy as np
import pandas as pd
import sys
import pickle


def create_csv(inpath, outpath):
    sg = {tfea.COMPRESSED_HISTOGRAMS: 1,
          tfea.IMAGES: 1,
          tfea.AUDIO: 1,
          tfea.SCALARS: 0,
          tfea.HISTOGRAMS: 1}
    ea = tfea.EventAccumulator(inpath, size_guidance=sg)
    ea.Reload()
    scalar_tags = ea.Tags()['scalars']
    df = pd.DataFrame(columns=scalar_tags)
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        scalars = np.array(map(lambda x: x.value, events))
        df.loc[:, tag] = scalars
    df.to_csv(outpath)


def create_pkl(inpath, outpath=None):
    """
    Parse the scalars in a directory of events or a single event file
    """
    # assert tfea.IsTensorFlowEventsFile(inpath), 'File is not a tf event'
    sg = {tfea.COMPRESSED_HISTOGRAMS: 1,
          tfea.IMAGES: 1,
          tfea.AUDIO: 1,
          tfea.SCALARS: 0,
          tfea.HISTOGRAMS: 1}
    ea = tfea.EventAccumulator(inpath, size_guidance=sg)
    ea.Reload()
    scalar_tags = ea.Tags()['scalars']
    print('Columns:', scalar_tags)
    track = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        scalars = np.array([x.value for x in events])
        track[tag] = scalars
    if not outpath:
        modelname = "_".join(inpath.split('/')[1:])
        print(modelname)
        outpath = "Results/%s_events.pkl" % modelname
        print('Dumping pickled stats in %s' % outpath)
    pickle.dump(track,
                open(outpath, 'wb'))

if __name__ == '__main__':
    args = sys.argv
    inpath = args[1]
    # outpath = args[2]
    # create_csv(inpath, outpath)
    create_pkl(inpath)
