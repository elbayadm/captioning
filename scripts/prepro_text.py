import os
import json
import argparse
import random
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize

def tokenize(sent):
    return str(sent).lower().translate(None, string.punctuation).strip().split()

def build_vocab(sentences, params):
    """
    Build vocabulary
    """
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for sent in sentences:
        txt = tokenize(sent)
        nw = len(txt)
        sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
        for w in txt:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str, cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w, n in counts.iteritems() if n <= count_thr]
    vocab = [w for w, n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    print 'number of words in vocab would be %d' % (len(vocab), )
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)
    max_len = max(sent_lengths.keys())
    print 'max length sentence in raw data: ', max_len
    print 'sentence length distribution (count, number of words):'
    sum_len = sum(sent_lengths.values())
    for i in xrange(max_len+1):
        print '%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0)*100.0/sum_len)

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print 'inserting the special UNK token'
        vocab.append('UNK')
    processed_sentences = []
    for sent in sentences:
        txt = tokenize(sent)
        proc = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
        processed_sentences.append(proc)
    return vocab, processed_sentences


def encode_sentences(sentences, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """
    max_length = params['max_length']
    M = len(sentences)
    label_length = np.zeros(M, dtype='uint32')
    L = np.zeros((M, max_length), dtype='uint32')
    for i, sent in enumerate(sentences):
        label_length[i] = min(max_length, len(sent)) # record the length of this sequence
        for k, w in enumerate(sent):
            if k < max_length:
                L[i, k] = wtoi[w]
    # note: word indices are 1-indexed, and captions are padded with zeros
    assert np.all(label_length > 0), 'error: some caption had no words?'
    return L, label_length


def main(params):
    """
    Main preprocessing
    """
    with open(params['input_txt'], 'r') as f:
        sentences = f.readlines()
    print "Read %d lines from %s" % (len(sentences), params['input_txt'])
    seed(123) # make reproducible
    # create the vocab
    vocab, proc_sentences = build_vocab(sentences, params)
    itow = {i+1:w for i, w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i, w in enumerate(vocab)} # inverse table
    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_length = encode_sentences(proc_sentences, params, wtoi)
    # create output h5 file
    N = len(sentences)
    f = h5py.File(params['output_h5'], "w")
    f.create_dataset("labels", dtype='uint32', data=L)
    f.create_dataset("label_length", dtype='uint32', data=label_length)
    print 'wrote ', params['output_h5']
    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['sentences'] = []
    for i, sent in enumerate(proc_sentences):
        jsent = {'tokens': sent}
        coin = random.uniform(0,1)
        if coin < .75:
            jsent['split'] = 'train'
        elif coin < .88:
            jsent['split'] = 'val'
        else:
            jsent['split'] = 'test'
        out['sentences'].append(jsent)
    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']


if __name__ == "__main__":
    #  $ python scripts/prepro.py --input_json .../dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk.h5 --images_root ...
    parser = argparse.ArgumentParser()
    # input json
    DATA_DIR = 'data/BookCorpus/'
    parser.add_argument('--input_txt', default='%strain_selected.txt' % DATA_DIR,  help='input txt file to process into hdf5')
    parser.add_argument('--output_json', default='%sfreq50_books.json' % DATA_DIR, help='output json file')
    parser.add_argument('--output_h5', default='%sfreq50_books.h5' % DATA_DIR, help='output h5 file')
    # options
    parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=50, type=int, help='only words that occur more than this number of times will be put in vocab')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent=2)
    main(params)
