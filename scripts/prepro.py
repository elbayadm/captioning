"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""

import os
import json
import argparse
import random
from random import shuffle, seed
import string
import nltk
from nltk.corpus import stopwords
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize
from six.moves import cPickle as pickle

def tokenize(sentence):
    return nltk.word_tokenize(str(sentence).lower().translate(None, string.punctuation))


def build_vocab(imgs, params):
    """
    Build vocabulary
    """
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(list(counts.values()))
    print('total words:', total_words)
    stops = stopwords.words('english')
    # print('Eliminating stop words:', stops)
    # bad_words = [w for w, n in counts.items() if (n <= count_thr or w in stops)]
    # vocab = [w for w, n in counts.items() if (n > count_thr and w not in stops)]
    bad_words = [w for w, n in counts.items() if (n <= count_thr)]
    vocab = [w for w, n in counts.items() if (n > count_thr)]

    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('Vocab:', vocab)
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len+1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0)*100.0/sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')
    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab


def encode_captions_syn(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """
    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs) # total number of captions
    label_arrays = []
    label_syn_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    S  = pickle.load(open('data/Glove/nearest_neighbors.pkl', 'r'), encoding="iso-8859-1")
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'
        Li = np.zeros((n, max_length), dtype='uint32')
        Li_syn = np.zeros((n, max_length), dtype='uint32')

        for j, s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    index = int(wtoi[w])
                    print("%s(%d)" % (w, index))
                    Li[j, k] = index
                    if len(S[index]['neighbors']):
                        synindex = S[index]['neighbors'].keys()[0]
                        print("synonym %s(%d)" % (S[index]['neighbors'][synindex]['word'], synindex))
                        Li_syn[j, k] = synindex
                    else:
                        Li_syn[j, k] = index
        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_syn_arrays.append(Li_syn)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1
        counter += n
    L = np.concatenate(label_arrays, axis=0) # put all the labels together
    L_syn = np.concatenate(label_syn_arrays, axis=0) # put all the labels together

    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert L_syn.shape[0] == M, 'lengths don\'t match? that\'s weird (syn)'

    assert np.all(label_length > 0), 'error: some caption had no words?'
    print('encoded captions to array of size ', L.shape)
    return L, L_syn, label_start_ix, label_end_ix, label_length


def encode_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """
    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs) # total number of captions
    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'
        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]
        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1
        counter += n
    L = np.concatenate(label_arrays, axis=0) # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'
    print('encoded captions to array of size ', L.shape)
    return L, label_start_ix, label_end_ix, label_length


def encode_extra_scored_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """
    max_length = params['max_length']
    N = len(imgs)
    # Load additional captions
    extra_ = json.load(open(params['gen_source'], 'r'))
    extra = {}
    scored = True
    no_scores = []
    with_scores = []
    for im in extra_:
        try:
            extra[im['id']] = [[[w if w in wtoi else 'UNK' for w in sent.split()], sc] for sent, sc in zip(im['captions'], im['scores'])]
            extra[im['id']] = [s for s in extra[im['id']] if len(s[0])]
            with_scores.append(im['id'])
            #  print extra[im['id']]
        except:
            print('No scores found in the generated json')
            no_scores.append(im['id'])
            extra[im['id']] = [[[w if w in wtoi else 'UNK' for w in sent.split()], 1] for sent in im['captions']]
        #  print im['id']

    print("Len extra:", len(extra), "vs extra_", len(extra_))
    print("with scores:", len(with_scores), "unique:", len(np.unique(np.array(with_scores))))
    print("Captions =", sum(len(img['final_captions']) for img in imgs)) # total number of captions
    missing_indices = []
    found_indices = []
    for img in imgs:
        #  print img['cocoid'], img['split']
        if img['split'] == 'train':
            #  print "Pre:", img['final_captions']
            #  img["final_captions"] = [[c, 3.] for c in img['final_captions']]
            try:
                img["final_captions"] = extra[img['cocoid']]  # I should append!!!!!
                found_indices.append(img['cocoid'])
            except:
                #  assert img['cocoid'] in extra
                img["final_captions"] = [[c, 1.] for c in img['final_captions']]
                missing_indices.append(img['cocoid'])
            #  print "Post:",  img['final_captions']
        else:
            #  print "skipping val/restval/test"
            img["final_captions"] = [[c, 1.] for c in img['final_captions']]


    M = sum(len(img['final_captions']) for img in imgs) # total number of captions
    print("Missing %d extra scores vs %d found" % (len(missing_indices), len(found_indices)))
    print("Missing from extra: %d" % len(no_scores))
    print("Encoding %d captions" % M)
    label_arrays = []
    score_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'
        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            score_arrays.append(s[1])
            s_ = s
            s = s[0]
            label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
            assert len(s) > 0, "Sentence of length 0 %s %s %s" % (s, str(s_), str(img['final_captions']))
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]
        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1
        counter += n
    L = np.concatenate(label_arrays, axis=0) # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    if scored:
        assert len(score_arrays) == M, "Missing scores"
    score_arrays = np.array(score_arrays)
    assert np.all(label_length > 0), 'error: some caption had no words?'
    print('encoded captions to array of size ', L.shape)
    return L, score_arrays, label_start_ix, label_end_ix, label_length




def encode_extra_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """
    max_length = params['max_length']
    N = len(imgs)
    # Load additional captions
    extra_ = json.load(open(params['gen_source'], 'r'))
    extra = {}
    scored = True
    for im in extra_:
        try:
            extra[im['id']] = [[[w if w in wtoi else 'UNK' for w in sent.split(' ')], sc] for sent, sc in zip(im['sampled'], im['scores'])]
            #  print extra[im['id']]
        except:
            print('No scores found in the generated json')
            extra[im['id']] = [[[w if w in wtoi else 'UNK' for w in sent.split(' ')], 1] for sent in im['sampled']]
        #  print im['id']

    print("Captions =", sum(len(img['final_captions']) for img in imgs)) # total number of captions

    for img in imgs:
        print(img['cocoid'], img['split'])
        try:
            if img['split'] == 'train':
                #  print "Gt:",  img["final_captions"]
                #  print "Added:", extra[img['cocoid']]
                print("Pre:", img['final_captions'])
                img["final_captions"] = [[c, 3.] for c in img['final_captions']]
                img["final_captions"] += extra[img['cocoid']]
                print(img['final_captions'])
            else:
                print("skipping val/restval/test")
                img["final_captions"] = [[c, 1.] for c in img['final_captions']]
        except:
            print("########## No addtional captions provided")


    M = sum(len(img['final_captions']) for img in imgs) # total number of captions
    print("Encoding %d captions" % M)
    label_arrays = []
    score_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'
        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            print("s(pre)", s)
            score_arrays.append(s[1])
            s = s[0]
            print("s:", s)
            label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]
        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1
        counter += n
    L = np.concatenate(label_arrays, axis=0) # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    if scored:
        assert len(score_arrays) == M, "Missing scores"
    score_arrays = np.array(score_arrays)
    assert np.all(label_length > 0), 'error: some caption had no words?'
    print('encoded captions to array of size ', L.shape)
    return L, score_arrays, label_start_ix, label_end_ix, label_length


def main(params):
    """
    Main preprocessing
    """
    imgs = json.load(open(params['input_json'], 'r'))
    #  imgs = random.sample(imgs['images'], 3000)
    imgs = imgs['images']
    seed(123) # make reproducible
    # create the vocab
    vocab = build_vocab(imgs, params)
    itow = {i+1:w for i, w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i, w in enumerate(vocab)} # inverse table
    #------------- Reuse stored vocab
    # data = json.load(open('data/coco/cocotalk.json', "r"))
    # itow = data['ix_to_word']
    # wtoi = {itow[k]: k for k in itow}
    # encode captions in large arrays, ready to ship to hdf5 file
    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if w in wtoi else 'UNK' for w in txt]
            img['final_captions'].append(caption)
            #  if len(img['final_captions']) >= 5:
            #      break

    ################################################################################################################
    if params['gen'] == '':
        L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)
        Scores = []
    else:
        #  L, Scores, label_start_ix, label_end_ix, label_length = encode_extra_captions(imgs, params, wtoi)
        L, Scores, label_start_ix, label_end_ix, label_length = encode_extra_scored_captions(imgs, params, wtoi)
    #
    # create output h5 file
    N = len(imgs)
    f = h5py.File(params['output_h5'], "w")
    f.create_dataset("labels", dtype='uint32', data=L)
    f.create_dataset("scores", dtype='float32', data=Scores)
    #  f.create_dataset("labels_syn", dtype='uint32', data=Lsyn)
    f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f.create_dataset("label_length", dtype='uint32', data=label_length)
    dset = f.create_dataset("images", (N, 3, params['imsize'],  params['imsize']), dtype='uint8') # space for resized images
    for i, img in enumerate(imgs):
        # load the image
        if 'filepath' not in img:
            img['filepath'] = ''
        I = imread(os.path.join(params['images_root'], img['filepath'], img['filename']))
        try:
            Ir = imresize(I, ( params['imsize'], params['imsize']))
        except:
            print('failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],))
            raise
        # handle grayscale input images
        if len(Ir.shape) == 2:
            Ir = Ir[:, :, np.newaxis]
            Ir = np.concatenate((Ir, Ir, Ir), axis=2)
        # and swap order of axes from (256,256,3) to (3,256,256)
        Ir = Ir.transpose(2, 0, 1)
        # write to h5
        dset[i] = Ir
        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
    f.close()
    print('wrote ', params['output_h5'])
    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):
        jimg = {}
        jimg['split'] = img['split']
        if 'filename' in img:
            jimg['file_path'] = os.path.join(img['filepath'], img['filename']) # copy it over, might need
            jimg['id'] = img['imgid']
        if 'cocoid' in img:
            jimg['id'] = img['cocoid'] # copy over & mantain an id, if present (e.g. coco ids, useful)
        out['images'].append(jimg)
    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])


if __name__ == "__main__":
    #  $ python scripts/prepro.py --input_json .../dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk.h5 --images_root ...
    parser = argparse.ArgumentParser()
    # input json
    DATA_DIR = 'coco'
    parser.add_argument('--input_json', default='data/%s/dataset_%s.json' % (DATA_DIR, DATA_DIR),  help='input json file to process into hdf5')
    parser.add_argument('--gen', default='', help='Source of additional captions')
    #  parser.add_argument('--output_json', default='%sgen_cocotalk.json' % DATA_DIR, help='output json file')
    parser.add_argument('--output', default='', help='output name')
    #  parser.add_argument('--output_h5', default='%sgen_cocotalk.h5' % DATA_DIR, help='output h5 file')
    # options
    parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--imsize', default=256, type=int, help='image size.')
    parser.add_argument('--images_root', default='data/%s/images' % DATA_DIR, help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    params['output_json'] = 'data/%s/%s.json' % (DATA_DIR, params['output'])
    params['output_h5'] = 'data/%s/%s.h5' % (DATA_DIR, params['output'])
    params['gen_source'] = "data/%s/%s.json" % (DATA_DIR, params['gen'])
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
