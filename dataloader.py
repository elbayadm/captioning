from __future__ import absolute_import
from __future__ import division
#  from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import torch
from torchvision import transforms as trn

preprocess = trn.Compose([
    #trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class textDataLoader(object):
    """
    Text data iterator class
    """
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        # load the json file which contains additional information about the dataset
        self.opt.logger.warn('DataLoader loading json file: %s' % opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        self.opt.logger.warn('vocab size is %d ' % self.vocab_size)
        # open the hdf5 file
        self.opt.logger.warn('DataLoader loading h5 file: %s' % opt.input_h5)
        self.h5_file = h5py.File(self.opt.input_h5)
        # load in the sequence data
        seq_size = self.h5_file['labels'].shape
        self.seq_length = seq_size[1]
        self.opt.logger.warn('max sequence length in data is %d' % self.seq_length)

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['sentences'])):
            sent = self.info['sentences'][ix]
            if sent['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif sent['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif sent['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        self.opt.logger.warn('assigned %d sentences to split train' % len(self.split_ix['train']))
        self.opt.logger.warn('assigned %d sentences to split val' % len(self.split_ix['val']))
        self.opt.logger.warn('assigned %d sentences to split test' % len(self.split_ix['test']))
        self.iterators = {'train': 0, 'val': 0, 'test': 0}

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_batch(self, split, batch_size=None):
        split_ix = self.split_ix[split]
        batch_size = batch_size or self.batch_size
        label_batch = np.zeros([batch_size, self.seq_length + 2], dtype ='int')
        score_batch = np.ones([batch_size, ], dtype ='float32')
        mask_batch = np.zeros([batch_size, self.seq_length + 2], dtype ='float32')
        max_index = len(split_ix)
        wrapped = False
        infos = []
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            ix = split_ix[ri]
            label_batch[i, 1 : self.seq_length + 1] = self.h5_file['labels'][ix, :self.seq_length]
            try:
                score_batch[i,] = self.h5_file['scores'][ix]
            except:
                print 'No scores found'
            # record associated info as well
            infos.append(ix)
        # generate mask
        nonzeros = np.array(map(lambda x: (x != 0).sum()+2, label_batch))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data = {}
        data['labels'] = label_batch
        data['masks'] = mask_batch
        data['scores'] = score_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(split_ix), 'wrapped': wrapped}
        data['infos'] = infos
        #  print infos[0], ">>" ,infos[-1]

        return data

    def reset_iterator(self, split):
        self.iterators[split] = 0

class DataLoader(object):
    """
    data iterator class
    """
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = self.opt.seq_per_img
        #  self.load_syn = opt.use_synonyms

        # load the json file which contains additional information about the dataset
        self.opt.logger.warn('DataLoader loading json file: %s' % opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        self.opt.logger.warn('vocab size is %d ' % self.vocab_size)
        # open the hdf5 file
        self.opt.logger.warn('DataLoader loading h5 file: %s' % opt.input_h5)
        self.h5_file = h5py.File(self.opt.input_h5)


        # extract image size from dataset
        images_size = self.h5_file['images'].shape
        assert len(images_size) == 4, 'images should be a 4D tensor'
        assert images_size[2] == images_size[3], 'width and height must match'
        self.num_images = images_size[0]
        self.num_channels = images_size[1]
        self.max_image_size = images_size[2]
        self.opt.logger.warn('read %d images of size %dx%dx%d' %
                             (self.num_images,
                              self.num_channels,
                              self.max_image_size,
                              self.max_image_size))

        # load in the sequence data
        seq_size = self.h5_file['labels'].shape
        self.seq_length = seq_size[1]
        self.opt.logger.warn('max sequence length in data is %d' % self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_file['label_start_ix'][:]
        self.label_end_ix = self.h5_file['label_end_ix'][:]

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        self.opt.logger.warn('assigned %d images to split train' % len(self.split_ix['train']))
        self.opt.logger.warn('assigned %d images to split val' % len(self.split_ix['val']))
        self.opt.logger.warn('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        split_ix = self.split_ix[split]
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img
        #  print "get_batch seq_per_img:", seq_per_img

        img_batch = np.ndarray([batch_size, 3, 224,224], dtype ='float32')
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype ='int')
        score_batch = np.ones([batch_size * seq_per_img, ], dtype ='float32')

        #  label_syn_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype ='int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype ='float32')
        max_index = len(split_ix)
        wrapped = False

        infos = []

        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            ix = split_ix[ri]

            # fetch image
            #img = self.load_image(self.image_info[ix]['filename'])
            img = self.h5_file['images'][ix, :, :, :]
            img_batch[i] = preprocess(torch.from_numpy(img[:, 16:-16, 16:-16].astype('float32')/255.0)).numpy()

            # fetch the sequence labels
            ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1 # number of captions available for this image
            assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'
            #  print "Reading %d captions (%d required)" % (ncap, seq_per_img)
            #  scores = np.ones([seq_per_img,], dtype='float32')
            if ncap < seq_per_img:
                # we need to subsample (with replacement)
                seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
                scores = np.ones([seq_per_img,], dtype = 'float32')
                #  if self.load_syn:
                #      seq_syn = np.zeros([self.seq_per_img, self.seq_length], dtype = 'int')

                for q in range(seq_per_img):
                    ixl = random.randint(ix1,ix2)
                    seq[q, :] = self.h5_file['labels'][ixl, :self.seq_length]
                    try:
                        scores[q] = self.h5_file['scores'][ixl]
                    except:
                        if self.opt.less_confident < 1 and self.opt.less_confident:
                            if ixl - ix1 > 4:
                                scores[q] = self.opt.less_confident
            else:
                #  ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
                ixl = ix1
                seq = self.h5_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]
                try:
                    scores = self.h5_file['scores'][ixl: ixl + seq_per_img]
                except:
                    scores = np.ones([seq_per_img,], dtype='float32')
                    if self.opt.less_confident < 1 and self.opt.less_confident:
                        scores[5:] *= self.opt.less_confident
                #  print "Seq:", seq
                #  if self.load_syn:
                #      seq_syn = self.h5_file['labels_syn'][ixl: ixl + self.seq_per_img, :self.seq_length]

            label_batch[i * seq_per_img : (i + 1) * seq_per_img, 1 : self.seq_length + 1] = seq
            score_batch[i * seq_per_img : (i + 1) * seq_per_img] = scores
            #  if self.load_syn:
            #      label_syn_batch[i * self.seq_per_img : (i + 1) * self.seq_per_img, 1 : self.seq_length + 1] = seq_syn

            # record associated info as well
            info_dict = {}
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        # generate mask
        nonzeros = np.array(map(lambda x: (x != 0).sum()+2, label_batch))

        for ix, row in enumerate(mask_batch):
            row[1:nonzeros[ix]] = 1

        data = {}
        data['images'] = img_batch
        data['labels'] = label_batch
        data['scores'] = score_batch
        #  data['labels_syn'] = label_syn_batch
        data['masks'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(split_ix), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    def reset_iterator(self, split):
        self.iterators[split] = 0
