#  from dataloader import *
#  from opts import create_logger
#
#
#  logger  = create_logger('./tmp_log')
#  opt = {"batch_size": 10000, "train_only": 1, "logger": logger,
#         "input_json": "data/BookCorpus/freq5_books.json",
#         "input_h5": "data/BookCorpus/freq5_books.h5"}
#
#
#  class Struct:
#      def __init__(self, **entries):
#          self.__dict__.update(entries)
#
#  opts = Struct(**opt)
#  loader = textDataLoader(opts)
#  i = 0
#  while i < 50:
#      data = loader.get_batch('test')
#      i += 1
#

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import numpy as np
LOG_DIR = "save/tmptf"

init_op = tf.initialize_all_variables()

with tf.Session() as session:
    N = 10000 # Number of items (vocab size).
    D = 200 # Dimensionality of the embedding.
    embedding_var = tf.constant(np.zeros((N, D)), name='word_embedding')
    x = tf.Variable([42.0, 42.1, 42.3], name='x')
    session.run(init_op)
    print(session.run(tf.global_variables()))

    saver = tf.train.Saver([x])
    saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), 0)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    projector.visualize_embeddings(summary_writer, config)
