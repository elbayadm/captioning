from dataloader import *
from opts import create_logger


logger  = create_logger('./tmp_log')
opt = {"batch_size": 10000, "train_only": 1, "logger": logger,
       "input_json": "data/BookCorpus/freq5_books.json",
       "input_h5": "data/BookCorpus/freq5_books.h5"}


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

opts = Struct(**opt)
loader = textDataLoader(opts)
i = 0
while i < 50:
    data = loader.get_batch('test')
    i += 1

