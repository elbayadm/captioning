from dataloader import *
import opts
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # Load the tokenized captions:
    opt = opts.parse_opt()
    loader = DataLoader(opt)
    labels = loader.h5_file['labels'][()]
    print('Labels shape:', labels.shape)
    print(type(labels))
    cnt = Counter()
    for l in labels:
        cnt.update(l)
    cnt = cnt.most_common()
    words = loader.ix_to_word
    words['0'] = 'EOS'
    K = []
    V = []
    with open('words_count.txt', 'w') as f:
        for k, v in cnt:
            # print(k, words[str(k)], cnt[k])
            print(k, words[str(k)], v)
            f.write("%s: %d\n" % (words[str(k)], v))
            K.append(words[str(k)])
            V.append(v)
    print(V)
    plt.figure()
    K = K[1:]
    V = V[1:]
    plt.bar(np.arange(len(K)),
            V,
            0.99,  # bar width
            c='r')
    plt.yscale("log")
    plt.savefig('words_count.png')


