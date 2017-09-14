from dataloader import *
import opts
from collections import Counter

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
    words['0'] = "UNK"
    with open('words_count.txt', 'w') as f:
        for k, v in cnt:
            # print(k, words[str(k)], cnt[k])
            print(k, words[str(k)], v)
            f.write("%s: %d\n" % (words[str(k)], v))


