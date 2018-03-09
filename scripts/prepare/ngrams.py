from collections import defaultdict
import pickle
import json


def ngrams(words, k=2):
    ngrams = []
    for i in range(len(words)-k+1):
        ngram = "_".join(words[i:i+k])
        ngrams.append(ngram)
    return " ".join(ngrams)


def precook(words, nmin=1, nmax=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    counts = defaultdict(int)
    for k in range(nmin, nmax+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def compute_doc_freq(crefs):
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram,count) in list(ref.items())]):
            print('ngram:', ngram)
            document_frequency[ngram] += 1
    return document_frequency


def build_ngrams(n=2, output='train_bigrams.txt'):
    coco = json.load(open('data/coco/dataset_coco.json', 'r'))
    imgs = coco['images']
    with open('data/coco/%s' % output, 'w') as f:
        for img in imgs:
            for sent in img['sentences']:
                if n == 1:
                    f.write('%s\n' % " ".join(sent['tokens']))
                else:
                    f.write("%s\n" % ngrams(sent['tokens']))


def read_coco(nmin=1, nmax=4):
    crefs = []
    coco = json.load(open('data/coco/dataset_coco.json', 'r'))
    imgs = coco['images']
    coco_refs = {}
    for img in imgs:
        for sent in img['sentences']:
            try:
                coco_refs[img['imgid']].append(sent['tokens'])
            except:
                coco_refs[img['imgid']] = [sent['tokens']]

    for imid in coco_refs:
        crefs.append([precook(ref, nmin=nmin, nmax=nmax) for ref in coco_refs[imid]])
    return crefs


if __name__ == "__main__":
    # crefs = read_coco()
    # tfidf = compute_doc_freq(crefs)
    # pickle.dump({'freq': tfidf, 'length': len(crefs)}, open('data/coco-train-tok-df.p', 'wb'))
    # Dump bigrams as corpus:
    build_ngrams(n=1, output="train_unigrams.txt")


