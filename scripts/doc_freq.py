from collections import defaultdict
import pickle
import json

def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
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


def read_coco(n=4):
    crefs = []
    coco = json.load(open('data/coco/annotations/captions_train2014.json', 'r'))
    annot = coco['annotations']
    coco_refs = {}
    for a in annot:
        try:
            coco_refs[a['image_id']].append(a['caption'])
        except:
            coco_refs[a['image_id']] = [a['caption']]

    for imid in coco_refs:
        crefs.append([precook(ref, n) for ref in coco_refs[imid]])
    return crefs


if __name__ == "__main__":
    crefs = read_coco()
    tfidf = compute_doc_freq(crefs)
    pickle.dump({'freq': tfidf, 'length': len(crefs)}, open('data/coco-train-df.p', 'wb'))

