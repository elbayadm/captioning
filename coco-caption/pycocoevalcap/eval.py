__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice

class COCOEvalCap:
    def __init__(self, coco, cocoRes, all_metrics=False):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.all_metrics = all_metrics
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        if self.all_metrics:
            scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                # (Meteor(), "METEOR"),
                # (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
                # (Spice(), "SPICE")
            ]
        else:
            scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Cider(), "CIDEr")
            ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print(('computing %s score...'%(scorer.method())))
            try:
                score, scores = scorer.compute_score(gts, res)
                if type(method) == list:
                    for sc, scs, m in zip(score, scores, method):
                        self.setEval(sc, m)
                        self.setImgToEvalImgs(scs, list(gts.keys()), m)
                        print(("%s: %0.3f"%(m, sc)))
                else:
                    self.setEval(score, method)
                    self.setImgToEvalImgs(scores, list(gts.keys()), method)
                    print(("%s: %0.3f"%(method, score)))
            except:
                print('Could not evaluate %s score' % (scorer.method()))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(sorted(imgIds), scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [self.imgToEval[imgId] for imgId in sorted(self.imgToEval.keys())]
