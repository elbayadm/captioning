"""
Samplers for data augmentation
"""

from .greedy import GreedySampler
from .hamming import HammingSampler
from .hamming_sim import HammingSimSampler
from .hamming_unigram import HammingUnigramSampler
from .ngram import NgramSampler


def init_sampler(select, opt):
    """
    Wrapper for sampler selection
    """
    if select == 'greedy':
        return GreedySampler()
    elif select == 'hamming':
        return HammingSampler(opt)
    elif select == 'hamming-sim':
        return HammingSimSampler(opt)
    elif select == 'hamming-unigram':
        return HammingUnigramSampler(opt)
    elif select == 'ngram':
        return NgramSampler(opt)
    else:
        raise ValueError('Unknonw sampler %s' % select)



