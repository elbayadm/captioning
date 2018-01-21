import scipy.misc as misc
import numpy as np

len_target = 16
v = 60  # Vocabulary size
T = .9  # Temperature
max_edits = len_target
def edit_dist():
    x = np.zeros(max_edits)
    for n_edits in range(max_edits):
        total_n_edits = 0  # total edits with n_edits edits without v^n_edits term
        for n_substitutes in range(min(len_target, n_edits)+1):
            # print(n_substitutes)
            n_insert = n_edits - n_substitutes
            current_edits = misc.comb(len_target, n_substitutes, exact=False) * \
                            misc.comb(len_target+n_insert-n_substitutes, n_insert, exact=False)
            total_n_edits += current_edits
        x[n_edits] = np.log(total_n_edits) + n_edits * np.log(v)
        # log(tot_edits * v^n_edits)
        x[n_edits] = x[n_edits] - n_edits / T - n_edits / T * np.log(v)
        # log(tot_edits * v^n_edits * exp(-n_edits / T) * v^(-n_edits / T))

    p = np.exp(x)
    p /= np.sum(p)
    print('Probas:', p)


def hamming_dist(v=60):
    x = [np.log(misc.comb(len_target, d, exact=False)) + d * np.log(v) - d/T * np.log(v) - d/T for d in range(len_target + 1)]
    x = np.array(x)
    p = np.exp(x)
    p /= np.sum(p)
    print('Probas:', p)


hamming_dist()
hamming_dist(9000)
