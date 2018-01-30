"""
Pyre Python implementation of the PPMI Transform code.
Use this only if the Cython code will not compile--this will
be considerably slower.
"""

import numpy as np

import bz2
from collections import Counter
from time import sleep

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from gensim.corpora.dictionary import Dictionary
from tqdm import tqdm, trange

def cooccurence_matrix(infile, total, window, smoothing):
    """
    Generates a co-occurrence matrix using symmetric-window skip-grams of
    length window.  Then generates a PPMI transform using smoothed probabilities.

    :param infile: bz2-compressed file to read.
    :param total: the total number of files, if known, for TQDM to use.
    :param window: symmetric window size to use.
    :param smoothing: smoothing value for smoothed prior distributions
    :param no_below: no_below arg for Gensim dict.
    :param no_above: no_above arg for Gensim dict.
    :return: SVD vectors
    """

    with bz2.open(infile, "r") as F:
        # gensim Dictionary for word<->id mappings
        vocab = Dictionary(
            i.split()[1:]
            for i in tqdm(F, total=total, desc=f"{infile}: {'Gathering Vocabulary':<25s}")
        )
        vocab.compactify()
        sleep(.5)
        print("\nVOCAB SIZE: {}".format(len(vocab)))
        sleep(.5)

    with bz2.open(infile, "r") as F:
        INDS = Counter(
            (DOC[i], DOC[i+j])
            for DOC in (
                np.array(vocab.doc2idx(J.split()[1:]))
                for J in tqdm(F, total=total, desc=f"{infile}: {'Co-occurrence Matrix':<25s}")
            )
            for i in range(1, len(DOC))
            for j in range(min(window, len(DOC) - i))
        )

    # Convert {(A, B):C} dict structure to np.array([C, A, B]) for
    # sparse matrix construction.
    INDS = np.array([
        [INDS[I], I[0], I[1]]
        for I in tqdm(INDS.keys(), desc=f"{infile}: {'Generating Indices':<25s}")
        if I[0] != I[1]
        and I[0] > 0
        and I[1] > 0
    ])
    print(INDS.shape)
    ppmi_mat = csr_matrix(
        (INDS[:,0], (INDS[:,1], INDS[:,2])),
        shape=(len(vocab), len(vocab))
    )

    print("PPMI matrix shape: {}".format(ppmi_mat.shape))
    del INDS
    # ppmi_mat.eliminate_zeros()
    # Add transpose, since PPMI is symmetric--PPMI(i,j) = PPMI(j,i)
    ppmi_mat = ppmi_mat + ppmi_mat.transpose()

    ### PPMI TRANSFORMATION ###
    print("Generating matrices for PPMI transform...")
    # We'll use these more than once, so only calculate them the one time
    POW = ppmi_mat.power(smoothing)
    TOT = np.sum(ppmi_mat)
    p_i_star = np.array(np.sum(ppmi_mat, axis=1) / TOT).astype(np.float32).reshape((-1,))
    p_star_j = np.array(np.sum(POW, axis=0) / np.sum(POW)).astype(np.float32).reshape((-1,))
    ppmi_mat = ppmi_mat / TOT

    ### PPMI TRANSFORM ###
    data = ppmi_mat.data.astype(np.float32)
    indices = ppmi_mat.indices.astype(np.int32)
    indptr = ppmi_mat.indptr.astype(np.int32)
    for i in trange(indptr.shape[0] - 1, desc=f"{infile}: {'PPMI Transform':<25s}"):
        data[indptr[i]:indptr[i+1]] = \
            np.maximum(
                0,
                np.log2(data[indptr[i]:indptr[i+1]] / (p_i_star[i] * p_star_j[indices[indptr[i]:indptr[i+1]]]))
        )
    ppmi_mat = csr_matrix((data, indices, indptr))
    ppmi_mat.eliminate_zeros()

    ### SVD ###
    sleep(.5)
    print("SVD...")
    # per https://web.stanford.edu/~jurafsky/slp3/16.pdf we only
    # use the raw left singular values as the word embedding vectors
    U = svds(ppmi_mat, k=300, return_singular_vectors="u")[0]

    return U, vocab

def main(infile, total, window, smoothing):
    return cooccurence_matrix(
        infile,
        window=window,
        smoothing=smoothing,
        total=total,
    )
