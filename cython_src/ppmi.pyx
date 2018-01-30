# cython: profile=True

import numpy as np
cimport numpy as np
cimport cython

import bz2
from collections import Counter
from time import sleep

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from gensim.corpora.dictionary import Dictionary
from tqdm import tqdm, trange

cdef cooccurence_matrix_testing(
        str infile,
        int total,
        int window,
        float smoothing,
    ):
    """
    Generates a raw window-based co-occurrence matrix of vocabulary
    terms based on the input text.

    THIS VERSION MANUALLY CONSTRUCTS A CSR_MATRIX FROM DATA/INDICES/INDPTR
    ARRAYS.

    :param infile: bz2-compressed file to read.
    :param total: the total number of files, if known, for TQDM to use.
    :param window: symmetric window size to use.
    :param smoothing: smoothing value for smoothed prior distributions
    :param no_below: no_below arg for Gensim dict.
    :param no_above: no_above arg for Gensim dict.
    :return: SVD vectors
    """

    # Cython--initialize all variables used in this function
    cdef np.ndarray[dtype=np.int32_t, ndim=2] pairs
    cdef np.ndarray[dtype=np.int32_t, ndim=1] indices, indptr
    cdef np.ndarray[dtype=np.float32_t, ndim=1] data, p_i_star, p_star_j
    cdef np.ndarray[dtype=np.int_t, ndim=1] DOC
    cdef int i, j, _
    cdef bytes J, S
    cdef tuple TT

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
    ppmi_mat = csr_matrix(
        (INDS[:,0], (INDS[:,1], INDS[:,2])),
        shape=(len(vocab), len(vocab))
    )

    print("PPMI matrix shape: {}".format(ppmi_mat.shape))
    del INDS
    ppmi_mat.eliminate_zeros()

    # Cast to CSR matrix to allow easier and MUCH faster acess to the
    # underlying data for PPMI transformations
    print("Casting to CSR matrix...")
    ppmi_mat = csr_matrix(ppmi_mat, dtype=np.float32)
    # Add transpose, since PPMI is symmetric--PPMI(i,j) = PPMI(j,i)
    ppmi_mat = ppmi_mat + ppmi_mat.transpose()

    ### PPMI TRANSFORMATION ###
    print("Generating matrices for PPMI transform...")
    # We'll use these more than once, so only calculate them the one time
    # Smoothing per https://web.stanford.edu/~jurafsky/slp3/15.pdf page 8
    POW = ppmi_mat.power(smoothing)
    TOT = np.sum(ppmi_mat)
    p_i_star = np.array(np.sum(ppmi_mat, axis=1) / TOT).astype(np.float32).reshape((-1,))
    # smooth the context probabilities
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
    # ignore the right singular values--the left singular ones are
    # what Hamilton et al use, so we only care about those.
    # This will also give a speedup compared to computing both matrices.
    U = svds(ppmi_mat, k=300, return_singular_vectors="u")[0]

    return U, vocab

def main(
    str infile,
    int total,
    int window,
    float smoothing,
):
    return cooccurence_matrix_testing(
        infile,
        window=window,
        smoothing=smoothing,
        total=total,
    )
