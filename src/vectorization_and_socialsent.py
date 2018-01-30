"""
Performs a SocialSent run over each community independently,
then filters the vocabulary to pick ones that occur in all communities
with a user-specified frequency (absolute or relative), and runs a
series of ML classifiers on the terms and their polarities to determine
which ones are most important.  Finally, returns sample posts from each
subreddit using those terms.
"""

from functools import partial
from multiprocessing import Pool
import os
import pickle

from gensim.corpora.dictionary import Dictionary
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import sentprop
# Try importing Cython versions of the PPMI code,
# which will be significantly faster, but also require compilation.
# Fall back on pure Python implementations, which are liable to be
# significantly slower.
try:
    import ppmi
except Exception as e:
    print(e)
    print("Importing pure Python PPMI transform; this will be slower than the Cython version.")
    from src import ppmi_purepython as ppmi


def make_vectors(
        window=5,
        smoothing=0.75
):
    """
    Wrapper function to vectorize each file individually.
    This function is NOT multithreaded, because the scipy implementation
    of SVDS already seems to be.

    :return:
    """
    # Get a list of total numbers of posts, so that tqdm can print expected
    # time to completion and a progress bar when reading data in.
    totals = pd.read_csv("Word Counts by Subreddit.csv")
    totals = dict(zip(totals["File"], totals["Number of Posts"]))

    files = [
        i.path
        for i in os.scandir("Processed Files")
        if i.name.endswith("_processed.txt.bz2")
        and not (
            os.path.isfile(i.path.replace(".txt.bz2", ".npy").replace("Processed Files", "Processed Files/MODELS"))
            and os.path.isfile(i.path.replace(".txt.bz2", ".vocab").replace("Processed Files", "Processed Files/MODELS"))
         )
    ]
    files = sorted(files, key=os.path.getsize)
    for i in files:
        out = i.replace("Processed Files","Processed Files/MODELS")
        vecs, vocab = ppmi.main(
            infile=i,
            total=totals[i.split("\\")[-1].replace("_processed.txt.bz2", "")],
            window=window,
            smoothing=smoothing,
        )
        np.save(
            out.replace(".txt.bz2", ".npy"),
            vecs
        )
        vocab.save(
            out.replace(".txt.bz2", ".vocab")
        )

    return 0

def make_sentprop(
        vecfile,
        pos_seeds,
        neg_seeds,
        no_below=50,
        no_above=0.5,
        filter_extremes=True,
        tol=1e-6,
        beta=0.9,
        nn=10,
        maxiter=500,
):
    """
    Run the SentProp algorithm, per the paper.

    :param vecfile: path to the .npy file containing word vectors
    :param vocabfile: path to the Gensim dictionary file.
    :param no_below: no_below paramter for the
    :param mode: 'tf' or 'sp'.  'tf' to use TensorFlow for GPU accelerated
        matrix multiplication.  'sp' to use scipy+numpy in Cython for CPU,
        but more memory-efficient, matrix multiplication.  'sp' recommended
        for most uses.
    :param filter_extremes: bool; True to remove tokens with extreme document
        frequencies.  Can be useful to enable if your corpus is large and will not
        fit into memory.
    :return:
    """
    vocabfile = vecfile.replace(".npy", ".vocab")
    id2word = Dictionary.load(vocabfile)
    vecs = np.load(vecfile).astype(np.float32)
    fname = vocabfile.split("\\")[-1].rsplit("_", maxsplit=1)[0]

    # If filter_extremes is true, remove vocabulary items.
    # This can help remove extremely low-frequency terms that
    # might introduce noise; even very loose parameters can
    # remove a large portion of the vocabulary, substantially
    # speeding up computation, though at the price of accuracy.
    if filter_extremes == True:
        tmp = {id2word[i]:vecs[i] for i in range(vecs.shape[0])}
        id2word.filter_extremes(
            no_below=no_below,
            no_above=no_above,
            keep_n=len(id2word) # defaults to 100k; this removes this limit
        )
        tmp = {i:tmp[i] for i in tmp if i in id2word.token2id}
        before_size = vecs.shape[0]
        vecs = np.array([
            tmp[i]
            for i in id2word.token2id.keys()
        ])
        print(f"""{fname}: Original vocab size: {before_size}
{fname}: Filtered vocab size: {vecs.shape[0]}
{fname}: {before_size - vecs.shape[0]} items with document frequency <{no_below} removed.""")
        del tmp

    scores = sentprop.main(
        id2word,
        vecs,
        pos_seeds,
        neg_seeds,
        tol=tol,
        beta=beta,
        nn=nn,
        maxiter=maxiter,
        print_name=vocabfile.replace("\\", "/").split("/")[-1].split("_")[0]
    )

    scores_file = vecfile\
        .replace('.npy', '_scores.p')\
        .replace("\\", "/")\
        .replace('MODELS/', 'MODELS/SCORES/')
    pickle.dump(scores, open(f"{scores_file}", "wb"))
    return scores

def main(
        pos_seeds,
        neg_seeds,
        no_below=50,
        no_above=0.5,
        filter_extremes=True,
        tol=1e-6,
        beta=0.9,
        nn=10,
        maxiter=500,
        n_threads=1,
):
    """
    Run the sentprop, multithreaded.
    :return:
    """

    # Pull the list of vocab/vector files for SentPropping
    files = [
        i.path # ignore .vocab suffix
        for i in os.scandir("Processed Files/MODELS")
        if i.path.endswith(".npy")
        and not os.path.isfile(i.path.replace(".npy", "_scores.p").replace("\\", "/").replace("MODELS/", "MODELS/SCORES/"))
    ]
    files = sorted(files, key=os.path.getsize)
    # make a partial function so we can apply it more easily
    # for multiprocessing
    mappable = partial(
        make_sentprop,
        pos_seeds=pos_seeds,
        neg_seeds=neg_seeds,
        no_below=no_below,
        no_above=no_above,
        filter_extremes=filter_extremes,
        tol=tol,
        beta=beta,
        nn=nn,
        maxiter=maxiter
    )

    with Pool(n_threads) as P:
        res = list(P.imap(mappable, files))