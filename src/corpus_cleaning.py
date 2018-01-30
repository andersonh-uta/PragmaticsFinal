"""
Functions containing various bits of functionality to do a first pass on the
Reddit Public Comments corpus and split posts into their own file, one per
subreddit, with each file containing all posts from that subreddit.
"""

import bz2
from functools import partial
import json
import multiprocessing
import os
from shutil import move
from tempfile import TemporaryDirectory as TD

import bounter
import pandas as pd
from tqdm import tqdm

from src import preprocessing


############################################################
### FUNCTIONS TO COUNT THE NUMBER OF POSTS PER SUBREDDIT ###
############################################################
def recursive_scan(dirname):
    """
    Recursive version of os.scandir, which returns all filenames in a directory.
    :param dirname: the path to the directory to be walked.
    :return: a list of filenames
    """
    for i in os.scandir(dirname):
        if os.path.isfile(i.path):
            yield i.path
        elif os.path.isdir(i.path):
            yield from recursive_scan(i.path)


def count_number_worker(fname, size_mb=200):
    """
    Worker for multithreading the counting operation for speed.

    :param fname: the bz2 file to open.
    :param size_mb: the maximum memory footprint of the bounter.
    :return: bounter object on the subreddit keys
    """
    b = bounter.bounter(size_mb=size_mb)
    with bz2.open(fname, "rt", encoding="utf8") as F:
        b.update(json.loads(i)["subreddit"] for i in F)
    return b

def count_posts(num_threads):
    """
    Main function to multithread the subreddit counting.  Saves the counts
    out to a new csv file.

    :param num_threads: int
        How many threads to use when processing the data.  More --> faster,
        but also uses more memory.
    :param reddit_dir: str
        The folder containing the Reddit Public Comment Corpus files in bz2
        format.  The files can be contained in subfolders, but they should
        be the ONLY files in this directory.
    :param outfile: str, ends with .csv
        The path to save the output file to.  Must be a .csv file.
    :return: bounter object with subreddit counts
    """
    # files = []
    # for i in os.walk("reddit"):
    #     if i[0] == "by subreddit": continue
    #     for j in i[2]:
    #         files.append(f"{i[0]}\\{j}")
    files = list(recursive_scan("reddit"))

    # multiprocess the counting of posts per subreddit.
    with multiprocessing.Pool(processes=num_threads) as pool:
        bounters = list(tqdm(pool.imap(count_number_worker, files), desc="Counting Posts", total=len(files)))
    big_bounter = bounter.bounter(size_mb=200)
    for i in bounters: big_bounter.update(i)
    df = pd.Series(dict(big_bounter.items()))
    df.to_csv("Subreddit Post Counts.csv")

    return big_bounter

#############################################
### FUNCTIONS TO SPLIT POSTS BY SUBREDDIT ###
#############################################
def subreddit_worker(infile, keep, min_len=5, delete_orig=False):
    """
    Reads the input file, removes lines from the wrong subreddits,
    and writes each subreddit to its own file.  New temporary files
    are created, and populated with posts by each subreddit.  They
    are then copied out of the temporary directory, back to the main Reddit
    directory.

    :param keep: set or frozenset
        A list of subreddit names to keep.
    :param infile: str
        Path to a .bz2 file from the Reddit corpus to filter through.
    :param min_len: int
        Minimum length of a post, in number of words.  Posts shorter than
        this will be discarded.
    :param delete_orig: bool
        True to delete the original .bz2 file (from Archive.org) after processing,
        False to keep it.  Deleting it can save disk space.
    :param output_dir: str
        Directory to save the intermediate split files to.
    :return: 0 on success.

    """
    with bz2.open(infile, "rt", encoding="utf8") as IN, TD() as DIR:
        # Open temporary files
        files = {
            i : bz2.open(f"{DIR}\\{i}.txt.bz2", "wt", encoding="utf8")
            for i in keep
        }
        # Quick pre-filtering based on length.  More aggressive filtering
        # will be done later, during NLP preprocessing.
        filt = filter(
            lambda x: x["subreddit"] in keep and len(x["body"].split()) >= min_len,
            map(json.loads, IN)
        )
        for i in filt:
            # write repr() to keep each post to strictly one line, regardless
            # of newline breaks, etc in the original
            files[i["subreddit"]].write(repr(i["body"].strip()) + '\n')

        # make sure we explicitly close all files, since we're not using
        # a with-block for this!
        for i in files:
            files[i].close()

        # move non-empty files out of the tempdir
        for i in os.scandir(DIR):
            out = infile.replace("\\", "/") \
                .split("/")[-1] \
                .rsplit(".", maxsplit=1)[0]
            if bz2.open(i.path, "rt", encoding="utf8").readline().strip():
                move(
                    i.path,
                    f"By Subreddit/{i.name.rsplit('.', maxsplit=1)[0]} {out}.txt.bz2"
                )
    if delete_orig == True:
        os.remove(infile)

    return 0

def split_by_subreddit(keep=(), min_len=5, delete_orig=False, num_threads=1):
    """
    Reads the raw corpus data, and split posts into new files such that
    each file contains ONLY a single subreddit's posts.  Each input file
    from the original dataset will have a corresponding file for each subreddit
    contained in that file.  These should be concatenated together using
    the concatenate_subreddit_files() function.

    :param reddit_dir: str
        Relative or absolute path to the directory where the Reddit files
        are saved.
    :param keep: set or frozenset
        A set or frozenset of subreddit IDs to keep.  Any post in a subreddit
        NOT in this collection will be discarded.
    :param min_len: int
        Minimum length of a post, in number of words.  Posts shorter than
        this will be discarded.
    :param delete_orig: bool
        True to delete the original .bz2 file (from Archive.org) after processing,
        False to keep it.  Deleting it can save disk space.
    :param output_dir: str
        Directory to save the intermediate split files to.
    :return: 0 on success
    """

    # files = [f"{i[0]}\\{j}" for i in os.walk("reddit") for j in i[2]]
    files = list(recursive_scan("reddit"))

    with multiprocessing.Pool(processes=num_threads) as pool:
        worker = partial(subreddit_worker, keep=keep, min_len=min_len, delete_orig=delete_orig)
        res = list(tqdm(pool.imap(worker, files), desc="Splitting by Subreddit", total=len(files)))

    return 0

def concatenation_worker(prefix):
    """
    Takes all files in the By Subreddit folder that start with the given
    prefix and concatenates them together, deleting the original files
    as it finishes with them.  Concatenated files will be saved to
    input_dir/FINAL/

    :param prefix: str
        The subreddit name to concatenate.  This will be the first substring
        of the filename.
    :param input_dir: str
        The directory containing the by-subreddit files needing concatenation.
    :param delete_orig: bool
        True to delete the original .bz2 file (from splitting by subreddit)
        after processing, False to keep it.  Deleting it can save disk space.
    :return: 0 on success
    """
    files = [
        i.path
        for i in os.scandir("By Subreddit")
        if i.name.startswith(prefix)
    ]
    if files == []: return 0

    with bz2.open(f"By Subreddit/FINAL/{prefix}.txt.bz2", "wt", encoding="utf8") as OUT:
        for F in files:
            with bz2.open(F, "rt", encoding="utf8") as IN:
                for i in IN:
                    OUT.write(i)
            os.remove(F)
    return 0

def concatenate_subreddit_files(num_threads=1):
    """
    After splitting the raw data by subreddit, concatenate all of the sub-files
    together.  Files will be output to input_dir/FINAL/

    :param prefixes: array-like of strings
        File prefix strings to concatenate.  These should correspond to subreddit
        names.
    :param input_dir: str
        The directory containing the by-subreddit files needing concatenation.
    :return:
    """
    prefixes = set(
        i.name.split()[0]
        for i in os.scandir("By Subreddit/")
        if i.name.endswith(".txt.bz2")
    )

    with multiprocessing.Pool(processes=num_threads) as pool:
        worker=partial(concatenation_worker)
        res = list(tqdm(pool.imap(concatenation_worker, sorted(prefixes)), desc="Concatenating Files", total=len(prefixes)))

    return 0

def split_by_subreddit_worker_testing(line):
    J = json.loads(line)
    return (J["subreddit"], J["body"], len(preprocessing.process_string(J["body"])))

def split_by_subreddit_testing(keep=(), min_len=5, delete_orig=False, num_threads=1):
    keep = set(keep)

    with multiprocessing.Pool(num_threads) as P:
        # in_files = [f"{i[0]}\\{j}" for i in os.walk("reddit") for j in i[2]]
        in_files = list(recursive_scan("reddit"))
        out_files = {
            i:bz2.open(f"By Subreddit/FINAL/{i}.txt.bz2", "wt", encoding="utf8")
            for i in keep
        }
        res = P.imap(
            split_by_subreddit_worker_testing,
            (
                j
                for i in in_files
                for j in bz2.open(i, "rt", encoding="utf8")
             )
        )

        for i in tqdm(res):
            if i[2] < min_len or i[0] not in keep: continue
            out_files[i[0]].write(f"{repr(i[1])}\n")

    for i in out_files:
        out_files[i].close()