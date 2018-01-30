"""
Python program to run all of the Reddit processing steps in one go.

Please see requirements.txt for all required libraries.
"""

import configparser
import re
from time import time
import os

import pandas as pd
from tqdm import tqdm

from src import corpus_cleaning
from src import preprocessing
from src import tokens_and_plotting
from src import vectorization_and_socialsent as socialsent


def collect_posts(config):
    """
    Run the corpus cleaning.  This takes a pass over the full Reddit corpus,
    does some very basic filtering of posts by length, and saves them out
    to new files with one subreddit's posts per file.

    :param config: a CongifParser() object run on the config file.
    :return: 0 on success
    """
    C = config["CorpusCleaning"]

    num_threads = int(config["Global"]["num_threads"])
    min_len = int(config["Global"]["window_size"])
    keep = re.split(r"\W+", C["keep"])
    delete_orig =\
        True if C["delete_original_archives"].lower() == "true" else False

    corpus_cleaning.split_by_subreddit(
        keep=keep,
        min_len=min_len,
        delete_orig=delete_orig,
        num_threads=num_threads
    )
    corpus_cleaning.concatenate_subreddit_files(
        num_threads=num_threads
    )

    return 0

def preprocess_text(config):
    """
    Run the NLP preprocessing pipeline on the split corpora.

    :param config: a CongifParser() object run on the config file.
    :return: 0 on success
    """
    C = config["Preprocessing"]
    # Grab the config options.
    min_len = int(config["Global"]["window_size"])
    num_threads = int(config["Global"]["num_threads"])
    delete_orig = True if  C["delete_unprocessed_archives"].lower() == "true" else False
    threshold = float(C["threshold"])
    min_count = int(C["min_count"])
    num_phrasing_rounds = int(C["num_phrasing_rounds"])


    preprocessing.process_and_save(
        threshold=threshold,
        min_count=min_count,
        min_len=min_len,
        delete_orig=delete_orig,
        num_phrasing_rounds=num_phrasing_rounds,
        num_threads=num_threads
    )

    preprocessing.word_counter(
        num_threads=num_threads
    )
    return 0

def socialsent_modeling(config):
    """
    Do the PPMI transform and SocialSent modeling.

    :param config:  a CongifParser() object run on the config file.
    :return: 0 on success
    """
    C = config["SocialSent"]
    window = int(config["Global"]["window_size"])
    smoothing = float(C["smoothing"])
    pos_seeds = re.split(r"\W+", C["pos_seeds"].lower())
    neg_seeds = re.split(r"\W+", C["neg_seeds"].lower())
    no_below = int(C["no_below"])
    no_above = float(C["no_above"])
    filter_extremes = True if C["filter_extremes"].lower() == "true" else False
    tol = float(C["tol"])
    beta = float(C["beta"])
    nn = int(C["nn"])
    maxiter = int(C["maxiter"])

    # Run PPMI vectorization and SVD embedding
    socialsent.make_vectors(
        window=window,
        smoothing=smoothing
    )

    socialsent.main(
        pos_seeds,
        neg_seeds,
        no_below=no_below,
        no_above=no_above,
        filter_extremes=filter_extremes,
        tol=tol,
        beta=beta,
        nn=nn,
        maxiter=maxiter,
        n_threads=int(config["Global"]["NUM_THREADS"])
    )

    # # Pull the list of vocab/vector files for SentPropping
    # files = [
    #     i.path # ignore .vocab suffix
    #     for i in os.scandir("Processed Files/MODELS")
    #     if i.path.endswith(".npy")
    #     and not os.path.isfile(i.path.replace(".npy", "_scores.p").replace("\\", "/").replace("MODELS/", "MODELS/SCORES/"))
    # ]
    # files = sorted(files, key=os.path.getsize)
    #
    # for i in files:
    #     socialsent.make_sentprop(
    #         vecfile=i,
    #         vocabfile=i.replace(".npy", ".vocab"),
    #         pos_seeds=pos_seeds,
    #         neg_seeds=neg_seeds,
    #         no_below=no_below,
    #         no_above=no_above,
    #         filter_extremes=filter_extremes,
    #         tol=tol,
    #         beta=beta,
    #         nn=nn,
    #         maxiter=maxiter,
    #         fname=i.split("\\")[-1].split("_")[0]
    #     )

    return 0

def rank_tokens_and_generate_plots(config):
    """
    Makes plots of tokken_frequency-score values and ranked lists
    of token scores for all subreddits queried.
    :param config: parsed config file
    :return: 0 on success
    """
    dfs = {}
    for i in tqdm(list(os.scandir("Processed Files/MODELS/")), desc="Ranking tokens and generating plots"):
        if os.path.isdir(i.path): continue
        if i.name.endswith(".npy"): continue
        dfs[i.name] = tokens_and_plotting.rank_tokens(
            i.path,
            i.path.replace("/MODELS/", "/MODELS/SCORES/").replace(".vocab", "_scores.p")
        )
        tokens_and_plotting.make_plot(
            i.path,
            i.path.replace("/MODELS/", "/MODELS/SCORES/").replace(".vocab", "_scores.p")
        )

    with pd.ExcelWriter("Outputs/Ranked Tokens.xlsx", engine="xlsxwriter") as W:
        for i in dfs:
            sheet = i.replace("_processed.vocab", "")
            if len(sheet) > 31: sheete = sheet[:31]
            dfs[i].to_excel(W, sheet_name=sheet, index=False)

    return 0

def pull_example_posts(config):
    """
    Retrieves example posts contianing words as specified in the reddit.conf file.
    :param config: parsed config file
    :return: 0 on success
    """
    C = config["ExamplePosts"]
    for i in tqdm(C, desc="Pulling example posts..."):
        if i == "n_examples": continue
        tokens_and_plotting.example_puller(
            f"Processed Files/{i}_raw.txt.bz2",
            re.split(r"\W+", C[i])
        )

    return 0

if __name__ == "__main__":
    # read the config file
    config = configparser.ConfigParser()
    config.read_file(open("reddit.conf"))

    # create all necessary folders
    if not os.path.isdir("By Subreddit/FINAL"):
        os.makedirs("By Subreddit/FINAL")
    if not os.path.isdir("Processed Files/MODELS/SCORES"):
        os.makedirs("Processed Files/MODELS/SCORES")
    if not os.path.isdir("Outputs"):
        os.mkdir("Outputs")

    start = time()
    # only run main modeling if NOT pulling example posts
    if config["ExamplePosts"]["pull_example_posts"].lower() == "false":
        # if config["CorpusCleaning"]["count_posts"].lower() == "true":
        #     corpus_cleaning.count_posts(num_threads=int(config["Global"]["num_threads"]))
        # collect_posts(config)
        # preprocess_text(config)
        socialsent_modeling(config)
        rank_tokens_and_generate_plots(config)

    elif config["ExamplePosts"]["pull_example_posts"].lower() == "true":
        pull_example_posts(config)

    end = time()
    dur = end - start
    time_taken = f"{dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds"
    print("Time taken: ", time_taken)