"""
Functions for basic NLP preprocessing on the Reddit corps.
"""

from ast import literal_eval
import bz2
from functools import partial
import multiprocessing
import os
import string
from shutil import move
import re

import pandas as pd
from gensim.models.phrases import Phrases, Phraser
from tqdm import tqdm

RE_PUNCT = re.compile("([{}])+".format(re.escape(string.punctuation)))
RE_TAGS = re.compile(r"<([^>]+)>")
RE_NUMERIC = re.compile(r"\b[0-9]+\b")
RE_NONALPHA = re.compile(r"[^0-9A-z ]+")
RE_WHITESPACE = re.compile(r"(\s)+")
RE_URL = re.compile(r"[http[s]?://]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

STOPWORDS = """
a about above across after afterwards again against all almost alone along already also although always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at back be
became because become becomes becoming been before beforehand behind being below beside besides between beyond bill both bottom but by call can
cannot cant co computer con could couldnt cry de describe
detail did didn do does doesn doing don done down due during
each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except few fifteen
fify fill find fire first five for former formerly forty found four from front full further get give go
had has hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how however hundred i ie
if in inc indeed interest into is it its itself keep last latter latterly least less ltd
just
kg km
made make many may me meanwhile might mill mine more moreover most mostly move much must my myself name namely
neither never nevertheless next nine no nobody none noone nor not nothing now nowhere of off
often on once one only onto or other others otherwise our ours ourselves out over own part per
perhaps please put rather re
quite
rather really regarding
same say see seem seemed seeming seems serious several she should show side since sincere six sixty so some somehow someone something sometime sometimes somewhere still such system take ten
than that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin third this those though three through throughout thru thus to together too top toward towards twelve twenty two un under
until up unless upon us used using
various very very via
was we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would yet you
your yours yourself yourselves
"""
STOPWORDS = frozenset(w for w in STOPWORDS.split() if w)

def file_yielder(infile):
    """
    Simple generator to do low-memory file reading, one row at a time.
    Used as a helper function elsewhere in this file to keep indentation
    down and code cleaner.

    :param infile: str
        Path to file.
    :yield: bytes
        lines from the file
    """
    with bz2.open(infile, "rt", encoding="utf8") as F:
        yield from F

def process_string(s, minsize=2):
    """
    Perform a simple Gensim-like preprocessing pipeline on a string of text.
    --Lowercase
    --Remove HTML tags
    --Remove punctuation
    --Replace all whitespaces with a single space
    --Remove numbers, unless they're part of words.

    :param s: str
        String to process
    :return: str
        The processed string
    """
    # Define custom pipeline--cuts down on the number of calls
    # to certain functions to boost speed.
    # This mirrors the Gensim preprocess_string method, but without
    # the stemming and with fewer checks.
    s = filter(
        lambda x: len(x) >= minsize
                  and x not in STOPWORDS,
        RE_NUMERIC.sub(
            " ",
            RE_WHITESPACE.sub(
                " ",
                RE_NONALPHA.sub(
                    " ",
                    RE_TAGS.sub(
                        "",
                        RE_URL.sub(
                            " ",
                            s.lower()
                        )
                    )
                )
            )
        ).split()
    )

    return tuple(s)

def process_file(infile, min_len=5):
    """
    Read a single-subreddit .bz2 file.  Yield a generator over processed
    texts, applying the gensim-like process_string() to each non-empty string.

    :param infile: str
        path to .txt.bz2 file to process
    :param min_len: int
        minimum number of words in a post to keep it.  Posts
        shorter than min_len words AFTER PROCESSING are discarded.
    :return: yields documents
    """
    yield from filter(
        lambda x: len(x.split()) >= min_len,
        map(
            lambda x: process_string(x, min_len=min_len),
            filter(lambda x: x.strip(), file_yielder(infile))
        )
    )

def process_and_save_worker(
        infile,
        threshold=10,
        min_count=50,
        min_len=5,
        delete_orig=False,
        num_phrasing_rounds=2,
):
    """
    Single threaded worker for the text preprocessing and saving of files.
    Called by process_and_save().

    :param infile: str
        Path to the .txt.bz2 file to be processed.
    :param threshold: float
        The threshold kwarg of Gensim's Phrases() object.
    :param min_count: int
        The min_count kwarg of Gensim's Phrases() object.
    :param min_len: int
        minimum number of words in a post to keep it.  Posts
        shorter than min_len words AFTER PROCESSING are discarded.
    :param delete_orig: bool
        True to delete the original .bz2 file (from Archive.org) after processing,
        False to keep it.  Deleting it can save disk space.
    :return:
    """
    # Grab two temporary files, so we can shunt data between them,
    # processing it while it's in memory.
    suff = infile.replace("\\", "/").split("/")[-1][:-8]
    S = f"{suff:<20s}"
    raw_out = infile\
        .replace("\\", "/") \
        .replace(".txt.bz2", "_raw.txt.bz2") \
        .replace("By Subreddit/FINAL/", "Processed Files/")
    processed_out = infile \
        .replace("\\", "/") \
        .replace(".txt.bz2", "_processed.txt.bz2") \
        .replace("By Subreddit/FINAL/", "Processed Files/")
    working_file = processed_out.replace("_processed.txt.bz2", "_working.txt.bz2")

    # Create a total count variable that we'll use to update tqdm appropriately.
    total = 0
    with bz2.open(infile, "rt", encoding="utf8") as I, \
            bz2.open(raw_out, "wt", encoding="utf8") as R, \
            bz2.open(processed_out, "wt", encoding="utf8") as P:
        for i in tqdm(I, desc=f"{S}: Preprocessing", mininterval=5, position=1):
            # Preprocess, and skip if length is too low.
            text = process_string(literal_eval(i))
            if len(text) < min_len: continue
            text = " ".join(text)
            # write repr() for raw files to ensure one line per post;
            # write the ID as a fixed length string.
            R.write(f"{i.strip()}\n")
            P.write(f"{text}\n")
            total += 1

    # Stream the processed files through a Gensim Phrases() object.
    # If there are any phrases to be found, stream through a Phraser()
    # object and into a temp file.  Then overwrite the original
    # processed file and repeat.
    for i in range(num_phrasing_rounds):
        with bz2.open(processed_out, "rt", encoding="utf8") as IN, \
                bz2.open(working_file, "wt", encoding="utf8") as OUT:
            p = Phrases(
                (i.strip().split()
                 for i in tqdm(
                    IN,
                    total=total,
                    desc=f"{S} Phrase-finding {i+1}",
                    mininterval=5,
                    position=1
                )),
                threshold=threshold,
                min_count=min_count,
                # for some reason I get errors if the delimiter isn't a bytestring
                delimiter=b'_'
            )
            IN.seek(0)
            # See if there were any phrases found.  If not, abort phrasing early.
            try:
                next(p.export_phrases(i.strip().split() for i in IN))
            except StopIteration:
                break
            pp = Phraser(p)
            IN.seek(0)
            for i in tqdm(IN, total=total, desc=f"{S} Applying phraser 1", mininterval=5, position=1):
                OUT.write(f"{' '.join(list(pp[i.strip().split()]))}\n")
        move(working_file, processed_out)

    # Now, do a final pass to filter posts by length again.  This second pass
    # is because the length of a file may have changed considerably after
    # processing.  As before, stream to a temporary working file, then
    # overwrite the original when done.
    raw_working = raw_out.replace(".txt.bz2", "_working.txt.bz2")
    processed_working = processed_out.replace(".txt.bz2", "_working.txt.bz2")
    with bz2.open(raw_out, "rt", encoding="utf8") as RAW_IN, \
        bz2.open(processed_out, "rt", encoding="utf8") as PROC_IN, \
        bz2.open(raw_working, "wt", encoding="utf8") as RAW_OUT,  \
        bz2.open(processed_working, "wt", encoding="utf8") as PROC_OUT:

        for i in zip(PROC_IN, RAW_IN):
            if len(i[0].split()) >= min_len:
                PROC_OUT.write(i[0])
                RAW_OUT.write(i[1])

    move(processed_working, processed_out)
    move(raw_working, raw_out)


    if delete_orig == True:
        os.remove(infile)

    return 0

def process_and_save(
        threshold=10,
        min_count=50,
        min_len=5,
        delete_orig=False,
        num_phrasing_rounds=2,
        num_threads=1,
):
    """
    Run the processing steps on the provided file, then save
    the results.  NOTE: for memory use reasons, this will
    avoid reading the whole files into memory; it will read
    off disk and write back to disk at each step.

    Dumps the raw text directly to a _raw.txt.bz2 file in the Processed Files/
    directory.  Them dumps the text to process to a temporary file.

    :param infile: str
        Path to the .txt.bz2 file to be processed.
    :param threshold: float
        The threshold kwarg of Gensim's Phrases() object.
    :param min_count: int
        The min_count kwarg of Gensim's Phrases() object.
    :param min_len: int
        minimum number of words in a post to keep it.  Posts
        shorter than min_len words AFTER PROCESSING are discarded.
    :param delete_orig: bool
        True to delete the original .bz2 file (from Archive.org) after processing,
        False to keep it.  Deleting it can save disk space.
    :return:
    """
    # Grab the list of files to preprocess.
    files = [i.path for i in os.scandir("By Subreddit/FINAL")]
    files = sorted(files, key=os.path.getsize)

    worker = partial(
        process_and_save_worker,
        threshold=threshold,
        min_count=min_count,
        min_len=min_len,
        delete_orig=delete_orig,
        num_phrasing_rounds=num_phrasing_rounds,
    )
    with multiprocessing.Pool(processes=num_threads) as pool:
        res = list(tqdm(pool.imap(worker, files), total=len(files)))

    return 0

def word_counter_worker(infile):
    """
    Simple function to count the number of words and posts in an input file.
    Called by word_counter().

    :param infile: str
        Path to a .bz2 text file
    :return: dict
        Formatted as:
        {posts:n_posts, raw_words:n_raw_words, processed_words:n_processed_words}
    """
    name = infile \
        .replace("_raw.txt.bz2", "") \
        .replace("_processed.txt.bz2", "") \
        .rsplit("\\", maxsplit=1)[1]#.rsplit("_", maxsplit=1)[0]
    raw = infile.replace("_processed.txt.bz2", "_raw.txt.bz2")
    processed = infile.replace("_raw.txt.bz2", "_processed.txt.bz2")
    counts = {
        "File":name,
        "Number of Posts":0,
        "Number of Raw Words":0,
        "Number of Processed Words":0
    }

    with bz2.open(raw, "rt", encoding="utf8") as RAW:
        for i in tqdm(RAW, position=1, desc=f"{name} raw", mininterval=10):
            counts["Number of Raw Words"] += len(i.split())
    with bz2.open(processed, "rt", encoding="utf8") as PRO:
        for i in tqdm(PRO, position=1, desc=f"{name} pro", mininterval=10):
            counts["Number of Posts"] += 1
            counts["Number of Processed Words"] += len(i.split())

    return counts

def word_counter(num_threads=1):
    """
    Multithreaded function to count the number of words per file.

    :param num_threads: int
        Number of threads to use for word counting.
    :return: 0 on success
    """
    # Now, count words in each subreddit
    files = [
        i.path
        for i in os.scandir("Processed Files")
        if i.path.endswith("_raw.txt.bz2")
    ]
    with multiprocessing.Pool(processes=num_threads) as pool:
        res = list(tqdm(pool.imap(word_counter_worker, files), total=len(files)))
    df = pd.DataFrame(res)
    df.to_csv("Word Counts by Subreddit.csv")

    return 0