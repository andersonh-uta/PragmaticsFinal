from ast import literal_eval
import bz2
from collections import defaultdict
import re
import string
import os
import pickle
from pprint import pprint
import random

from gensim.corpora import Dictionary
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

RE_PUNCT = re.compile('([{}])+'.format(re.escape(string.punctuation)), re.UNICODE)
RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
RE_NUMERIC = re.compile(r"\b[0-9]+\b", re.UNICODE)
RE_NONALPHA = re.compile(r"\W", re.UNICODE)
RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)
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

def process_string(s, minsize=2):
    """
    Perform a simple Gensim-like preprocessing pipeline on a string of text.
    --Lowercase
    --Remove HTML tags
    --Remove punctuation
    --Replace all whitespaces with a single space
    --Remove numbers, unless they're part of words.

    :param s: string to process
    :return: the processed string
    """
    # Define custom pipeline--cuts down on the number of calls
    # to certain functions to boost speed.
    # This mirrors the Gensim preprocess_string method, but without
    # the stemming and with fewer checks.
    s = filter(
        lambda x: len(x) >= minsize
                  and x not in STOPWORDS,
        # Remove numbers
        RE_NUMERIC.sub(
            " ",
            # remove multiple whitespaces
            RE_WHITESPACE.sub(
                " ",
                # remove punctuation
                RE_PUNCT.sub(
                    " ",
                    # remove HTML tags
                    RE_TAGS.sub(
                        "",
                        s.lower()
                    )
                )
            )
        ).split()
    )

    return set(s)

def rank_tokens(vocabfile, sentimentfile):
    """
    Generate a pandas dataframe and a .csv file of all the input vocabulary
    terms, their document frequencies, and their associated weights.  Weights
    are sqrt(document_frequency) * sentiment.

    :param vocabfile: str
        Path to a saved Gensim dictionary file.
    :param sentimentfile: str
        Path to a .p file storing a dictionary in the form {token:sentiment}.
    :return: pandas DataFrame object
    """

    if not os.path.isdir("Outputs"): os.mkdir("Outputs")
    title = vocabfile.split("_")[0].split("/")[-1]
    sent = pickle.load(open(sentimentfile, "rb"))
    vocab = Dictionary.load(vocabfile)
    plotting = pd.DataFrame(
        [{"Token":i,
         "Sentiment":sent[i],
          "Sentiment Magnitude":abs(sent[i]),
         "Document Frequency":vocab.dfs[vocab.token2id[i]]} for i in sent]
    )
    plotting["Weight"] = np.abs(plotting["Sentiment"] * np.log(plotting["Document Frequency"]))
    plotting = plotting.sort_values("Weight", ascending=False)
    plotting.iloc[:20].to_csv(f"Outputs/{title}.csv", index=False, float_format="%.4f")
    return plotting

def make_plot(vocabfile, sentimentfile):
    """
    Scatter plot tokens by log document frequency and sentiment.
    :param vocabfile: .vocab file for a subreddit, containing a Gensim Dictionary() object
        trained on the subreddit.
    :param sentimentfile: a .p file, containing a Python dict() object with word-sentiment pairs.
    :return: 0 on success
    """
    title = vocabfile.split("_")[0].split("/")[-1]
    sent = pickle.load(open(sentimentfile, "rb"))
    vocab = Dictionary.load(vocabfile)
    plotting = np.array([[i, sent[i], vocab.dfs[vocab.token2id[i]]] for i in sent])

    x = plotting[:,1].astype(np.float)
    y = plotting[:,2].astype(int)

    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=1, alpha=0.15)
    plt.yscale("log")
    plt.title(f"r/{title}")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Document Frequencies (logarithmic)")
    plt.xlim([-3.2, 3.2])
    plt.ylim([6, 10**6.5])
    plt.savefig(f"Outputs/{title}.png", dpi=300)
    # plt.show()
    plt.close()

    return 0

def example_puller(infile, words, n_examples=10):
    """
    Randomly sample some examples of the given words from the specified subreddit.
    Gathers the first n_examples posts per word, then for each successive one,
    randomly replaces one of the existing ones

    :param subreddit: str
        path to the raw subreddit file
    :param words: array-like
        A list of words to pick from
    :param n_examples: int
        How many posts to pull for each word.
    :return:
    """
    rownums = defaultdict(list)
    results = {i:[] for i in words}
    # get the row numbers for the processed files.  Use this later to pull
    # the raw posts for examination.
    with bz2.open(infile, "rt", encoding="utf8") as F:
        counter = 0
        for i in tqdm(F, desc=f"{infile} First Pass"):
            s = set(i.split())
            for W in words:
                if W in s: rownums[counter].append(W)
            counter += 1
    rownums = dict(rownums)

    with bz2.open(infile, "rt", encoding="utf8") as F:
        counter = -1
        for i in tqdm(F, desc=f"{infile} Second Pass"):
            counter += 1
            if counter not in rownums: continue
            s = literal_eval(i)
            for W in rownums[counter]:
                results[W].append(s)

    results = {i:random.choices(results[i], k=min(n_examples, len(results[i]))) for i in results}
    out = infile.replace("\\", "/").split("/")[-1].split("_")[0]
    pickle.dump(results, open(f"Outputs/{out} sample posts.p", "wb"))
    with open(f"Outputs/{out} sample posts.txt", "w", encoding="utf8") as F:
        for i in results:
            for j in results[i]:
                F.write("""{}\n\t{}\n\n""".format(i, j.replace('\n', '\n\n\t')))

if __name__ == "__main__":
    files = [
        i.path
        for i in os.scandir("Processed Files/MODELS/SCORES")
        if i.name.endswith(".p")
    ]
    for i in files:
        make_plot(i.replace("\\", "/").replace("SCORES/", "").replace("_scores.p", ".vocab"), i)
    dfs = {}
    for i in files:
        vocabfile = i.replace("\\", "/").replace("SCORES/", "").replace("_scores.p", ".vocab")
        dfs[i.replace("\\", "/").split("/")[-1].split("_")[0]] = rank_tokens(vocabfile, i)
    with pd.ExcelWriter("Outputs/Ranked Tokens.xlsx", engine="xlsxwriter") as W:
        for i in dfs:
            dfs[i].to_excel(W, sheet_name=i, index=False)

    # latex_tables()
    #
    files = {
        # "2007scape":"noob poor kid free scammed trolling immature minor easyscape inb4 giving new\_players huge logic perfectproper properly math defend software".split(),
        # "linux":"linux work desktop gnome kde mint bloated windows unity mac osx xp vista windows\_xp win7 widely\_available".split(),
        # "runescape":"gwd poor pvm gf soloing pvming gg xp slayer jagex combat runescape game boss bosses bossing rich corp bandos rares million millions".split(),
        # "darksouls":"connection lost\_izalith ugly dark\_souls damage matchmaking help work giant\_skeletons ng boss sword armor fair chosen\_undead demon\_ruins duke\_archives bells titanite\_slabs invaded ease titanite\_slab worthwhile".split(),
        # "4chan":"good\_job freedom muh pay buy poor rights suicide black reddit account fucking money bastard stupid hate father rich got\_banned abortion".split(),
        # "politics":"hitler koch\_brothers nazis richest richer genocide billionaires trickle upper\_class trickle\_economics government poor rich middle\_class jew book backs job\_creators sympathy".split(),
        # "atheism":"violence rich moral morality dangerous genocide slavery oppression insane poor rights freedom food constitution africa guilty separation\_church\_state amendment equal\_rights".split(),
        # "pcmasterrace":"pcgiveaway framerates\_high low\_temperatures temps\_low dust bugs monitor overclock oc gigabyte msi 60hz asus 144hz dell psu".split(),
    }
    # for i in files:
    #     example_puller(i, files[i], 25)
        # break


