IMPORTANT NOTE: Final test run is not yet complete as of uploading this.  Full stability and functionality is not guaranteed.

This code is the code used for my final project in Pragmatics, studying whether the sentiment of words used in a given subreddit (per Hamilton et al. 2016) encodes any meaningful notion of "community values."

Since I've already finished with this project, I do not intend to provide much or any support for/maintenance of this code.  However, you are free to do what you wish with it, subject to the GNU GPL v2 license terms.

# REQUIREMENTS
Python 3.6+, 64-bit  (all command line commands below assume this is the only version on your system--if you have multiple Python versions installed, you *must* use the appropriate command for Python 3.6 to run these commands)

Some of the code in this project uses Cython to accelerate some matrix calculations.  If you run this code without Cython, it will be considerably slower.  However, Cython needs to be installed (via pip), AND a C compiler needs to be installed and configured for your system.  Please refer to the Cython documentation for more information: http://www.cython.org

Other required Python libraries can be installed from the requirements.txt file included in this project.

# INSTALLATION
1. Download or clone the repository to your local machine.
2. RECOMMENDED: Create a Python virtual environment in the folder and activate it before continuing.
3. Install the requirements from the requirements.txt file: `pip install -r requirements.txt` (note: requires Python 3!)
4. Download the Reddit Public Comments Corpus from Archive.org (https://archive.org/details/2015_reddit_comments_corpus), or any subset of it.  Place these files in the reddit/ directory (you can nest them in sub-folders if you want).  Do not unzip them!  Leave them as .bz2 files--the Python code expects .bz2 archives and will crash otherwise, and unzipping the whole dataset would take up around a terabyte of space.

# BUILDING THE CYTHON CODE
If you're using a virtual environment, activate it.  From the command line, in the project folder, run the setup.py file: `python setup.py build_ext --inplace`

This will build the Cython code, creating a .pyd file that will run considerably faster for some of the matrix calculations than a pure python implementation.  If you have trouble building the Cython code, there is a pure Python implementation of this same functionality that the program will fall back on.

# RUNNING
Simply execute the FinalPaper.py file using a 64-bit Python 3.6 or later interpreter.  You may want to change some of the config file options (see next section).

# CONFIG FILE OPTIONS
The options for the program are stored in the `reddit.conf` file.  All options should  be provided in the form `name=value`.

## Global
These are global options that will apply across numerous sections of the program.

`NUM_THREADS`: this is the number of processing threads you want to use.  More threads --> runs faster, but also consumes more memory (on machines with low memory, this can cause errors and crash the program).  In general, set this to, at most, one less than the number of _physical_ CPU cores in your system, unless you're running into MemoryErrors or serious performance issues, in which case set this to 1.

`window_size`: the symmetric window size to use when computing the PPMI transform for the text.  This is also used to filter posts by word count at several steps.


## CorpusCleaning
These options apply to the corpus cleaning stage, i.e. collecting posts, splitting by subreddit, and doing preliminary filtering.

`count_posts`: must be True or False.  If True, the program will do a pass over the raw corpus files and count the total number of posts and record this in a .csv file.  (This step will take several hours and is not necessary--the Subreddit Post Counts.csv file already packaged here contains the results of just such a run).

`delete_original_archives`: True or False.  If True, the program will delete the original .bz2 archives as it processes them.  this can save space if you're processing a very large number of subreddits, but note that you'll have to re-download the original archives if you want to make changes to, e.g., the window size or other parameters in this section.

`keep`: A list of subreddit names, which are capitalization-sensitive, to keep.  The program will collect posts from these subreddits into individual files for quicker processing later, and will ignore any other subreddits.  The list can be either a whitespace-separated list (e.g.: AskReddit funny pics gaming), or each subreddit name can be on a new line, but each new line must be indented (see the default reddit.conf for an example of this formatting).


## Preprocessing
These options apply to the preprocessing stage, where text is cleaned and filtered more aggressively before final processing.

`threshold`: the threshold argument in Gensim's Phrases() object (consult Gensim documentation for more information; 10 is the default value).

`min_count`: the min_count argument in Gensim's Phrases() object.  Multi-word phrases must occur at least this many _total_ times to be considered a valid phrase.

`delete_unprocessed_archives`: True or False.  If True, the files split by subreddit in the Corpus Cleaning steps are deleted once they've been processed, leaving only the processed and raw text by subreddit.

`num_phrasing_rounds`: How many passes over each subreddit to do in order to find phrases.  Each phrasing pass allows two adjacent tokens to be joined, so multiple passes allows the discovery of longer multi-word phrases.  However, passes are fairly time consuming for larger subreddits.


## SocialSent
These are options for the SocialSent analysis.  Many of the options are parameters in Hamilton et al's original code.

`smoothing`: Exponential smoothing parameter for the PPMI transformation.  0.75 is a good value.

`filter_extremes`: True or False.  If True, a vocabulary count constructed with a Gensim Dictionary() object is used, and its .filter_extremes() method is called to filter our words with extreme (low and high) document frequencies.  This may reduce accuracy of the models, but may also substantially decrease runtime.

`no_below`: If filter_extremes is True, any tokens occurring in fewer than `no_below` documents are ignored.

`no_above`: If filter_extremes is True, any tokens occurring in more than `no_above` documents (as a proportion of the total corpus) are ignored.

`pos_seeds`: a space-separated list of words to use as the positive-sentiment seeds.  The values in the default config file are as in Hamilton et al.

`pos_seeds`: a space-separated list of words to use as the negative-sentiment seeds.  The values in the default config file are as in Hamilton et al.

`tol`: tolerance value for the SocialSent runs.  Once the scores change by less than this threshold, the model is determined to have reached convergence, and thus, the sentiment scores are finalized.

`beta`: The beta parameter in the SocialSent model.

`nn`: The number of nearest neighbors to use when simulating the random walk in the SocialSent algorithm.  10 is the value used in the original paper.

`maxiter`: Maximum number of iterations for the SocialSent matrix multiplication.  After this number of iterations, even if tolerance has not been reached, the model terminates and the sentiment scores are finalized.


## ExamplePosts
These options control the selection of example posts containing terms of interest.

`pull_example_posts`: True or False.  If True, the above processing steps are assumed to have already been run, and thus will not be run again.  The only processing will be pulling example posts from the corpus, containing user-specified words from user-specified subreddits.  If False, the above processing steps are performed, and example posts are NOT pulled.  (you'd need to run the above steps to determine the words you want to pull usage examples of, anyways).

`n_examples`: how many example posts to pull for each word.  Posts are randomly sampled from the corpus.

You should also provide a list of subdreddit-word combinations here in the form of `subreddit=word1 word2 word3` etc for the words you want to examine in that subreddit.
