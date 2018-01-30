import os

def recursive_walk(dirname):
    """
    Recursive version of os.walk, which returns all filenames in a directory.
    :param dirname: the path to the directory to be walked.
    :return: a list of filenames
    """
    for i in os.scandir(dirname):
        if os.path.isfile(i.path):
            yield i.path
        elif os.path.isdir(i.path):
            yield from recursive_walk(i.path)


print(list(recursive_walk("reddit")))