import codecs
import glob
import os

import numpy


def read(filename, mode="r", encoding="utf-8", errors='ignore'):
    """Load a text file."""
    with codecs.open(filename, mode=mode, encoding=encoding, errors=errors) as f:
        return f.read()


def directory_exists(directory):
    if not os.path.exists:
        raise Exception(f"The directory {directory} doesn't exist.")


def is_file(path):
    return os.path.isfile(path)


def is_dir(path):
    return os.path.isdir(path)


def get_files_count(path):
    return sum([len(files) for r, d, files in os.walk(path)])


def get_dir_iterator(directory, extension='*', recursive=True):
    directory_exists(directory)
    return glob.iglob(directory + f"{'/**' if recursive else None}/{extension}", recursive=recursive)


def get_dir_files_recursive(directory, extension='*') -> []:
    directory_exists(directory)
    return [y for x in os.walk(directory) for y in glob.glob(os.path.join(x[0], f"*.{extension}"))]


def get_randomized_files_list(path):
    f_list = get_dir_files_recursive(path)
    numpy.random.shuffle(f_list)
    return f_list
