import glob
import os

import numpy


class File(object):

    @staticmethod
    def read(filename, mode="r", encoding="utf-8", errors='ignore'):
        """Load a text file."""
        with open(filename, mode=mode, encoding=encoding, errors=errors) as f:
            return f.read()

    @staticmethod
    def get_files_count(directory: str):
        return len([name for name in os.listdir(directory) if os.path.isfile(name)])

    @staticmethod
    def create_directory(directory: str):
        if not File.directory_exists(directory):
            os.makedirs(directory)

    @staticmethod
    def create_file(filename: str):
        file = open(filename, 'a')
        file.close()

    @staticmethod
    def write_file(filename: str, content: str, mode: str = "w"):
        if not File.file_exists(filename):
            File.create_file(filename)

        file = open(filename, mode)
        file.write(content)
        file.close()

    @staticmethod
    def directory_exists(directory):
        return os.path.exists(directory)

    @staticmethod
    def file_exists(filename):
        return os.path.exists(filename)

    @staticmethod
    def is_file(path):
        return os.path.isfile(path)

    @staticmethod
    def is_dir(path):
        return os.path.isdir(path)

    @staticmethod
    def get_files_count(path):
        return sum([len(files) for r, d, files in os.walk(path)])

    @staticmethod
    def length(file_name):
        return sum(1 for line in open(file_name) if line.rstrip())

    @staticmethod
    def get_dir_iterator(directory, extension='*', recursive=True):
        File.directory_exists(directory)
        return glob.iglob(directory + f"{'/**' if recursive else None}/{extension}", recursive=recursive)

    @staticmethod
    def get_dir_files_recursive(directory, extension='*') -> []:
        File.directory_exists(directory)
        return [y for x in os.walk(directory) for y in glob.glob(os.path.join(x[0], f"*.{extension}"))]

    @staticmethod
    def get_randomized_files_list(path):
        f_list = File.get_dir_files_recursive(path)
        numpy.random.shuffle(f_list)
        return f_list
