from typing import List

from Data import Data, DatasetCategory, Dataset
from Data import DataLabel
from Utils import Log, File, Assert, Text


class CorpusReader(object):
    dataset: Dataset
    excluded: List[str]

    def __init__(self, dataset: Dataset = None):
        self.dataset = dataset if dataset is not None else Dataset()
        self.excluded = []

    def add_exclusion(self, string):
        self.excluded.append(string)

    def add_dir_to_dataset(self,
                           directory, dataset_category: DatasetCategory = None, data_label: DataLabel = None,
                           load_ratio: float = 1):
        """
        Recursively add to the dataset the corpus contained in a source_directory.
        :param load_ratio:
        :param directory: the source_directory containing the corpus
        :param dataset_category: the category of the data (training or testing)
        :param data_label: 
        :return:
        """

        Log.info(f"# Adding source_directory '{directory}'... ", newline=False)
        files = File.get_dir_files_recursive(directory)
        Assert.not_empty(files, f"The source_directory {directory} is empty.")
        no_of_files = len(files)

        for filename in files:
            if self.__can_add_more(no_of_files, load_ratio):
                if len(self.excluded) == 0 or [word for word in self.excluded if word not in filename]:
                    self.add_file_to_dataset(filename, dataset_category, data_label)

        Log.info("done.", timestamp=False)

    def add_file_to_dataset(self, filename, dataset_category: DatasetCategory = None, data_label: DataLabel = None):
        """
        Add a single file from a corpus to the dataset.
        :param filename:
        :param dataset_category:
        :param data_label:
        :return:
        """

        content = File.read(filename)
        clean = Text.clean(content)

        if len(clean) < 3:
            return

        label = data_label
        data = Data(content, label)

        self.dataset.put(data, dataset_category)

    def __can_add_more(self, no_of_files: int, load_ratio: float):
        """
        Check if the current load ratio is lower than the defined.
        :param no_of_files:
        :param load_ratio:
        :return:
        """
        current_ratio = self.dataset.get_total_size() / no_of_files
        return True if current_ratio < load_ratio else False
