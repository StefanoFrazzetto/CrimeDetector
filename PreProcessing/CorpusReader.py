from typing import List

from Classification import Data, DatasetCategory, Dataset
from Classification.Data import DataLabel
from PreProcessing import CorpusName, CorpusLabeler
from Utils import Log, File, Assert


class CorpusReader(object):
    dataset: Dataset
    excluded: List[str]

    def __init__(self, corpus_name: CorpusName, dataset: Dataset = None):
        self.dataset = dataset if dataset is not None else Dataset()
        self.excluded = []
        self.data_labeler = CorpusLabeler.factory(corpus_name)

    def add_exclusion(self, string):
        self.excluded.append(string)

    def add_dir_to_dataset(self, directory, dataset_category: DatasetCategory = None, data_label: DataLabel = None):
        """
        Recursively add to the dataset the corpus contained in a directory.
        :param directory: the directory containing the corpus
        :param dataset_category: the category of the data (training or testing)
        :param data_label: 
        :return:
        """

        Log.info(f"# Adding directory '{directory}'... ", newline=False)

        files = File.get_dir_files_recursive(directory)
        Assert.not_empty(files, f"The directory {directory} is empty.")
        for filename in files:
            if [word for word in self.excluded if word not in filename]:
                file_content = File.read(filename)

                # Autodetect label from file using CorpusLabeler
                label = self.data_labeler.get_label(filename) if data_label is None else data_label
                content = File.read(file_content)
                data = Data(content, filename, label)

                self.dataset.put(data, dataset_category)

        Log.info("done.", timestamp=False)

    def add_file_to_dataset(self, filename, category: DatasetCategory = None, data_label: DataLabel = None):
        file_content = File.read(filename)

        email_body = File.read(file_content)
        label = self.data_labeler.get_label(filename) if data_label is None else data_label
        data = Data(email_body, filename, label)

        self.dataset.put(data, category)
