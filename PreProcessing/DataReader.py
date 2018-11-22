from typing import List

from Classification import Data, DatasetCategory, Dataset
from Utils.Email import Email
from PreProcessing import CorpusName, CorpusLabeler
from Classification.Data import DataLabel
from Utils import Log, File, Assert


class DataReader(object):
    dataset: Dataset
    excluded: List[str]

    def __init__(self, corpus_name: CorpusName, dataset: Dataset = None):
        self.dataset = dataset if dataset is not None else Dataset()
        self.excluded = []
        self.data_labeler = CorpusLabeler.factory(corpus_name)

    def add_exclusion(self, string):
        self.excluded.append(string)

    @staticmethod
    def email_body_from_content(content):
        return Email.from_content(content)

    def add_dir_to_dataset(self, directory, category: DatasetCategory = None, data_label: DataLabel = None):
        Log.info(f"# Adding directory '{directory}'... ", newline=False)

        files = File.get_dir_files_recursive(directory)
        Assert.not_empty(files, f"The directory {directory} is empty.")
        for filename in files:
            if [word for word in self.excluded if word not in filename]:
                file_content = File.read(filename)

                # Autodetect label from filename if not manually assigned
                label = self.data_labeler.get_label(filename) if data_label is None else data_label
                email_body = self.email_body_from_content(file_content)
                data = Data(email_body, filename, label)

                self.dataset.put(data, category)

        Log.info("done.", timestamp=False)

    def add_file_to_dataset(self, filename, category: DatasetCategory = None, data_label: DataLabel = None):
        file_content = File.read(filename)

        email_body = self.email_body_from_content(file_content)
        label = self.data_labeler.get_label(filename) if data_label is None else data_label
        data = Data(email_body, filename, label)

        self.dataset.put(data, category)

