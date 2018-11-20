from typing import List

from Classification.Data import DataLabel, Data, DatasetCategory, Dataset, Email
from Utils import Log, File, Assert


class DataReader(object):
    dataset: Dataset
    excluded: List[str]

    def __init__(self, dataset: Dataset = None):
        self.dataset = dataset if dataset is not None else Dataset()
        self.excluded = []

    @staticmethod
    def get_file_category(filename):
        if "spmsg" in filename:
            return DataLabel.SPAM
        else:
            return DataLabel.HAM

    def add_exclusion(self, string):
        self.excluded.append(string)

    @staticmethod
    def email_body_from_content(content):
        return Email.from_content(content)

    def add_dir_to_dataset(self, directory, category: DatasetCategory = None, data_label: DataLabel = None):
        Log.info(f"# Adding directory '{directory}'... ", newline=False)

        files = File.get_dir_files_recursive(directory)
        Assert.not_empty(files, f"The directory {directory} is empty.")
        for file in files:
            if [word for word in self.excluded if word not in file]:
                file_content = File.read(file)

                # Autodetect label from filename if not manually assigned
                label = DataReader.get_file_category(file) if data_label is None else data_label
                email_body = self.email_body_from_content(file_content)
                data = Data(email_body, file, label)

                self.dataset.put(data, category)

        Log.info("Directory added.", timestamp=False)

    def add_file_to_dataset(self, filename, category: DatasetCategory = None, data_label: DataLabel = None):
        file_content = File.read(filename)

        email_body = self.email_body_from_content(file_content)
        label = DataReader.get_file_category(filename) if data_label is None else data_label
        data = Data(email_body, filename, label)

        self.dataset.put(data, category)

    def print_info(self):
        Log.info(f"Training samples: {self.dataset.get_training_size()}.")
        Log.info(f"Testing samples: {self.dataset.get_testing_size()}.")
        Log.info(f"Total samples: {self.dataset.get_total_size()}.")
        Log.info(f"Dataset split ratio: {self.dataset.get_current_split_ratio()}.")
