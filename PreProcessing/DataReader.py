from Classification.Data import DataLabel, Data, DatasetCategory, Dataset
from DataStructures.Text import Email
from Utils import Log, File


class DataReader(object):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
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
        Log.info(f"Adding directory '{directory}'...")

        for file in File.get_dir_files_recursive(directory):
            if [word for word in self.excluded if word not in file]:
                file_content = File.read(file)

                # Autodetect label from filename if not manually assigned
                label = DataReader.get_file_category(file) if data_label is None else data_label
                email_body = self.email_body_from_content(file_content)
                data = Data(email_body, file, label)

                self.dataset.put(data, category)

        Log.info("Directory added.")
        Log.info(f"Training size: {self.dataset.get_training_size()}.")
        Log.info(f"Testing size: {self.dataset.get_testing_size()}.")
        Log.info(f"Total size: {self.dataset.get_total_size()}.")
        Log.info(f"Split ratio: {self.dataset.get_current_split_ratio()}.")
