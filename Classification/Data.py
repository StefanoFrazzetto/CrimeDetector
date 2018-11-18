from enum import Enum
from typing import List

import pandas as pd


class DataLabel(Enum):
    HAM = 0
    SPAM = 1

    def __str__(self):
        return self.name


class DatasetCategory(Enum):
    TRAINING = 0
    TESTING = 1


class Data(object):
    def to_dict(self):
        return {
            'label': self.label.value,
            'message': self.message,
            'file': self.file
        }

    def unpack(self) -> (str, str, str):
        return str(self.label), str(self.message), self.file

    def __init__(self, message: str, file: str, label: DataLabel):
        self.message = message
        self.file = file
        self.label = label

    def __str__(self):
        return self.message

    @staticmethod
    def list_to_dictionary_list(data: List['Data']):
        content = []
        for element in data:
            content.append(element.to_dict())
        return content

    @staticmethod
    def list_to_dataframe(data: List['Data']):
        data = Data.list_to_dictionary_list(data)
        return pd.DataFrame(data)


class Dataset(object):
    """
    Category-agnostic dataset.

    The put operation automatically puts the data into either
    training or testing according to the defined split ratio.
    """
    training: List[Data]
    testing: List[Data]

    def __init__(self, split_ratio=0.7, language='english'):
        self.language = language
        self.split_ratio = split_ratio

        self.training = []
        self.testing = []

    """
    INSPECTION.
    """

    @staticmethod
    def __get_spam_ham_set(subset):
        ham = 0
        spam = 0
        for data in subset:
            if data.label == DataLabel.HAM:
                ham += 1
            else:
                spam += 1
        return ham, spam

    def get_ham_spam_count(self):
        ham_training, spam_training = self.__get_spam_ham_set(self.training)
        ham_testing, spam_testing = self.__get_spam_ham_set(self.testing)

        return (ham_training + ham_testing), (spam_training + spam_testing)

    def get_training_size(self):
        return len(self.training)

    def get_testing_size(self):
        return len(self.testing)

    def get_total_size(self):
        return self.get_training_size() + self.get_testing_size()

    def get_current_split_ratio(self):
        if self.get_training_size() == 0 or self.get_testing_size() == 0:
            return 0

        return self.get_training_size() / self.get_total_size()

    """
    INSERTION.
    """

    def add_to_training(self, data):
        self.__put_data(self.training, data)

    def add_to_testing(self, data):
        self.__put_data(self.testing, data)

    @staticmethod
    def __put_data(subset, data):
        subset.append(data)

    def put(self, data: Data, data_category: DatasetCategory = None):
        """
        Add data to the dataset in a seemingly balanced way.
        :param data_category:
        :param data:
        """
        # Automatically split sets using split ratio
        if data_category is None:
            if len(self.training) == 0:
                self.add_to_training(data)
            elif len(self.testing) == 0:
                self.add_to_testing(data)
            else:
                self.add_to_training(data) \
                    if self.get_current_split_ratio() <= self.split_ratio \
                    else self.add_to_testing(data)
        # Manually assign data
        elif data_category == DatasetCategory.TRAINING:
            self.add_to_training(data)
        elif data_category == DatasetCategory.TESTING:
            self.add_to_testing(data)
