import math
from enum import Enum
from typing import List

import pandas as pd

from Interfaces import Serializable, Analyzable
from Utils import Log, Numbers


class DatasetCategory(Enum):
    TRAINING = 0
    TESTING = 1


class Dataset(Serializable):
    """
    Category-agnostic dataset.

    The put operation automatically puts the data into either
    training or testing according to the defined split ratio.
    """
    training: pd.DataFrame
    testing: pd.DataFrame

    __training: List[Analyzable]
    __testing: List[Analyzable]

    def __init__(self, split_ratio=0.7, max_data=math.inf, language='english'):
        self.split_ratio = split_ratio
        self.max_data = max_data
        self.language = language

        self.negative = 0
        self.positive = 0

        self.__training = []
        self.__testing = []

        self.training = None
        self.testing = None

        self.finalized = False

    """
    Private Methods
    """

    def __add_to_training(self, data: Analyzable):
        self.__training.append(data.to_dictionary())

    def __add_to_testing(self, data: Analyzable):
        self.__testing.append(data.to_dictionary())

    def __get_training_size(self):
        return len(self.__training)

    def __get_testing_size(self):
        return len(self.__testing)

    def __get_total_size(self):
        return self.__get_training_size() + self.__get_testing_size()

    def __get_current_split_ratio(self):
        if self.__get_training_size() == 0 or self.__get_testing_size() == 0:
            return 0

        return self.__get_training_size() / self.__get_total_size()

    """
    Public Methods
    """

    def get_training_size(self):
        return len(self.training)

    def get_testing_size(self):
        return len(self.testing)

    def get_total_size(self):
        return self.get_training_size() + self.get_testing_size()

    def get_split_ratio(self):
        if self.get_training_size() == 0 or self.get_testing_size() == 0:
            return 0
        return self.get_training_size() / self.get_total_size()

    def finalize(self):
        self.finalized = True
        self.training = pd.DataFrame(self.__training)
        self.testing = pd.DataFrame(self.__testing)

        self.__training = None
        self.__testing = None

    def put(self, data: Analyzable, data_category: DatasetCategory = None):
        """
        Add data to the dataset in a seemingly balanced way.
        :param data_category:
        :param data:
        """

        if self.finalized:
            raise RuntimeError("Cannot add more elements to a finalized database.")

        # If full
        if self.__get_total_size() >= self.max_data:
            return

        if data.is_negative():
            self.negative += 1
        else:
            self.positive += 1

        # Automatically split sets using split ratio
        if data_category is None:
            if self.__get_training_size() == 0:
                self.__add_to_training(data)
            elif self.__get_testing_size() == 0:
                self.__add_to_testing(data)
            else:
                self.__add_to_training(data) \
                    if self.__get_current_split_ratio() <= self.split_ratio \
                    else self.__add_to_testing(data)

        # Manually assign data
        elif data_category == DatasetCategory.TRAINING:
            self.__add_to_training(data)
        elif data_category == DatasetCategory.TESTING:
            self.__add_to_testing(data)

    def get_positives(self):
        return (self.training['label'] == 1).sum()

    def get_negatives(self):
        return (self.training['label'] == 0).sum()

    def log_info(self):
        Log.info(f"### DATASET SAMPLES ###", header=True)
        Log.info(f"Total: {self.get_training_size() + self.get_testing_size()} - "
                 f"Training: {self.get_training_size()} / "
                 f"Testing: {self.get_testing_size()}.")

        Log.info(f"Positive (P): {self.get_positives()} / "
                 f"Negative (N): {self.get_negatives()} - "
                 f"Ratio (P/Total): {Numbers.get_formatted_percentage(self.get_positives(), self.get_positives() + self.get_negatives())} %")

        split_ratio = Numbers.get_formatted_percentage(self.get_training_size(), self.get_total_size())
        Log.info(f"Dataset split ratio: {split_ratio} %")

    def under_sample(self, ratio=0.5):
        self.training = self.training.drop(self.training.query('label == 0').sample(frac=ratio).index)

    # def under_sample(self):
    #     positives = self.positive
    #     count = 0
    #     temp_dataset = []
    #
    #     # Reset count
    #     self.positive = 0
    #     self.negative = 0
    #
    #     for element in self.training:
    #         if element.is_negative() and count < positives:
    #             count += 1
    #             temp_dataset.append(element)
    #             self.negative += 1
    #         elif element.is_positive():
    #             temp_dataset.append(element)
    #             self.positive += 1
    #         else:
    #             continue
    #     self.training = temp_dataset
