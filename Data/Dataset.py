import math
from enum import Enum
from typing import List

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
    training: List[Analyzable]
    testing: List[Analyzable]

    def __init__(self, split_ratio=0.7, max_data=math.inf, language='english'):
        self.split_ratio = split_ratio
        self.max_data = max_data
        self.language = language

        self.negative = 0
        self.positive = 0
        self.training = []
        self.testing = []

    """
    INSPECTION.
    """

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

    def add_to_training(self, data: Analyzable):
        self.training.append(data)

    def add_to_testing(self, data: Analyzable):
        self.testing.append(data)

    def put(self, data: Analyzable, data_category: DatasetCategory = None):
        """
        Add data to the dataset in a seemingly balanced way.
        :param data_category:
        :param data:
        """

        # If full
        if self.get_total_size() >= self.max_data:
            return

        if data.is_negative():
            self.negative += 1
        else:
            self.positive += 1

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

    def log_info(self):
        Log.info(f"### DATASET SAMPLES ###", header=True)
        Log.info(f"Total: {self.get_total_size()} - "
                 f"Training: {self.get_training_size()} / "
                 f"Testing: {self.get_testing_size()}.")

        Log.info(f"Positive (P): {self.positive} / "
                 f"Negative (N): {self.negative} - "
                 f"Ratio (P/Total): {Numbers.get_formatted_percentage(self.positive, self.positive + self.negative)} %")

        split_ratio = "{0:.2f}".format(self.get_current_split_ratio() * 100)
        Log.info(f"Dataset split ratio: {split_ratio} %")
