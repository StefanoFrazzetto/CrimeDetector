import math
from enum import Enum
from typing import List

import pandas as pd

from Interfaces import Serializable, Analyzable
from PreProcessing import CorpusName
from Utils import Log, Numbers, Hashing


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

    def __init__(self, dataset_id: str, corpus_name: CorpusName, split_ratio=0.85, max_data=math.inf, language='english'):
        self.dataset_id = dataset_id
        self.corpus_name = corpus_name.name

        self.split_ratio = split_ratio
        self.max_data = max_data
        self.language = language

        self.__training = []
        self.__testing = []

        self.training = None
        self.testing = None

        self.finalized = False

    def __hash__(self):
        dataset_hash = f"{self.dataset_id}{self.split_ratio}{self.max_data}{self.language}"
        return Hashing.sha256_digest(dataset_hash)

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

    @staticmethod
    def get_positives(dataset):
        return (dataset['label'] == 1).sum()

    @staticmethod
    def get_negatives(dataset):
        return (dataset['label'] == 0).sum()

    def log_info(self):
        Log.info(f"### DATASET SAMPLES ###", header=True)
        split_ratio = Numbers.get_formatted_percentage(self.get_training_size(), self.get_total_size())
        Log.info(f"Total: {self.get_training_size() + self.get_testing_size()} - "
                 f"Training (Tr): {self.get_training_size()} / "
                 f"Testing (Ts): {self.get_testing_size()} - "
                 f"Split Ratio (Tr/Ts): {split_ratio} %")

        Log.info("# TRAINING")
        ratio = Numbers.get_formatted_percentage(
            self.get_positives(self.training), self.get_positives(self.training) + self.get_negatives(self.training))
        Log.info(f"Positive (P): {self.get_positives(self.training)} / "
                 f"Negative (N): {self.get_negatives(self.training)} - "
                 f"Ratio (P/Total): {ratio} %")

        Log.info("# TESTING")
        ratio = Numbers.get_formatted_percentage(
            self.get_positives(self.testing),
            self.get_positives(self.testing) + self.get_negatives(self.testing))
        Log.info(f"Positive (P): {self.get_positives(self.testing)} / "
                 f"Negative (N): {self.get_negatives(self.testing)} - "
                 f"Ratio (P/Total): {ratio} %")

    def balance_negatives(self):
        """
        Remove negatives to match the positives number.
        """

        training_positives = self.get_positives(self.training)
        training_negatives = self.get_negatives(self.training)
        training_frac = 1 - (training_positives / training_negatives)
        self.training = self.training.drop(self.training.query('label == 0')
                                           .sample(frac=training_frac, random_state=42).index)

        testing_positives = self.get_positives(self.testing)
        testing_negatives = self.get_negatives(self.testing)
        testing_frac = 1 - (testing_positives / testing_negatives)
        self.testing = self.testing.drop(self.testing.query('label == 0')
                                         .sample(frac=testing_frac, random_state=42).index)
