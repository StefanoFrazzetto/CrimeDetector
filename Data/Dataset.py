import math
from enum import Enum
from typing import List

import pandas as pd
from sklearn.utils import shuffle

from Interfaces import Serializable, Analyzable, AnalyzableLabel
from PreProcessing import CorpusName
from Utils import Log, Numbers, Hashing, Assert


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

    RANDOM_STATE = 42

    def __init__(self, dataset_id: str,
                 corpus_name: CorpusName,
                 split_ratio=0.85,
                 max_data=math.inf,
                 language='english'
                 ):
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
        dataset_hash = f"" \
            f"{self.dataset_id}" \
            f"{self.split_ratio}" \
            f"{self.max_data}" \
            f"{self.language}"
        return f"{self.corpus_name}_{Hashing.sha256_digest(dataset_hash)}"

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

        self.training = shuffle(self.training, random_state=42)
        self.testing = shuffle(self.testing, random_state=42)

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
        Log.fine(f"### DATASET SAMPLES ###", header=True)

        # Table header
        data = [["Subset", "Positive", "Negative", "Positive %"]]

        # Training
        positives = self.get_positives(self.training)
        negatives = self.get_negatives(self.training)
        ratio = Numbers.percentage(
            self.get_positives(self.training),
            self.get_positives(self.training) + self.get_negatives(self.training)
        )

        training = ["Training", positives, negatives, ratio]
        data.append(training)

        # Testing
        positives = self.get_positives(self.testing)
        negatives = self.get_negatives(self.testing)
        ratio = Numbers.percentage(
            self.get_positives(self.testing),
            self.get_positives(self.testing) + self.get_negatives(self.testing)
        )
        testing = ["Testing", positives, negatives, ratio]
        data.append(testing)

        Log.tabulate(data)

        # Total
        positives = self.get_positives(self.training) + self.get_positives(self.testing)
        negatives = self.get_negatives(self.training) + self.get_negatives(self.testing)
        training = Numbers.percentage(self.get_training_size(), self.get_total_size())
        testing = Numbers.percentage(self.get_testing_size(), self.get_total_size())
        total = [
            ["Total positives", "Total negatives", "Training %", "Testing %"],
            [positives, negatives, training, testing]
        ]

        Log.tabulate(total)

    def balance_training(self,
                         majority_minority_ratio: int or float = 1,
                         random_state: int or None = RANDOM_STATE
                         ):
        positive_samples = self.get_positives(self.training)
        negative_samples = self.get_negatives(self.training)

        if negative_samples == 0 or positive_samples == 0:
            raise RuntimeError("Invalid number of samples (0).")

        majority_samples = negative_samples if negative_samples >= positive_samples else positive_samples
        majority_label = AnalyzableLabel.NEGATIVE.value \
            if negative_samples >= positive_samples \
            else AnalyzableLabel.POSITIVE.value

        minority_samples = negative_samples if negative_samples < positive_samples else positive_samples
        minority_label = AnalyzableLabel.NEGATIVE.value \
            if negative_samples < positive_samples \
            else AnalyzableLabel.POSITIVE.value

        # Drop samples
        Assert.different(majority_label, minority_label)
        drop_frac = 1 - (minority_samples / majority_samples * majority_minority_ratio)
        self.training = self._drop_subset_samples(self.training, "training", drop_frac, majority_label, random_state)

    def balance_testing(self,
                        majority_minority_ratio: int or float = 1,
                        random_state: int or None = RANDOM_STATE
                        ):

        positive_samples = self.get_positives(self.testing)
        negative_samples = self.get_negatives(self.testing)

        majority_samples = negative_samples if negative_samples >= positive_samples else positive_samples
        majority_label = AnalyzableLabel.NEGATIVE.value \
            if negative_samples >= positive_samples \
            else AnalyzableLabel.POSITIVE.value

        minority_samples = negative_samples if negative_samples < positive_samples else positive_samples
        minority_label = AnalyzableLabel.NEGATIVE.value \
            if negative_samples < positive_samples \
            else AnalyzableLabel.POSITIVE.value

        # Drop samples
        Assert.different(majority_label, minority_label)
        drop_frac = 1 - (minority_samples / majority_samples * majority_minority_ratio)
        self.testing = self._drop_subset_samples(self.testing, "testing", drop_frac, majority_label, random_state)

    def _drop_subset_samples(self,
                             subset: pd.DataFrame, subset_name: str,
                             drop_fraction: float, majority_label: str,
                             random_state):
        """
        Ensure that the sample drop ratio is achievable:
        it might be negative if the specified majority-minority ratio is higher than the current subset ratio.
        E.g. over-sampling ratio 5, current subset ratio 3.5
        :param subset:
        :param drop_fraction:
        :param majority_label:
        :param random_state:
        :return:
        """
        if drop_fraction > 0:
            Log.info(f"Dropping {subset_name} samples for the majority class.")
            return subset.drop(subset.query(f'label == {majority_label}')
                               .sample(frac=drop_fraction, random_state=random_state).index)
        else:
            Log.warning("Cannot drop samples: the specified majority-minority ratio is higher than the achievable.")
            return subset

    def balance_all(self,
                    majority_minority_ratio: int = 1,
                    random_state: int or None = RANDOM_STATE
                    ):
        """
        Automatically remove the necessary amount of elements of
        the majority class to match the number of minority ones.
        """

        self.balance_training(majority_minority_ratio, random_state=random_state)
        self.balance_testing(majority_minority_ratio, random_state=random_state)
