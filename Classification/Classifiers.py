import abc
import copy
from enum import Enum
from math import floor

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from PreProcessing.PreProcessor import PreProcessor
from Utils import Log
from Utils.Text import TextUtils


class ClassifierType(Enum):
    NAIVE_BAYES = 0,
    SVM = 1,
    MLP = 2,


class Classifier(metaclass=abc.ABCMeta):
    """
    Define an abstract Classifier class containing the base methods for classifiers.

    The class exposes a factory method that allows to instantiate the specific available classifiers.
    """

    # The split ratio between training and testing data.
    DEFAULT_SPLIT_RATIO = 0.7

    @staticmethod
    def factory(classifier_type, split_ratio, corpus_directory, load_percentage):
        """Define factory method for classifiers."""
        assert classifier_type in ClassifierType, f"Unrecognised classifier {classifier_type.name}"

        # Initialize pre-processor
        pre_processor = PreProcessor(
            directory=corpus_directory,
            load_percentage=load_percentage,
            language=PreProcessor.DEFAULT_LANGUAGE,
            stop_words=PreProcessor.DEFAULT_STOP_WORDS
        )

        if classifier_type == ClassifierType.NAIVE_BAYES:
            return NaiveBayes(pre_processor, split_ratio=split_ratio)

        if classifier_type == ClassifierType.MLP:
            return MLP(pre_processor, split_ratio=split_ratio)

    @abc.abstractmethod
    def load(self):
        """Abstract method used by the concrete classes to perform all the necessary operations."""
        pass

    def __init__(self, pre_processor: PreProcessor, split_ratio=DEFAULT_SPLIT_RATIO):
        """Initialize the object."""
        self.split_ratio = split_ratio

        # Split the dataset into training and testing.
        dataset = copy.deepcopy(pre_processor.dataset)
        training_last_element_index = floor((len(dataset) * split_ratio) - 1)
        self.training_set = dataset[0:training_last_element_index]
        self.testing_set = dataset[training_last_element_index + 1:]
        self.pre_processor = pre_processor

        # Use abstract method to perform custom loading for subclasses
        self.load()


class NaiveBayes(Classifier):
    """Multinomial Naive Bayes (MNB) classifier."""

    def get_accuracy(self):
        return self.accuracy_score

    def extract_features(self, dataset):
        """
        Convert a dataset to a matrix of token counts using CountVectorizer from sklearn.

        A corpus of documents can thus be represented by a matrix with one row per document
        and one column per token (e.g. word) occurring in the corpus.

        It would be possible to use FeatureHashing to increase speed and reduce memory usage.
        """
        vectorizer = CountVectorizer(
            stop_words=f'{self.pre_processor.language if self.pre_processor.stop_words else None}',
            preprocessor=TextUtils.clean
        )

        # Fit the vectorizer on the training set
        np.random.shuffle(dataset)
        vectorizer.fit(dataset)

        # Get the vocabulary...
        vec_vocabulary = {v: k for k, v in vectorizer.vocabulary_.items()}

        # ...and take only a number of elements equal to the length of the dataset.
        vocabulary = [vec_vocabulary[i] for i in range(len(dataset))]

        return vectorizer.transform(dataset), vocabulary

    def __train_and_test(self):
        td_matrix_training, training_labels = self.extract_features(self.training_set)

        model = MultinomialNB()
        model.fit(td_matrix_training, training_labels)
        self.model = model

        td_matrix_testing, test_labels = self.extract_features(self.testing_set)
        predicted_labels = self.model.predict(td_matrix_testing)
        self.accuracy_score = accuracy_score(test_labels, predicted_labels)

    def load(self):
        self.__train_and_test()


class SVM(Classifier):
    """Support Vector Machine."""

    def load(self):
        pass

    def __init__(self, pre_processor, split_ratio=Classifier.DEFAULT_SPLIT_RATIO):
        super(SVM, self).__init__(pre_processor, split_ratio)


class MLP(Classifier):
    """MLP classifier class."""

    def load(self):
        pass

    def __init__(self, pre_processor, split_ratio=Classifier.DEFAULT_SPLIT_RATIO):
        super(MLP, self).__init__(pre_processor, split_ratio)
