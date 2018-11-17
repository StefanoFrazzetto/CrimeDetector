import abc
from enum import Enum
from typing import List

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from Classification.Data import Dataset, Data


class ClassifierType(Enum):
    MultinomialNaiveBayes = 0,
    SupportVectorMachine = 1,
    MultiLayerPerceptron = 2,


class Classifier(metaclass=abc.ABCMeta):
    """
    Define an abstract Classifier class containing the base methods for classifiers.

    The class exposes a factory method that allows to instantiate the specific available classifiers.
    """

    def __init__(self):
        """Initialize the object."""
        self.accuracy = 0
        self.model = None

    @staticmethod
    def factory(classifier_type):
        """Define factory method for classifiers."""
        assert classifier_type in ClassifierType, f"Unrecognised classifier {classifier_type.name}"

        if classifier_type == ClassifierType.MultiLayerPerceptron:
            return MultiLayerPerceptron()

        if classifier_type == ClassifierType.MultinomialNaiveBayes:
            return MultinomialNB()

        if classifier_type == ClassifierType.SupportVectorMachine:
            return SupportVectorMachine()

    """
    Common methods for classifiers.
    """

    def fit(self, dataset: Dataset):
        """Fit the model according to the given training data."""
        pass

    def get_params(self):
        """Get parameters for the classifier."""
        pass

    def predict(self, data: List[Data]):
        """Perform classification of the samples in 'data'."""
        pass

    def score(self):
        """Return the mean accuracy on the given test data and labels."""
        return self.accuracy


class MultinomialNaiveBayes(Classifier):
    """Multinomial Naive Bayes (MNB) classifier."""

    def __init__(self):
        super(MultinomialNaiveBayes, self).__init__()
        self.model = MultinomialNB()

    # def train_and_test(self, dataset: Dataset):
    #     td_matrix_training, training_labels = self.extract_features()
    #
    #     model = MultinomialNB()
    #     model.fit(td_matrix_training, training_labels)
    #     self.model = model
    #
    #     td_matrix_testing, test_labels = self.extract_features(self.testing_set)
    #     predicted_labels = self.model.predict(td_matrix_testing)
    #     self.accuracy_score = accuracy_score(test_labels, predicted_labels)


class SupportVectorMachine(Classifier):
    """Support Vector Machine."""

    def __init__(self):
        super(SupportVectorMachine, self).__init__()
        self.model = SVC()


class MultiLayerPerceptron(Classifier):
    """MLP classifier class."""

    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.model = MLPClassifier()
