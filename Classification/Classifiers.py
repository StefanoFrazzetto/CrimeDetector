import abc
from enum import Enum
from typing import List

import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from Utils import Assert


class ClassifierType(Enum):
    MultinomialNaiveBayes = 0,
    SupportVectorMachine = 1,
    MultiLayerPerceptron = 2,


class Classifier(metaclass=abc.ABCMeta):
    from Classification import Dataset

    """
    Define an abstract Classifier class containing the base methods for classifiers.

    The class exposes a factory method that allows to instantiate the specific available classifiers.
    """

    dataset: Dataset

    def __init__(self):
        """Initialize the object."""
        self.model = None
        self.dataset = None
        self.trained = False

    @staticmethod
    def factory(classifier_type):
        """Define factory method for classifiers."""
        assert classifier_type in ClassifierType, f"Unrecognised classifier {classifier_type.name}"

        if classifier_type == ClassifierType.MultiLayerPerceptron:
            return MultiLayerPerceptron()

        if classifier_type == ClassifierType.MultinomialNaiveBayes:
            return MultinomialNaiveBayes()

        if classifier_type == ClassifierType.SupportVectorMachine:
            return SupportVectorMachine()

    """
    Common methods for classifiers.
    """

    def _assert_trained(self):
        Assert.true(self.trained, "The classifier model is not trained yet!")

    def fit(self, term_document_matrix, labels: List):
        """Fit the model according to the given training data."""
        self.trained = True
        return self.model.fit(term_document_matrix, labels)

    def get_accuracy(self, true_labels: List, predicted_labels: List):
        self._assert_trained()
        return sklearn.metrics.accuracy_score(true_labels, predicted_labels)

    def get_model(self):
        return self.model

    def get_params(self):
        """Get parameters for the classifier."""
        pass

    def predict(self, term_document_matrix):
        """Perform classification of the **'data' vectors** and return the **predicted labels**."""
        return self.model.predict(term_document_matrix)


class MultinomialNaiveBayes(Classifier):
    """Multinomial Naive Bayes (MNB) classifier."""

    def __init__(self):
        super(MultinomialNaiveBayes, self).__init__()
        self.model = MultinomialNB()


class SupportVectorMachine(Classifier):
    """Support Vector Machine."""

    def __init__(self):
        super(SupportVectorMachine, self).__init__()
        self.model = SVC(
            kernel='linear',
            gamma='auto',
        )


class MultiLayerPerceptron(Classifier):
    """MLP classifier class."""

    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.model = MLPClassifier(
            hidden_layer_sizes=[8],
            solver='lbfgs',
        )
