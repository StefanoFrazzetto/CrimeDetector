import abc
from enum import Enum

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


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

    def fit(self, term_document_matrix, labels: list):
        """Fit the model according to the given training data."""
        return self.model.fit(term_document_matrix, labels)

    def get_model(self):
        return self.model

    def get_params(self):
        """Get parameters for the classifier."""
        pass

    def predict(self, data):
        """Perform classification of the samples in 'data'."""
        return self.model.predict(data)

    def score(self):
        """Return the mean accuracy on the given test data and labels."""
        pass


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
