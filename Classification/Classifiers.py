import abc
from enum import Enum
from typing import List

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from Interfaces import Serializable, Factorizable
from Utils import Assert, Log


class ClassifierType(Enum):
    MultinomialNaiveBayes = "mnb"
    SupportVectorMachine = "svm"
    MultiLayerPerceptron = "mlp"
    RandomForest = "rf"


class Classifier(Serializable, Factorizable, metaclass=abc.ABCMeta):
    """
    Define an abstract Classifier class containing the base methods for classifiers.

    The class exposes a factory method that allows to instantiate the specific available classifiers.
    """

    classifier: 'Classifier'
    type: ClassifierType
    trained: bool
    parameters: dict

    def __init__(self):
        """Initialize the object."""
        self.type = None

        self.classifier = None

        # Whether the classifier has been trained already
        self.trained = False

        # Parameters for testing the classifier
        self.parameters = {}

    @staticmethod
    def factory(classifier_type: ClassifierType) -> 'Classifier':
        """Define factory method for classifiers."""
        assert classifier_type in ClassifierType, f"Unrecognised classifier type {classifier_type.name}"

        classifier = None

        if classifier_type == ClassifierType.MultiLayerPerceptron:
            classifier = MultiLayerPerceptron()

        if classifier_type == ClassifierType.MultinomialNaiveBayes:
            classifier = MultinomialNaiveBayes()

        if classifier_type == ClassifierType.SupportVectorMachine:
            classifier = SupportVectorMachine()

        classifier.type = classifier_type

        return classifier

    def _assert_fitted(self):
        Assert.true(self.trained, "The classifier has not been fitted with data yet!")

    """
    Wrapper methods
    """

    def fit(self, term_document_matrix, labels: List):
        """Fit the classifier according to the given training data."""
        Log.info(f"Fitting classifier with data...")
        self.trained = True
        data = self.classifier.fit(term_document_matrix, labels)
        Log.info("Done fitting classifier.")
        return data

    def predict(self, term_document_matrix) -> List:
        """Perform classification of the **'data' vectors** and return the **predicted labels**."""
        return self.classifier.predict(term_document_matrix)

    """
    Metrics
    """

    def get_accuracy(self, true_labels: List, predicted_labels: List):
        self._assert_fitted()
        return metrics.accuracy_score(true_labels, predicted_labels)

    """
    Getters
    """

    def get_short_name(self):
        return self.type.value

    def get_params(self):
        """Get parameters for the classifier."""
        pass


class MultinomialNaiveBayes(Classifier):
    """
    Multinomial Naive Bayes (MNB) classifier.

    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    """

    def __init__(self):
        super(MultinomialNaiveBayes, self).__init__()
        self.classifier = MultinomialNB()
        self.parameters = {}


class SupportVectorMachine(Classifier):
    """
    Support Vector Machine classifier.

    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def __init__(self):
        super(SupportVectorMachine, self).__init__()
        self.classifier = SVC(
            # kernel='linear',
            # gamma='auto',
            # max_iter=-1
        )

        self.parameters = {
            'kernel': ['linear', 'rbf'],
            'gamma': ['auto', 'scale', 0.1, 100, 1000],
            'max_iter': [-1, 100, 1000]
        }


class MultiLayerPerceptron(Classifier):
    """
    Multi-layer Perceptron classifier.

    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    """

    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()

        self.classifier = MLPClassifier(
            # hidden_layer_sizes=[8],
            # solver='lbfgs',
            # max_iter=200
        )

        self.parameters = {
            'activation': ['relu', 'logistic', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'max_iter': [200],
            'early_stopping': [False, True]
        }


class RandomForest(Classifier):
    """
    Random Forest classifier.

    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    def __init__(self):
        super(RandomForest, self).__init__()
        self.classifier = RandomForestClassifier()

        self.parameters = {
            'n_estimators': [10, 50, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10],
            'max_features': ['auto', None]
        }
