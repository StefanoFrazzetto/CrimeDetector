import abc
from enum import Enum
from typing import List

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from Interfaces import Serializable, Factorizable
from Utils import Assert, Log


class ClassifierType(Enum):
    MultinomialNaiveBayes = "mnb"
    GaussianNaiveBayes = "gnb"
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

        # Hyper-parameters for tuning the classifier
        self.search_parameters = {}

    @staticmethod
    def factory(classifier_type: ClassifierType) -> 'Classifier':
        """Define factory method for classifiers."""
        assert classifier_type in ClassifierType, f"Unrecognised classifier type {classifier_type.name}"

        classifier = None

        if classifier_type == ClassifierType.MultiLayerPerceptron:
            classifier = MultiLayerPerceptron()

        if classifier_type == ClassifierType.GaussianNaiveBayes:
            classifier = GaussianNaiveBayes()

        if classifier_type == ClassifierType.MultinomialNaiveBayes:
            classifier = MultinomialNaiveBayes()

        if classifier_type == ClassifierType.RandomForest:
            classifier = RandomForest()

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

    def get_name(self):
        return self.type.name

    def get_short_name(self):
        return self.type.value

    def get_search_params(self):
        """Get parameters for the classifier."""
        return self.search_parameters

    def get_best_params(self):
        """Return the best parameters for the classifier."""
        return self.best_parameters


class MultinomialNaiveBayes(Classifier):
    """
    Multinomial Naive Bayes (MNB) classifier.

    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    """

    def __init__(self):
        super(MultinomialNaiveBayes, self).__init__()
        self.classifier = MultinomialNB(
            alpha=0.05,
            fit_prior=True
        )

        self.search_parameters = {
            # 'alpha': [0.01, 0.1, 1],
            # 'alpha': [0.05, 0.1, 0.15, 1],
            'alpha': [0.05, 0.1, 0.2],
            'fit_prior': [True, False]
        }


class GaussianNaiveBayes(Classifier):
    """
        Gaussian Naive Bayes (GNB) classifier.

        https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
        """

    def __init__(self):
        super(GaussianNaiveBayes, self).__init__()
        self.classifier = GaussianNB()
        self.search_parameters = {}


class SupportVectorMachine(Classifier):
    """
    Support Vector Machine classifier.

    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def __init__(self):
        super(SupportVectorMachine, self).__init__()
        self.classifier = SVC(
            kernel='linear',
            gamma='auto',
            max_iter=-1
        )

        self.search_parameters = {
            'kernel': ['linear', 'rbf'],
            'gamma': ['auto', 'scale'],
            'max_iter': [-1, 100, 200]
        }


class MultiLayerPerceptron(Classifier):
    """
    Multi-layer Perceptron classifier.

    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    """

    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()

        self.classifier = MLPClassifier(
            activation='tanh',
            early_stopping=True,
            hidden_layer_sizes=(10,),
            max_iter=200,
            solver='adam'
        )

        self.search_parameters = {
            'activation': ['relu', 'logistic', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'hidden_layer_sizes': [(8,), (10,)],
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
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            criterion='entropy',
            max_depth=None,
            max_features='auto'
        )

        self.search_parameters = {
            'n_estimators': [10, 50, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10],
            'max_features': ['auto', None]
        }
