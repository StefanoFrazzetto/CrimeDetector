import abc
from enum import Enum
from typing import List

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from Classification import DataLabel
from Interfaces import Serializable
from Utils import Assert, Log


class ClassifierType(Enum):
    MultinomialNaiveBayes = 0,
    SupportVectorMachine = 1,
    MultiLayerPerceptron = 2,


class Classifier(Serializable, metaclass=abc.ABCMeta):
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

    def _assert_trained(self):
        Assert.true(self.trained, "The classifier model is not trained yet!")

    @staticmethod
    def _get_data_labels():
        return [label.name for label in DataLabel]

    """
    Wrapper methods for classifiers.
    """

    def fit(self, term_document_matrix, labels: List):
        """Fit the model according to the given training data."""
        Log.info("Fitting model with data...")
        self.trained = True
        data = self.model.fit(term_document_matrix, labels)
        Log.info("Done fitting model.")
        return data

    def get_model(self):
        return self.model

    def predict(self, term_document_matrix) -> List:
        """Perform classification of the **'data' vectors** and return the **predicted labels**."""
        return self.model.predict(term_document_matrix)

    """
    Metrics.
    """

    def get_accuracy(self, true_labels: List, predicted_labels: List):
        self._assert_trained()
        return metrics.accuracy_score(true_labels, predicted_labels)

    def get_params(self):
        """Get parameters for the classifier."""
        pass

    def get_confusion_matrix(self, true_labels: List, predicted_labels: List):
        self._assert_trained()
        return metrics.confusion_matrix(true_labels, predicted_labels)

    def get_prediction_report(self, true_labels: List, predicted_labels: List):
        self._assert_trained()
        return metrics.classification_report(true_labels, predicted_labels, target_names=Classifier._get_data_labels())

    def get_metrics(self, true_labels: List, predicted_labels: List):
        """
        Return the following metrics for the trained model:
            - Accuracy score
            - Precision score
            - Recall score
        :param true_labels: the correct labels for the dataset.
        :param predicted_labels:  the labels predicted by the model.
        :return: accuracy_score, precision_score, recall_score
        """
        self._assert_trained()
        return {
            'accuracy': metrics.accuracy_score(true_labels, predicted_labels),
            'precision': metrics.precision_score(true_labels, predicted_labels),
            'recall': metrics.recall_score(true_labels, predicted_labels)
        }

    def get_f_scores(self, true_labels: List, predicted_labels: List):
        self._assert_trained()
        return \
            metrics.fbeta_score(true_labels, predicted_labels, 0.5), \
            metrics.fbeta_score(true_labels, predicted_labels, 1), \
            metrics.fbeta_score(true_labels, predicted_labels, 2)


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
