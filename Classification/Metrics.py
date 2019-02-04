from typing import List

from sklearn import metrics


class Metrics(object):
    def __init__(self, classifier: str, true_labels, predicted_labels, samples=None):
        self.classifier = classifier
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.samples = samples

    def get_confusion_matrix(self):
        return metrics.confusion_matrix(self.true_labels, self.predicted_labels)

    def get_classification_report(self, labels: List = None):
        return metrics.classification_report(self.true_labels, self.predicted_labels, labels)

    def get_all(self):
        """
        Return the following metrics for the trained classifier:
            - Accuracy score
            - Precision score
            - Recall score
        :return: accuracy_score, precision_score, recall_score
        """
        return {
            'classifier': self.classifier,
            'accuracy': metrics.accuracy_score(self.true_labels, self.predicted_labels),
            'precision': metrics.precision_score(self.true_labels, self.predicted_labels),
            'recall': metrics.recall_score(self.true_labels, self.predicted_labels),
            'f0.5': metrics.fbeta_score(self.true_labels, self.predicted_labels, 0.5),
            'f1': metrics.fbeta_score(self.true_labels, self.predicted_labels, 1),
            'f2': metrics.fbeta_score(self.true_labels, self.predicted_labels, 2),
            'f3': metrics.fbeta_score(self.true_labels, self.predicted_labels, 3),
            'samples': self.samples
        }

