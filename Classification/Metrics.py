from enum import Enum

import pandas as pd
from sklearn import metrics as skmetrics
from sklearn.metrics import auc, roc_curve

from Classification import Classifier


class MetricType(Enum):
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'

    F05 = 'f0.5'
    F1 = 'f1'
    F2 = 'f2'
    F3 = 'f3'

    TPR = 'tpr'
    FPR = 'fpr'
    AUC = 'auc'

    def __str__(self):
        return self.value


class Metrics(object):
    metrics: set
    values: pd.DataFrame

    def __init__(self, *metric_types: MetricType):
        metrics = set()
        metrics.add('classifier')
        metrics.add('samples')

        # Add all if none specified
        if not metric_types:
            for metric_type in MetricType:
                metrics.add(metric_type.value)

        # Add only the selected ones
        else:
            for metric_type in metric_types:
                metrics.add(metric_type.value)

        self.metrics = metrics
        self.values = pd.DataFrame(columns=metrics)

    def append(self, classifier: Classifier, true_labels, predicted_labels):
        values = self.generate_all(true_labels, predicted_labels)
        values['classifier'] = classifier.get_short_name()
        values['samples'] = len(true_labels)
        dataframe = pd.DataFrame(values, index=[0])

        self.values = pd.concat([self.values, dataframe], sort=True)

    def sort(self, by: str = None):
        sort_by = 'classifier' if by is None else by
        self.values.sort_values(sort_by, inplace=True)

    def get_values(self) -> pd.DataFrame:
        return self.values

    @staticmethod
    def get_confusion_matrix(true_labels, predicted_labels):
        return skmetrics.confusion_matrix(true_labels, predicted_labels)

    @staticmethod
    def get_classification_report(true_labels, predicted_labels, labels=None):
        return skmetrics.classification_report(true_labels, predicted_labels, labels)

    def has_metric(self, metric_type: MetricType):
        return metric_type.value in self.metrics

    def generate_all(self, true_labels, predicted_labels) -> dict:
        values = {}

        # Precision
        if self.has_metric(MetricType.PRECISION):
            values[MetricType.PRECISION.value] = skmetrics.precision_score(true_labels, predicted_labels)

        # Recall
        if self.has_metric(MetricType.RECALL):
            values[MetricType.RECALL.value] = skmetrics.recall_score(true_labels, predicted_labels)

        # Accuracy
        if self.has_metric(MetricType.ACCURACY):
            values[MetricType.ACCURACY.value] = skmetrics.accuracy_score(true_labels, predicted_labels)

        # F0.5
        if self.has_metric(MetricType.F05):
            values[MetricType.F05.value] = skmetrics.fbeta_score(true_labels, predicted_labels, 0.5)

        # F1
        if self.has_metric(MetricType.F1):
            values[MetricType.F1.value] = skmetrics.fbeta_score(true_labels, predicted_labels, 1)

        # F2
        if self.has_metric(MetricType.F2):
            values[MetricType.F2.value] = skmetrics.fbeta_score(true_labels, predicted_labels, 2)

        # F3
        if self.has_metric(MetricType.F3):
            values[MetricType.F3.value] = skmetrics.fbeta_score(true_labels, predicted_labels, 3)

        # AUC - TPR - FPR
        if self.has_metric(MetricType.AUC) or self.has_metric(MetricType.TPR) or self.has_metric(MetricType.FPR):
            false_positive_rate, true_positive_rate, thresholds = roc_curve(true_labels, predicted_labels)
            values[MetricType.AUC.value] = auc(false_positive_rate, true_positive_rate)
            values[MetricType.TPR.value] = true_positive_rate[1]    # get only the value, 0 to 1
            values[MetricType.FPR.value] = false_positive_rate[1]   # get only the value, 0 to 1

        return values
