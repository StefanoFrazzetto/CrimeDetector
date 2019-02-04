import pandas as pd
from sklearn import metrics

from Classification import Classifier


# class MetricsColumns(Enum):
#     ACCURACY = 'accuracy'
#     PRECISION = 'precision'
#     RECALL = 'recall'

class Metrics(object):
    metrics: pd.DataFrame

    def __init__(self):
        columns = {
            'classifier',
            'accuracy',
            'precision',
            'recall',
            'f0.5',
            'f1',
            'f2',
            'f3',
            'samples'
        }
        self.metrics = pd.DataFrame(columns=columns)

    def add(self, classifier: Classifier, true_labels, predicted_labels):
        this_metrics = self.generate(true_labels, predicted_labels)
        this_metrics['classifier'] = classifier.get_short_name()
        dataframe = pd.DataFrame(this_metrics, index=[0])

        self.metrics = pd.concat([self.metrics, dataframe], sort=True)

    def sort(self, by: str = None):
        sort_by = 'classifier' if by is None else by
        self.metrics.sort_values(sort_by, inplace=True)

    def get(self) -> pd.DataFrame:
        return self.metrics

    @staticmethod
    def get_confusion_matrix(true_labels, predicted_labels):
        return metrics.confusion_matrix(true_labels, predicted_labels)

    @staticmethod
    def get_classification_report(true_labels, predicted_labels, labels):
        return metrics.classification_report(true_labels, predicted_labels, labels)

    @staticmethod
    def generate(true_labels, predicted_labels):
        return {
            'accuracy': metrics.accuracy_score(true_labels, predicted_labels),
            'precision': metrics.precision_score(true_labels, predicted_labels),
            'recall': metrics.recall_score(true_labels, predicted_labels),
            'f0.5': metrics.fbeta_score(true_labels, predicted_labels, 0.5),
            'f1': metrics.fbeta_score(true_labels, predicted_labels, 1),
            'f2': metrics.fbeta_score(true_labels, predicted_labels, 2),
            'f3': metrics.fbeta_score(true_labels, predicted_labels, 3),
            'samples': len(true_labels)
        }
