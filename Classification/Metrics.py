from typing import Set

import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
from sklearn.metrics import auc, roc_curve

from Classification import Classifier
from Classification import MetricType
from Utils import Plot, Numbers
from Utils.Plot import PlotType


class Metrics(object):
    metrics: Set[MetricType]
    values: pd.DataFrame

    # @formatter:off
    metric_plot = {
        MetricType.ACCURACY:    PlotType.BOXPLOT,
        MetricType.PRECISION:   PlotType.BOXPLOT,
        MetricType.RECALL:      PlotType.BOXPLOT,
        MetricType.F05:         PlotType.BOXPLOT,
        MetricType.F1:          PlotType.BOXPLOT,
        MetricType.F2:          PlotType.BOXPLOT,
        MetricType.F3:          PlotType.BOXPLOT,
        MetricType.AUC:         PlotType.CATPLOT,
        MetricType.TPR:         PlotType.NONE,
        MetricType.FPR:         PlotType.NONE,
        MetricType.ROC:         PlotType.ROC_CURVE,
        MetricType.THRESHOLDS:  PlotType.NONE,
    }
    # @formatter:on

    def __init__(self, *metrics: MetricType):
        # Add all if none specified
        if not metrics:
            self.metrics = set()
            for metric_type in MetricType:
                self.metrics.add(metric_type)

        # Add only the selected ones
        else:
            self.metrics = set(metrics)

        columns = self._get_dataframe_columns()
        self.values = pd.DataFrame(columns=columns)

        self.true_labels = None
        self.predicted_labels = None

    @staticmethod
    def _get_dataframe_columns() -> dict:
        """
        Construct the dictionary containing the columns for
        the metrics dataframe.
        :return:
        """
        columns = dict()
        columns['classifier'] = None
        columns['samples'] = None
        columns['training time'] = None

        for metric_type in MetricType:
            columns[metric_type.value] = None

        columns[MetricType.FPR.value] = np.zeros(shape=(3,))
        columns[MetricType.TPR.value] = np.zeros(shape=(3,))
        columns[MetricType.THRESHOLDS.value] = np.zeros(shape=(3,))

        return columns

    def append(self, classifier: Classifier, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels

        values = self.generate_all(true_labels, predicted_labels)
        values['classifier'] = classifier.get_short_name()
        values['samples'] = len(true_labels)
        values['training time'] = Numbers.format_float(classifier.training_time, 0)

        # Create a new dataframe and append the values
        columns = self._get_dataframe_columns()
        dataframe = pd.DataFrame(columns=columns)
        dataframe = dataframe.append(values, ignore_index=True)

        # Merge the new dataframe into the metrics one
        self.values = pd.concat([self.values, dataframe], sort=True)

    def sort(self, by: str = None):
        sort_by = 'classifier' if by is None else by
        self.values.sort_values(sort_by, inplace=True)

    def get_values(self) -> pd.DataFrame:
        return self.values

    def get_classifier_values(self, classifier: Classifier):
        return self.values.query(f"classifier == '{classifier.get_short_name()}'")

    def get_classifier_metrics(self, classifier: Classifier, metric_type: MetricType = None):
        """
        Return all or a specific metric for the specified classifier.
        :param classifier: the classifier to get the metrics for.
        :param metric_type: the metric to return
        :return:
        """
        values = self.get_classifier_values(classifier)

        if metric_type is None:
            return values
        else:
            return values.loc[:, f"{metric_type.value}"]

    @staticmethod
    def get_confusion_matrix(true_labels, predicted_labels):
        return skmetrics.confusion_matrix(true_labels, predicted_labels)

    @staticmethod
    def get_classification_report(true_labels, predicted_labels, labels=None):
        return skmetrics.classification_report(true_labels, predicted_labels, labels)

    def has_metric(self, metric_type: MetricType):
        return metric_type in self.metrics

    def generate_all(self, true_labels, predicted_labels) -> dict:
        """
        Return all the metrics as a dictionary.
        :param true_labels:
        :param predicted_labels:
        :return:
        """
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
        if self.has_metric(MetricType.ROC) \
                or self.has_metric(MetricType.TPR) \
                or self.has_metric(MetricType.FPR) \
                or self.has_metric(MetricType.AUC):
            false_positive_rate, true_positive_rate, thresholds = roc_curve(true_labels, predicted_labels,
                                                                            drop_intermediate=False)
            values[MetricType.AUC.value] = auc(false_positive_rate, true_positive_rate)
            values[MetricType.TPR.value] = true_positive_rate
            values[MetricType.FPR.value] = false_positive_rate

        return values

    @staticmethod
    def _get_plottype_for_metric(metric_type: MetricType) -> PlotType:
        return Metrics.metric_plot.get(metric_type)

    def _get_plot_obj(self) -> Plot:
        return Plot(self.values)

    def visualize(self, *metric_types: MetricType):
        plot = self._get_plot_obj()

        for metric_type in self.metrics:
            plot_type = self._get_plottype_for_metric(metric_type)
            if plot_type == PlotType.NONE:
                continue

            plot.view(metric_type.value, plot_type)

    def save(self, path: str, *metric_types: MetricType):
        plot = self._get_plot_obj()

        for metric_type in self.metrics:
            plot_type = self._get_plottype_for_metric(metric_type)
            if plot_type == PlotType.NONE:
                continue

            plot.save(metric_type.value, plot_type=plot_type, path=path)

    def get_means_table(self):
        from tabulate import tabulate
        # Get mean grouping by classifier
        df_mean = self.values.groupby(['classifier']).mean()

        # Drop NaN columns
        df_mean = df_mean.dropna(axis=1, how='all')

        # Sort by values (descending order)
        df_mean = df_mean.sort_values(by=[MetricType.PRECISION.value, MetricType.RECALL.value], ascending=False)

        return tabulate(df_mean, headers='keys', tablefmt='psql', showindex=True)
