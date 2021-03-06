from typing import Set

import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
from sklearn.metrics import auc, roc_curve, matthews_corrcoef

from Classification import Classifier
from Classification import MetricType
from Utils import Plot, Numbers
from Utils.Plot import PlotType


class Metrics(object):
    """
    Metrics allows to automatically generate the metrics to compare the machine
    learning algorithms. It is necessary to select the metrics to be used and
    to provide the true samples and the predicted samples from the classifier.
    """

    metrics: Set[MetricType]
    values: pd.DataFrame

    # @formatter:off
    metric_plot = {
        MetricType.ACCURACY:            PlotType.BOXPLOT,
        MetricType.PRECISION:           PlotType.BOXPLOT,
        MetricType.RECALL:              PlotType.BOXPLOT,
        MetricType.F05:                 PlotType.BOXPLOT,
        MetricType.F1:                  PlotType.BOXPLOT,
        MetricType.F2:                  PlotType.BOXPLOT,
        MetricType.F3:                  PlotType.BOXPLOT,
        MetricType.AUC:                 PlotType.CATPLOT,
        MetricType.TPR:                 PlotType.NONE,
        MetricType.FPR:                 PlotType.NONE,
        MetricType.ROC:                 PlotType.ROC_CURVE,
        MetricType.THRESHOLDS:          PlotType.NONE,
        MetricType.MCC:                 PlotType.BOXPLOT,
        MetricType.CONFUSION_MATRIX:    PlotType.CONFUSION_MATRIX
    }
    # @formatter:on

    def __init__(self):

        columns = self._get_dataframe_columns()
        self.values = pd.DataFrame(columns=columns)

        self.true_labels = None
        self.predicted_labels = None
        self.probabilities = None

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

        columns['true_labels'] = np.zeros(shape=(3,))
        columns['predicted_labels'] = np.zeros(shape=(3,))

        return columns

    def append(self, classifier: Classifier, true_labels, predicted_labels, probabilities):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.probabilities = probabilities

        values = self.generate_all(true_labels, predicted_labels, probabilities)
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

    def generate_all(self, true_labels, predicted_labels, probabilities) -> dict:
        """
        Return all the metrics as a dictionary.
        :param true_labels:
        :param predicted_labels:
        :return:
        """
        values = {
            'true_labels': true_labels.values, 'predicted_labels': predicted_labels,
            MetricType.PRECISION.value: skmetrics.precision_score(true_labels, predicted_labels),
            MetricType.RECALL.value: skmetrics.recall_score(true_labels, predicted_labels),
            MetricType.ACCURACY.value: skmetrics.accuracy_score(true_labels, predicted_labels),
            MetricType.F05.value: skmetrics.fbeta_score(true_labels, predicted_labels, 0.5),
            MetricType.F1.value: skmetrics.fbeta_score(true_labels, predicted_labels, 1),
            MetricType.F2.value: skmetrics.fbeta_score(true_labels, predicted_labels, 2),
            MetricType.F3.value: skmetrics.fbeta_score(true_labels, predicted_labels, 3)
        }

        false_positive_rate, true_positive_rate, thresholds = roc_curve(true_labels, predicted_labels)
        values[MetricType.AUC.value] = auc(false_positive_rate, true_positive_rate)
        values[MetricType.TPR.value] = true_positive_rate
        values[MetricType.FPR.value] = false_positive_rate
        values[MetricType.MCC.value] = matthews_corrcoef(true_labels, predicted_labels)

        return values

    @staticmethod
    def _get_plottype_for_metric(metric_type: MetricType) -> PlotType:
        return Metrics.metric_plot.get(metric_type)

    def _get_plot_obj(self) -> Plot:
        return Plot(self.values)

    def visualize(self, *metric_types: MetricType):
        """
        View the plot(s).
        :param metric_types: the metrics for which plots will be created.
        """
        plot = self._get_plot_obj()

        for metric_type in metric_types:
            plot_type = self._get_plottype_for_metric(metric_type)
            if plot_type == PlotType.NONE:
                continue

            plot.view(metric_type.value, plot_type)

    def save(self, path: str, *metric_types: MetricType):
        """
        Save the metrics to path.
        :param path:
        :param metric_types:
        :return:
        """

        plot = self._get_plot_obj()

        for metric_type in metric_types:
            plot_type = self._get_plottype_for_metric(metric_type)
            if plot_type == PlotType.NONE:
                continue

            plot.save(metric_type.value, plot_type=plot_type, path=path)

    def get_means_table(self):
        """
        Generate the mean values for all the classifiers' results.
        :return:
        """

        from tabulate import tabulate
        # Get mean grouping by classifier
        df_mean = self.values.groupby(['classifier']).mean()

        # Drop NaN columns
        df_mean = df_mean.dropna(axis=1, how='all')

        # Sort by values (descending order)
        df_mean = df_mean.sort_values(by=[MetricType.PRECISION.value, MetricType.RECALL.value], ascending=False)

        return tabulate(df_mean, headers='keys', tablefmt='github', showindex=True)
