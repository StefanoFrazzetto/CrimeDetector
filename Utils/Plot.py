from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scikitplot import metrics as skplt
from scipy.interpolate import pchip


class PlotType(Enum):
    CATPLOT = 'catplot'
    BOXPLOT = 'boxplot'
    ROC_CURVE = 'roc_curve'
    NONE = None


class Plot(object):
    sns.set(font_scale=1.2)

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def save(self, metric: str, plot_type: PlotType, path: str):
        """
        Save the specified plot to path.
        :param metric:
        :param plot_type:
        :param path:
        :return:
        """
        plot = self._create_plot(metric, plot_type)
        plot.savefig(f"{path}/{metric}.png")
        plot.clf()

    def view(self, metric: str, plot_type: PlotType):
        """
        View the specified plot.
        :param metric:
        :param plot_type:
        :return:
        """
        plot = self._create_plot(metric, plot_type)
        plot.show()
        plot.clf()

    def _create_plot(self, metric: str, plot_type: PlotType):
        if plot_type == PlotType.CATPLOT:
            return self._catplot(metric)

        if plot_type == PlotType.BOXPLOT:
            return self._boxplot(metric)

        if plot_type == PlotType.ROC_CURVE:
            return self._roc_curve()

    def _catplot(self, metric: str) -> Any:
        plot = sns.catplot(x='classifier', y=metric, jitter=False, data=self.data, palette='rainbow')
        plot.set(title=metric.capitalize())
        # plot.set(ylim=(0.7, 1), yticks=np.arange(0.0, 1.1, 0.025))

        return plot.fig

    def _boxplot(self, metric: str):
        # plt.figure(figsize=(8, 6))
        boxplot = sns.boxplot(x='classifier', y=metric, data=self.data, palette='rainbow')
        boxplot.set(ylim=(0.5, 1), yticks=np.arange(0.0, 1.1, 0.05))
        boxplot.set_title(metric.capitalize())

        return boxplot.figure

    def _roc_curve(self):
        classifiers = self.data['classifier'].unique().tolist()

        for classifier in classifiers:
            plt.figure()
            plt.title(f"Receiver Operating Characteristic ({classifier})")
            sns.set_style("darkgrid")
            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

            classifier_data = self.data.query(f"classifier == '{classifier}'")
            # classifier_data.sort_values('fpr_value', inplace=True)

            roc_data = pd.DataFrame(
                dict(
                    fpr=[el[1] for el in classifier_data['fpr'].values],
                    tpr=[el[1] for el in classifier_data['tpr'].values],
                    auc=classifier_data['auc'].values.tolist()
                ),
            ).reset_index()

            for index, row in classifier_data.iterrows():
                plt.plot(
                    row['fpr'], row['tpr'],
                    lw=1,
                    alpha=0.5
                    # label='ROC curve (area = %0.2f)' % classifier_data['auc'].mean(),
                )

            plt.plot(
                classifier_data['fpr'].mean(), classifier_data['tpr'].mean(),
                color='black',
                lw=1,
                label='ROC curve (mean area = %0.2f)' % classifier_data['auc'].mean(),
            )

            plt.legend(loc="lower right")
            plt.show()

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        skplt.plot_confusion_matrix(y_true, y_pred, normalize=True)
        plt.show()
