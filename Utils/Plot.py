from enum import Enum
from typing import Generator

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scikitplot import metrics as skplt

from Classification.MetricType import MetricType
from Utils import Log


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
        plots = self._create_plot(metric, plot_type)
        for i, plot in enumerate(plots):
            plot.savefig(f"{path}/{metric}_{i}.png")
            plot.clf()

    def view(self, metric: str, plot_type: PlotType):
        """
        View the specified plot.
        :param metric:
        :param plot_type:
        :return:
        """
        plots = self._create_plot(metric, plot_type)
        for plot in plots:
            plot.show()
            plot.clf()

    def _create_plot(self, metric: str, plot_type: PlotType) -> Generator:
        Log.debug(f"Plotting {metric} on {plot_type.name}.")
        if plot_type == PlotType.CATPLOT:
            return self._catplot(metric)

        if plot_type == PlotType.BOXPLOT:
            return self._boxplot(metric)

        if plot_type == PlotType.ROC_CURVE:
            return self._roc_curve()

    def _boxplot(self, metric: str):
        # plt.figure(figsize=(8, 6))
        boxplot = sns.boxplot(x='classifier', y=metric, data=self.data, palette='rainbow')
        boxplot.set(ylim=(0.5, 1), yticks=np.arange(0.0, 1.1, 0.05))
        boxplot.set_title(metric.capitalize())

        yield boxplot.figure

    def _catplot(self, metric: str):
        plot = sns.catplot(x='classifier', y=metric, jitter=False, data=self.data, palette='rainbow')
        plot.set(title=metric.capitalize())
        # plot.set(ylim=(0.7, 1), yticks=np.arange(0.0, 1.1, 0.025))

        yield plot.fig

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

            for index, row in classifier_data.iterrows():
                plt.plot(
                    row[MetricType.FPR.value],
                    row[MetricType.TPR.value],
                    lw=1,
                    alpha=0.5
                )

            plt.plot(
                classifier_data[MetricType.FPR.value].mean(),
                classifier_data[MetricType.TPR.value].mean(),
                color='black',
                lw=1,
                label='ROC curve (mean area = %0.2f)' % classifier_data[MetricType.AUC.value].mean(),
            )

            plt.legend(loc="lower right")
            yield plt

    @staticmethod
    def scatter2D(data, labels, centers=None, save_path=None):
        plt.figure(figsize=(25, 25))

        # Draw points
        plt.scatter(
            data[:, 0], data[:, 1],
            c=labels.map({0: 'green', 1: 'red'}),
            linewidths=0.05,
            s=2
        )

        # Draw centroids
        if centers is not None:
            plt.scatter(
                centers[:, 0], centers[:, 1],
                marker='x', s=500, linewidths=4,
                c=pd.Series(['magenta', 'cyan'], index=[0, 1]),
            )

        if save_path is None:
            plt.show()
        else:
            plt.savefig(f"{save_path}/scatter_2D.svg")

    @staticmethod
    def scatter3D(data, labels, centers=None, save_path=None):
        fig = plt.figure(figsize=(25, 25))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('ortho')
        ax.view_init(-22.5, -45)

        # Draw points
        ax.scatter(
            data[:, 0], data[:, 1], data[:, 2],
            c=labels.map({0: 'green', 1: 'red'}),
            marker='o',
            linewidths=0.05,
            s=2
        )

        # Draw centroids
        if centers is not None:
            plt.scatter(
                centers[:, 0], centers[:, 1],
                marker='x', s=500, linewidths=4,
                c=pd.Series(['magenta', 'cyan'], index=[0, 1]),
            )

        if save_path is None:
            plt.show()
        else:
            plt.savefig(f"{save_path}/scatter_3D.svg")

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        skplt.plot_confusion_matrix(y_true, y_pred, normalize=True)
        plt.show()
