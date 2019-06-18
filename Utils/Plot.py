from enum import Enum
from typing import Generator

import matplotlib
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
    CONFUSION_MATRIX = 'confusion_matrix'
    NONE = None


class Plot(object):
    sns.set(font_scale=1.2)
    matplotlib.use('TkAgg')

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
            # plot.clf()

    def _create_plot(self, metric: str, plot_type: PlotType) -> Generator:
        Log.debug(f"Plotting {metric} on {plot_type.name}.")
        if plot_type == PlotType.CATPLOT:
            return self._catplot(metric)

        if plot_type == PlotType.BOXPLOT:
            return self._boxplot(metric)

        if plot_type == PlotType.ROC_CURVE:
            return self._roc_curve()

        if plot_type == PlotType.CONFUSION_MATRIX:
            return self._confusion_matrix()

    def _boxplot(self, metric: str):
        # plt.figure(figsize=(8, 6))
        boxplot = sns.boxplot(x='classifier', y=metric, data=self.data, palette='rainbow')
        boxplot.set(ylim=(0.0, 1), yticks=np.arange(0.0, 1.1, 0.1))
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

    def _confusion_matrix(self, normalize: bool = True):
        from sklearn.metrics import confusion_matrix
        from sklearn.utils.multiclass import unique_labels
        """
            This method plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            Source: Scikit-learn documentation
        """
        classifiers = self.data['classifier'].unique().tolist()
        classes = ['negative', 'positive']

        for classifier in classifiers:
            classifier_data = self.data.query(f"classifier == '{classifier}'")
            mean_mcc = classifier_data[MetricType.MCC.value].mean()

            # TODO: Get values using the mean MCC as search query.
            # y_true = classifier_data.loc[classifier_data[MetricType.MCC.value] == mean_mcc]['true_labels']
            # y_pred = classifier_data.loc[classifier_data[MetricType.MCC.value] == mean_mcc]['predicted_labels']
            y_true = classifier_data.iloc[0]['true_labels']
            y_pred = classifier_data.iloc[0]['predicted_labels']


            title = f"{classifier} - "
            if normalize:
                title += 'Normalized confusion matrix'
            else:
                title += 'Confusion matrix, without normalization'

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            # Only use the labels that appear in the data
            classes = ['negative', 'positive']
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            # We want to show all ticks...
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   # ... and label them with the respective list entries
                   xticklabels=classes, yticklabels=classes,
                   title=title,
                   ylabel='True label',
                   xlabel='Predicted label')

            # Remove grid lines
            plt.grid(b=None)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black"
                            )
            fig.tight_layout()
            yield fig

    @staticmethod
    def scatter2D(data, labels, centers=None, save_path=None):
        """
        Create a 2D scatter plot.
        :param data:
        :param labels:
        :param centers:
        :param save_path:
        :return:
        """
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
        """
        Create a 3D scatter plot.
        :param data:
        :param labels:
        :param centers:
        :param save_path:
        :return:
        """
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
        """
        Plot a confusion matrix using skplot.
        :param y_true:
        :param y_pred:
        :return:
        """
        skplt.plot_confusion_matrix(y_true, y_pred, normalize=True)
        plt.show()
