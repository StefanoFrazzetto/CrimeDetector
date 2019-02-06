from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns


class PlotType(Enum):
    CATPLOT = 'catplot'
    BOXPLOT = 'boxplot'


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
