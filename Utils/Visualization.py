import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import FacetGrid


class Visualization(object):
    sns.set(font_scale=1.2)

    @staticmethod
    def plot(label1, label2, data):
        sns.lmplot(label1, label2, data=data, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
        plt.show()

    @staticmethod
    def plot_metrics(x_label: str, y_label: str, data, title: str = None, save_path: str = None):
        plt.figure(figsize=(8, 6))
        boxplot = sns.boxplot(x=x_label, y=y_label, data=data[[x_label, y_label]], palette='rainbow')

        boxplot.set(ylim=(0.5, 1), yticks=np.arange(0.0, 1.1, 0.05))

        # boxplot.ax.set_xticks(np.arange(0.0, 1.0, 0.05), minor=True)
        # FacetGrid.set(boxplot, yticks=np.arange(1, 4, 1))

        if title is not None:
            boxplot.set_title(title)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(f"{save_path}/{title}.png")
