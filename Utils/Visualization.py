import seaborn as sns
import matplotlib.pyplot as plt


class Visualization(object):
    sns.set(font_scale=1.2)

    @staticmethod
    def plot(label1, label2, data):
        sns.lmplot(label1, label2, data=data, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
        plt.show()

    @staticmethod
    def plot_metrics(x_label: str, y_label: str, data):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=x_label, y=y_label, data=data, palette='rainbow')
        plt.show()
