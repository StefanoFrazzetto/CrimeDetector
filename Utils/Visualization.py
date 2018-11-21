import seaborn as sns


class Visualization(object):
    sns.set(font_scale=1.2)

    @staticmethod
    def plot(label1, label2, data):
        sns.lmplot(label1, label2, data=data, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})

    @staticmethod
    def plot_metrics(self, x_label: str, y_label: str, data: list):
        sns.boxplot(x=x_label, y=y_label, data=data, palette='rainbow')
