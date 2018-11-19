import seaborn as sns


class Visualization(object):
    sns.set(font_scale=1.2)

    @staticmethod
    def plot(label1, label2, data):
        sns.lmplot(label1, label2, data=data, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
