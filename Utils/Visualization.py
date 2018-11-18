import seaborn as sns;

sns.set(font_scale=1.2)


def plot(label1, label2, data):
    sns.lmplot(label1, label2, data=data, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
