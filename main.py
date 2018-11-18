"""
Useful links:
- https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-2-coding-5966f25f1475
- https://medium.com/data-from-the-trenches/text-classification-the-first-step-toward-nlp-mastery-f5f95d525d73
"""
import sklearn
from collections import defaultdict
from pprint import pprint

import pandas as pd

from Classification.Classifiers import Classifier, ClassifierType
from Classification.Data import Dataset, DatasetCategory, Data
from PreProcessing.DataReader import DataReader
from PreProcessing.FeatureExtraction import SKCountVectorizer
from Utils import Visualization

dataset = Dataset()
dr = DataReader(dataset)
dr.add_exclusion("Acknowldegement")
dir = "/home/stefano/Downloads/spam-non-spam-dataset"
dr.add_dir_to_dataset(f"{dir}/train-mails", DatasetCategory.TRAINING)
dr.add_dir_to_dataset(f"{dir}/test-mails", DatasetCategory.TESTING)

# dr.add_dir_to_dataset(f"/home/stefano/Downloads/enron_with_categories", DatasetCategory.TESTING)


preproc = SKCountVectorizer(dataset.training)
mt = preproc.fit_transform()
classifier = Classifier.factory(ClassifierType.MultiLayerPerceptron)
trained = classifier.fit(mt, preproc.get_labels())
predicted = classifier.predict(preproc.transform(dataset.testing))

actual_labels = Data.list_to_dataframe(dataset.testing)['label']

res = sklearn.metrics.accuracy_score(actual_labels, predicted)

pprint(res)