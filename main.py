"""
Useful links:
- https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-2-coding-5966f25f1475
- https://medium.com/data-from-the-trenches/text-classification-the-first-step-toward-nlp-mastery-f5f95d525d73
"""
from collections import defaultdict
from pprint import pprint

import pandas as pd

from Classification.Classifiers import Classifier, ClassifierType
from Classification.Data import Dataset, DatasetCategory
from PreProcessing.DataReader import DataReader
from PreProcessing.FeatureExtraction import SKCountVectorizer
from Utils import Visualization

dataset = Dataset()
dr = DataReader(dataset)
dr.add_exclusion("Acknowldegement")
dir = "/home/stefano/Downloads/spam-non-spam-dataset"
dr.add_dir_to_dataset(f"{dir}/train-mails", DatasetCategory.TRAINING)
dr.add_dir_to_dataset(f"{dir}/test-mails", DatasetCategory.TESTING)


preproc = SKCountVectorizer(dr.dataset)
mt = preproc.fit_transform()
classifier = Classifier.factory(ClassifierType.MultinomialNaiveBayes)
classifier.fit(mt)

# hamc, spamc = dataset.get_ham_spam_count()

# df = Dataset.to_dataframe(dataset.training)
#
# print(f"Ham count: {hamc} - Spam count: {spamc}")