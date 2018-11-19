"""
Useful links:
- https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-2-coding-5966f25f1475
- https://medium.com/data-from-the-trenches/text-classification-the-first-step-toward-nlp-mastery-f5f95d525d73
"""
from pprint import pprint

import sklearn

from Classification import Classifier, ClassifierType
from Classification import Dataset, DatasetCategory, Data
from PreProcessing import DataReader
from PreProcessing import CountVectorizer

dataset = Dataset()
dr = DataReader(dataset)
dr.add_exclusion("Acknowldegement")
dir = "/home/stefano/Downloads/spam-non-spam-dataset"
dr.add_dir_to_dataset(f"{dir}/train-mails", DatasetCategory.TRAINING)
dr.add_dir_to_dataset(f"{dir}/test-mails", DatasetCategory.TESTING)

# dr.add_dir_to_dataset(f"/home/stefano/Downloads/enron_with_categories", DatasetCategory.TESTING)


preproc = CountVectorizer(dataset.training)
vectors = preproc.fit_transform()
classifier = Classifier.factory(ClassifierType.MultiLayerPerceptron)
classifier.fit(vectors, preproc.get_labels())
predicted = classifier.predict(preproc.transform(dataset.testing))

actual_labels = Data.list_to_dataframe(dataset.testing)['label']

res = sklearn.metrics.accuracy_score(actual_labels, predicted)

pprint(res)
