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
data_reader = DataReader(dataset)
data_reader.add_exclusion("Acknowldegement")

base_dir = "/home/stefano/Downloads/spam-non-spam-dataset"
data_reader.add_dir_to_dataset(f"{dir}/train-mails", DatasetCategory.TRAINING)
data_reader.add_dir_to_dataset(f"{dir}/test-mails", DatasetCategory.TESTING)


vectorizer = CountVectorizer(dataset.training)
vectors = vectorizer.fit_transform()
classifier = Classifier.factory(ClassifierType.MultiLayerPerceptron)
classifier.fit(vectors, vectorizer.get_labels())
predicted = classifier.predict(vectorizer.transform(dataset.testing))

actual_labels = Data.list_to_dataframe(dataset.testing)['label']

res = sklearn.metrics.accuracy_score(actual_labels, predicted)

pprint(res)
