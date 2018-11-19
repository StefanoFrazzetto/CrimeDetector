"""
Useful links:
- https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-2-coding-5966f25f1475
- https://medium.com/data-from-the-trenches/text-classification-the-first-step-toward-nlp-mastery-f5f95d525d73
"""
from pprint import pprint

from Classification import Classifier, ClassifierType, Dataset, Data
from Classification import DatasetCategory
from PreProcessing import CountVectorizer
from PreProcessing import DataReader

dataset = Dataset()
data_reader = DataReader(dataset)
data_reader.add_exclusion("Acknowldegement")

base_dir = "/home/stefano/Downloads/spam-non-spam-dataset"
data_reader.add_dir_to_dataset(f"{base_dir}/train-mails", DatasetCategory.TRAINING)
data_reader.add_dir_to_dataset(f"{base_dir}/test-mails", DatasetCategory.TESTING)
data_reader.get_info()

vectorizer = CountVectorizer()
training_vectors = vectorizer.fit_transform(dataset.training)
testing_vectors = vectorizer.transform(dataset.testing)

classifier = Classifier.factory(ClassifierType.SupportVectorMachine)
classifier.fit(training_vectors, vectorizer.get_labels(dataset.training))

true_labels = Data.list_to_dataframe(dataset.testing, 'label')
predicted_labels = classifier.predict(testing_vectors)

accuracy = classifier.get_accuracy(true_labels, predicted_labels)

pprint(accuracy)
