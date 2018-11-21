"""
Useful links:
- https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-2-coding-5966f25f1475
- https://medium.com/data-from-the-trenches/text-classification-the-first-step-toward-nlp-mastery-f5f95d525d73
"""

from Classification import Classifier, ClassifierType
from Classification import Dataset, Data, DataLabel
from Classification import DatasetCategory
from PreProcessing import CountVectorizer, CorpusName
from PreProcessing import DataReader

# CREATE DATASET
dataset = Dataset.deserialize() if Dataset.is_serialized() else Dataset()
if dataset.is_serialized() is False:
    # Create the dataset and serialize it.
    data_reader = DataReader(CorpusName.KDNUGGETS, dataset)
    data_reader.add_exclusion("Acknowldegement")
    data_reader.add_exclusion("Summary.txt")

    base_dir = "/home/stefano/Downloads/spam-non-spam-dataset"
    data_reader.add_dir_to_dataset(f"{base_dir}/train-mails", DatasetCategory.TRAINING)
    data_reader.add_dir_to_dataset(f"{base_dir}/test-mails", DatasetCategory.TESTING)
    data_reader.add_dir_to_dataset(f"/home/stefano/Documents/University/CSCU9YS/Datasets/enron1/ham", data_label=DataLabel.HAM)
    data_reader.add_dir_to_dataset(f"/home/stefano/Documents/University/CSCU9YS/Datasets/enron1/spam", data_label=DataLabel.SPAM)

    data_reader.add_dir_to_dataset(f"/home/stefano/Documents/University/CSCU9YS/Datasets/enron2/ham", data_label=DataLabel.HAM)
    data_reader.add_dir_to_dataset(f"/home/stefano/Documents/University/CSCU9YS/Datasets/enron2/spam", data_label=DataLabel.SPAM)

    data_reader.add_dir_to_dataset(f"/home/stefano/Documents/University/CSCU9YS/Datasets/enron3/ham", data_label=DataLabel.HAM)
    data_reader.add_dir_to_dataset(f"/home/stefano/Documents/University/CSCU9YS/Datasets/enron3/spam", data_label=DataLabel.SPAM)

    data_reader.add_dir_to_dataset(f"/home/stefano/Documents/University/CSCU9YS/Datasets/enron4/ham", data_label=DataLabel.HAM)
    data_reader.add_dir_to_dataset(f"/home/stefano/Documents/University/CSCU9YS/Datasets/enron4/spam", data_label=DataLabel.SPAM)

    data_reader.add_dir_to_dataset(f"/home/stefano/Documents/University/CSCU9YS/Datasets/enron5/ham", data_label=DataLabel.HAM)
    data_reader.add_dir_to_dataset(f"/home/stefano/Documents/University/CSCU9YS/Datasets/enron5/spam", data_label=DataLabel.SPAM)

    data_reader.add_dir_to_dataset(f"/home/stefano/Documents/University/CSCU9YS/Datasets/enron6/ham", data_label=DataLabel.HAM)
    data_reader.add_dir_to_dataset(f"/home/stefano/Documents/University/CSCU9YS/Datasets/enron6/spam", data_label=DataLabel.SPAM)
    data_reader.print_info()
    dataset.serialize()

# CREATE VECTORS
training_vectors = None
testing_vectors = None
vectorizer = CountVectorizer.deserialize() if CountVectorizer.is_serialized() else CountVectorizer()
if vectorizer.is_serialized():
    training_vectors = vectorizer.vectors
    testing_vectors = vectorizer.transform(dataset.testing)
else:
    training_vectors = vectorizer.fit_transform(dataset.training)
    vectorizer.serialize()

# CREATE CLASSIFIER
classifier = Classifier.factory(ClassifierType.SupportVectorMachine)
if classifier.is_serialized() is False:
    classifier.fit(training_vectors, vectorizer.get_labels(dataset.training))
    classifier.serialize()
else:
    classifier = classifier.deserialize()

true_labels = Data.list_to_dataframe(dataset.testing, 'label')
predicted_labels = classifier.predict(testing_vectors)

metrics = classifier.get_metrics(true_labels, predicted_labels)
confusion_matrix = classifier.get_confusion_matrix(true_labels, predicted_labels)

print(metrics)
print(confusion_matrix)

# Visualization.plot("ham", "spam", )
