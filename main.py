"""
Useful links:
- https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-2-coding-5966f25f1475
- https://medium.com/data-from-the-trenches/text-classification-the-first-step-toward-nlp-mastery-f5f95d525d73
"""

from Classification import ClassifierType, Benchmark
from Classification import Dataset
from Classification import DatasetCategory
from PreProcessing import CorpusName
from PreProcessing import DataReader

# CREATE DATASET

dataset = Dataset.deserialize() if Dataset.is_serialized() else Dataset()
if dataset.is_serialized() is False:
    # Create the dataset and serialize it.
    data_reader = DataReader(CorpusName.KDNUGGETS, dataset)
    base_dir = "/home/stefano/Documents/University/CSCU9YS/Datasets"

    # KDNUGGETS
    data_reader.add_exclusion("Acknowldegement")
    data_reader.add_dir_to_dataset(f"{base_dir}/kdnuggets/train-mails", DatasetCategory.TRAINING)
    data_reader.add_dir_to_dataset(f"{base_dir}/kdnuggets/test-mails", DatasetCategory.TESTING)

    # ENRON
    # data_reader.add_exclusion("Summary.txt")
    # data_reader.add_dir_to_dataset(f"{base_dir}/enron1/ham", data_label=DataLabel.HAM)
    # data_reader.add_dir_to_dataset(f"{base_dir}/enron1/spam", data_label=DataLabel.SPAM)
    #
    # data_reader.add_dir_to_dataset(f"{base_dir}/enron2/ham", data_label=DataLabel.HAM)
    # data_reader.add_dir_to_dataset(f"{base_dir}/enron2/spam", data_label=DataLabel.SPAM)
    #
    # data_reader.add_dir_to_dataset(f"{base_dir}/enron3/ham", data_label=DataLabel.HAM)
    # data_reader.add_dir_to_dataset(f"{base_dir}/enron3/spam", data_label=DataLabel.SPAM)
    #
    # data_reader.add_dir_to_dataset(f"{base_dir}/enron4/ham", data_label=DataLabel.HAM)
    # data_reader.add_dir_to_dataset(f"{base_dir}/enron4/spam", data_label=DataLabel.SPAM)
    #
    # data_reader.add_dir_to_dataset(f"{base_dir}/enron5/ham", data_label=DataLabel.HAM)
    # data_reader.add_dir_to_dataset(f"{base_dir}/enron5/spam", data_label=DataLabel.SPAM)
    #
    # data_reader.add_dir_to_dataset(f"{base_dir}/enron6/ham", data_label=DataLabel.HAM)
    # data_reader.add_dir_to_dataset(f"{base_dir}/enron6/spam", data_label=DataLabel.SPAM)
    data_reader.print_info()
    dataset.serialize()

benchmarks = Benchmark(dataset)
benchmarks.add_classifier(ClassifierType.SupportVectorMachine)
benchmarks.add_classifier(ClassifierType.MultinomialNaiveBayes)
benchmarks.add_classifier(ClassifierType.MultiLayerPerceptron)
benchmarks.run()
