from typing import Dict, Set

from Classification import Classifier, ClassifierType
from Classification import Metrics
from Data import Dataset
from Interfaces import Analyzable
from PreProcessing import CountVectorizer
from Utils import Visualization, DataConverter, Log, Assert


class Benchmark(object):
    classifier_types: Set[ClassifierType]
    classifiers: Dict[ClassifierType, Classifier]
    vectorizer = CountVectorizer

    def __init__(self, dataset: Dataset, subset_split=0.2):
        self.dataset = dataset
        self.subset_split = subset_split
        self.classifier_types = set()

        self.classifiers = dict()
        self.vectorizer = None
        self.training_vectors = None

        self.metrics = None

    def add_classifier(self, classifier_type: ClassifierType):
        self.classifier_types.add(classifier_type)

    def initialize_classifiers(self, training_vectors, training_labels):
        """
        Initialize all the classifiers with the provided training vectors and labels.
        :param training_vectors:
        :param training_labels:
        """
        for classifier_type in self.classifier_types:
            classifier = Classifier.factory(classifier_type)
            classifier.fit(training_vectors, training_labels)

            self.classifiers[classifier_type] = classifier

    # def __generate_subsets(self, data: list) -> list:
    #     chunks_size = round(len(data) * self.subset_split)
    #     return list(DataConverter.list_chunks(data, chunks_size))

    def run(self, validation_vectors, validation_labels):
        """
        Run each classifier and get its metrics.
        """
        # Assert.same_length(validation_vectors, validation_labels)

        # subsets = self.__generate_subsets(self.dataset.validation)

        metrics = []
        for classifier_type, classifier in self.classifiers.items():
            Log.info(f"Benchmarking {classifier_type.name}... ", newline=False)

            predicted_labels = classifier.predict(validation_vectors)
            current_metrics = Metrics(
                classifier_type=classifier_type,
                true_labels=validation_labels,
                predicted_labels=predicted_labels,
                # samples=len(validation_vectors)
            )
            metrics.append(current_metrics.get_all())

            Log.info("done.", timestamp=False)

        self.metrics = metrics

    def plot_metrics(self):
        Log.info("Generating plots... ", newline=False)
        metrics_dataframe = DataConverter.dictionary_list_to_dataframe(self.metrics)
        Visualization.plot_metrics('classifier', 'accuracy', metrics_dataframe, 'Accuracy')
        Visualization.plot_metrics('classifier', 'precision', metrics_dataframe, 'Precision')
        Visualization.plot_metrics('classifier', 'recall', metrics_dataframe, 'Recall')
        Visualization.plot_metrics('classifier', 'f0.5', metrics_dataframe, 'F0.5 score')
        Visualization.plot_metrics('classifier', 'f1', metrics_dataframe, 'F1 score')
        Visualization.plot_metrics('classifier', 'f2', metrics_dataframe, 'F2 score')
        Visualization.plot_metrics('classifier', 'f3', metrics_dataframe, 'F3 score')
        Log.info("done.", timestamp=False)
