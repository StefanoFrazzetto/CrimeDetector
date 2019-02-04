from typing import Dict, Set

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from Classification import Classifier, ClassifierType
from Classification import Metrics
from Data import Dataset
from Utils import Visualization, DataStructures, Log


class Benchmark(object):
    classifier_types: Set[ClassifierType]
    classifiers: Dict[ClassifierType, Classifier]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.classifier_types = set()
        self.classifiers = dict()
        self.metrics = None

        self.count_vectorizer = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()

    def add_classifier(self, classifier_type: ClassifierType):
        self.classifier_types.add(classifier_type)

    def _initialize_transformer(self):
        training_data = self.dataset.training['text']
        training_labels = self.dataset.training['label']

        training_vectors = self.count_vectorizer.fit_transform(training_data, training_labels)
        training_vectors = self.tfidf_transformer.fit_transform(training_vectors, training_labels)

        return training_vectors, training_labels

    def _transform_data(self, vectors):
        vectors = self.count_vectorizer.transform(vectors)
        vectors = self.tfidf_transformer.transform(vectors)
        return vectors

    def initialize_classifiers(self):
        """
        Initialize all the classifiers with the provided training vectors and labels.
        """
        training_vectors, training_labels = self._initialize_transformer()

        for classifier_type in self.classifier_types:
            classifier = Classifier.factory(classifier_type)
            classifier.fit(training_vectors, training_labels)

            self.classifiers[classifier_type] = classifier

    def run(self, folds: int = 1):
        """
        Run each classifier and get its metrics.
        """
        data = self.dataset.validation['text']
        labels = self.dataset.validation['label']
        testing_data_subsets = np.array_split(data, folds)
        testing_labels_subsets = np.array_split(labels, folds)

        metrics = Metrics()
        for classifier_type, classifier in self.classifiers.items():
            Log.info(f"Benchmarking {classifier_type.name}... ", newline=False)

            for i in range(len(testing_data_subsets)):
                vectors = self._transform_data(testing_data_subsets[i])
                predicted_labels = classifier.predict(vectors)

                metrics.add(
                    classifier,
                    true_labels=testing_labels_subsets[i],
                    predicted_labels=predicted_labels
                )

            Log.info("done.", timestamp=False)

        self.metrics = metrics.get()

    def plot_metrics(self, save_path: str = None):
        Log.info("Generating plots... ", newline=False)
        Visualization.plot_metrics('classifier', 'accuracy', self.metrics, 'Accuracy', save_path)
        Visualization.plot_metrics('classifier', 'precision', self.metrics, 'Precision', save_path)
        Visualization.plot_metrics('classifier', 'recall', self.metrics, 'Recall', save_path)
        Visualization.plot_metrics('classifier', 'f0.5', self.metrics, 'F0.5 score', save_path)
        Visualization.plot_metrics('classifier', 'f1', self.metrics, 'F1 score', save_path)
        Visualization.plot_metrics('classifier', 'f2', self.metrics, 'F2 score', save_path)
        Visualization.plot_metrics('classifier', 'f3', self.metrics, 'F3 score', save_path)
        Log.info("done.", timestamp=False)
