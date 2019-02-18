from typing import Dict, Set

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from Classification import Classifier, ClassifierType, MetricType, FeatureExtraction
from Classification import Metrics
from Data import Dataset
from Utils import Log
from Utils import Numbers


class Benchmark(object):
    classifier_types: Set[ClassifierType]
    classifiers: Dict[ClassifierType, Classifier]
    metrics: Metrics

    def __init__(self, dataset: Dataset, feature_extraction: FeatureExtraction):
        self.dataset = dataset
        self.classifier_types = set()
        self.classifiers = dict()
        self.metrics = Metrics()
        self.features = feature_extraction

        Log.info("### BENCHMARK ###", header=True)

    def add_classifier(self, classifier_type: ClassifierType):
        self.classifier_types.add(classifier_type)
        Log.debug(f"Selected classifier {classifier_type} for benchmarking.")

    def select_metrics(self, *metric_types: MetricType):
        self.metrics = Metrics(*metric_types)

    def initialize_classifiers(self):
        """
        Initialize all the classifiers with the provided training vectors and labels.
        """
        training_vectors, training_labels = self.features.fit_transform()

        Log.info("Initializing classifiers.")
        for classifier_type in self.classifier_types:
            classifier = Classifier.factory(classifier_type)
            classifier.fit(training_vectors, training_labels)
            self.classifiers[classifier_type] = classifier

        Log.info("Classifiers initialized correctly.")

    def run(self, folds: int = 1):
        """
        Run each classifier and get its values.
        """
        Log.info("Starting benchmarking process.", header=True)

        data = self.dataset.validation['text']
        labels = self.dataset.validation['label']
        testing_data_subsets = np.array_split(data, folds)
        testing_labels_subsets = np.array_split(labels, folds)

        for classifier_type, classifier in self.classifiers.items():
            Log.debug(f"Benchmarking {classifier_type.name}... ", newline=False)

            for i in range(len(testing_data_subsets)):
                vectors = self.features.transform(testing_data_subsets[i])
                predicted_labels = classifier.predict(vectors)

                self.metrics.append(
                    classifier,
                    true_labels=testing_labels_subsets[i],
                    predicted_labels=predicted_labels
                )

            Log.debug("done.", timestamp=False)

        self.metrics.sort()
        Log.info("Benchmark process completed.")

    def get_info(self):
        Log.info("### CLASSIFIERS INFO ###", header=True)

        for classifier_type, classifier in self.classifiers.items():
            Log.debug(f"{classifier.get_name()}", header=True)

            Log.debug(f"\tTraining time:     \t"
                      f"{Numbers.format_float(classifier.training_time, 0)} ms")

        Log.info("Mean values:")
        Log.info(self.metrics.get_means_table(), timestamp=False)

    def plot_metrics(self, *metrics: MetricType):
        Log.info("Generating plots... ", header=True)
        self.metrics.visualize(*metrics)
        Log.info("done.")

    def save_metrics(self, path: str, *metrics: MetricType):
        Log.info(f"Saving plots to '{path}'... ", header=True)
        self.metrics.save(path, *metrics)
        Log.info("done.")

    def clustering(self):
        vectors, labels = self.features.fit_transform()(dense=True)
        pca = PCA(n_components=2, random_state=42).fit(vectors)
        data2D = pca.transform(vectors)
        plt.figure(figsize=(56, 40))
        # plt.figure(figsize=(10, 6))
        plt.title("k-means after dimensionality reduction using PCA")
        plt.scatter(
            data2D[:, 0], data2D[:, 1],
            c=labels.map({0: 'green', 1: 'red'}),
            linewidths=0.05,
        )
        # plt.show()

        from sklearn.cluster import KMeans
        true_k = 2
        kmeans = KMeans(n_clusters=true_k, random_state=42).fit(vectors)
        centers2D = pca.transform(kmeans.cluster_centers_)

        # plt.hold(True)
        plt.scatter(
            centers2D[:, 0], centers2D[:, 1],
            marker='x', s=300, linewidths=4,
            c=pd.Series(['magenta', 'cyan'], index=[0, 1]),
        )
        plt.show()  # not required if using ipython notebook

        Log.info("Top terms per cluster:", header=True)
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = self.features.get_names()
        for i in range(true_k):
            Log.info("Cluster %d:" % i, newline=False)
            for ind in order_centroids[i, :10]:
                Log.info(f" {terms[ind]}", timestamp=False)
