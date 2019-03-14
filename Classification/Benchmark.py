from typing import Dict, Set

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from Classification import Classifier, ClassifierType, MetricType, FeatureExtraction
from Classification import Metrics
from Data import Dataset
from Utils import Log, Plot
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

        subsets = np.array_split(self.dataset.testing, folds)

        for classifier_type, classifier in self.classifiers.items():
            Log.debug(f"Benchmarking {classifier_type.name}... ", newline=False)

            for i in range(len(subsets)):
                data_subset = subsets[i]['data']
                labels_subset = subsets[i]['label']

                vectors = self.features.transform(data_subset)
                predicted_labels = classifier.predict(vectors)

                self.metrics.append(
                    classifier,
                    true_labels=labels_subset,
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

    def clustering(self, draw_centroids=True, three_dimensional=False, save_path=None):
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D
        from sklearn.cluster import KMeans

        Log.info("Performing PCA on training dataset.", header=True)

        n_components = 2 if three_dimensional is False else 3

        vectors, labels = self.features.fit_transform(dense=True)
        pca = PCA(n_components=n_components, random_state=42, whiten=True).fit(vectors)
        data = pca.transform(vectors)
        centers = None

        if draw_centroids:
            Log.info("Calculating centroids positions.")
            kmeans = KMeans(n_clusters=2, random_state=42).fit(vectors)
            centers = pca.transform(kmeans.cluster_centers_)
            self.get_top_cluster_terms(kmeans)

        if not three_dimensional:
            Plot.scatter2D(data, labels, centers, save_path)
        else:
            Plot.scatter3D(data, labels, centers, save_path)

    def get_top_cluster_terms(self, kmeans):
        Log.info("Top terms per cluster:", header=True)
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = self.features.get_names()

        # Cluster 0
        Log.info(f"Cluster 0:", timestamp=False)
        for ind in order_centroids[0, :10]:
            Log.info(f" {terms[ind]}", timestamp=False)

        # Cluster 1
        Log.info("Cluster 1:", timestamp=False)
        for ind in order_centroids[1, :10]:
            Log.info(f" {terms[ind]}", timestamp=False)
