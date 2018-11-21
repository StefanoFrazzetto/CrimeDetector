from typing import List

from Classification import Classifier, ClassifierType, Dataset, Data
from PreProcessing import CountVectorizer
from Utils import Visualization


class Benchmark(object):
    classifier_types: List[ClassifierType]
    classifier: List[Classifier]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        self.classifier_types = []
        self.classifiers = []
        self.scores = []

        self.vectorizer = None
        self.training_vectors = None
        self.testing_vectors = None

        self.generate_vectors()

    def add_algorithm(self, classifier_type: ClassifierType):
        self.classifier_types.append(classifier_type)

    def fit_classifier(self, classifier: Classifier):
        classifier.fit(self.training_vectors, self.vectorizer.get_labels(self.dataset.testing))

    def generate_vectors(self):
        vectorizer = CountVectorizer.deserialize() if CountVectorizer.is_serialized() else CountVectorizer()
        if vectorizer.is_serialized():
            self.training_vectors = vectorizer.vectors
            self.testing_vectors = vectorizer.transform(self.dataset.testing)
        else:
            self.training_vectors = vectorizer.fit_transform(self.dataset.training)
            vectorizer.serialize()

    def run(self):
        for classifier_type in self.classifier_types:
            classifier = Classifier.factory(classifier_type)
            self.fit_classifier(classifier)
            self.classifiers.append(classifier)

    def plot_metrics(self):
        for classifier in self.classifiers:
            true_labels = Data.list_to_dataframe(self.dataset.testing, 'label')
            predicted_labels = classifier.predict(self.testing_vectors)
            metrics = classifier.get_metrics(classifier.get_metrics(true_labels, predicted_labels))
            Visualization.plot_metrics('metrics', 'score', metrics)
