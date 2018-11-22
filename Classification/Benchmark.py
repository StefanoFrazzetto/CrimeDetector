from typing import List, Dict, Set

from Classification import Classifier, ClassifierType, Dataset, Data
from Classification import Metrics
from PreProcessing import CountVectorizer
from Utils import Visualization, DataTools


class Benchmark(object):
    classifier_types: Set[ClassifierType]
    classifiers: Dict[ClassifierType, Classifier]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.classifier_types = set()

        self.classifiers = dict()
        self.vectorizer = None
        self.training_vectors = None
        self.testing_vectors = None

    def __configure(self):
        """
        Configure the class by generating feature vectors and initializing classifiers.
        """
        self.__generate_vectors()
        self.__initialize_classifiers()

    def __fit_classifier(self, classifier: Classifier):
        """
        Fit a classifier using the training vectors and their labels.
        :param classifier: the classifier to fit.
        """
        classifier.fit(self.training_vectors, self.vectorizer.get_labels(self.dataset.training))

    def __generate_vectors(self):
        """
        Generate the training and testing vectors using the provided datasets.
        """
        vectorizer = CountVectorizer.instantiate()
        if vectorizer.is_serialized():
            self.training_vectors = vectorizer.vectors
            self.testing_vectors = vectorizer.transform(self.dataset.testing)
        else:
            self.training_vectors = vectorizer.fit_transform(self.dataset.training)
            vectorizer.serialize()
        self.vectorizer = vectorizer

    def __initialize_classifiers(self):
        for classifier_type in self.classifier_types:
            classifier = Classifier.factory(classifier_type)

            # Load serialized classifier, or fit a new one
            if classifier.is_serialized():
                classifier = classifier.deserialize()
            else:
                self.__fit_classifier(classifier)

            self.classifiers[classifier_type] = classifier

    def add_classifier(self, classifier_type: ClassifierType):
        self.classifier_types.add(classifier_type)

    def run(self):
        self.__configure()

        apr = []
        for classifier_type, classifier in self.classifiers.items():
            true_labels = Data.list_to_dataframe(self.dataset.testing, 'label')
            predicted_labels = classifier.predict(self.testing_vectors)

            current_metrics = Metrics(classifier_type, true_labels, predicted_labels)
            apr.append(current_metrics.get_apr())

        Visualization.plot_metrics('classifier', 'accuracy', DataTools.dictionary_list_to_dataframe(apr))
