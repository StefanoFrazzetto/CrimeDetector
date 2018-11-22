from typing import Dict, Set

from Classification import Classifier, ClassifierType, Dataset, Data
from Classification import Metrics
from PreProcessing import CountVectorizer
from Utils import Visualization, DataTools, Log


class Benchmark(object):
    classifier_types: Set[ClassifierType]
    classifiers: Dict[ClassifierType, Classifier]
    vectorizer = CountVectorizer

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.classifier_types = set()

        self.classifiers = dict()
        self.vectorizer = None
        self.training_vectors = None

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
        classifier.serialize()

    def __generate_vectors(self):
        """
        Generate the training vectors using the provided datasets.
        """
        vectorizer = CountVectorizer.instantiate()
        if vectorizer.is_serialized():
            self.training_vectors = vectorizer.vectors
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

    def __generate_subsets(self, data: list) -> list:
        chunks_size = round(len(data) * 0.20)
        return list(DataTools.list_chunks(data, chunks_size))

    def add_classifier(self, classifier_type: ClassifierType):
        self.classifier_types.add(classifier_type)

    def run(self):
        self.__configure()
        subsets = self.__generate_subsets(self.dataset.testing)

        apr = []
        for classifier_type, classifier in self.classifiers.items():
            Log.info(f"Benchmarking {classifier_type.name}... ", newline=False)
            for subset in subsets:
                true_labels = Data.list_to_dataframe(subset, 'label')
                predicted_labels = classifier.predict(self.vectorizer.transform(subset))

                current_metrics = Metrics(
                    classifier_type=classifier_type,
                    true_labels=true_labels,
                    predicted_labels=predicted_labels,
                    samples=len(subset)
                )
                apr.append(current_metrics.get_apr())
            Log.info("done.", timestamp=False)

        Log.info("Generating plots... ", newline=False)
        apr_dataframe = DataTools.dictionary_list_to_dataframe(apr)
        title = f"Testing with {len(subsets)} subsets of {len(subsets[0])} samples"
        Visualization.plot_metrics('classifier', 'accuracy', apr_dataframe, title)
        Visualization.plot_metrics('classifier', 'precision', apr_dataframe, title)
        Visualization.plot_metrics('classifier', 'recall', apr_dataframe, title)
        Log.info("done.", timestamp=False)
