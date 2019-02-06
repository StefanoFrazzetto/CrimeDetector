import string
from pprint import pprint
from typing import Dict, Set

import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from Classification import Classifier, ClassifierType
from Classification import Metrics, MetricType
from Data import Dataset
from PreProcessing.NLTKStemmer import StemmerType, NLTKStemmer
from Utils import Log

nltk.download('punkt')


class Benchmark(object):
    classifier_types: Set[ClassifierType]
    classifiers: Dict[ClassifierType, Classifier]
    metrics: Metrics

    def stem_tokens(self, tokens):
        stemmed = []
        for item in tokens:
            stemmed.append(self.stemmer.stem(item))
        return stemmed

    def tokenize(self, text):
        text = "".join([ch for ch in text if ch not in string.punctuation])
        tokens = nltk.word_tokenize(text)
        stems = self.stem_tokens(tokens)
        return stems

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.classifier_types = set()
        self.classifiers = dict()
        self.metrics = None

        self.stemmer = NLTKStemmer.factory(StemmerType.SNOWBALL)
        # self.count_vectorizer = CountVectorizer(tokenizer=self.tokenize)

        self.count_vectorizer = CountVectorizer(
            analyzer='word',
            max_df=1.0,
            ngram_range=(1, 1),
            # tokenizer=self.tokenize,
            # strip_accents='ascii',
        )
        self.tfidf_transformer = TfidfTransformer()

    def add_classifier(self, classifier_type: ClassifierType):
        self.classifier_types.add(classifier_type)

    def _initialize_transformer(self):
        training_data = self.dataset.training['text']
        training_labels = self.dataset.training['label']

        training_vectors = self.count_vectorizer.fit_transform(training_data, training_labels)
        training_vectors = self.tfidf_transformer.fit_transform(training_vectors, training_labels)

        Log.info("Stop words", header=True)
        pprint(self.count_vectorizer.stop_words_)

        Log.info("Vocabulary", header=True)
        pprint(self.count_vectorizer.get_feature_names())

        Log.info(f"Number of features: {len(self.count_vectorizer.vocabulary_)}", header=True)

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
        Run each classifier and get its values.
        """
        Log.info("Starting benchmarking process.", header=True)

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

                metrics.append(
                    classifier,
                    true_labels=testing_labels_subsets[i],
                    predicted_labels=predicted_labels
                )

            Log.info("done.", timestamp=False)

        metrics.sort()
        self.metrics = metrics

    def get_info(self):
        Log.info("Classifiers information", header=True)

        for classifier_type, classifier in self.classifiers.items():
            Log.info(f"Classifier: {classifier.get_name()} - "
                     f"median: {self.metrics.get_classifier_metrics(classifier, MetricType.PRECISION).median()}")

    def plot_metrics(self):
        Log.info("Generating plots... ", newline=False, header=True)
        self.metrics.visualize(
            MetricType.PRECISION,
            MetricType.ACCURACY,
            MetricType.RECALL,
            MetricType.F05,
            MetricType.F1,
            MetricType.F2,
            MetricType.F3,
            MetricType.AUC,
        )
        Log.info("done.", timestamp=False)

    def save_metrics(self, path: str):
        Log.info(f"Saving plots to {path}... ", newline=False, header=True)
        self.metrics.save(
            path,
            MetricType.PRECISION,
            MetricType.ACCURACY,
            MetricType.RECALL,
            MetricType.F05,
            MetricType.F1,
            MetricType.F2,
            MetricType.F3,
            MetricType.AUC,
        )
        Log.info("done.", timestamp=False)
