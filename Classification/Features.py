import string
from enum import Enum
from typing import List

import nltk
from imblearn.over_sampling import ADASYN
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from Data import Dataset
from PreProcessing.NLTKStemmer import NLTKStemmer, StemmerType


class PipelineProcess(Enum):
    VECTORIZE_ONLY = 0
    VECTORIZE_AND_TRANSFORM = 1


class Pipeline(object):
    processors: List

    def __init__(self, pipeline_process: PipelineProcess):
        self.processors = []

        if pipeline_process == PipelineProcess.VECTORIZE_ONLY:
            self.processors.append(self._get_count_vectorizer())

        if pipeline_process == PipelineProcess.VECTORIZE_AND_TRANSFORM:
            self.processors.append(self._get_count_vectorizer())
            self.processors.append(self._get_tfidf_transformer())

    def fit_transform(self, X, y):
        vectors = X
        for i in range(len(self.processors)):
            # Apply consecutive transformations on the same vectors
            vectors = self.processors[i].fit_transform(vectors, y)
            # If last processor, return data
            if i == (len(self.processors) - 1):
                return vectors

    def transform(self, X):
        vectors = X
        for i in range(len(self.processors)):
            # Apply consecutive transformations on the same vectors
            vectors = self.processors[i].transform(vectors)
            # If last processor, return data
            if i == (len(self.processors) - 1):
                return vectors

    def get_vocabulary(self):
        for processor in self.processors:
            if type(processor) is CountVectorizer:
                return processor.vocabulary_

    def get_feature_names(self):
        for processor in self.processors:
            if type(processor) is CountVectorizer:
                return processor.get_feature_names()

    @staticmethod
    def _get_count_vectorizer() -> CountVectorizer:
        return CountVectorizer(
            # max_df=0.5,
            # max_features=5000,
            # analyzer='word',
            ngram_range=(1, 1),
            tokenizer=Features.tokenize,
            strip_accents='ascii',
        )

    @staticmethod
    def _get_tfidf_transformer() -> TfidfTransformer:
        return TfidfTransformer()


class Features(object):
    dataset: Dataset
    pipeline: Pipeline

    def __init__(self, dataset: Dataset, oversample: bool = False):
        self.dataset = dataset
        self.oversample = oversample

        self.pipeline = Pipeline(PipelineProcess.VECTORIZE_AND_TRANSFORM)

        self.vectors = None
        self.labels = None

        nltk.download('punkt')

    def fit_transform(self, dense: bool = False):
        if self.vectors is None:
            training_data = self.dataset.training['text']
            training_labels = self.dataset.training['label']

            self.vectors = self.pipeline.fit_transform(X=training_data, y=training_labels)
            self.labels = training_labels

            if self.oversample:
                oversampler = Features._get_oversampler()
                self.vectors, self.labels = oversampler.fit_resample(self.vectors, training_labels)

        return self.vectors.todense() if dense else self.vectors, self.labels

    def transform(self, data: list):
        return self.pipeline.transform(data)

    def get_names(self):
        return self.pipeline.get_feature_names()

    def get_vocabulary(self):
        return self.pipeline.get_vocabulary()

    @staticmethod
    def stem_tokens(tokens, stemmer):
        stemmed = []

        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    @staticmethod
    def tokenize(text):
        stemmer = NLTKStemmer.factory(StemmerType.SNOWBALL)
        text = "".join([ch for ch in text if ch not in string.punctuation])
        tokens = nltk.word_tokenize(text)
        stems = Features.stem_tokens(tokens, stemmer)
        return stems

    @staticmethod
    def _get_oversampler():
        # return SMOTE(n_jobs=4)
        return ADASYN(
            sampling_strategy='minority',
            n_jobs=4
        )
