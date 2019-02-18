import string
from enum import Enum
from typing import List

import nltk
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from Data import Dataset
from PreProcessing.NLTKStemmer import NLTKStemmer, StemmerType
from Utils import Log


class Pipeline(object):
    processors: List

    def __init__(self, *extraction_steps: 'FeatureExtractionStep'):
        self.processors = []
        self.extraction_steps = extraction_steps

    def fit_transform(self, X, y):
        Log.info("### FEATURE EXTRACTION ###", header=True)
        Log.info("Executing fit_transform using the following pipeline:")
        for step in self.extraction_steps:
            Log.info(f"\t{step.value}")

        self.processors.clear()
        vectors = X
        labels = y

        # CountVectorizer
        if FeatureExtractionStep.VECTORIZE in self.extraction_steps:
            if FeatureExtractionStep.TOKENIZE in self.extraction_steps:
                processor = self._get_count_vectorizer(Pipeline.tokenize)
            else:
                processor = self._get_count_vectorizer()
            vectors = processor.fit_transform(vectors, labels)
            self.processors.append(processor)

        # TF-IDF
        if FeatureExtractionStep.TFIDF in self.extraction_steps:
            processor = self._get_tfidf_transformer()
            vectors = processor.fit_transform(vectors, labels)
            self.processors.append(processor)

        # Oversampling with ADASYN
        if FeatureExtractionStep.OVERSAMPLE_ADASYN in self.extraction_steps:
            processor = self._get_adasyn()
            vectors, labels = processor.fit_resample(vectors, labels)

        # Oversampling with SMOTE
        if FeatureExtractionStep.OVERSAMPLE_SMOTE in self.extraction_steps:
            processor = self._get_smote()
            vectors, labels = processor.fit_resample(vectors, labels)

        return vectors, labels

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
    def _get_count_vectorizer(tokenizer=None) -> CountVectorizer:
        return CountVectorizer(
            # max_df=0.5,
            max_features=9200,
            # analyzer='word',
            ngram_range=(1, 1),
            tokenizer=tokenizer,
            strip_accents='ascii',
        )

    @staticmethod
    def _get_tfidf_transformer() -> TfidfTransformer:
        return TfidfTransformer()

    @staticmethod
    def _get_adasyn():
        return ADASYN(
            sampling_strategy='auto',
            n_jobs=4
        )

    @staticmethod
    def _get_smote():
        return SMOTE(
            sampling_strategy='auto',
            n_jobs=4
        )

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
        stems = Pipeline.stem_tokens(tokens, stemmer)
        return stems


class FeatureExtractionStep(Enum):
    VECTORIZE = 'CountVectorizer'
    TOKENIZE = 'Tokenization'
    TFIDF = 'TfidfTransformer'
    OVERSAMPLE_ADASYN = 'Oversampling_ADASYN'
    OVERSAMPLE_SMOTE = 'Oversampling_SMOTE'


class FeatureExtraction(object):
    dataset: Dataset
    pipeline: Pipeline

    def __init__(self, dataset: Dataset, *steps: FeatureExtractionStep):
        self.dataset = dataset
        self.pipeline = Pipeline(*steps)

        self.vectors = None
        self.labels = None

        nltk.download('punkt')

    def fit_transform(self, dense: bool = False):
        if self.vectors is None:
            training_data = self.dataset.training['text']
            training_labels = self.dataset.training['label']

            self.vectors, self.labels = self.pipeline.fit_transform(X=training_data, y=training_labels)

        return self.vectors.todense() if dense else self.vectors, self.labels

    def transform(self, data: list):
        return self.pipeline.transform(data)

    def get_names(self):
        return self.pipeline.get_feature_names()

    def get_vocabulary(self):
        return self.pipeline.get_vocabulary()