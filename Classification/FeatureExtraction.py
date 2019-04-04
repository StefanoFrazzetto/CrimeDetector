import string
from enum import Enum
from typing import List
import numpy as np

import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from Data import Dataset
from PreProcessing.NLTKStemmer import NLTKStemmer, StemmerType
from Utils import Log


class Pipeline(object):
    RANDOM_STATE = 42
    processors: List

    def __init__(self, *extraction_steps: 'FeatureExtractionStep', **kwargs):
        self.max_features = kwargs['max_features']

        self.processors = []
        self.extraction_steps = extraction_steps

    def fit_transform(self, X, y):
        Log.info("Executing fit_transform using the following pipeline:")

        self.processors.clear()
        vectors = X
        labels = y

        # CountVectorizer
        # if FeatureExtractionStep.VECTORIZE in self.extraction_steps:
        Log.info(f"\t- CountVectorizer")
        if FeatureExtractionStep.TOKENIZE in self.extraction_steps:
            Log.info(f"\t- Tokenizer")
            processor = self._get_count_vectorizer(max_features=self.max_features, tokenizer=Pipeline.tokenize)
        else:
            processor = self._get_count_vectorizer(max_features=self.max_features)
        vectors = processor.fit_transform(vectors, labels)
        self.processors.append(processor)

        # TF-IDF
        if FeatureExtractionStep.TFIDF in self.extraction_steps:
            Log.info(f"\t- TF-IDF")
            processor = self._get_tfidf_transformer()
            vectors = processor.fit_transform(vectors, labels)
            self.processors.append(processor)

        # Undersampling with RandomUnderSampler
        if FeatureExtractionStep.UNDERSAMPLE in self.extraction_steps:
            Log.info(f"\t- Under-sampling with RandomUnderSampler")
            processor = self._get_random_undersampler()
            vectors, labels = processor.fit_resample(vectors, labels)

        # Oversampling with ADASYN
        if FeatureExtractionStep.OVERSAMPLE_ADASYN in self.extraction_steps:
            Log.info(f"\t- Over-sampling with ADASYN")
            processor = self._get_adasyn()
            vectors, labels = processor.fit_resample(vectors, labels)

        # Oversampling with SMOTE
        if FeatureExtractionStep.OVERSAMPLE_SMOTE in self.extraction_steps:
            Log.info(f"\t- Over-sampling with SMOTE")
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
    def _get_count_vectorizer(max_features=None, tokenizer=None) -> CountVectorizer:
        return CountVectorizer(
            # max_df=0.5,
            max_features=max_features,
            analyzer='word',
            ngram_range=(1, 1),
            tokenizer=tokenizer,
            strip_accents='ascii',
        )

    @staticmethod
    def _get_tfidf_transformer() -> TfidfTransformer:
        return TfidfTransformer()

    @staticmethod
    def _get_random_undersampler():
        from imblearn.under_sampling import RandomUnderSampler
        return RandomUnderSampler(
            random_state=Pipeline.RANDOM_STATE
        )

    @staticmethod
    def _get_adasyn():
        from imblearn.over_sampling import ADASYN
        return ADASYN(
            sampling_strategy='auto',
            n_jobs=4,
            random_state=Pipeline.RANDOM_STATE
        )

    @staticmethod
    def _get_smote():
        from imblearn.over_sampling import SMOTE
        return SMOTE(
            sampling_strategy='auto',
            n_jobs=4,
            random_state=Pipeline.RANDOM_STATE
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
        clean_text = "".join([ch for ch in text if ch not in string.punctuation])  # Remove punctuation
        tokens = nltk.word_tokenize(clean_text)
        stems = Pipeline.stem_tokens(tokens, stemmer)
        return stems


class FeatureExtractionStep(Enum):
    VECTORIZE = 'CountVectorizer'
    TOKENIZE = 'Tokenization'
    TFIDF = 'TfidfTransformer'
    UNDERSAMPLE = 'Undersampling_RandomUnderSampler'
    UNDERSAMPLE_DROP = 'Undersampling_RandomlyRemovingSamples'
    OVERSAMPLE_ADASYN = 'Oversampling_ADASYN'
    OVERSAMPLE_SMOTE = 'Oversampling_SMOTE'


class FeatureExtraction(object):
    dataset: Dataset
    pipeline: Pipeline

    # noinspection PyTypeChecker
    def __init__(self, *steps: FeatureExtractionStep, dataset: Dataset, **kwargs):
        Log.info("### FEATURE EXTRACTION ###", header=True)

        self.max_features = kwargs['max_features'] if 'max_features' in kwargs else None

        self.dataset = dataset
        self.pipeline = Pipeline(*steps, max_features=self.max_features)
        self.vectors = None
        self.labels = None

        # Random undersampling by dropping samples
        if FeatureExtractionStep.UNDERSAMPLE_DROP in steps:
            Log.info("Undersampling by dropping samples [TRAINING, TESTING]")
            self.dataset.balance_training(5)
            self.dataset.balance_testing(5)

        # Download punctuation vocabulary
        nltk.download('punkt', halt_on_error=False)

        # Create vectors
        self.fit_transform()

        # Number of vectors
        Log.info(f"# vectors: {self.vectors.shape[0]}")

        # The number of features is equal to the number of columns for the vectors
        Log.info(f"# features: {self.vectors.shape[1]}")

        # Labels info
        unique, counts = np.unique(self.labels, return_counts=True)
        for i in range(len(unique)):
            Log.info(f"# label {unique[i]}: {counts[i]}")

    def fit_transform(self, dense: bool = False):
        if self.vectors is None:
            training_data = self.dataset.training['data']
            training_labels = self.dataset.training['label']

            self.vectors, self.labels = self.pipeline.fit_transform(X=training_data, y=training_labels)

        return self.vectors.todense() if dense else self.vectors, self.labels

    def transform(self, data: list):
        return self.pipeline.transform(data)

    def get_names(self):
        return self.pipeline.get_feature_names()

    def get_vocabulary(self):
        return self.pipeline.get_vocabulary()
