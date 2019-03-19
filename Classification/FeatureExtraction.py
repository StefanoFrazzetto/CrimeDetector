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

    def __init__(self, *extraction_steps: 'FeatureExtractionStep', max_features: int = None):
        self.processors = []
        self.max_features = max_features
        self.extraction_steps = extraction_steps

    def fit_transform(self, X, y):
        Log.info("Executing fit_transform using the following pipeline:")

        self.processors.clear()
        vectors = X
        labels = y

        # CountVectorizer
        if FeatureExtractionStep.VECTORIZE in self.extraction_steps:
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

    def __init__(self, *steps: FeatureExtractionStep, dataset: Dataset, max_features: int = None):
        Log.info("### FEATURE EXTRACTION ###", header=True)

        self.dataset = dataset
        self.max_features = max_features
        self.pipeline = Pipeline(*steps, max_features=max_features)
        self.vectors = None
        self.labels = None

        # Download punctuation vocabulary
        nltk.download('punkt')

        # Create vectors
        self.fit_transform()

        Log.info(f"Number of features: {len(self.get_vocabulary())}")

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
