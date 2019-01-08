from typing import List

from sklearn.feature_extraction.text import CountVectorizer as SKCountVectorizer

from Interfaces import Serializable, Analyzable
from Utils import Text, Assert, Log


class CountVectorizer(Serializable):
    data: List[Analyzable]
    vectorizer: SKCountVectorizer

    def __init__(self, data: List[Analyzable] = None, stop_words: str = 'english', preprocessor=None):
        self.data = data
        self.vectors = None
        self.vectorizer = SKCountVectorizer(
            analyzer='word',
            # ngram_range=(1, 3),
            preprocessor=preprocessor if preprocessor is not None else Text.clean,
            stop_words=stop_words,
        )

    def _check_data(self, data: List[Analyzable] = None):
        if data is None:
            Assert.not_none(self.data, "The vectorizer was not initialized with any data.")
            return self.data
        else:
            return data

    def fit(self, data: List[Analyzable] = None):
        """
        Convert a dataset to a matrix of token counts using CountVectorizer from sklearn.

        A corpus of documents can thus be represented by a matrix with one row per document
        and one column per token (e.g. word) occurring in the corpus.

        It would be possible to use FeatureHashing to increase speed and reduce memory usage.
        """

        data = self._check_data(data)
        dataframe = Analyzable.list_to_dataframe(data, 'data')
        return self.vectorizer.fit(dataframe)

    def fit_transform(self, data: List[Analyzable] = None):
        Log.info("Generating vectors from data... ", newline=False)

        data = self._check_data(data)
        dataframe = Analyzable.list_to_dataframe(data, 'data')
        self.vectors = self.vectorizer.fit_transform(dataframe)

        Log.info("done.", timestamp=False)
        return self.vectors

    def get_features(self):
        return self.vectorizer.get_feature_names()

    def get_labels(self, data: List[Analyzable]):
        dataframe = Analyzable.list_to_dataframe(data, 'label')
        return dataframe

    def transform(self, data: List[Analyzable]):
        dataframe = Analyzable.list_to_dataframe(data, 'data')
        return self.vectorizer.transform(dataframe)
