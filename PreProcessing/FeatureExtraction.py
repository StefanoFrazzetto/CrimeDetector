from typing import List

from sklearn.feature_extraction.text import CountVectorizer as SKCountVectorizer

from Classification import Data
from Utils import Text, Assert


class CountVectorizer(object):
    data: List[Data]
    vectorizer: SKCountVectorizer

    def __init__(self, data: List[Data] = None, stop_words: str = 'english', preprocessor=None):
        self.data = data
        self.vectorizer = SKCountVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            preprocessor=preprocessor if preprocessor is not None else Text.clean,
            stop_words=stop_words,
        )

    def _check_data(self, data: List[Data] = None):
        if data is None:
            Assert.not_none(self.data, "The vectorizer was not initialized with any data.")
            return self.data
        else:
            return data

    def fit(self, data: List[Data] = None):
        """
        Convert a dataset to a matrix of token counts using CountVectorizer from sklearn.

        A corpus of documents can thus be represented by a matrix with one row per document
        and one column per token (e.g. word) occurring in the corpus.

        It would be possible to use FeatureHashing to increase speed and reduce memory usage.
        """

        data = self._check_data(data)
        dataframe = self.__get_dataframe_from_data(data)
        return self.vectorizer.fit(dataframe['message'])

    def fit_transform(self, data: List[Data] = None):
        data = self._check_data(data)
        dataframe = self.__get_dataframe_from_data(data, 'message')
        return self.vectorizer.fit_transform(dataframe)

    def get_features(self):
        return self.vectorizer.get_feature_names()

    def get_labels(self, data: List[Data]):
        dataframe = self.__get_dataframe_from_data(data, 'label')
        return dataframe

    def transform(self, data: List[Data]):
        dataframe = self.__get_dataframe_from_data(data, 'message')
        return self.vectorizer.transform(dataframe)

    @staticmethod
    def __get_dataframe_from_data(data: List[Data], key: str = None):
        dataframe = Data.list_to_dataframe(data)
        if key is None:
            return dataframe
        else:
            return dataframe[key]
