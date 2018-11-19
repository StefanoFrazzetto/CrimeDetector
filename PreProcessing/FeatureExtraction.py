from typing import List

from sklearn.feature_extraction.text import CountVectorizer as SKCountVectorizer

from Classification import Data
from Utils import Text


class CountVectorizer:
    vectorizer: SKCountVectorizer

    def __init__(self, data: List[Data], stop_words: str = 'english', preprocessor=None):
        self.data = data
        self.vectorizer = SKCountVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            preprocessor=preprocessor if preprocessor is not None else Text.clean,
            stop_words=stop_words,
        )

    def fit(self, data: List[Data] = None):
        """
        Convert a dataset to a matrix of token counts using CountVectorizer from sklearn.

        A corpus of documents can thus be represented by a matrix with one row per document
        and one column per token (e.g. word) occurring in the corpus.

        It would be possible to use FeatureHashing to increase speed and reduce memory usage.
        """

        self.data = data
        dataframe = self.__get_dataframe_from_data(data)
        return self.vectorizer.fit(dataframe['message'])

    def fit_transform(self, data: List[Data] = None):
        self.data = data
        dataframe = self.__get_dataframe_from_data(data)
        return self.vectorizer.fit_transform(dataframe['message'])

    def get_features(self):
        return self.vectorizer.get_feature_names()

    def get_labels(self, data: List[Data] = None):
        dataframe = self.__get_dataframe_from_data(data, 'label')
        return dataframe

    def transform(self, data: List[Data]):
        dataframe = self.__get_dataframe_from_data(data, 'message')
        return self.vectorizer.transform(dataframe)

    def __get_dataframe_from_data(self, data: List[Data] = None, key: str = None):
        # Create the dataframe from a list of dictionaries
        if data is None:
            data = self.data

        dataframe = Data.list_to_dataframe(self.data if data is None else data)
        if key is None:
            return dataframe
        else:
            return dataframe[key]
