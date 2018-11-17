from typing import List

from sklearn.feature_extraction.text import CountVectorizer

from Classification.Data import Dataset, Data
from Utils.Text import TextUtils


class SKCountVectorizer:
    vectorizer: CountVectorizer

    def __init__(self, dataset: Dataset, stop_words: str = 'english', preprocessor=None):
        self.dataset = dataset
        self.vectorizer = CountVectorizer(
            stop_words=stop_words,
            preprocessor=preprocessor if preprocessor is not None else TextUtils.clean
        )

    def fit(self, dataset: Dataset = None):
        """
        Convert a dataset to a matrix of token counts using CountVectorizer from sklearn.

        A corpus of documents can thus be represented by a matrix with one row per document
        and one column per token (e.g. word) occurring in the corpus.

        It would be possible to use FeatureHashing to increase speed and reduce memory usage.
        """

        dataframe = self.__get_dataset_dataframe(dataset)
        self.vectorizer.fit(dataframe['message'])

    def fit_transform(self, dataset: Dataset = None):
        dataframe = self.__get_dataset_dataframe(dataset)
        dataframe.drop('file', axis=1, inplace=True)
        return self.vectorizer.fit_transform(dataframe['message'])

    def __get_dataset_dataframe(self, dataset=None):
        # Create the dataframe from a list of dictionaries
        return Dataset.to_dataframe(self.dataset.training if dataset is None else dataset.training)

    def get_features(self):
        return self.vectorizer.get_feature_names()

    def transform(self, data: List[Data]):
        return self.vectorizer.transform(data)
