import abc
from enum import Enum
from typing import List

import pandas as pd


class AnalyzableLabel(Enum):
    NEGATIVE = 0
    POSITIVE = 1

    def __str__(self):
        return self.name


class Analyzable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_label(self) -> AnalyzableLabel:
        pass

    @abc.abstractmethod
    def get_data(self):
        pass

    def is_negative(self):
        return self.get_label() == AnalyzableLabel.NEGATIVE

    def is_positive(self):
        return self.get_label() == AnalyzableLabel.POSITIVE

    def to_dictionary(self):
        return {
            'label': self.get_label().value,
            'data': self.get_data(),
        }

    @staticmethod
    def list_to_dataframe(elements: List['Analyzable'], key: str = None):
        tmp_list = [element.to_dictionary() for element in elements]
        dataframe = pd.DataFrame(tmp_list)
        if key is None:
            return dataframe

        if key not in dataframe.index:
            err_msg = f"The dataframe does not contain the column '{key}'"
            raise ValueError(err_msg)

        return dataframe[key]
