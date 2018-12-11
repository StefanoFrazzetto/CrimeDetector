import abc
from enum import Enum
from typing import List

import pandas as pd


class DataLabel(Enum):
    """
    The labels for the data contained in the corpus.
    """
    NEGATIVE = 0
    POSITIVE = 1

    def __str__(self):
        return self.name


class ClassifiableContent(metaclass=abc.ABCMeta):
    def __init__(self):
        pass


class Data(object):
    def __init__(self, content: str, label: DataLabel):
        self.content = content
        self.label = label

    def __str__(self):
        return self.content

    def to_dict(self):
        return {
            'label': self.label.value,
            'content': self.content,
        }

    def unpack(self) -> (str, str, str):
        """Unpack the data as label, content, and file name."""
        return str(self.label), str(self.content)

    @staticmethod
    def list_to_dictionary_list(data: List['Data']):
        content = []
        for element in data:
            content.append(element.to_dict())
        return content

    @staticmethod
    def list_to_dataframe(data: List['Data'], key: str = None) -> pd.DataFrame:
        data = Data.list_to_dictionary_list(data)
        if key is None:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(data)[key]
