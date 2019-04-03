import abc
from enum import Enum


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
