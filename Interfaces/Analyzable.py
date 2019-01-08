import abc
from enum import Enum


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
