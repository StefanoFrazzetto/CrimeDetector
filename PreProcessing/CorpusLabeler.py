import abc
from enum import Enum


class CorpusName(Enum):
    PAN12 = 0


class CorpusLabeler(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @staticmethod
    def factory(corpus_name: CorpusName):
        if corpus_name == CorpusName.ENRON:
            return PAN12()

        assert f"Unknown corpus labeler {corpus_name}"

    @abc.abstractmethod
    def get_label(self, *args):
        pass


class PAN12(CorpusLabeler):

    def get_label(self, *args):
        pass
