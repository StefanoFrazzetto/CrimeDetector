import abc
from enum import Enum

from Data import PAN12
from Interfaces import Factorizable
from Utils import Assert


class CorpusName(Enum):
    PAN12 = 0


class Corpus(Factorizable, metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @staticmethod
    def factory(corpus_name: CorpusName):
        Assert.valid_enum(corpus_name)

        if corpus_name == CorpusName.PAN12:
            return PAN12()

