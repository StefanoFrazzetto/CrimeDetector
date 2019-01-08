import abc
import hashlib
from enum import Enum

from Data import Dataset
from Interfaces import Serializable


class CorpusType(Enum):
    SEXUAL_PREDATORS = 0
    CYBER_BULLYING = 1
    BANKING_FRAUD = 2

    def __str__(self):
        return self.name


class CorpusName(Enum):
    PAN12 = 0


class CorpusParser(Serializable, metaclass=abc.ABCMeta):

    def __init__(self, corpus_type: CorpusType):
        self.corpus_type = corpus_type
        self.source_directory = ""

    def __eq__(self, other: 'CorpusParser'):
        return self.source_directory == other.source_directory

    def __hash__(self):
        obj_hash = str(self.source_directory).encode('utf-8')
        return hashlib.sha256(obj_hash).hexdigest()

    @staticmethod
    def factory(corpus_name: CorpusName):
        if corpus_name == CorpusName.PAN12:
            from PreProcessing.PAN12 import Parser
            return Parser()

        assert f"Unknown corpus labeler {corpus_name}"

    def set_source_directory(self, source_directory: str):
        self.source_directory = source_directory

    @abc.abstractmethod
    def parse(self):
        pass

    @abc.abstractmethod
    def dump(self, directory, *args):
        pass

    @abc.abstractmethod
    def get_dataset(self) -> Dataset:
        pass


