import abc
import hashlib
from enum import Enum

from Data import Dataset
from Interfaces import Serializable
from Utils import File, Hashing


class CorpusType(Enum):
    SEXUAL_PREDATORS = 0
    CYBER_BULLYING = 1
    BANKING_FRAUD = 2

    def __str__(self):
        return self.name


class CorpusName(Enum):
    PAN12 = 0


class CorpusParser(Serializable, metaclass=abc.ABCMeta):

    def __init__(self, corpus_type: CorpusType, merge_messages: bool = True):
        self.corpus_type = corpus_type
        self.source_directory = ""
        self.merge_messages = merge_messages

    def __eq__(self, other: 'CorpusParser'):
        return self.source_directory == other.source_directory

    def __hash__(self):
        parser_hash = f"DIR: {self.source_directory} - MERGE_MESSAGES: {self.merge_messages}"
        return Hashing.sha256_digest(parser_hash)

    @staticmethod
    def factory(corpus_name: CorpusName, merge_messages: bool = True):
        if corpus_name == CorpusName.PAN12:
            from PreProcessing.PAN12 import Parser
            return Parser(merge_messages)

        raise ValueError(f"Unknown corpus name {corpus_name}")

    def set_source_directory(self, source_directory: str):
        if not File.directory_exists(source_directory):
            raise OSError(f"The directory {source_directory} does not exist.")
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

    @abc.abstractmethod
    def log_info(self):
        pass
