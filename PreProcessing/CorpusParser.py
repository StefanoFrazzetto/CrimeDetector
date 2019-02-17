import abc
from enum import Enum

from Data import Dataset
from Interfaces import Serializable
from Utils import File, Hashing


class CorpusName(Enum):
    PAN12 = 0


class CorpusParser(Serializable, metaclass=abc.ABCMeta):

    def __init__(self, merge_messages: bool = True):
        self.corpus_name = None
        self.source_directory = None
        self.merge_messages = merge_messages

    def __eq__(self, other: 'CorpusParser'):
        return self.source_directory == other.source_directory

    def __hash__(self):
        parser_hash = f"DIR: {self.source_directory} - MERGE_MESSAGES: {self.merge_messages}"
        return Hashing.sha256_digest(parser_hash)

    @staticmethod
    def factory(corpus_name: CorpusName, merge_messages: bool = True):
        if corpus_name not in CorpusName:
            raise ValueError(f"Unknown corpus name {corpus_name}")

        corpus_parser = None

        if corpus_name == CorpusName.PAN12:
            from PreProcessing.PAN12 import Parser
            corpus_parser = Parser(merge_messages)

        corpus_parser.corpus_name = corpus_name

        return corpus_parser

    def set_source_directory(self, source_directory: str):
        if not File.directory_exists(source_directory):
            raise OSError(f"The directory {source_directory} does not exist.")
        self.source_directory = source_directory

    def get_params(self):
        return f"Corpus name: {self.corpus_name} - " \
            f"Source dir: {self.source_directory} - " \
            f"Merge messages: {str(self.merge_messages)}"

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
