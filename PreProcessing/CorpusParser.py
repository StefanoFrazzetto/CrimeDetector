import abc
from enum import Enum

from Data import Dataset
from Interfaces import Serializable
from Utils import File, Hashing


class CorpusName(Enum):
    PAN12 = 0


class CorpusParser(Serializable, metaclass=abc.ABCMeta):

    def __init__(self):
        self.corpus_name = None
        self.source_path = None
        self.merge_messages = False

    def __eq__(self, other: 'CorpusParser'):
        return self.source_path == other.source_path

    def __hash__(self):
        parser_hash = f"DIR: {self.source_path} - MERGE_MESSAGES: {self.merge_messages}"
        return Hashing.sha256_digest(parser_hash)

    @staticmethod
    def factory(corpus_name: CorpusName, source_path: str, merge_messages: bool = True):
        if corpus_name not in CorpusName:
            raise ValueError(f"Unknown corpus name {corpus_name}")

        corpus_parser = None

        if corpus_name == CorpusName.PAN12:
            from PreProcessing.PAN12 import PAN12Parser
            corpus_parser = PAN12Parser()

        corpus_parser.corpus_name = corpus_name
        corpus_parser.merge_messages = merge_messages
        corpus_parser.source_path = source_path

        return corpus_parser

    def get_params(self):
        return f"Corpus name: {self.corpus_name} - " \
            f"Source path: {self.source_path} - " \
            f"Merge messages: {str(self.merge_messages)}"

    @abc.abstractmethod
    def parse(self):
        pass

    @abc.abstractmethod
    def dump(self, directory, *args):
        pass

    @abc.abstractmethod
    def add_to_dataset(self, dataset: Dataset):
        pass

    @abc.abstractmethod
    def log_info(self):
        pass
