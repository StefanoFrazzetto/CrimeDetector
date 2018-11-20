import abc
from enum import Enum

from Classification import DataLabel


class CorpusName(Enum):
    KDNUGGETS = 0
    ENRON = 1


class CorpusLabeler(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @staticmethod
    def factory(corpus_name: CorpusName):
        if corpus_name == CorpusName.KDNUGGETS:
            return KDNuggetsLabeler()

        if corpus_name == CorpusName.ENRON:
            return EnronLabeler()

        assert f"Unknown corpus labeler {corpus_name}"

    @abc.abstractmethod
    def get_label(self, *args):
        pass


class KDNuggetsLabeler(CorpusLabeler):
    """
    Parser for https://aclweb.org/aclwiki/Spam_filtering_datasets
    """

    def get_label(self, filename):
        if "spmsg" in filename:
            return DataLabel.SPAM
        else:
            return DataLabel.HAM


class EnronLabeler(CorpusLabeler):

    def get_label(self, *args):
        pass
