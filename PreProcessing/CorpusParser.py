import abc
from enum import Enum

from Data import Dataset
from Interfaces import Serializable
from Utils import Hashing, Log


class CorpusName(Enum):
    PAN12 = "PAN-12"
    FORMSPRING = "FORMSPRING_v4"

    def __str__(self):
        return self.name


class CorpusParser(Serializable, metaclass=abc.ABCMeta):
    """
    Abstract class to parse the corpora into a representation that
    can be fed to classifiers.
    """

    def __init__(self):
        self.corpus_name = None
        self.source_path = None
        self.kwargs = None
        self._parsed = False

    def __eq__(self, other: 'CorpusParser'):
        return self.source_path == other.source_path

    def __hash__(self):
        parser_hash = f"{self.corpus_name.name}{self.kwargs}"
        return Hashing.sha256_digest(parser_hash)

    @staticmethod
    def factory(corpus_name: CorpusName, source_path: str, **kwargs):
        if corpus_name not in CorpusName:
            raise ValueError(f"Unknown corpus name {corpus_name}")

        corpus_parser = None

        if corpus_name == CorpusName.PAN12:
            from PreProcessing.PAN12 import PAN12Parser
            corpus_parser = PAN12Parser(**kwargs)

        if corpus_name == CorpusName.FORMSPRING:
            from PreProcessing.Formspring import FormspringParser
            corpus_parser = FormspringParser(**kwargs)

        corpus_parser.corpus_name = corpus_name
        corpus_parser.source_path = source_path
        corpus_parser.kwargs = kwargs

        # Deserialize if requested and possible.
        if kwargs.get('deserialize', False) and corpus_parser.is_serialized():
            Log.fine("Deserializing parser...")
            corpus_parser = corpus_parser.deserialize()
            Log.fine("Deserialization completed.")

        return corpus_parser

    def get_params(self):
        return f"Corpus name: {self.corpus_name} - " \
            f"Source path: {self.source_path} - " \
            f"KWArguments: {self.kwargs}"

    def parse(self, serialize: bool = True):
        """
        Parse the dataset provided.
        :return:
        """

        # If already parsed, skip.
        if self._parsed:
            return

        # Delegate the actual parsing the concrete class.
        Log.info(f"Parsing {self.corpus_name} corpus... ")
        self._do_parse()
        self._parsed = True
        Log.info("Parsing done.")

        if serialize:
            self.serialize()

    @abc.abstractmethod
    def _do_parse(self):
        pass

    @abc.abstractmethod
    def dump(self, directory, *args):
        pass

    @abc.abstractmethod
    def add_to_dataset(self, dataset: Dataset):
        """
        Add the parsed information to a dataset object.
        :param dataset:
        :return:
        """
        pass

    @abc.abstractmethod
    def log_info(self):
        """
        Log parsing information.
        :return:
        """
        Log.fine("### PARSER INFO ###", header=True)
        pass
