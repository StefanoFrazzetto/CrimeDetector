import abc
from enum import Enum

from nltk import LancasterStemmer, SnowballStemmer, PorterStemmer


class StemmerType(Enum):
    PORTER = 0,
    SNOWBALL = 1,
    LANCASTER = 2


class Stemmer(metaclass=abc.ABCMeta):
    @staticmethod
    def factory(stemmer_type=StemmerType.SNOWBALL, language="english", ignore_stop_words=False):
        # Check stemmer_type is recognised
        assert stemmer_type in StemmerType, f"Unrecognised stemmer type {stemmer_type.name}"

        stemmer = NLTKStemmer(language, ignore_stop_words)
        if stemmer_type == StemmerType.PORTER:
            return stemmer.get_porter_stemmer()

        if stemmer_type == StemmerType.SNOWBALL:
            return stemmer.get_snowball_stemmer()

        if stemmer_type == StemmerType.LANCASTER:
            return stemmer.get_lancaster_stemmer()


class NLTKStemmer(Stemmer):

    def __init__(self, language="english", ignore_stop_words=False):
        self.language = language
        self.ignore_stop_words = ignore_stop_words

    @staticmethod
    def get_porter_stemmer():
        """Get Porter 2 stemmer with NLTK extensions."""
        return PorterStemmer(mode=nltk.PorterStemmer.NLTK_EXTENSIONS)

    def get_snowball_stemmer(self):
        """Get Snowball (Porter 2) stemmer."""
        return SnowballStemmer(
            language=self.language,
            ignore_stopwords=self.ignore_stop_words
        )

    @staticmethod
    def get_lancaster_stemmer():
        """Get Lancaster (Paice-Husk) stemmer."""
        return LancasterStemmer()
