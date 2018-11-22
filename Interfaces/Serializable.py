import abc
import codecs
import pickle

from Interfaces import Factorizable
from Utils import Log, File


class Serializable(metaclass=abc.ABCMeta):
    @classmethod
    def instantiate(cls, *args, **kwargs):
        """
        Automatically instantiate the class.
        First attempts to deserialize the class, if serialized, then checks
        if the class implements Factorizable and attempts to use its factory,
        lastly invokes the class constructor.
        :param args:
        :param kwargs:
        :return: the instantiated class.
        """
        if cls.is_serialized():
            return cls.deserialize()

        # Check if implements Factorizable
        if isinstance(cls, type(Factorizable)):
            cls: Factorizable
            return cls.factory(args, kwargs)
        else:
            return cls()

    @classmethod
    def _get_class_name(cls):
        return cls.__name__

    @classmethod
    def _get_filename(cls):
        return f"./.data/{cls._get_class_name().lower()}.dat"

    @classmethod
    def is_serialized(cls):
        return File.file_exists(cls._get_filename())

    def serialize(self):
        try:
            file = self._get_filename()
            Log.info(f"Serializing to '{file}'... ", newline=False)
            with codecs.open(file, mode='wb+') as f:
                pickle.dump(self, f)
                f.close()
            Log.info("done.", timestamp=False)
        except IOError:
            Log.warning(f"Cannot serialize '{self._get_class_name()}'")

    @classmethod
    def deserialize(cls):
        try:
            file = cls._get_filename()
            Log.info(f"Deserializing {cls._get_class_name()} from '{file}'... ", newline=False)
            with codecs.open(file, mode='rb+') as f:
                deserialized = pickle.load(f)
                f.close()
            Log.info("done.", timestamp=False)
            return deserialized
        except IOError:
            Log.warning(f"Cannot deserialize '{cls._get_class_name()}'")
