import codecs
import pickle

from Utils import Log, File


class Serializable(object):
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
            Log.info(f"Deserializing class from '{file}'... ", newline=False)
            with codecs.open(file, mode='rb+') as f:
                deserialized = pickle.load(f)
                f.close()
            Log.info("done.", timestamp=False)
            return deserialized
        except IOError:
            Log.warning(f"Cannot deserialize '{cls._get_class_name()}'")
