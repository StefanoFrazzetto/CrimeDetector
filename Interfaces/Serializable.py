import abc
import pickle

from Utils import Log, File


class Serializable(metaclass=abc.ABCMeta):

    @classmethod
    def _get_class_name(cls):
        return cls.__name__

    def __get_filename(self):
        return f"{self._get_class_name().lower()}_{self.__hash__()}.dat"

    def _get_file_path(self):
        return f"./.data/{self.__get_filename()}"

    def is_serialized(self):
        return File.file_exists(self._get_file_path())

    def serialize(self):
        try:
            file = self._get_file_path()
            Log.info(f"Serializing to '{file}'... ", newline=False)
            with open(file, mode='wb+') as f:
                pickle.dump(self, f)
                f.close()
            Log.info("done.", timestamp=False)
        except IOError:
            Log.warning(f"Cannot serialize '{self._get_class_name()}'")

    def deserialize(self):
        try:
            file = self._get_file_path()
            Log.info(f"Deserializing {self._get_class_name()} from '{file}'... ", newline=False)
            with open(file, mode='rb+') as f:
                deserialized = pickle.load(f)
                f.close()
            Log.info("done.", timestamp=False)
            return deserialized
        except IOError as e:
            Log.warning(f"Cannot deserialize '{self._get_class_name()}'. {e}")

        except Exception as e:
            Log.warning(str(e))
