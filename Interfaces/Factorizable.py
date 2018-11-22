import abc


class Factorizable(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def factory(*args, **kwargs):
        pass
