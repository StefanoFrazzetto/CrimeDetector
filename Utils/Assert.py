from typing import Any


class Assert:
    @staticmethod
    def not_empty(obj: list, message: str = None):
        assert len(obj) > 0, "The list is empty." if message is None else message

    @staticmethod
    def not_none(obj: Any, message: str = None):
        assert obj is not None, "The object value is None." if message is None else message

    @staticmethod
    def same_length(obj1: Any, obj2: Any, message: str = None):
        assert len(obj1) == len(obj2), "The objects have different lengths." if message is None else message

    @staticmethod
    def true(var: Any, message: str = None):
        assert var is True, "The variable is not true." if message is None else message
