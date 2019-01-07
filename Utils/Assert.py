from enum import Enum
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
    def equal(obj1: Any, obj2: Any, message: str = None):
        assert obj1 == obj2, f"The objects are not equal. {str(obj1)}, {str(obj2)}" if message is None else message

    @staticmethod
    def true(var: Any, message: str = None):
        assert var is True, "The variable is not true." if message is None else message

    @staticmethod
    def valid_enum(var: Any, message: str = None):
        assert var in var.__class__, f"Unrecognised enum {var.name}" if message is None else message
