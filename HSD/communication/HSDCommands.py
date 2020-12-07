import abc
from typing import Any

class HSDCmd(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def from_dict(self, obj: Any):
        pass

    @abc.abstractmethod
    def to_dict(self):
        pass