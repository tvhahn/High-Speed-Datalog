from typing import Any

from .HSDCommands import HSDCmd

class OtherDevHSDCmd(HSDCmd):
    @staticmethod
    def from_dict(self, obj: Any):
        pass

    def to_dict(self):
        pass


class OtherDevHSDGetCmd(OtherDevHSDCmd):
    pass


class OtherDevHSDSetCmd(OtherDevHSDCmd):
    pass


class OtherDevHSDControlCmd(OtherDevHSDCmd):
    pass
