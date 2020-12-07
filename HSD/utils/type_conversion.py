from typing import Any, List, TypeVar, Type, Callable, cast
T = TypeVar("T")


class TypeConversion:

    # utilities for data type conversion from Datalog to Python
    @staticmethod
    def check_type(check_type):
        switcher = {
            'uint8_t': 'uint8',
            'uint16_t': 'uint16',
            'uint32_t': 'uint32',
            'int8_t': 'int8',
            'int16_t': 'int16',
            'int32_t': 'int32',
            'float': 'float32',
            'double': 'double',
        }
        return switcher.get(check_type, "error")

    @staticmethod
    def check_type_length(check_type):
        switcher = {
            'uint8_t': 1,
            'int8_t': 1,
            'uint16_t': 2,
            'int16_t': 2,
            'uint32_t': 4,
            'int32_t': 4,
            'float': 4,
            'double': 8,
        }
        return switcher.get(check_type, "error")

    @staticmethod
    def from_str(x: Any) -> str:
        assert isinstance(x, str)
        return x

    @staticmethod
    def from_int(x: Any) -> int:
        assert isinstance(x, int) and not isinstance(x, bool)
        return x

    @staticmethod
    def from_float(x: Any) -> float:
        assert isinstance(x, (float, int)) and not isinstance(x, bool)
        return float(x)

    @staticmethod
    def from_bool(x: Any) -> bool:
        assert isinstance(x, bool)
        return x

    @staticmethod
    def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
        assert isinstance(x, list)
        return [f(y) for y in x]

    @staticmethod
    def to_class(c: Type[T], x: Any) -> dict:
        assert isinstance(x, c)
        return cast(Any, x).to_dict()
