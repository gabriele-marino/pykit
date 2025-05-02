from typing import Any, Type, TypeGuard

from kitpy.type_hints import T


def is_list_of_type(input: Any, expected_type: Type[T]) -> TypeGuard[list[T]]:
    if not isinstance(input, list):
        return False
    return all(isinstance(item, expected_type) for item in input)
