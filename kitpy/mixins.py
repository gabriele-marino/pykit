from typing import Type

import yaml

from kitpy.type_hints import T


class YAMLMixin:
    @classmethod
    def from_yaml(cls: Type[T], yaml_file: str) -> T:
        with open(yaml_file, "r") as f:
            configs = yaml.safe_load(f)
        return cls(**configs)
