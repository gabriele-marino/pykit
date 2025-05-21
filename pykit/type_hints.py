from typing import Sequence, TypeVar, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

T = TypeVar("T")

Numeric = float | int

OneOrSequence = Union[T | Sequence[T]]
OneOrMatrix = Union[T | Sequence[Sequence[T]]]

Vector = OneOrSequence[Numeric]

ArrayLike = list[float] | NDArray[np.float64] | pd.Series[float]
