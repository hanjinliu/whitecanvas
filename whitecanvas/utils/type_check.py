from __future__ import annotations

import sys
from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from typing_extensions import TypeGuard


def is_not_array(x) -> bool:
    return np.isscalar(x) or isinstance(x, Enum)


def is_real_number(x) -> TypeGuard[float]:
    return isinstance(x, (int, float, Real, np.number))


def is_pandas_dataframe(df) -> TypeGuard[pd.DataFrame]:
    typ = type(df)
    if (
        typ.__name__ != "DataFrame"
        or "pandas" not in sys.modules
        or typ.__module__.split(".")[0] != "pandas"
    ):
        return False
    import pandas as pd

    return isinstance(df, pd.DataFrame)


def is_polars_dataframe(df) -> TypeGuard[pl.DataFrame]:
    typ = type(df)
    if (
        typ.__name__ != "DataFrame"
        or "polars" not in sys.modules
        or typ.__module__.split(".")[0] != "polars"
    ):
        return False
    import polars as pl

    return isinstance(df, pl.DataFrame)
