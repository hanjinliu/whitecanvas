from __future__ import annotations

import sys
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Any, Iterator, TypeVar, Generic

import numpy as np

from ._utils import unique_product

if TYPE_CHECKING:
    from typing_extensions import Self
    import pandas as pd
    import polars as pl
    import pyarrow as pa

_T = TypeVar("_T")


class DataFrameWrapper(ABC, Generic[_T]):
    def __init__(self, data: _T):
        self._data = data

    def __repr__(self) -> str:
        return f"{type(self).__name__} of {self._data!r}"

    def __len__(self) -> int:
        such_as = next(iter(self.iter_values()), None)
        if such_as is None:
            return 0
        else:
            return such_as.size

    @property
    def shape(self) -> tuple[int, int]:
        such_as = next(iter(self.iter_values()), None)
        if such_as is None:
            return 0, 0
        else:
            return such_as.size, len(self.iter_keys())

    def get_native(self) -> _T:
        return self._data

    @abstractmethod
    def __getitem__(self, item: str) -> np.ndarray:
        ...

    @abstractmethod
    def iter_keys(self) -> Iterator[str]:
        ...

    def iter_values(self) -> Iterator[np.ndarray]:
        for k in self.iter_keys():
            yield self[k]

    def iter_items(self) -> Iterator[tuple[str, np.ndarray]]:
        for k in self.iter_keys():
            yield k, self[k]

    @abstractmethod
    def select(self, columns: list[str]) -> Self:
        ...

    @abstractmethod
    def sort(self, by: str) -> Self:
        ...

    @abstractmethod
    def filter(
        self,
        by: tuple[str, ...],
        values: tuple[Any, ...],
    ) -> Self:
        ...

    @abstractmethod
    def group_by(self, by: tuple[str, ...]) -> Iterator[tuple[tuple[Any, ...], Self]]:
        ...


class DictWrapper(DataFrameWrapper[dict[str, np.ndarray]]):
    def __getitem__(self, item: str) -> np.ndarray:
        return self._data[item]

    def iter_keys(self) -> Iterator[str]:
        return iter(self._data.keys())

    def select(self, columns: list[str]) -> Self:
        return DictWrapper({k: v for k, v in self._data.items() if k in columns})

    def sort(self, by: str) -> Self:
        arr = self[by]
        indices = np.argsort(arr)
        return DictWrapper({k: v[indices] for k, v in self._data.items()})

    def filter(
        self,
        by: tuple[str, ...],
        values: tuple[Any, ...],
    ) -> Self:
        sers = np.column_stack([self._data[b] == val for b, val in zip(by, values)])
        sl = sers.all(axis=1)
        return DictWrapper({k: v[sl] for k, v in self._data.items()})

    def group_by(self, by: tuple[str, ...]) -> Iterator[tuple[tuple[Any, ...], Self]]:
        for values in unique_product(*[self._data[b] for b in by]):
            yield values, self.filter(by, values)


class PandasWrapper(DataFrameWrapper["pd.DataFrame"]):
    def __getitem__(self, item: str) -> np.ndarray:
        series = self._data[item]
        if series.size > 0 and isinstance(series[0], str):
            return series.to_numpy().astype(str)
        else:
            return series.to_numpy()

    def iter_keys(self) -> Iterator[str]:
        return iter(self._data.columns)

    def select(self, columns: list[str]) -> Self:
        return PandasWrapper(self._data[columns])

    def sort(self, by: str) -> Self:
        return PandasWrapper(self._data.sort_values(by))

    def filter(
        self,
        by: tuple[str, ...],
        values: tuple[Any, ...],
    ) -> Self:
        sers = np.column_stack([self._data[b] == val for b, val in zip(by, values)])
        return PandasWrapper(self._data[sers])

    def group_by(self, by: tuple[str, ...]) -> Iterator[tuple[tuple[Any, ...], Self]]:
        for sl, sub in self._data.groupby(by, observed=True):
            yield sl, PandasWrapper(sub)


class PolarsWrapper(DataFrameWrapper["pl.DataFrame"]):
    def __getitem__(self, item: str) -> np.ndarray:
        try:
            return self._data[item].to_numpy()
        except Exception as e:
            import polars as pl

            if isinstance(e, pl.ColumnNotFoundError):
                raise KeyError(item) from None
            raise e

    def iter_keys(self) -> Iterator[str]:
        return iter(self._data.columns)

    def select(self, columns: list[str]) -> Self:
        return PolarsWrapper(self._data.select(columns))

    def sort(self, by: str) -> Self:
        return PolarsWrapper(self._data.sort(by))

    def filter(
        self,
        by: tuple[str, ...],
        values: tuple[Any, ...],
    ) -> Self:
        kwargs = {b: val for b, val in zip(by, values)}
        df = self._data.filter(**kwargs)
        return PolarsWrapper(df)

    def group_by(self, by: tuple[str, ...]) -> Iterator[tuple[tuple[Any, ...], Self]]:
        for sl, sub in self._data.group_by(by, maintain_order=True):
            yield sl, PolarsWrapper(sub)


class PyArrowWrapper(DataFrameWrapper["pa.Table"]):
    def __getitem__(self, item: str) -> np.ndarray:
        return self._data[item].to_numpy()

    def iter_keys(self) -> Iterator[str]:
        return iter(self._data.column_names)

    def select(self, columns: list[str]) -> Self:
        return PyArrowWrapper(self._data.select(columns))

    def sort(self, by: str) -> Self:
        return PyArrowWrapper(self._data.sort_by(by))

    def filter(
        self,
        by: tuple[str, ...],
        values: tuple[Any, ...],
    ) -> Self:
        kwargs = {b: val for b, val in zip(by, values)}
        df = self._data.filter(**kwargs)
        return PyArrowWrapper(df)

    def group_by(self, by: tuple[str, ...]) -> Iterator[tuple[tuple[Any, ...], Self]]:
        for sl, sub in self._data.group_by(by, maintain_order=True):
            yield sl, PyArrowWrapper(sub)


def parse(data: Any) -> DataFrameWrapper:
    """Parse a data object into a DataFrameWrapper."""
    if isinstance(data, dict):
        df = {k: np.asarray(v) for k, v in data.items()}
        if len(df) > 0:  # check length
            if len({v.size for v in df.values()}) > 1:
                raise ValueError("All columns must have the same length")
        return DictWrapper(df)
    elif _is_pandas_dataframe(data):
        return PandasWrapper(data)
    elif _is_polars_dataframe(data):
        return PolarsWrapper(data)
    elif _is_pyarrow_table(data):
        return PyArrowWrapper(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def _is_pandas_dataframe(df) -> bool:
    typ = type(df)
    if (
        typ.__name__ != "DataFrame"
        or "pandas" not in sys.modules
        or typ.__module__.split(".")[0] != "pandas"
    ):
        return False
    import pandas as pd

    return isinstance(df, pd.DataFrame)


def _is_polars_dataframe(df) -> bool:
    typ = type(df)
    if (
        typ.__name__ != "DataFrame"
        or "polars" not in sys.modules
        or typ.__module__.split(".")[0] != "polars"
    ):
        return False
    import polars as pl

    return isinstance(df, pl.DataFrame)


def _is_pyarrow_table(df) -> bool:
    typ = type(df)
    if (
        typ.__name__ != "Table"
        or "pyarrow" not in sys.modules
        or typ.__module__.split(".")[0] != "pyarrow"
    ):
        return False
    import pyarrow as pa

    return isinstance(df, pa.Table)
