from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Iterator, TypeVar

import numpy as np
from numpy.typing import NDArray

from whitecanvas.utils.type_check import is_pandas_dataframe, is_polars_dataframe

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401
    import polars as pl  # noqa: F401
    from typing_extensions import Self

_T = TypeVar("_T")


class DataFrameWrapper(ABC, Generic[_T]):
    def __init__(self, data: _T):
        self._data = data

    def __repr__(self) -> str:
        return f"{type(self).__name__} of {self._data!r}"

    def __len__(self) -> int:
        such_as = next(self.iter_values(), None)
        if such_as is None:
            return 0
        else:
            return such_as.size

    @property
    def shape(self) -> tuple[int, int]:
        such_as = next(self.iter_values(), None)
        if such_as is None:
            return 0, 0
        else:
            return such_as.size, len(self.columns)

    def get_native(self) -> _T:
        return self._data

    @abstractmethod
    def __getitem__(self, item: str) -> NDArray[np.generic]: ...

    def __contains__(self, item: str) -> bool:
        return item in self.iter_keys()

    @abstractmethod
    def iter_keys(self) -> Iterator[str]: ...

    def iter_values(self) -> Iterator[NDArray[np.generic]]:
        for k in self.iter_keys():
            yield self[k]

    def iter_items(self) -> Iterator[tuple[str, NDArray[np.generic]]]:
        for k in self.iter_keys():
            yield k, self[k]

    @abstractmethod
    def select(self, columns: list[str]) -> Self:
        """Select columns from the data frame and return a new one."""

    @abstractmethod
    def sort(self, by: str) -> Self:
        """Sort the data frame by a column and return the sorted one."""

    @abstractmethod
    def get_rows(self, indices: list[int]) -> Self:
        """Get rows by indices and return a new data frame."""

    @abstractmethod
    def filter(
        self,
        by: tuple[str, ...],
        values: tuple[Any, ...],
    ) -> Self: ...

    @abstractmethod
    def group_by(
        self, by: tuple[str, ...]
    ) -> Iterator[tuple[tuple[Any, ...], Self]]: ...

    @abstractmethod
    def agg_by(self, by: tuple[str, ...], on: list[str], method: str) -> Self: ...

    @abstractmethod
    def melt(self, id_vars: list[str], value_vars: list[str]) -> Self: ...

    @abstractmethod
    def value_count(self, by: tuple[str, ...]) -> Self:
        """Return the count of each group."""

    @abstractmethod
    def value_first(self, by: tuple[str, ...], on: str) -> Self:
        """Return the first value of a column for each group."""

    @property
    def columns(self) -> list[str]:
        """Column names of the data frame."""
        return list(self.iter_keys())


class DictWrapper(DataFrameWrapper[dict[str, np.ndarray]]):
    def __getitem__(self, item: str) -> np.ndarray:
        if not isinstance(item, str):
            raise TypeError(f"Unsupported type: {type(item)}")
        try:
            return self._data[item]
        except KeyError:
            raise KeyError(
                f"{item!r} not in the keys. Valid keys are: {list(self._data.keys())}."
            ) from None

    def iter_keys(self) -> Iterator[str]:
        return iter(self._data.keys())

    def select(self, columns: list[str]) -> Self:
        return DictWrapper({k: v for k, v in self._data.items() if k in columns})

    def sort(self, by: str) -> Self:
        arr = self[by]
        indices = np.argsort(arr)
        return DictWrapper({k: v[indices] for k, v in self._data.items()})

    def get_rows(self, indices: list[int]) -> Self:
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
        if by == ():
            yield (), self
            return
        observed = set()
        for row in zip(*[self._data[b] for b in by]):
            if row in observed:
                continue
            yield row, self.filter(by, row)
            observed.add(row)

    def agg_by(self, by: tuple[str, ...], on: list[str], method: str) -> Self:
        if method not in ("min", "max", "mean", "median", "sum", "std"):
            raise ValueError(f"Unsupported aggregation method: {method}")
        agg = getattr(np, method)
        out = {k: [] for k in (*by, *on)}
        for sl, sub in self.group_by(by):
            for b, s in zip(by, sl):
                out[b].append(s)
            for o in on:
                out[o].append(agg(sub[o]))
        return DictWrapper({k: np.array(v) for k, v in out.items()})

    def melt(
        self,
        id_vars: list[str],
        value_vars: list[str],
        var_name: str | None = None,
        value_name: str | None = None,
    ) -> Self:
        if var_name is None:
            var_name = "variable"
        if value_name is None:
            value_name = "value"
        out = {k: [] for k in [*id_vars, var_name, value_name]}
        for k, v in self._data.items():
            if k in id_vars:
                out[k].extend(v)
            elif k in value_vars:
                out[var_name] += [k] * len(v)
                out[value_name].extend(v)
            else:
                pass
        return DictWrapper({k: np.array(v) for k, v in out.items()})

    def value_count(self, by: tuple[str, ...]) -> Self:
        out = {k: [] for k in [*by, "size"]}
        for sl, sub in self.group_by(by):
            for b, s in zip(by, sl):
                out[b].append(s)
            out["size"].append(len(sub[by[0]]))
        return DictWrapper({k: np.array(v) for k, v in out.items()})

    def value_first(self, by: tuple[str, ...], on: str) -> Self:
        out = {k: [] for k in [*by, on]}
        for sl, sub in self.group_by(by):
            for b, s in zip(by, sl):
                out[b].append(s)
            out[on].append(sub[on][0])
        return DictWrapper({k: np.array(v) for k, v in out.items()})


class PandasWrapper(DataFrameWrapper["pd.DataFrame"]):
    def __getitem__(self, item: str) -> np.ndarray:
        if not isinstance(item, str):
            raise TypeError(f"Unsupported type: {type(item)}")
        series = self._data[item]
        if series.size > 0 and isinstance(series.iloc[0], str):
            return series.to_numpy().astype(str)
        else:
            return series.to_numpy()

    def iter_keys(self) -> Iterator[str]:
        return iter(self._data.columns)

    def select(self, columns: list[str]) -> Self:
        return PandasWrapper(self._data[columns])

    def sort(self, by: str) -> Self:
        return PandasWrapper(self._data.sort_values(by))

    def get_rows(self, indices: list[int]) -> Self:
        return PandasWrapper(self._data.iloc[indices])

    def filter(
        self,
        by: tuple[str, ...],
        values: tuple[Any, ...],
    ) -> Self:
        sers = np.column_stack([self._data[b] == val for b, val in zip(by, values)])
        return PandasWrapper(self._data[sers])

    def group_by(self, by: tuple[str, ...]) -> Iterator[tuple[tuple[Any, ...], Self]]:
        if by == ():
            yield (), self
            return
        for sl, sub in self._data.groupby(list(by), observed=True):
            yield sl, PandasWrapper(sub)

    def agg_by(self, by: tuple[str, ...], on: list[str], method: str) -> Self:
        on = list(on)
        return PandasWrapper(self._data.groupby(list(by))[on].agg(method).reset_index())

    def melt(
        self,
        id_vars: list[str],
        value_vars: list[str],
        var_name: str | None = None,
        value_name: str | None = None,
    ) -> Self:
        if value_name is None:
            value_name = "value"
        return PandasWrapper(
            self._data.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=var_name,
                value_name=value_name,
                ignore_index=True,
            )
        )

    def value_count(self, by: tuple[str, ...]) -> Self:
        import pandas as pd  # noqa: F811, RUF100

        rows = []
        for sl, sub in self._data.groupby(list(by), observed=True):
            rows.append((*sl, len(sub)))
        return PandasWrapper(pd.DataFrame(rows, columns=[*by, "size"]))

    def value_first(self, by: tuple[str, ...], on: str) -> Self:
        return PandasWrapper(self._data.groupby(list(by)).first().reset_index())


class PolarsWrapper(DataFrameWrapper["pl.DataFrame"]):
    def __getitem__(self, item: str) -> np.ndarray:
        if not isinstance(item, str):
            raise TypeError(f"Unsupported type: {type(item)}")
        try:
            return self._data[item].to_numpy()
        except Exception as e:
            import polars as pl  # noqa: F811, RUF100

            if isinstance(e, pl.ColumnNotFoundError):
                raise KeyError(item) from None
            raise e

    def iter_keys(self) -> Iterator[str]:
        return iter(self._data.columns)

    def select(self, columns: list[str]) -> Self:
        return PolarsWrapper(self._data.select(columns))

    def sort(self, by: str) -> Self:
        return PolarsWrapper(self._data.sort(by))

    def get_rows(self, indices: list[int]) -> Self:
        return PolarsWrapper(self._data[indices])

    def filter(
        self,
        by: tuple[str, ...],
        values: tuple[Any, ...],
    ) -> Self:
        kwargs = dict(zip(by, values))
        df = self._data.filter(**kwargs)
        return PolarsWrapper(df)

    def group_by(self, by: tuple[str, ...]) -> Iterator[tuple[tuple[Any, ...], Self]]:
        if by == ():
            yield (), self
            return
        for sl, sub in self._data.group_by(by, maintain_order=True):
            yield sl, PolarsWrapper(sub)

    def agg_by(self, by: tuple[str, ...], on: list[str], method: str) -> Self:
        import polars as pl  # noqa: F811, RUF100

        exprs = [getattr(pl.col(o), method)() for o in on]
        return PolarsWrapper(self._data.group_by(by, maintain_order=True).agg(*exprs))

    def melt(
        self,
        id_vars: list[str],
        value_vars: list[str],
        var_name: str | None = None,
        value_name: str | None = None,
    ) -> Self:
        return PolarsWrapper(
            self._data.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=var_name,
                value_name=value_name,
            )
        )

    def value_count(self, by: tuple[str, ...]) -> Self:
        return PolarsWrapper(
            self._data.group_by(by, maintain_order=True)
            .count()
            .rename({"count": "size"})
        )

    def value_first(self, by: tuple[str, ...], on: str) -> Self:
        return PolarsWrapper(self._data.group_by(by, maintain_order=True).first())


def parse(data: Any) -> DataFrameWrapper:
    """Parse a data object into a DataFrameWrapper."""
    if isinstance(data, DataFrameWrapper):
        return data
    if isinstance(data, dict):
        df = {k: np.asarray(v) for k, v in data.items()}
        if len(df) > 0:  # check length
            if len({v.size for v in df.values()}) > 1:
                raise ValueError("All columns must have the same length")
        return DictWrapper(df)
    elif is_pandas_dataframe(data):
        return PandasWrapper(data)
    elif is_polars_dataframe(data):
        return PolarsWrapper(data)
    elif hasattr(data, "__dataframe__"):
        df_interchangable = data.__dataframe__()
        return DictWrapper({k: np.asarray(v) for k, v in df_interchangable.items()})
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
