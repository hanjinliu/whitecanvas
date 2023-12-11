from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Iterator, Mapping
from cmap import Color
import numpy as np
from numpy.typing import NDArray

from whitecanvas.types import FacePattern
from ._plans import OffsetPlan, ColorPlan, HatchPlan
from ._utils import unique, unique_product

if TYPE_CHECKING:
    from typing_extensions import TypeGuard
    import pandas as pd
    import polars as pl


class GroupBy(Mapping[tuple[str, ...], dict[str, NDArray[np.number]]]):
    """Generalized groupby object."""

    def __init__(
        self,
        data: dict[tuple[str, ...], dict[str, NDArray[np.number]]],
        by: list[SingleBy],
        labels: np.ndarray,
    ):
        self._obj = data
        self._labels = labels  # keys in array form
        self._by = by
        self._label_counts = {k: len(val) for k, val in self._obj.items()}

    @classmethod
    def parse(cls, data, by: list[SingleBy]) -> GroupBy:
        """
        Parse data into a GroupBy object.

        Note that in the case of multi-level group-by, the left-most column is completely
        sorted. e.g. {"a": [1, 1, 2, 2], "b": [1, 2, 1, 2]} for by=("a", "b").
        """
        nested = len(by) > 0
        bylist = [b._title for b in by]  # TODO: consider labels
        if isinstance(data, dict):
            if nested:
                ar_dict: dict[str, NDArray[np.number]] = {}
                lengths: set[int] = set()
                for k, v in data.items():
                    arr = np.asarray(v)
                    ar_dict[k] = arr
                    lengths.add(arr.size)
                if len(lengths) > 1:
                    raise ValueError(f"Length of array data not consistent: {lengths}.")
                each_uniques = [unique(ar_dict[b]) for b in bylist]
                full = unique_product(each_uniques)
                obj: dict[tuple[str, ...], dict[str, NDArray[np.number]]] = {}
                for unique_val in full:
                    sl = np.all(
                        np.column_stack(
                            [ar_dict[b] == v for b, v in zip(bylist, unique_val)]
                        ),
                        axis=1,
                    )
                    if np.any(sl):
                        dict_filt = {k: v[sl] for k, v in ar_dict.items()}
                        obj[tuple(unique_val)] = dict_filt
            else:
                obj = {(k,): {"value": np.asarray(v)} for k, v in data.items()}
                uniques = np.array(list(data.keys())).reshape(-1, 1)
        elif _is_pandas_dataframe(data):
            # NOTE: pandas groupby sorts the keys
            if nested:
                obj = {cat: val for cat, val in data.groupby(bylist)}
                uniques = np.array(list(obj.keys()))
            else:
                obj = {(c,): data[[c]] for c in data.columns}
                uniques = np.array(list(data.columns)).reshape(-1, 1)
        elif _is_polars_dataframe(data):
            if nested:
                return cls.parse({c.name: c for c in data.iter_columns()}, by)
            else:
                obj = {(c,): data.select(c) for c in data.columns}
                uniques = np.array(list(data.columns)).reshape(-1, 1)
        else:
            raise TypeError(f"{type(data)} cannot be categorized.")
        return GroupBy(obj, bylist, uniques)

    @property
    def by(self) -> list[str]:
        return [b._title for b in self._by]

    def __getitem__(self, key: tuple[str, ...]) -> dict[str, NDArray[np.number]]:
        return self._obj[key]

    def __iter__(
        self,
    ) -> Iterator[tuple[tuple[Any, ...], dict[str, NDArray[np.number]]]]:
        return iter(self._obj.items())

    def __len__(self) -> int:
        return len(self._obj)

    def keys(self) -> Iterator[tuple[Any, ...]]:
        return self._obj.keys()

    def values(self) -> Iterator[dict[str, NDArray[np.number]]]:
        return self._obj.values()

    def get_offsets(self, plan: OffsetPlan) -> list[float]:
        return plan.generate(self._labels, self._by)

    def get_colors(self, plan: ColorPlan) -> list[Color]:
        return plan.generate(self._labels, self._by)

    def get_hatches(self, plan: HatchPlan) -> list[FacePattern]:
        return plan.generate(self._labels, self._by)


class SingleBy:
    def __init__(self, title: str, labels: list[Any] | None = None):
        self._title = title
        self._labels = labels

    def copy(self) -> SingleBy:
        return SingleBy(self._title, self._labels)


class GroupByTask:
    """Object that is ready to be grouped."""

    def __init__(self, data: Any, by: list[SingleBy]):
        if isinstance(data, dict):
            self._columns = list(data.keys())
        elif _is_pandas_dataframe(data):
            self._columns = list(data.columns)
        elif _is_polars_dataframe(data):
            self._columns = list(data.columns)
        else:
            raise TypeError(f"{type(data)} cannot be categorized.")

        # check by is a subset of the columns
        not_found = [b._title for b in by if b._title not in self._columns]
        if len(not_found) > 0:
            raise ValueError(
                f"Column(s) {not_found} not found in data columns {self._columns}."
            )
        self._data = data
        self._by = by

    def more_by(self, by: list[SingleBy]) -> GroupByTask:
        not_found = [b._title for b in by if b._title not in self._columns]
        if len(not_found) > 0:
            raise ValueError(
                f"Column(s) {not_found} not found in data columns {self._columns}."
            )
        by2 = self._by + [b for b in by if b not in self._by]
        return GroupByTask(self._data, by2)

    def as_groupby(self) -> GroupBy:
        return GroupBy.parse(self._data, self._by)

    def with_selections(self, kwargs: dict[str, list[Any]]) -> GroupByTask:
        by = [b.copy() for b in self._by]
        for b in by:
            if (labels := kwargs.pop(b._title, None)) is not None:
                b._labels = labels
        if not_found := list(kwargs.keys()):
            raise ValueError(
                f"Columns {not_found} not found in groupby columns {self._columns}."
            )
        return GroupByTask(self._data, by)

    @property
    def by(self) -> list[str]:
        return [b._title for b in self._by]


def _is_pandas_dataframe(df) -> TypeGuard[pd.DataFrame]:
    typ = type(df)
    if (
        typ.__name__ != "DataFrame"
        or "pandas" not in sys.modules
        or typ.__module__.split(".")[0] != "pandas"
    ):
        return False
    import pandas as pd

    return isinstance(df, pd.DataFrame)


def _is_polars_dataframe(df) -> TypeGuard[pl.DataFrame]:
    typ = type(df)
    if (
        typ.__name__ != "DataFrame"
        or "polars" not in sys.modules
        or typ.__module__.split(".")[0] != "polars"
    ):
        return False
    import polars as pl

    return isinstance(df, pl.DataFrame)


def _sort_by_key(d: dict[str, Any]) -> dict[str, Any]:
    return dict(sorted(d.items(), key=lambda x: x[0]))
