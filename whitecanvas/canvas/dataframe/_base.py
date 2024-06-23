from __future__ import annotations

import weakref
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Iterator,
    Literal,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np

from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas.layers.tabular import parse
from whitecanvas.utils.collections import OrderedSet

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.layers.tabular._dataframe import DataFrameWrapper

_C = TypeVar("_C")  # NOTE: don't have to be a canvas
_T = TypeVar("_T")
_DF = TypeVar("_DF")
NStr = Union[str, Sequence[str]]
AggMethods = Literal["min", "max", "mean", "median", "sum", "std"]


class BaseCatPlotter(Generic[_C, _DF]):
    def __init__(
        self,
        canvas: _C,
        df: _DF,
    ):
        self._canvas_ref = weakref.ref(canvas)
        self._df: DataFrameWrapper[_DF] = parse(df)

    def _canvas(self) -> _C:
        canvas = self._canvas_ref()
        if canvas is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        return canvas

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        pass


class CatIterator(Generic[_DF]):
    def __init__(
        self,
        df: DataFrameWrapper[_DF],
        offsets: tuple[str, ...],
        numeric: bool = False,
        sort_func: Callable[[Sequence[_T]], Sequence[_T]] | None = None,
    ):
        self._df = df
        self._offsets = tuple(offsets)
        self._cat_map_cache = {}
        self._numeric = numeric
        if sort_func is None:
            sort_func = lambda x: x  # noqa: E731
        elif numeric:
            raise ValueError("sort_func is not allowed for numeric data.")
        self._sort_func = sort_func

    @property
    def df(self) -> DataFrameWrapper[_DF]:
        return self._df

    @property
    def offsets(self) -> tuple[str, ...]:
        return self._offsets

    def category_map(self, columns: tuple[str, ...]) -> dict[tuple, int]:
        """Calculate how to map category columns to integers."""
        columns = tuple(columns)
        if columns in self._cat_map_cache:
            return self._cat_map_cache[columns]
        if len(columns) == 0:
            return {(): 0}
        serieses = [self._df[c] for c in columns]
        if columns == self._offsets:
            categories = self._sort_func(OrderedSet(zip(*serieses)))
        else:
            categories = OrderedSet(zip(*serieses))
        _map = {uni: i for i, uni in enumerate(categories)}
        self._cat_map_cache[columns] = _map
        return _map

    def iter_arrays(
        self,
        by: tuple[str, ...],
        dodge: tuple[str, ...] | None = None,
        width: float = 0.8,
    ) -> Iterator[tuple[tuple, float, DataFrameWrapper[_DF]]]:
        """
        Iterate over the groups of the DataFrame for plotting.

        Returns
        -------
        Iterator of (tuple, float, DataFrameWrapper[_DF])
            The first tuple is the group key, the second float is the x (or y)
            coordinate the group should be plotted at, and the third is the subset
            of the DataFrame that corresponds to the group.
        """
        if dodge is None:
            dodge = ()
        if set(self._offsets) > set(by):
            raise ValueError(
                f"offsets must be a subset of by, got offsets={self._offsets!r} and "
                f"by={by!r}"
            )
        indices = [by.index(d) for d in self._offsets]
        if self._numeric:
            _map = NumericMap()
        else:
            _map = self.category_map(self._offsets)
        if not dodge:
            for sl, group in self._df.group_by(by):
                key = tuple(sl[i] for i in indices)
                yield sl, _map[key], group
        else:
            if set(self._offsets) & set(dodge):
                raise ValueError(
                    f"offsets and dodge must be disjoint, got offsets={self._offsets!r}"
                    f" and dodge={dodge!r}"
                )
            if self._numeric:
                _pos = list(self.prep_position_map(self._offsets, dodge=False).values())
                _width = np.diff(np.sort(_pos)).min() * width
            else:
                _width = width
            inv_indices = [by.index(d) for d in dodge]
            _res_map = self.category_map(dodge)
            _nres = len(_res_map)
            dmax = (_nres - 1) / 2 / _nres * _width
            dd = np.linspace(-dmax, dmax, _nres)
            for sl, group in self._df.group_by(by):
                key = tuple(sl[i] for i in indices)
                res = tuple(sl[i] for i in inv_indices)
                yield sl, dd[_res_map[res]] + _map[key], group

    def prep_arrays(
        self,
        by: tuple[str, ...],
        value: str,
        dodge: tuple[str, ...] | None = None,
        width: float = 0.8,
    ) -> tuple[list[float], list[np.ndarray], list[tuple]]:
        x = []
        arrays = []
        categories = []
        for sl, offset, group in self.iter_arrays(by, dodge, width=width):
            x.append(offset)
            arrays.append(group[value])
            categories.append(sl)
        return x, arrays, categories

    def prep_position_map(
        self,
        by: tuple[str],
        dodge: tuple[str, ...] | None = None,
        width: float = 0.8,
    ) -> dict[tuple, float]:
        out = {}
        for sl, offset, _ in self.iter_arrays(by, dodge=dodge, width=width):
            out[sl] = offset
        return out

    def axis_ticks(self) -> tuple[list[float], list[str]]:
        """Prepare the axis ticks and labels for the category plot."""
        pos: list[float] = []
        labels: list[str] = []
        for k, v in self.category_map(self._offsets).items():
            pos.append(v)
            labels.append("\n".join(map(str, k)))
        pos_indices = np.argsort(pos)
        return [pos[i] for i in pos_indices], [labels[i] for i in pos_indices]

    def axis_label(self) -> str:
        return "/".join(self._offsets)

    def zoom_factor(
        self,
        dodge: tuple[str, ...] | None = None,
        width: float = 0.8,
    ) -> float:
        """Return the zoom factor for the given dodge."""
        if dodge:
            _res_map = self.category_map(dodge)
            _nres = len(_res_map)
            if _nres == 1:
                return 1.0
            dmax = (_nres - 1) / 2 / _nres * width
            return 2 * dmax / (_nres - 1)
        else:
            return 1.0

    def categories(self) -> list[tuple]:
        return list(self.category_map(self._offsets).keys())


class NumericMap:
    def __getitem__(self, key: tuple[float]) -> float:
        return key[0]
