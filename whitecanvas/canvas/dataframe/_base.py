from __future__ import annotations

import weakref
from typing import (
    TYPE_CHECKING,
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
    ):
        self._df = df
        self._offsets = offsets
        self._cat_map_cache = {}

    @property
    def df(self) -> DataFrameWrapper[_DF]:
        return self._df

    @property
    def offsets(self) -> tuple[str, ...]:
        return self._offsets

    def category_map(self, columns: tuple[str, ...] | None = None) -> dict[tuple, int]:
        """Calculate how to map category columns to integers."""
        if columns is None:
            key = self._offsets
        else:
            key = tuple(columns)
        if key in self._cat_map_cache:
            return self._cat_map_cache[key]
        if len(key) == 0:
            return {(): 0}
        columns = [self._df[c] for c in key]
        _map = {uni: i for i, uni in enumerate(OrderedSet(zip(*columns)))}
        self._cat_map_cache[key] = _map
        return _map

    def iter_arrays(
        self,
        by: tuple[str, ...],
        dodge: tuple[str, ...] | None = None,
    ) -> Iterator[tuple[tuple, float, DataFrameWrapper[_DF]]]:
        if dodge is None:
            dodge = ()
        if set(self._offsets) > set(by):
            raise ValueError(
                f"offsets must be a subset of by, got offsets={self._offsets!r} and "
                f"by={by!r}"
            )
        indices = [by.index(d) for d in self._offsets]
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
            inv_indices = [by.index(d) for d in dodge]
            _res_map = self.category_map(dodge)
            _nres = len(_res_map)
            _width = 0.8
            dmax = (_nres - 1) / 2 / _nres * _width
            dd = np.linspace(-dmax, dmax, _nres)
            for sl, group in self._df.group_by(by):
                key = tuple(sl[i] for i in indices)
                res = tuple(sl[i] for i in inv_indices)
                yield sl, dd[_res_map[res]] + _map[key], group

    def category_map_with_dodge(
        self,
        by: tuple[str, ...],
        dodge: tuple[str, ...] | None = None,
    ) -> dict[tuple, int]:
        """Category mapping considering dodge."""
        if dodge is None:
            dodge = ()
        if set(self._offsets) > set(by):
            raise ValueError(
                f"offsets must be a subset of by, got offsets={self._offsets!r} and "
                f"by={by!r}"
            )
        indices = [by.index(d) for d in self._offsets]
        _map = self.category_map(self._offsets)
        if not dodge:
            return _map
        out = {}
        if set(self._offsets) & set(dodge):
            raise ValueError(
                f"offsets and dodge must be disjoint, got offsets={self._offsets!r}"
                f" and dodge={dodge!r}"
            )
        inv_indices = [by.index(d) for d in dodge]
        _res_map = self.category_map(dodge)
        _nres = len(_res_map)
        for sl, _ in self._df.group_by(by):
            key = tuple(sl[i] for i in indices)
            res = tuple(sl[i] for i in inv_indices)
            out[sl] = _res_map[res] + _map[key] * _nres
        return out

    def prep_arrays(
        self,
        by: tuple[str, ...],
        value: str,
        dodge: tuple[str, ...] | None = None,
    ) -> tuple[list[float], list[np.ndarray], list[tuple]]:
        x = []
        arrays = []
        categories = []
        for sl, offset, group in self.iter_arrays(by, dodge):
            x.append(offset)
            arrays.append(group[value])
            categories.append(sl)
        return x, arrays, categories

    def prep_position_map(
        self,
        by: tuple[str],
        dodge: tuple[str, ...] | None = None,
    ) -> dict[tuple, float]:
        out = {}
        for sl, offset, _ in self.iter_arrays(by, dodge):
            out[sl] = offset
        return out

    def axis_ticks(self) -> tuple[list[float], list[str]]:
        pos = []
        labels = []
        for k, v in self.category_map(self._offsets).items():
            pos.append(v)
            labels.append("\n".join(map(str, k)))
        return pos, labels

    def axis_label(self) -> str:
        return "/".join(self._offsets)

    def zoom_factor(self, dodge: tuple[str, ...] | None = None) -> float:
        """Return the zoom factor for the given dodge."""
        if dodge:
            _res_map = self.category_map(dodge)
            _nres = len(_res_map)
            if _nres == 1:
                return 1.0
            _width = 0.8
            dmax = (_nres - 1) / 2 / _nres * _width
            return 2 * dmax / (_nres - 1)
        else:
            return 1.0

    def categories(self) -> list[tuple]:
        return list(self.category_map(self._offsets).keys())
