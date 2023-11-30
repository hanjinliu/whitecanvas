from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterator,
    Literal,
    Sequence,
    Generic,
    TypeVar,
)
import weakref
from cmap import Color
import numpy as np
from numpy.typing import NDArray

from whitecanvas import layers as _l
from whitecanvas.layers import group as _lg
from whitecanvas.types import (
    LineStyle,
    Symbol,
    ColorType,
    ColormapType,
    FacePattern,
    Orientation,
)
from whitecanvas.canvas._palette import ColorPalette
from whitecanvas.utils.normalize import as_array_1d

from whitecanvas._exceptions import ReferenceDeletedError

if TYPE_CHECKING:
    from typing_extensions import TypeGuard
    from ._base import CanvasBase
    import pandas as pd
    import polars as pl

_C = TypeVar("_C", bound="CanvasBase")
_T = TypeVar("_T")


class CategorizedStruct(Generic[_C, _T]):
    def __init__(
        self,
        canvas: _C,
        obj: dict[str, dict[str, _T]],
        offsets: NDArray[np.floating],
        orient: Orientation = Orientation.VERTICAL,
        palette: ColormapType | None = None,
    ):
        self._canvas_ref = weakref.ref(canvas)
        self._offsets = offsets
        self._offsets_initial = offsets.copy()
        self._color_palette = palette
        self._obj = obj
        self._orient = orient

    def _canvas(self) -> _C:
        canvas = self._canvas_ref()
        if canvas is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        return canvas

    @property
    def n_categories(self) -> int:
        """Number of categories."""
        return len(self._obj)

    @property
    def categories(self) -> list[Any]:
        """List of categories."""
        return list(self._obj.keys())

    def _generate_colors(self, color) -> list[Color]:
        if color is not None:
            return color
        return self._canvas()._color_palette.nextn(self.n_categories, update=False)

    def _generate_x(self) -> NDArray[np.floating]:
        x = np.arange(self.n_categories, dtype=np.float64)
        return x + self._offsets

    def _get_backend(self):
        return self._canvas()._get_backend()

    def keys(self) -> Iterator[str]:
        """Iterate over categories."""
        return self._obj.keys()

    def values(self) -> Iterator[dict[str, _T]]:
        """Iterate over data."""
        return self._obj.values()

    def items(self) -> Iterator[tuple[str, dict[str, _T]]]:
        """Iterate over (category, data) pairs."""
        return self._obj.items()


class CategorizedDataPlotter(CategorizedStruct[_C, NDArray[np.number]]):
    def __init__(
        self,
        canvas: _C,
        data: Any,
        by: str | None = None,
        *,
        orient: Orientation = Orientation.VERTICAL,
        offsets=None,
        palette: ColormapType | None = None,
        update_label: bool = False,
        unsafe: bool = False,
    ):
        if unsafe:
            _color_palette = palette
            obj = data
            ncats = len(obj)
        else:
            if palette is None:
                _color_palette = canvas._color_palette.copy()
            else:
                _color_palette = ColorPalette(palette)
            obj = _norm_input(data, by)
            ncats = len(obj)
        if offsets is None:
            offsets = np.zeros(ncats)
        elif isinstance(offsets, (int, float, np.number)):
            offsets = np.full(ncats, offsets)
        else:
            offsets = np.asarray(offsets)
            if offsets.shape != (ncats,):
                raise ValueError("Shape of offset is wrong")
        self._update_label = update_label
        super().__init__(canvas, obj, offsets, orient, _color_palette)

    def with_offset(self, offset: float) -> CategorizedDataPlotter[_C]:
        """Update offset of the plotter."""
        return CategorizedDataPlotter(
            self._canvas(),
            self._obj,
            offsets=offset,
            orient=self._orient,
            palette=self._color_palette,
            update_label=self._update_label,
            unsafe=True,
        )

    def mean(self) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, np.mean) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._orient, self._color_palette
        )

    def min(self) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, np.min) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._orient, self._color_palette
        )

    def max(self) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, np.max) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._orient, self._color_palette
        )

    def std(self, ddof: int = 1) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, np.std, ddof=ddof) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._orient, self._color_palette
        )

    def var(self, ddof: int = 1) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, np.var, ddof=ddof) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._orient, self._color_palette
        )

    def sum(self) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, np.sum) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._orient, self._color_palette
        )

    def count(self) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, len) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._orient, self._color_palette
        )

    def sem(self, ddof: int = 1) -> CategorizedAggDataPlotter[_C]:
        agged = {
            k: _aggregate(v, lambda x: np.std(x, ddof=ddof) / np.sqrt(len(x)))
            for k, v in self.items()
        }
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._orient, self._color_palette
        )

    def to_stripplot(
        self,
        y: str | None = None,
        *,
        name: str | None = None,
        strip_width: float = 0.3,
        color: ColorType | Sequence[ColorType] | None = None,
        alpha: float = 1.0,
        symbol: str | Symbol = Symbol.CIRCLE,
        size: float = 10,
        seed: int | None = 0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _lg.MarkerCollection:
        canvas = self._canvas()
        name = canvas._coerce_name("stripplot", name)
        color = self._generate_colors(color)
        data = self._generate_y(y)
        group = _lg.MarkerCollection.build_strip(
            self._generate_x(), data, name=name, orient=self._orient,
            strip_width=strip_width, seed=seed, symbol=symbol, size=size,
            color=color, alpha=alpha, pattern=pattern, backend=self._get_backend()
        )  # fmt: skip
        self._relabel_axis(y)
        return canvas.add_layer(group)

    def to_swarmplot(
        self,
        y: str | None = None,
        *,
        name: str | None = None,
        strip_width: float = 0.3,
        color: ColorType | Sequence[ColorType] | None = None,
        alpha: float = 1.0,
        symbol: str | Symbol = Symbol.CIRCLE,
        size: float = 10,
        sort: bool = False,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _lg.MarkerCollection:
        canvas = self._canvas()
        name = canvas._coerce_name("swarmplot", name)
        color = self._generate_colors(color)
        data = self._generate_y(y)
        group = _lg.MarkerCollection.build_swarm(
            self._generate_x(), data, name=name, orient=self._orient,
            strip_width=strip_width, symbol=symbol, size=size, sort=sort,
            color=color, alpha=alpha, pattern=pattern, backend=self._get_backend()
        )  # fmt: skip
        self._relabel_axis(y)
        return canvas.add_layer(group)

    def to_boxplot(
        self,
        y: str | None = None,
        *,
        name: str | None = None,
        box_width: float = 0.3,
        capsize: float = 0.15,
        color: ColorType | Sequence[ColorType] | None = None,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _lg.BoxPlot:
        canvas = self._canvas()
        name = canvas._coerce_name(_lg.BoxPlot, name)
        color = self._generate_colors(color)
        data = self._generate_y(y)
        group = _lg.BoxPlot.from_arrays(
            self._generate_x(), data, name=name, orient=self._orient,
            box_width=box_width, capsize=capsize, color=color, alpha=alpha,
            pattern=pattern, backend=self._get_backend(),
        )  # fmt: skip
        self._relabel_axis(y)
        return canvas.add_layer(group)

    def to_violinplot(
        self,
        y: str | None = None,
        *,
        name: str | None = None,
        shape: Literal["both", "left", "right"] = "both",
        violin_width: float = 0.3,
        band_width: float | str = "scott",
        color: ColorType | Sequence[ColorType] | None = None,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _lg.ViolinPlot:
        canvas = self._canvas()
        name = canvas._coerce_name(_lg.ViolinPlot, name)
        color = self._generate_colors(color)
        data = [v[y] for v in self._obj.values()]
        group = _lg.ViolinPlot.from_arrays(
            self._generate_x(), data, name=name, shape=shape, violin_width=violin_width,
            orient=self._orient, kde_band_width=band_width, color=color, alpha=alpha,
            pattern=pattern, backend=self._get_backend(),
        )  # fmt: skip
        self._relabel_axis(y)
        return canvas.add_layer(group)

    def __enter__(self) -> CategorizedDataPlotter[_C]:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _generate_colors(self, color) -> list[Color]:
        if color is not None:
            return color
        return self._canvas()._color_palette.nextn(self.n_categories, update=False)

    def _generate_x(self) -> NDArray[np.floating]:
        x = np.arange(self.n_categories, dtype=np.float64)
        return x + self._offsets

    def _generate_y(self, y):
        if y is None:
            y = self.categories[0]
        return [v[y] for v in self._obj.values()]

    def _get_backend(self):
        return self._canvas()._get_backend()

    def _relabel_axis(self, y):
        if not self._update_label:
            return
        canvas = self._canvas()
        if y is None:
            y = self.categories[0]
        if self._orient.is_vertical:
            canvas.y.label.text = str(y)
        else:
            canvas.x.label.text = str(y)


class CategorizedAggDataPlotter(CategorizedStruct[_C, "Aggregator[Any]"]):
    def __init__(
        self,
        canvas: _C,
        data: dict[str, dict[str, Aggregator[Any]]],
        offsets: NDArray[np.number],
        orient: Orientation,
        palette: ColorPalette,
    ):
        super().__init__(canvas, data, offsets, orient, palette)

    def _generate_y(self, y: str | None = None) -> dict[str, NDArray[np.number]]:
        if y is None:
            y = self.categories[0]
        return [v[y].compute() for v in self.values()]

    def to_line(
        self,
        y: str | None = None,
        *,
        name: str | None = None,
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        color: ColorType | None = None,
        alpha: float = 1.0,
        antialias: bool = True,
    ) -> _l.Line:
        canvas = self._canvas()
        name = canvas._coerce_name(_l.Line, name)
        color = canvas._generate_colors(color)
        data = self._generate_y(y)
        if self._orient.is_vertical:
            x_, y_ = self._generate_x(), data
        else:
            x_, y_ = data, self._generate_x()
        layer = _l.Line(
            x_, y_, name=name, width=width, style=style, color=color,
            alpha=alpha, antialias=antialias, backend=self._get_backend(),
        )  # fmt: skip
        return canvas.add_layer(layer)

    def to_markers(
        self,
        y: str | None = None,
        *,
        name=None,
        color=None,
        alpha=1.0,
        size=10,
        symbol=Symbol.CIRCLE,
        pattern=FacePattern.SOLID,
    ) -> _l.HeteroMarkers:
        canvas = self._canvas()
        name = canvas._coerce_name("markers", name)
        color = self._generate_colors(color)
        data = self._generate_y(y)
        if self._orient.is_vertical:
            x_, y_ = self._generate_x(), data
        else:
            x_, y_ = data, self._generate_x()
        layer = _l.HeteroMarkers(
            x_, y_, name=name, symbol=symbol, size=size, color=color,
            alpha=alpha, pattern=pattern, backend=self._get_backend(),
        )  # fmt: skip
        return canvas.add_layer(layer)

    def to_bars(
        self,
        y: str | None = None,
        *,
        name=None,
        bar_width=0.8,
        color=None,
        alpha=1.0,
        pattern=FacePattern.SOLID,
    ) -> _l.HeteroBars:
        canvas = self._canvas()
        name = canvas._coerce_name(_l.Bars, name)
        if color is None:
            color = self._generate_colors()
        data = self._generate_y(y)
        layer = _l.HeteroBars(
            self._generate_x(), data, name=name, orient=self._orient,
            bar_width=bar_width, color=color, alpha=alpha, pattern=pattern,
            backend=self._get_backend()
        )  # fmt: skip
        return canvas.add_layer(layer)


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


def _norm_input(data: Any, by: Any | None):
    nested = by is not None
    if isinstance(data, dict):
        if nested:
            array_dict: dict[str, NDArray[np.number]] = {}
            lengths: set[int] = set()
            for k, v in data.items():
                arr = as_array_1d(v)
                array_dict[k] = arr
                lengths.add(arr.size)
            if len(lengths) > 1:
                raise ValueError(f"Length of array data not consistent: {lengths}.")
            uniques = np.unique(array_dict[by])
            obj: dict[Hashable, dict[str, NDArray[np.number]]] = {}
            for unique_val in uniques:
                sl = array_dict[by] == unique_val
                dict_filt = {k: v[sl] for k, v in array_dict.items()}
                obj[unique_val] = dict_filt
        else:
            obj = {k: {"value": as_array_1d(v)} for k, v in data.items()}
    elif _is_pandas_dataframe(data):
        if nested:
            obj = {cat: val for cat, val in data.groupby(by)}
        else:
            obj = {c: data[[c]] for c in data.columns}
    elif _is_polars_dataframe(data):
        if nested:
            obj = {cat: val for cat, val in data.group_by(by, maintain_order=True)}
        else:
            obj = {c: data.select(c) for c in data.columns}
    else:
        raise TypeError(f"{type(data)} cannot be categorized.")
    return obj


def _aggregate(
    d: dict[str, NDArray[np.number]],
    func,
    **kwargs,
) -> dict[str, NDArray[np.number]]:
    out = {}
    for k, v in d.items():
        out[k] = Aggregator(v, func, **kwargs)
    return out


_V = TypeVar("_V", bound=Any)


class Aggregator(Generic[_V]):
    def __init__(
        self,
        arr: NDArray[np.number],
        func: Callable[[NDArray[np.number]], _V],
        **kwargs,
    ):
        self._arr = arr
        self._func = func
        self._kwargs = kwargs

    def compute(self) -> _V:
        if self._arr.dtype.kind not in "biufc":
            raise TypeError(f"Cannot aggregate {self._arr.dtype}.")
        return self._func(self._arr, **self._kwargs)
