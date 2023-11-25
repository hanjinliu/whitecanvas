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
        offsets=None,
        palette: ColormapType | None = None,
    ):
        self._canvas_ref = weakref.ref(canvas)
        self._offsets = offsets
        self._color_palette = palette
        self._obj = obj

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

    def _generate_colors(self) -> list[Color]:
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
        offsets=None,
        palette: ColormapType | None = None,
    ):
        if palette is None:
            _color_palette = canvas._color_palette.copy()
        else:
            _color_palette = ColorPalette(palette)
        _nested = by is not None
        obj = _norm_input(data, by, _nested)
        ncats = len(obj)
        if offsets is None:
            offsets = np.zeros(ncats)
        elif isinstance(offsets, (int, float, np.number)):
            offsets = np.full(ncats, offsets)
        else:
            offsets = np.asarray(offsets)
            if offsets.shape != (ncats,):
                raise ValueError("Shape of offset is wrong")
        super().__init__(canvas, obj, offsets, _color_palette)

    def mean(self) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, np.mean) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._color_palette
        )

    def min(self) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, np.min) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._color_palette
        )

    def max(self) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, np.max) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._color_palette
        )

    def std(self, ddof: int = 1) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, np.std, ddof=ddof) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._color_palette
        )

    def var(self, ddof: int = 1) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, np.var, ddof=ddof) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._color_palette
        )

    def sum(self) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, np.sum) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._color_palette
        )

    def count(self) -> CategorizedAggDataPlotter[_C]:
        agged = {k: _aggregate(v, len) for k, v in self.items()}
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._color_palette
        )

    def sem(self, ddof: int = 1) -> CategorizedAggDataPlotter[_C]:
        agged = {
            k: _aggregate(v, lambda x: np.std(x, ddof=ddof) / np.sqrt(len(x)))
            for k, v in self.items()
        }
        return CategorizedAggDataPlotter(
            self._canvas(), agged, self._offsets, self._color_palette
        )

    def to_stripplot(
        self,
        y: str | None = None,
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        strip_width: float = 0.1,
        color: ColorType | None = None,
        alpha: float = 1.0,
        symbol: str | Symbol = Symbol.CIRCLE,
        size: float = 10,
        pattern: str | FacePattern = FacePattern.SOLID,
    ):
        canvas = self._canvas()
        name = canvas._coerce_name(_lg.StripPlot, name)
        if color is None:
            color = self._generate_colors()
        if y is None:
            y = self.categories[0]
        data = [v[y] for v in self._obj.values()]
        group = _lg.StripPlot.from_arrays(
            self._generate_x(), data, name=name, orient=orient,
            strip_width=strip_width, seed=None, symbol=symbol, size=size,
            color=color, alpha=alpha, pattern=pattern, backend=self._get_backend()
        )  # fmt: skip
        return canvas.add_layer(group)

    def to_boxplot(
        self,
        y: str | None = None,
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        box_width: float = 0.5,
        capsize: float = 0.3,
        face_color: ColorType | list[ColorType] | None = None,
        edge_color: ColorType = "black",
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _lg.BoxPlot:
        canvas = self._canvas()
        name = canvas._coerce_name(_lg.BoxPlot, name)
        if face_color is None:
            face_color = self._generate_colors()
        if y is None:
            y = self.categories[0]
        data = [v[y] for v in self._obj.values()]
        group = _lg.BoxPlot.from_arrays(
            self._generate_x(), data, name=name, orient=orient, box_width=box_width,
            capsize=capsize, face_color=face_color, edge_color=edge_color,
            alpha=alpha, pattern=pattern, backend=self._get_backend(),
        )  # fmt: skip
        return canvas.add_layer(group)

    def to_violinplot(
        self,
        y: str | None = None,
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        shape: Literal["both", "left", "right"] = "both",
        violin_width: float = 0.3,
        band_width: float | str = "scott",
        colors: ColorType | Sequence[ColorType] | None = None,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _lg.ViolinPlot:
        canvas = self._canvas()
        name = canvas._coerce_name(_lg.ViolinPlot, name)
        if colors is None:
            colors = self._generate_colors()
        data = [v[y] for v in self._obj.values()]
        group = _lg.ViolinPlot.from_arrays(
            self._generate_x(), data, name=name, shape=shape, violin_width=violin_width,
            orient=orient, kde_band_width=band_width, colors=colors, alpha=alpha,
            pattern=pattern, backend=self._get_backend(),
        )  # fmt: skip
        return canvas.add_layer(group)

    def _generate_colors(self) -> list[Color]:
        return self._canvas()._color_palette.nextn(self.n_categories, update=False)

    def _generate_x(self) -> NDArray[np.floating]:
        x = np.arange(self.n_categories, dtype=np.float64)
        return x + self._offsets

    def _get_backend(self):
        return self._canvas()._get_backend()


class CategorizedAggDataPlotter(CategorizedStruct[_C, "Aggregator[Any]"]):
    def __init__(
        self,
        canvas: _C,
        data: dict[str, dict[str, Aggregator[Any]]],
        offsets: NDArray[np.number],
        palette: ColorPalette,
    ):
        super().__init__(canvas, data, offsets, palette)

    def _get_plot_data(self, y: str | None = None) -> dict[str, NDArray[np.number]]:
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
        data = self._get_plot_data(y)
        layer = _l.Line(
            self._generate_x(),
            data,
            name=name,
            width=width,
            style=style,
            color=color,
            alpha=alpha,
            antialias=antialias,
            backend=self._get_backend(),
        )
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
        if color is None:
            color = self._generate_colors()
        data = self._get_plot_data(y)
        layer = _l.HeteroMarkers(
            self._generate_x(), data, name=name, symbol=symbol,
            size=size, color=color, alpha=alpha, pattern=pattern,
            backend=self._get_backend(),
        )  # fmt: skip
        return canvas.add_layer(layer)

    def to_bars(
        self,
        y: str | None = None,
        *,
        name=None,
        orient=Orientation.VERTICAL,
        bar_width=0.8,
        color=None,
        alpha=1.0,
        pattern=FacePattern.SOLID,
    ) -> _l.HeteroBars:
        canvas = self._canvas()
        name = canvas._coerce_name(_l.Bars, name)
        if color is None:
            color = self._generate_colors()
        data = self._get_plot_data(y)
        layer = _l.HeteroBars(
            self._generate_x(), data, name=name, orient=orient,
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


def _norm_input(data: Any, by: Any, nested: bool):
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
