from __future__ import annotations

import sys
from abc import ABC
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

from whitecanvas._exceptions import ReferenceDeletedError

if TYPE_CHECKING:
    from typing_extensions import TypeGuard, Self
    from ._base import CanvasBase
    import pandas as pd
    import polars as pl

_C = TypeVar("_C", bound="CanvasBase")
_T = TypeVar("_T")


class CategorizedStruct(ABC, Generic[_C, _T]):
    def __init__(
        self,
        canvas: _C,
        obj: dict[str, dict[str, _T]],
    ):
        self._canvas_ref = weakref.ref(canvas)
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

    def _generate_colors(self, color) -> list[Color]:
        if color is not None:
            return color
        return self._canvas()._color_palette.nextn(self.n_categories, update=False)

    def _generate_x(self) -> NDArray[np.floating]:
        x = np.arange(self.n_categories, dtype=np.float64)
        return x + self._offsets

    def _get_backend(self):
        return self._canvas()._get_backend()

    def _default_y_label(self) -> str:
        try:
            v = next(iter(self._obj.values()))
            y = next(iter(v.keys()))
        except StopIteration:
            y = "value"
        return y

    def sort(
        self,
        rule: Callable[[str], float],
        *,
        descending: bool = False,
    ) -> Self:
        """Sort categories by name."""
        names = sorted(self._obj.keys(), reverse=descending, key=rule)
        return self.select(names)

    def filter(
        self,
        rule: Callable[[str], bool],
    ) -> Self:
        """Filter categories by name."""
        names = [n for n in self._obj.keys() if rule(n)]
        return self.select(names)

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
        super().__init__(canvas, obj)
        self._offsets = offsets
        self._orient = orient
        self._color_palette = _color_palette
        self._update_label = update_label

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

    def select(self, *names) -> CategorizedDataPlotter[_C]:
        """
        Select categories by name.

        This method is used for filtering and sorting categories.

        >>> df = {
        ...     "values": [1, 2, 3, 4, 5, 6],
        ...     "vars": ["a", "b", "a", "b", "a", "b"],
        ... }
        >>> canvas.cat(df, by="vars").select(["b", "a"]).to_stripplot("values")
        """
        return super().select(*names)

    # Aggregators
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

    # Plotting methods
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
        name = canvas._coerce_name("boxplot", name)
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
        name = canvas._coerce_name("violinplot", name)
        color = self._generate_colors(color)
        data = self._generate_y(y)
        group = _lg.ViolinPlot.from_arrays(
            self._generate_x(), data, name=name, shape=shape, violin_width=violin_width,
            orient=self._orient, kde_band_width=band_width, color=color, alpha=alpha,
            pattern=pattern, backend=self._get_backend(),
        )  # fmt: skip
        self._relabel_axis(y)
        return canvas.add_layer(group)

    def to_countplot(
        self,
        *,
        name: str | None = None,
        bar_width: float = 0.8,
        color: ColorType | None = None,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _l.Bars:
        canvas = self._canvas()
        name = canvas._coerce_name("count", name)
        color = canvas._generate_colors(color)

        def _len(d: dict[str, Sequence[Any]]) -> int:
            key = next(iter(d.keys()), None)
            if key is None:
                raise ValueError("Empty data.")
            return len(d[key])

        counts = [_len(v) for v in self._obj.values()]
        layer = _l.Bars(
            self._generate_x(),
            counts,
            name=name,
            orient=self._orient,
            bar_width=bar_width,
            color=color,
            alpha=alpha,
            pattern=pattern,
            backend=self._get_backend(),
        )
        self._relabel_axis("count")
        return canvas.add_layer(layer)

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

    def select(self, *names) -> CategorizedDataPlotter[_C]:
        """Select categories by name."""
        if len(names) == 0:
            raise ValueError("At least one category name must be given.")
        elif len(names) == 1 and isinstance(names[0], list):
            names = names[0]
        try:
            obj = {name: self._obj[name] for name in names}
        except KeyError:
            not_found = [n for n in names if n not in self._obj]
            raise ValueError(f"Categories not found: {not_found!r}.") from None
        # reorder offsets
        offsets = np.asarray(
            [self._offsets[self.categories.index(name)] for name in names]
        )
        return CategorizedDataPlotter(
            self._canvas(),
            obj,
            offsets=offsets,
            orient=self._orient,
            palette=self._color_palette,
            update_label=self._update_label,
            unsafe=True,
        )

    def _relabel_axis(self, y):
        canvas = self._canvas()
        tick_pos = np.arange(len(self.categories))
        tick_labels = self.categories
        if self._orient.is_vertical:
            canvas.x.ticks.set_labels(tick_pos, tick_labels)
        else:
            canvas.y.ticks.set_labels(tick_pos, tick_labels)

        if not self._update_label:
            return
        if y is None:
            y = self._default_y_label()
        if self._orient.is_vertical:
            canvas.y.label.text = str(y)
        else:
            canvas.x.label.text = str(y)

    def _generate_y(self, y):
        if y is None:
            y = self._default_y_label()
        return [v[y] for v in self._obj.values()]


class CategorizedAggDataPlotter(CategorizedStruct[_C, "Aggregator[Any]"]):
    def __init__(
        self,
        canvas: _C,
        data: dict[str, dict[str, Aggregator[Any]]],
        offsets: NDArray[np.number],
        orient: Orientation,
        palette: ColorPalette,
    ):
        super().__init__(canvas, data)
        self._offsets = offsets
        self._orient = orient
        self._color_palette = palette

    def _generate_y(self, y: str | None = None) -> dict[str, NDArray[np.number]]:
        if y is None:
            y = self.categories[0]
        return [v[y].compute() for v in self.values()]

    def select(self, *names) -> CategorizedDataPlotter[_C]:
        """
        Select categories by name.

        This method is used for filtering and sorting categories.

        >>> df = {
        ...     "values": [1, 2, 3, 4, 5, 6],
        ...     "vars": ["a", "b", "a", "b", "a", "b"],
        ... }
        >>> canvas.cat(df, by="vars").mean().select(["b", "a"]).to_markers("values")
        """
        return super().select(*names)

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

    def select(self, *names) -> CategorizedAggDataPlotter[_C]:
        """Select categories by name."""
        if len(names) == 0:
            raise ValueError("At least one category name must be given.")
        elif len(names) == 1 and isinstance(names[0], list):
            names = names[0]
        try:
            obj = {name: self._obj[name] for name in names}
        except KeyError:
            not_found = [n for n in names if n not in self._obj]
            raise ValueError(f"Categories not found: {not_found!r}.") from None
        # reorder offsets
        offsets = np.asarray(
            [self._offsets[self.categories.index(name)] for name in names]
        )
        return CategorizedAggDataPlotter(
            self._canvas(),
            obj,
            offsets=offsets,
            orient=self._orient,
            palette=self._color_palette,
        )


class ColorizedPlotter(CategorizedStruct[_C, NDArray[np.number]]):
    def __init__(
        self,
        canvas: _C,
        data: Any,
        by: str | None = None,
        *,
        orient: Orientation = Orientation.VERTICAL,
        palette: ColormapType | None = None,
        update_label: bool = False,
        unsafe: bool = False,
    ):
        if unsafe:
            _color_palette = palette
            obj = data
        else:
            if palette is None:
                _color_palette = canvas._color_palette.copy()
            else:
                _color_palette = ColorPalette(palette)
            obj = _norm_input(data, by)
        super().__init__(canvas, obj)
        self._orient = orient
        self._color_palette = _color_palette
        self._update_label = update_label

    def to_markers(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        alpha: float = 1.0,
        symbol: str | Symbol = Symbol.CIRCLE,
        size: float = 10,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> list[_l.Markers]:
        canvas = self._canvas()
        name = canvas._coerce_name("scatter", name)
        color = self._generate_colors(color)
        xdatas = [v[x] for v in self._obj.values()]
        ydatas = [v[y] for v in self._obj.values()]
        layers: list[_l.Markers] = []
        for xdata, ydata, c, cat in zip(xdatas, ydatas, color, self.categories):
            layer = _l.Markers(
                xdata, ydata, name=f"{name}-{cat}", symbol=symbol, size=size,
                color=c, alpha=alpha, pattern=pattern, backend=self._get_backend(),
            )  # fmt: skip
            layers.append(layer)
        for layer in layers:
            canvas.add_layer(layer)
        self._relabel_axes(x, y)
        return layers

    def to_line(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        color: ColorType | None = None,
        alpha: float = 1.0,
        antialias: bool = True,
    ) -> list[_l.Line]:
        canvas = self._canvas()
        name = canvas._coerce_name("line", name)
        color = self._generate_colors(color)
        xdatas = [v[x] for v in self._obj.values()]
        ydatas = [v[y] for v in self._obj.values()]
        layers: list[_l.Line] = []
        for xdata, ydata, c, cat in zip(xdatas, ydatas, color, self.categories):
            layer = _l.Line(
                xdata, ydata, name=f"{name}-{cat}", width=width, style=style,
                color=c, alpha=alpha, antialias=antialias, backend=self._get_backend(),
            )  # fmt: skip
            layers.append(layer)
        for layer in layers:
            canvas.add_layer(layer)
        self._relabel_axes(x, y)
        return layers

    def to_hist(
        self,
        value_column: str | None = None,
        *,
        name: str | None = None,
        bins: int | Sequence[float] = 10,
        density: bool = False,
        range: tuple[float, float] | None = None,
        color: ColorType | None = None,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> list[_l.Bars]:
        canvas = self._canvas()
        name = canvas._coerce_name("histogram", name)
        color = self._generate_colors(color)
        data = self._generate_y(value_column)
        if hasattr(bins, "__iter__"):
            bins = np.asarray(bins)
        else:
            data_concat = np.concatenate(data)
            bins = np.linspace(data_concat.min(), data_concat.max(), bins + 1)
        layers = []
        for ydata, c, cat in zip(data, color, self.categories):
            layer = _l.Bars.from_histogram(
                ydata,
                bins=bins,
                density=density,
                range=range,
                name=f"{name}-{cat}",
                orient=self._orient,
                color=c,
                alpha=alpha,
                pattern=pattern,
                backend=self._get_backend(),
            )
            layers.append(layer)
        for layer in layers:
            canvas.add_layer(layer)
        self._relabel_axes(value_column, "count")
        return layers

    def to_cdf(
        self,
        value_column: str | None = None,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        alpha: float = 1.0,
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
    ) -> list[_l.Line]:
        canvas = self._canvas()
        name = canvas._coerce_name("cdf", name)
        color = self._generate_colors(color)
        data = self._generate_y(value_column)
        layers = []
        for ydata, c, cat in zip(data, color, self.categories):
            layer = _l.Line.from_cdf(
                ydata,
                name=f"{name}-{cat}",
                width=width,
                style=style,
                color=c,
                alpha=alpha,
                backend=self._get_backend(),
            )
            layers.append(layer)
        for layer in layers:
            canvas.add_layer(layer)
        self._relabel_axes(value_column, "density")
        return layers

    def _generate_y(self, y):
        if y is None:
            y = self._default_y_label()
        return [v[y] for v in self._obj.values()]

    def select(self, *names) -> ColorizedPlotter[_C]:
        """Select categories by name."""
        if len(names) == 0:
            raise ValueError("At least one category name must be given.")
        elif len(names) == 1 and isinstance(names[0], list):
            names = names[0]
        try:
            obj = {name: self._obj[name] for name in names}
        except KeyError:
            not_found = [n for n in names if n not in self._obj]
            raise ValueError(f"Categories not found: {not_found!r}.") from None
        return ColorizedPlotter(
            self._canvas(),
            obj,
            orient=self._orient,
            palette=self._color_palette,
        )

    def _relabel_axes(self, x: str, y: str):
        if not self._update_label:
            return
        canvas = self._canvas()
        if y is None:
            y = self._default_y_label()
        if not self._orient.is_vertical:
            x, y = y, x
        canvas.x.label.text = str(x)
        canvas.y.label.text = str(y)


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
                arr = np.asarray(v)
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
            obj = {k: {"value": np.asarray(v)} for k, v in data.items()}
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
