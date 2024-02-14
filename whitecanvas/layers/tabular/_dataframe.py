"""Layer with a dataframe bound to it."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    TypeVar,
    Union,
    overload,
)

import numpy as np
from cmap import Color, Colormap

from whitecanvas import layers as _l
from whitecanvas.backend import Backend
from whitecanvas.layers import group as _lg
from whitecanvas.layers.tabular import _jitter, _shared
from whitecanvas.layers.tabular import _plans as _p
from whitecanvas.layers.tabular._df_compat import DataFrameWrapper, parse
from whitecanvas.types import (
    ColormapType,
    ColorType,
    HistBinType,
    KdeBandWidthType,
    LineStyle,
    Orientation,
    _Void,
)
from whitecanvas.utils.hist import histograms

if TYPE_CHECKING:
    from typing_extensions import Self

_DF = TypeVar("_DF")
_Cols = Union[str, "tuple[str, ...]"]
_void = _Void()


class DFLines(_shared.DataFrameLayerWrapper[_lg.LineCollection, _DF], Generic[_DF]):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        segs: list[np.ndarray],
        labels: list[tuple[Any, ...]],
        color: _Cols | None = None,
        width: float = 1.0,
        style: _Cols | None = None,
        name: str | None = None,
        backend: str | Backend | None = None,
    ):
        splitby = _shared.join_columns(color, style, source=source)
        self._color_by = _p.ColorPlan.default()
        self._style_by = _p.StylePlan.default()
        self._labels = labels
        self._splitby = splitby
        base = _lg.LineCollection(segs, name=name, backend=backend)
        super().__init__(base, source)
        if color is not None:
            self.update_color(color)
        self.update_width(width)
        if style is not None:
            self.update_style(style)
        self.with_hover_template("\n".join(f"{k}: {{{k}!r}}" for k in self._splitby))

    @classmethod
    def from_table(
        cls,
        df: DataFrameWrapper[_DF],
        x: str | _jitter.JitterBase,
        y: str | _jitter.JitterBase,
        color: str | None = None,
        width: float | None = None,
        style: str | None = None,
        name: str | None = None,
        backend: str | Backend | None = None,
    ) -> DFLines[_DF]:
        splitby = _shared.join_columns(color, style, source=df)
        segs = []
        labels: list[tuple[Any, ...]] = []
        if isinstance(x, _jitter.JitterBase):
            xj = x
        else:
            xj = _jitter.IdentityJitter(x)
        if isinstance(y, _jitter.JitterBase):
            yj = y
        else:
            yj = _jitter.IdentityJitter(y)
        for sl, sub in df.group_by(splitby):
            labels.append(sl)
            segs.append(np.column_stack([xj.map(sub), yj.map(sub)]))
        return DFLines(
            df, segs, labels, name=name, color=color, width=width, style=style,
            backend=backend,
        )  # fmt: skip

    @overload
    def update_color(self, value: ColorType) -> Self:
        ...

    @overload
    def update_color(
        self,
        by: str | Iterable[str],
        palette: ColormapType | None = None,
    ) -> Self:
        ...

    def update_color(self, by, /, palette=None):
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        self._base_layer.color = color_by.generate(self._labels, self._splitby)
        self._color_by = color_by
        return self

    def update_width(self, value: float) -> Self:
        self._base_layer.width = value
        return self

    def update_style(self, by: str | Iterable[str], styles=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot style by a column other than {self._splitby}")
            style_by = _p.StylePlan.new(cov.columns, values=styles)
        else:
            style_by = _p.StylePlan.from_const(LineStyle(cov.value))
        self._base_layer.style = style_by.generate(self._labels, self._splitby)
        self._style_by = style_by
        return self

    def move(self, dx: float = 0.0, dy: float = 0.0, autoscale: bool = True) -> Self:
        """Add a constant shift to the layer."""
        for layer in self._base_layer:
            old_data = layer.data
            new_data = old_data[0] + dx, old_data[1] + dy
            layer.data = new_data
        if autoscale and (canvas := self._canvas_ref()):
            canvas._autoscale_for_layer(self, pad_rel=0.025)
        return self

    def with_hover_template(self, template: str) -> Self:
        extra = {}
        for i, key in enumerate(self._splitby):
            extra[key] = [row[i] for row in self._labels]
        self.base.with_hover_template(template, extra=extra)
        return self


class DFHeatmap(_shared.DataFrameLayerWrapper[_l.Image, _DF], Generic[_DF]):
    @property
    def cmap(self) -> Colormap:
        """Colormap of the heatmap."""
        return self._base_layer.cmap

    @cmap.setter
    def cmap(self, cmap: ColormapType):
        self._base_layer.cmap = Colormap(cmap)

    @property
    def clim(self) -> tuple[float, float]:
        """Color limits of the heatmap."""
        return self._base_layer.clim

    @clim.setter
    def clim(self, clim: tuple[float, float]):
        self._base_layer.clim = clim

    @classmethod
    def build_hist(
        cls,
        df: _DF,
        x: str,
        y: str,
        name: str | None = None,
        cmap: ColormapType = "gray",
        bins: HistBinType | tuple[HistBinType, HistBinType] = "auto",
        range=None,
        density: bool = False,
        backend: Backend | str | None = None,
    ) -> Self:
        src = parse(df)
        xdata = src[x]
        ydata = src[y]
        if xdata.dtype.kind not in "fiub":
            raise ValueError(f"Column {x!r} is not numeric.")
        if ydata.dtype.kind not in "fiub":
            raise ValueError(f"Column {y!r} is not numeric.")
        base = _l.Image.build_hist(
            xdata, ydata, name=name, cmap=cmap, bins=bins, range=range,
            density=density, backend=backend,
        )  # fmt: skip
        return cls(base, src)

    @classmethod
    def from_array(
        cls,
        src: DataFrameWrapper[_DF],
        arr: np.ndarray,
        name: str | None = None,
        cmap: ColormapType = "gray",
        clim: tuple[float | None, float | None] | None = None,
        backend: Backend | str | None = None,
    ) -> DFHeatmap[_DF]:
        return cls(_l.Image(arr, name=name, cmap=cmap, clim=clim, backend=backend), src)


class DFPointPlot2D(_shared.DataFrameLayerWrapper[_lg.LabeledPlot, _DF], Generic[_DF]):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        x: str,
        y: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        size: float | None = None,
        capsize: float = 0.15,
        name: str | None = None,
        backend: str | Backend | None = None,
    ):
        cols = _shared.join_columns(color, hatch, source=source)
        xdata = []
        ydata = []
        for _, sub in source.group_by(cols):
            xdata.append(sub[x])
            ydata.append(sub[y])
        base = _lg.LabeledPlot.from_arrays_2d(
            xdata, ydata, name=name, capsize=capsize, backend=backend
        )
        if size is not None:
            base.markers.size = size
        super().__init__(base, source)


class DFHistograms(
    _shared.DataFrameLayerWrapper[_lg.LayerCollectionBase[_lg.Histogram], _DF],
    Generic[_DF],
):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        base: _lg.LayerCollectionBase[_lg.Histogram],
        labels: list[tuple[Any, ...]],
        color: _Cols | None = None,
        width: str | None = None,
        style: _Cols | None = None,
    ):
        splitby = _shared.join_columns(color, style, source=source)
        self._color_by = _p.ColorPlan.default()
        self._width_by = _p.WidthPlan.default()
        self._style_by = _p.StylePlan.default()
        self._labels = labels
        self._splitby = splitby
        super().__init__(base, source)
        if color is not None:
            self.update_color(color)
        if isinstance(width, str):
            self.update_width(width)
        if style is not None:
            self.update_style(style)

    @classmethod
    def from_table(
        cls,
        df: DataFrameWrapper[_DF],
        value: str,
        bins: HistBinType = "auto",
        limits: tuple[float, float] | None = None,
        kind="count",
        shape="bars",
        color: str | None = None,
        width: float = 1.0,
        style: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ) -> DFHistograms[_DF]:
        splitby = _shared.join_columns(color, style, source=df)
        ori = Orientation.parse(orient)
        arrays: list[np.ndarray] = []
        labels: list[tuple] = []
        for sl, sub in df.group_by(splitby):
            labels.append(sl)
            arrays.append(sub[value])
        hist = histograms(arrays, bins, limits)

        layers = []
        for arr in arrays:
            each_layer = _lg.Histogram.from_array(
                arr, kind=kind, bins=hist.edges, limits=limits, width=width,
                orient=ori, shape=shape, backend=backend,
            )  # fmt: skip
            layers.append(each_layer)
        base = _lg.LayerCollectionBase(layers, name=name)
        return cls(df, base, labels, color=color, width=width, style=style)

    @overload
    def update_color(self, value: ColorType) -> Self:
        ...

    @overload
    def update_color(
        self,
        by: str | Iterable[str],
        palette: ColormapType | None = None,
    ) -> Self:
        ...

    def update_color(self, by, /, palette=None):
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        for i, col in enumerate(color_by.generate(self._labels, self._splitby)):
            self._base_layer[i].color = col
        self._color_by = color_by
        return self

    def update_width(self, value: float) -> Self:
        for hist in self._base_layer:
            hist.line.width = value
        return self

    def update_style(self, by: str | Iterable[str], styles=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot style by a column other than {self._splitby}")
            style_by = _p.StylePlan.new(cov.columns, values=styles)
        else:
            style_by = _p.StylePlan.from_const(LineStyle(cov.value))
        for i, st in enumerate(style_by.generate(self._labels, self._splitby)):
            self._base_layer[i].line.style = st
        self._style_by = style_by
        return self


class DFKde(
    _shared.DataFrameLayerWrapper[_lg.LayerCollectionBase[_lg.Kde], _DF],
    Generic[_DF],
):
    def __init__(
        self,
        source: DataFrameWrapper[_DF],
        base: _lg.LayerCollectionBase[_lg.Kde],
        labels: list[tuple[Any, ...]],
        color: _Cols | None = None,
        width: str | None = None,
        style: _Cols | None = None,
    ):
        splitby = _shared.join_columns(color, style, source=source)
        self._color_by = _p.ColorPlan.default()
        self._width_by = _p.WidthPlan.default()
        self._style_by = _p.StylePlan.default()
        self._labels = labels
        self._splitby = splitby
        super().__init__(base, source)
        if color is not None:
            self.update_color(color)
        if isinstance(width, str):
            self.update_width(width)
        if style is not None:
            self.update_style(style)

    @classmethod
    def from_table(
        cls,
        df: DataFrameWrapper[_DF],
        value: str,
        band_width: KdeBandWidthType = "scott",
        color: str | None = None,
        width: float = 1.0,
        style: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        backend: str | Backend | None = None,
    ) -> DFHistograms[_DF]:
        splitby = _shared.join_columns(color, style, source=df)
        ori = Orientation.parse(orient)
        arrays: list[np.ndarray] = []
        labels: list[tuple] = []
        for sl, sub in df.group_by(splitby):
            labels.append(sl)
            arrays.append(sub[value])
        layers = []
        for arr in arrays:
            each_layer = _lg.Kde.from_array(
                arr, width=width, band_width=band_width, orient=ori, backend=backend,
            )  # fmt: skip
            layers.append(each_layer)
        base = _lg.LayerCollectionBase(layers, name=name)
        return cls(df, base, labels, color=color, width=width, style=style)

    @overload
    def update_color(self, value: ColorType) -> Self:
        ...

    @overload
    def update_color(
        self,
        by: str | Iterable[str],
        palette: ColormapType | None = None,
    ) -> Self:
        ...

    def update_color(self, by, /, palette=None):
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot color by a column other than {self._splitby}")
            color_by = _p.ColorPlan.from_palette(cov.columns, palette=palette)
        else:
            color_by = _p.ColorPlan.from_const(Color(cov.value))
        for i, col in enumerate(color_by.generate(self._labels, self._splitby)):
            self._base_layer[i].color = col
        self._color_by = color_by
        return self

    def update_width(self, value: float) -> Self:
        for hist in self._base_layer:
            hist.line.width = value
        return self

    def update_style(self, by: str | Iterable[str], styles=None) -> Self:
        cov = _shared.ColumnOrValue(by, self._source)
        if cov.is_column:
            if set(cov.columns) > set(self._splitby):
                raise ValueError(f"Cannot style by a column other than {self._splitby}")
            style_by = _p.StylePlan.new(cov.columns, values=styles)
        else:
            style_by = _p.StylePlan.from_const(LineStyle(cov.value))
        for i, st in enumerate(style_by.generate(self._labels, self._splitby)):
            self._base_layer[i].line.style = st
        self._style_by = style_by
        return self


def default_template(it: Iterable[tuple[str, np.ndarray]], max_rows: int = 10) -> str:
    """
    Default template string for markers

    This template can only be used for those plot that has one tooltip for each data
    point, which includes markers, bars and rugs.
    """
    fmt_list = list[str]()
    for ikey, (key, value) in enumerate(it):
        if not key:
            continue
        if ikey >= max_rows:
            break
        if value.dtype.kind == "f":
            fmt_list.append(f"{key}: {{{key}:.4g}}")
        else:
            fmt_list.append(f"{key}: {{{key}!r}}")
    return "\n".join(fmt_list)
