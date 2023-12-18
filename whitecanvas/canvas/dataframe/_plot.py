from __future__ import annotations

from abc import ABC
from typing import (
    TYPE_CHECKING,
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
    Hatch,
    Orientation,
)
from whitecanvas.canvas._palette import ColorPalette
from whitecanvas.layers import tabular as _lt
from whitecanvas._exceptions import ReferenceDeletedError

if TYPE_CHECKING:
    from typing_extensions import Self
    from whitecanvas.canvas._base import CanvasBase

_C = TypeVar("_C", bound="CanvasBase")
_DF = TypeVar("_DF")


class DataFramePlotter(ABC, Generic[_C, _DF]):
    def __init__(
        self,
        canvas: _C,
        df: _DF,
        update_label: bool = False,
    ):
        self._canvas_ref = weakref.ref(canvas)
        self._df = df
        self._update_label = update_label

    def _canvas(self) -> _C:
        canvas = self._canvas_ref()
        if canvas is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        return canvas

    def add_line(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        color: str | tuple[str, ...] | None = None,
        width: str | None = None,
        style: str | tuple[str, ...] | None = None,
    ) -> _lt.WrappedLines[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedLines.from_table(
            self._df,
            x,
            y,
            name=name,
            color=color,
            width=width,
            style=style,
            backend=canvas._get_backend(),
        )
        if color is not None:
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            canvas.x.label.text = x
            canvas.y.label.text = y
        return canvas.add_layer(layer)

    def add_markers(
        self,
        x: str,
        y: str,
        *,
        name: str | None = None,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        size: str | None = None,
        symbol: str | tuple[str, ...] | None = None,
    ) -> _lt.WrappedMarkers[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedMarkers.from_table(
            self._df,
            x,
            y,
            name=name,
            color=color,
            hatch=hatch,
            size=size,
            symbol=symbol,
            backend=canvas._get_backend(),
        )
        if color is not None:
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            canvas.x.label.text = x
            canvas.y.label.text = y
        return canvas.add_layer(layer)

    def add_violinplot(
        self,
        offset: tuple[str, ...],
        value: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
    ) -> _lt.WrappedViolinPlot[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedViolinPlot(
            self._df,
            offset,
            value,
            name=name,
            color=color,
            hatch=hatch,
            orient=orient,
            backend=canvas._get_backend(),
        )
        if color is not None:
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            ...
        return canvas.add_layer(layer)

    def add_stripplot(
        self,
        offset: tuple[str, ...],
        value: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        symbol: str | tuple[str, ...] | None = None,
        size: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        seed: int | None = 0,
    ) -> _lt.WrappedMarkers[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedMarkers.build_stripplot(
            self._df,
            offset,
            value,
            name=name,
            color=color,
            hatch=hatch,
            symbol=symbol,
            size=size,
            orient=orient,
            extent=extent,
            seed=seed,
            backend=canvas._get_backend(),
        )
        if color is not None:
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            ...
        return canvas.add_layer(layer)

    def add_swarmplot(
        self,
        offset: tuple[str, ...],
        value: str,
        *,
        color: str | tuple[str, ...] | None = None,
        hatch: str | tuple[str, ...] | None = None,
        symbol: str | tuple[str, ...] | None = None,
        size: str | None = None,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        extent: float = 0.8,
        sort: bool = False,
    ) -> _lt.WrappedMarkers[_DF]:
        canvas = self._canvas()
        layer = _lt.WrappedMarkers.build_swarmplot(
            self._df,
            offset,
            value,
            name=name,
            color=color,
            hatch=hatch,
            symbol=symbol,
            size=size,
            orient=orient,
            extent=extent,
            sort=sort,
            backend=canvas._get_backend(),
        )
        if color is not None:
            layer.with_color(palette=canvas._color_palette)
        if self._update_label:
            ...
        return canvas.add_layer(layer)
