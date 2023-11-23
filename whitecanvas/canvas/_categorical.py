from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Hashable, Literal, Sequence, Generic, TypeVar
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
    Alignment,
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


class CategorizedDataPlotter(Generic[_C]):
    def __init__(
        self,
        canvas: _C,
        data: Any,
        by: str | None = None,
        offsets=None,
        palette: ColormapType | None = None,
    ):
        self._canvas_ref = weakref.ref(canvas)
        if palette is None:
            self._color_palette = canvas._color_palette.copy()
        else:
            self._color_palette = ColorPalette(palette)
        self._nested = by is not None
        self._obj = _norm_input(data, by, self._nested)
        if offsets is None:
            self._offsets = np.zeros(self.n_categories)
        elif isinstance(offsets, (int, float, np.number)):
            self._offsets = np.full(self.n_categories, offsets)
        else:
            self._offsets = np.asarray(offsets)
            if self._offsets.shape != (self.n_categories,):
                raise ValueError("Shape of offset is wrong")

    def _canvas(self) -> _C:
        canvas = self._canvas_ref()
        if canvas is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        return canvas

    @property
    def n_categories(self) -> int:
        return len(self._obj)

    @property
    def categories(self) -> list[Any]:
        return list(self._obj.keys())

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
        return self._canvas()._color_palette.nextn(self.n_categories)

    def _generate_x(self) -> NDArray[np.floating]:
        x = np.arange(self.n_categories, dtype=np.float64)
        return x + self._offsets

    def _get_backend(self):
        return self._canvas()._get_backend()


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
