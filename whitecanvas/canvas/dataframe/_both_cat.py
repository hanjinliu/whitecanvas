from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Sequence,
    TypeVar,
)

from whitecanvas.canvas.dataframe._base import BaseCatPlotter
from whitecanvas.layers import tabular as _lt
from whitecanvas.types import ColormapType

if TYPE_CHECKING:
    from whitecanvas.canvas._base import CanvasBase

    NStr = str | Sequence[str]

_C = TypeVar("_C", bound="CanvasBase")
_DF = TypeVar("_DF")


class BothAxesCatPlotter(BaseCatPlotter[_C, _DF]):
    def __init__(
        self,
        canvas: _C,
        df: _DF,
        x: str | tuple[str, ...],
        y: str | tuple[str, ...],
        update_label: bool = False,
    ):
        super().__init__(canvas, df, update_label)
        self._x = x
        self._y = y
        if update_label:
            self._update_xy_label(x, y)

    def _update_xy_label(
        self,
        x: str | tuple[str, ...],
        y: str | tuple[str, ...],
    ) -> None:
        """Update the x and y labels using the column names"""
        canvas = self._canvas()
        if not isinstance(x, str):
            x = "/".join(x)
        if not isinstance(y, str):
            y = "/".join(y)
        canvas.x.label.text = x
        canvas.y.label.text = y

    def add_heatmap(
        self,
        value: str,
        *,
        cmap: ColormapType = "inferno",
        clim: tuple[float, float] | None = None,
        name: str | None = None,
        fill: float = 0,
    ) -> _lt.DFHeatmap[_DF]:
        canvas = self._canvas()
        layer = _lt.DFHeatmap.build_heatmap(
            self._df, self._x, self._y, value, cmap=cmap, clim=clim, name=name,
            fill=fill, backend=canvas._get_backend(),
        )  # fmt: skip
        if self._update_label:
            canvas.x.ticks.set_labels(*layer._generate_xticks())
            canvas.y.ticks.set_labels(*layer._generate_yticks())
        return canvas.add_layer(layer)


# TODO: add this in agg plotter
# def add_heatmap(
#     self,
#     value: str,
#     *,
#     cmap: ColormapType = "inferno",
#     clim: tuple[float, float] | None = None,
#     name: str | None = None,
#     fill: float = 0,
# ) -> _lt.DFHeatmap[_DF]:
#     canvas = self._canvas()
#     df = parse(self._df)
#     df_agg = self._aggregate(df, (x, y), value)
#     layer = _lt.DFHeatmap.build_heatmap(
#         df_agg, x, y, value, cmap=cmap, clim=clim, name=name, fill=fill,
#         backend=canvas._get_backend(),
#     )  # fmt: skip
#     if self._update_label:
#         canvas.x.ticks.set_labels(*layer._generate_xticks())
#         canvas.y.ticks.set_labels(*layer._generate_yticks())
#     return canvas.add_layer(layer)
