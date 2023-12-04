from __future__ import annotations
from typing import TypeVar, Generic, TYPE_CHECKING
import weakref

import numpy as np
from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas.types import Orientation, LineStyle, ColorType, XYData
from whitecanvas.layers import MultiLine, Markers, Layer

if TYPE_CHECKING:
    from whitecanvas.canvas._base import CanvasBase

_C = TypeVar("_C", bound="CanvasBase")


class BetweenPlotter(Generic[_C]):
    def __init__(self, canvas: _C, layer1: Layer, layer2: Layer):
        self._canvas_ref = weakref.ref(canvas)
        self._layer1_ref = weakref.ref(layer1)
        self._layer2_ref = weakref.ref(layer2)

    def _canvas(self) -> _C:
        canvas = self._canvas_ref()
        if canvas is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        return canvas

    def _layers(self) -> tuple[Layer, Layer]:
        layer1 = self._layer1_ref()
        layer2 = self._layer2_ref()
        if layer1 is None or layer2 is None:
            raise ReferenceDeletedError("Layer has been deleted.")
        return layer1, layer2

    def connect_points(
        self,
        color: ColorType = "black",
        width: float = 1.0,
        alpha: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
    ) -> MultiLine:
        canvas = self._canvas()
        layer1, layer2 = self._layers()
        if not isinstance(data1 := getattr(layer1, "data", None), XYData):
            raise TypeError(f"Layer {layer1!r} is not a layer with XY data.")
        if not isinstance(data2 := getattr(layer2, "data", None), XYData):
            raise TypeError(f"Layer {layer2!r} is not a layer with XY data.")
        if data1.x.size != data2.x.size:
            raise ValueError("Data sizes must be equal.")
        xs = np.stack([data1.x, data2.x], axis=1)
        ys = np.stack([data1.y, data2.y], axis=1)
        segs = np.stack([xs, ys], axis=2)
        lines = MultiLine(
            segs,
            name=f"between<{layer1.name}, {layer2.name}>",
            color=color,
            width=width,
            alpha=alpha,
            style=style,
        )
        canvas.add_layer(lines, under=[layer1, layer2])
        return lines

    # def add_bracket(
    #     self,
    #     string: str = "",
    #     capsize: float | None = None,
    #     *,
    #     color: ColorType = "black",
    #     width: float = 1.0,
    #     alpha: float = 1.0,
    #     style: str | LineStyle = LineStyle.SOLID,
    #     orientation: str | Orientation = Orientation.HORIZONTAL,
    # ):
    #     from whitecanvas.layers.group import BracketText

    #     canvas = self._canvas()
    #     layer1, layer2 = self._layers()
    #     hint1 = layer1.bbox_hint()
    #     hint2 = layer2.bbox_hint()

    #     ori = Orientation.parse(orientation)
    #     if ori.is_vertical:
    #         xmax = max(hint1[1], hint2[1])
    #         ymin = min(hint1[2], hint2[2])
    #         ymax = max(hint1[3], hint2[3])
    #         y1 = (hint1[2] + hint1[3]) / 2
    #         y2 = (hint2[2] + hint2[3]) / 2
    #         y1, y2 = sorted([y1, y2])
    #         capsize = 0.03 * (ymax - ymin)
    #         dy = capsize * 2
    #         layer = BracketText(
    #             (xmax, y2 + dy), (xmax, y1 + dy), string, capsize=capsize,
    #         )
    #     else:
    #         ymax = max(hint1[3], hint2[3])
    #         xmin = min(hint1[0], hint2[0])
    #         xmax = max(hint1[1], hint2[1])
    #         x1 = (hint1[0] + hint1[1]) / 2
    #         x2 = (hint2[0] + hint2[1]) / 2
    #         x1, x2 = sorted([x1, x2])
    #         capsize = 0.03 * (xmax - xmin)
    #         dx = capsize * 2
    #         layer = BracketText(
    #             (x1 + dx, ymax), (x2 + dx, ymax), string, capsize=capsize,
    #         )
    #     layer.line.update(color=color, width=width, alpha=alpha, style=style)
    #     return canvas.add_layer(layer)
