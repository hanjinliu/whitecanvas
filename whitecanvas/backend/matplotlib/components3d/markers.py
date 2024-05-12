from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.transforms as mtransforms
import numpy as np
from mpl_toolkits.mplot3d import art3d
from numpy.typing import NDArray

from whitecanvas.backend.matplotlib._base import MplLayer, symbol_to_path
from whitecanvas.types import Hatch, LineStyle, Symbol
from whitecanvas.utils.normalize import as_color_array
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from whitecanvas.backend.matplotlib.components3d import Canvas3D


class Markers3D(art3d.Path3DCollection, MplLayer):
    def __init__(self, xdata, ydata, zdata):
        offsets = np.stack([xdata, ydata], axis=1)
        self._zdata = zdata
        self._symbol = Symbol.CIRCLE
        super().__init__(
            (symbol_to_path(self._symbol),),
            sizes=[6] * len(offsets),
            offsets=offsets,
            picker=True,
        )
        self.set_transform(mtransforms.IdentityTransform())
        self._edge_styles = [LineStyle.SOLID] * len(offsets)
        self.set_3d_properties(zdata, zdir="z")

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        offsets = self.get_offsets()
        return offsets[:, 0], offsets[:, 1], self._zdata

    def _plt_set_data(self, xdata, ydata, zdata):
        data = np.stack([xdata, ydata], axis=1)
        self.set_offsets(data)
        self._zdata = zdata
        self.set_3d_properties(zdata, zdir="z")

    ##### HasSymbol protocol #####

    def _plt_get_symbol(self) -> Symbol:
        return self._symbol

    def _plt_set_symbol(self, symbol: Symbol):
        path = symbol_to_path(symbol)
        self.set_paths([path] * len(self.get_offsets()))
        self._symbol = symbol

    def _plt_get_symbol_size(self) -> NDArray[np.floating]:
        return np.sqrt(self.get_sizes())

    def _plt_set_symbol_size(self, size: float | NDArray[np.floating]):
        if is_real_number(size):
            size = np.full(len(self.get_offsets()), size)
        self.set_sizes(size**2)

    ##### HasFaces protocol #####
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        return self.get_facecolor()

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, len(self.get_offsets()))
        self.set_facecolor(color)

    def _plt_get_face_hatch(self) -> Hatch:
        return [Hatch(self.get_hatch() or "")] * len(self.get_offsets())

    def _plt_set_face_hatch(self, pattern: Hatch | list[Hatch]):
        if not isinstance(pattern, Hatch):
            if len(set(pattern)) > 1:
                warnings.warn(
                    "matplotlib markers do not support multiple hatch patterns.",
                    UserWarning,
                    stacklevel=2,
                )
            pattern = pattern[0]
        if pattern is Hatch.SOLID:
            ptn = None
        else:
            ptn = pattern.value
        self.set_hatch(ptn)

    ##### HasEdges protocol #####

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return self.get_edgecolor()

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        color = as_color_array(color, len(self.get_offsets()))
        self.set_edgecolor(color)

    def _plt_get_edge_style(self) -> list[LineStyle]:
        return self._edge_styles

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            styles = [style.value] * len(self.get_offsets())
            self._edge_styles = [style] * len(self.get_offsets())
        else:
            styles = [s.value for s in style]
            self._edge_styles = style
        self.set_linestyle(styles)

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return self.get_linewidth()

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if is_real_number(width):
            width = np.full(len(self.get_offsets()), width)
        self.set_linewidth(width)

    def post_add(self, canvas: Canvas3D):
        self.set_offset_transform(canvas._axes.transData)
