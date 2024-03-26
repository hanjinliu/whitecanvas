from __future__ import annotations

import warnings
import weakref
from typing import TYPE_CHECKING

import numpy as np
from vispy.scene import visuals

from whitecanvas.backend.vispy.line import _make_connection, _safe_concat

if TYPE_CHECKING:
    from whitecanvas.backend.vispy.canvas import Canvas


class LineProperties:
    def __init__(self):
        self.visible = False
        self.color = np.ones(4, dtype=np.float32)
        self.width = 1.0
        self.style = "solid"


class GridLines(visuals.Compound):
    def __init__(self, canvas: Canvas):
        self._xscale = 0
        self._yscale = 0
        self._canvas_ref = weakref.ref(canvas)
        self._xlines = visuals.Line()
        self._ylines = visuals.Line()
        self._xprops = LineProperties()
        self._yprops = LineProperties()
        super().__init__([self._xlines, self._ylines])
        canvas._viewbox.add(self)
        self.order = -99999

    def set_x_grid_lines(self, visible: bool, color, width: float, style):
        self._xprops.visible = visible
        self._xprops.color = color
        self._xprops.width = width
        self._xprops.style = style
        self.update()

    def set_y_grid_lines(self, visible: bool, color, width: float, style):
        self._yprops.visible = visible
        self._yprops.color = color
        self._yprops.width = width
        self._yprops.style = style
        self.update()

    def _prepare_draw(self, view):
        if not (self._xprops.visible or self._yprops.visible):
            return super()._prepare_draw(view)
        if canvas := self._canvas_ref():
            rect = canvas._camera.rect
            xmin, xmax = rect.left, rect.right
            ymin, ymax = rect.bottom, rect.top
            if self._xprops.visible:
                xmajor_pos = canvas._xticks._get_ticker()._get_tick_frac_labels()[0]
                xmajor = xmajor_pos * (xmax - xmin) + xmin
                xdata = [np.array([[x, ymin], [x, ymax]]) for x in xmajor]
                self._xlines.set_data(
                    pos=_safe_concat(xdata),
                    color=self._xprops.color,
                    width=self._xprops.width,
                    connect=_make_connection(xdata),
                )
            if self._yprops.visible:
                ymajor_pos = canvas._yticks._get_ticker()._get_tick_frac_labels()[0]
                ymajor = ymajor_pos * (ymax - ymin) + ymin
                ydata = [np.array([[xmin, y], [xmax, y]]) for y in ymajor]
                self._ylines.set_data(
                    pos=_safe_concat(ydata),
                    color=self._yprops.color,
                    width=self._yprops.width,
                    connect=_make_connection(ydata),
                )
        return super()._prepare_draw(view)
