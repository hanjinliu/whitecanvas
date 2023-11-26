from __future__ import annotations
from functools import reduce
import numpy as np
from numpy.typing import NDArray

from qtpy import QtGui
import pyqtgraph as pg
from whitecanvas.protocols import LineProtocol, MultiLineProtocol, check_protocol
from whitecanvas.types import LineStyle
from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from ._qt_utils import array_to_qcolor, from_qt_line_style, to_qt_line_style


@check_protocol(LineProtocol)
class MonoLine(pg.PlotCurveItem, PyQtLayer):
    def __init__(self, xdata, ydata):
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        pen.setCosmetic(True)
        super().__init__(xdata, ydata, pen=pen, antialias=False)

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self.getData()

    def _plt_set_data(self, xdata, ydata):
        self.setData(xdata, ydata)

    ##### HasEdges #####
    def _get_pen(self) -> QtGui.QPen:
        return self.opts["pen"]

    def _plt_get_edge_width(self) -> float:
        return self._get_pen().widthF()

    def _plt_set_edge_width(self, width: float):
        pen = self._get_pen()
        pen.setWidthF(width)
        self.setPen(pen)

    def _plt_get_edge_style(self) -> LineStyle:
        return from_qt_line_style(self._get_pen().style())

    def _plt_set_edge_style(self, style: LineStyle):
        pen = self._get_pen()
        pen.setStyle(to_qt_line_style(style))
        self.setPen(pen)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.array(self._get_pen().color().getRgbF())

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        pen = self._get_pen()
        pen.setColor(array_to_qcolor(color))
        self.setPen(pen)

    def _plt_get_antialias(self) -> bool:
        return self.opts["antialias"]

    def _plt_set_antialias(self, antialias: bool):
        self.opts["antialias"] = antialias
        self.update()


@check_protocol(MultiLineProtocol)
class MultiLine(pg.ItemGroup, PyQtLayer):
    def __init__(self, data: list[NDArray[np.floating]]):
        super().__init__()
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        pen.setCosmetic(True)
        self._lines: list[pg.PlotCurveItem] = []
        for seg in data:
            item = pg.PlotCurveItem(seg[:, 0], seg[:, 1], pen=pen, antialias=True)
            self.addItem(item)
            self._lines.append(item)
        self._qpen = pen
        self._data = data
        self._bounding_rect_cache = None
        self._antialias = True

    def boundingRect(self):
        if self._bounding_rect_cache is not None:
            return self._bounding_rect_cache
        rect = reduce(lambda a, b: a | b, (item.boundingRect() for item in self._lines))
        self._bounding_rect_cache = rect
        return rect

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        return self._data

    def _plt_set_data(self, data: list[NDArray[np.floating]]):
        ndata = len(data)
        nitem = len(self._lines)
        scene = self.scene()
        if ndata < nitem:
            for item in self._lines[ndata:]:
                scene.removeItem(item)
            self._lines = self._lines[:ndata]
        else:
            pen = self._qpen
            for _ in range(ndata - nitem):
                item = pg.PlotCurveItem(pen=pen, antialias=True)
                self.addItem(item)
                self._lines.append(item)
        for item, seg in zip(self._lines, data):
            item.setData(seg[:, 0], seg[:, 1])
        self._bounding_rect_cache = None

    ##### HasEdges #####
    def _set_pen_to_curves(self):
        for item in self._lines:
            item.setPen(self._qpen)

    def _plt_get_edge_width(self) -> float:
        return self._qpen.widthF()

    def _plt_set_edge_width(self, width: float):
        self._qpen.setWidthF(width)
        self._set_pen_to_curves()

    def _plt_get_edge_style(self) -> LineStyle:
        return from_qt_line_style(self._qpen.style())

    def _plt_set_edge_style(self, style: LineStyle):
        self._qpen.setStyle(to_qt_line_style(style))
        self._set_pen_to_curves()

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.array(self._qpen.color().getRgbF())

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        self._qpen.setColor(array_to_qcolor(color))
        self._set_pen_to_curves()

    def _plt_get_antialias(self) -> bool:
        return self._antialias

    def _plt_set_antialias(self, antialias: bool):
        for item in self._lines:
            item.opts["antialias"] = antialias
            item.update()
        self._antialias = antialias
