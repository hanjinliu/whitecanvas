from __future__ import annotations
from functools import reduce
import numpy as np
from numpy.typing import NDArray

from qtpy import QtGui
import pyqtgraph as pg
from whitecanvas.protocols import LineProtocol, MultiLinesProtocol, check_protocol
from whitecanvas.types import LineStyle
from ._qt_utils import array_to_qcolor, from_qt_line_style, to_qt_line_style


@check_protocol(LineProtocol)
class MonoLine(pg.PlotCurveItem):
    def __init__(self, xdata, ydata):
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        pen.setCosmetic(True)
        super().__init__(xdata, ydata, pen=pen, antialias=False)

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.isVisible()

    def _plt_set_visible(self, visible: bool):
        self.setVisible(visible)

    def _plt_set_zorder(self, zorder: int):
        self.setZValue(zorder)

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


@check_protocol(MultiLinesProtocol)
class MultiLines(pg.ItemGroup):
    def __init__(self, data: list[NDArray[np.floating]]):
        super().__init__()
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        pen.setCosmetic(True)
        self._lines: list[pg.PlotCurveItem] = []
        for seg in data:
            item = pg.PlotCurveItem(seg[:, 0], seg[:, 1], pen=pen, antialias=True)
            self.addItem(item)
            self._lines.append(item)
        self._qpen_default = pen
        self._data = data
        self._bounding_rect_cache = None
        self._antialias = True

    def boundingRect(self):
        if self._bounding_rect_cache is not None:
            return self._bounding_rect_cache
        rect = reduce(lambda a, b: a | b, (item.boundingRect() for item in self._lines))
        self._bounding_rect_cache = rect
        return rect

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.isVisible()

    def _plt_set_visible(self, visible: bool):
        self.setVisible(visible)

    def _plt_set_zorder(self, zorder: int):
        self.setZValue(zorder)

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
            pen = self._qpen_default
            for _ in range(ndata - nitem):
                item = pg.PlotCurveItem(pen=pen, antialias=True)
                self.addItem(item)
                self._lines.append(item)
        for item, seg in zip(self._lines, data):
            item.setData(seg[:, 0], seg[:, 1])
        self._bounding_rect_cache = None

    ##### HasEdges #####
    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        return np.array([item.opts["pen"].widthF() for item in self._lines])

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        if np.isscalar(width):
            width = np.full(len(self._lines), width, dtype=np.float32)
        for item, w in zip(self._lines, width):
            pen: QtGui.QPen = item.opts["pen"]
            pen.setWidthF(w)
            item.setPen(pen)

    def _plt_get_edge_style(self) -> LineStyle:
        styles = []
        for item in self._lines:
            style = from_qt_line_style(item.opts["pen"].style())
            styles.append(style)
        return styles

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        if isinstance(style, LineStyle):
            qstyles = [to_qt_line_style(style)] * len(self._lines)
        else:
            qstyles = [to_qt_line_style(s) for s in style]

        for item, s in zip(self._lines, qstyles):
            pen: QtGui.QPen = item.opts["pen"]
            pen.setStyle(s)
            item.setPen(pen)

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        return np.stack(
            [item.opts["pen"].color().getRgbF() for item in self._lines],
            axis=0,
            dtype=np.float32,
        )

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        if color.ndim == 1:
            color = np.tile(color, (len(self._data), 1))
        for item, col in zip(self._lines, color):
            pen: QtGui.QPen = item.opts["pen"]
            pen.setColor(array_to_qcolor(col))
            item.setPen(pen)

    def _plt_get_antialias(self) -> bool:
        return self._antialias

    def _plt_set_antialias(self, antialias: bool):
        for item in self._lines:
            item.opts["antialias"] = antialias
            item.update()
        self._antialias = antialias
