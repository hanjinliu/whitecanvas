from functools import singledispatch
from typing import Generic, TypeVar

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ScatterPlotItem import drawSymbol
from qtpy import QtCore, QtGui

from whitecanvas.backend.pyqtgraph._qt_utils import (
    array_to_qcolor,
    to_qt_brush_style,
    to_qt_line_style,
    to_qt_symbol,
)
from whitecanvas.layers import _legend as _leg

_L = TypeVar("_L", bound=_leg.LegendItem)


@singledispatch
def make_sample_item(item) -> "QtItemSampleBase | None":
    return None


@make_sample_item.register
def _(item: _leg.LineLegendItem):
    return LineItemSample(item)


@make_sample_item.register
def _(item: _leg.MarkersLegendItem):
    return MarkerItemSample(item)


@make_sample_item.register
def _(item: _leg.BarLegendItem):
    return BarItemSample(item)


@make_sample_item.register
def _(item: _leg.PlotLegendItem):
    return PlotItemSample(item)


@make_sample_item.register
def _(item: _leg.ErrorLegendItem):
    return ErrorItemSample(item)


@make_sample_item.register
def _(item: _leg.LineErrorLegendItem):
    return LineErrorItemSample(item)


@make_sample_item.register
def _(item: _leg.MarkerErrorLegendItem):
    return MarkerErrorItemSample(item)


@make_sample_item.register
def _(item: _leg.PlotErrorLegendItem):
    return PlotErrorItemSample(item)


@make_sample_item.register
def _(item: _leg.StemLegendItem):
    return StemItemSample(item)


@make_sample_item.register
def _(item: _leg.TitleItem):
    return TitleItemSample(item)


class QtItemSampleBase(pg.GraphicsWidget, Generic[_L]):
    """Class responsible for drawing a single item in a LegendItem (sans label)"""

    def __init__(self, item: _L):
        pg.GraphicsWidget.__init__(self)
        self._item = item

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 20, 20)

    def paint(self, p, *args):
        pass


class LineItemSample(QtItemSampleBase[_leg.LineLegendItem]):
    def paint(self, p: QtGui.QPainter, *args):
        p.setPen(_edge_to_qpen(self._item))
        p.drawLine(0, 11, 20, 11)


class MarkerItemSample(QtItemSampleBase[_leg.MarkersLegendItem]):
    def paint(self, p: QtGui.QPainter, *args):
        p.translate(10, 10)
        drawSymbol(
            p,
            to_qt_symbol(self._item.symbol),
            self._item.size,
            _edge_to_qpen(self._item.edge),
            _face_to_qbrush(self._item.face),
        )


class BarItemSample(QtItemSampleBase[_leg.BarLegendItem]):
    def paint(self, p: QtGui.QPainter, *args):
        p.setBrush(_face_to_qbrush(self._item))
        p.setPen(_edge_to_qpen(self._item))
        p.drawRect(0, 4, 20, 12)


class PlotItemSample(QtItemSampleBase[_leg.PlotLegendItem]):
    def paint(self, p: QtGui.QPainter, *args):
        p.setPen(_edge_to_qpen(self._item.line))
        p.drawLine(0, 11, 20, 11)
        p.translate(10, 10)
        drawSymbol(
            p,
            to_qt_symbol(self._item.markers.symbol),
            self._item.markers.size,
            _edge_to_qpen(self._item.markers.edge),
            _face_to_qbrush(self._item.markers.face),
        )


class ErrorItemSample(QtItemSampleBase[_leg.ErrorLegendItem]):
    def paint(self, p: QtGui.QPainter, *args):
        p.setPen(_edge_to_qpen(self._item))
        p.drawLine(10, 4, 10, 16)
        if self._item.capsize > 0:
            p.drawLine(8, 4, 12, 4)
            p.drawLine(8, 16, 12, 16)


class LineErrorItemSample(QtItemSampleBase[_leg.LineErrorLegendItem]):
    def paint(self, p: QtGui.QPainter, *args):
        p.setPen(_edge_to_qpen(self._item.line))
        p.drawLine(0, 11, 20, 11)
        if self._item.xerr is not None:
            err = self._item.xerr
        elif self._item.yerr is not None:
            err = self._item.yerr
        else:
            return
        p.setPen(_edge_to_qpen(err))
        p.drawLine(10, 4, 10, 16)
        if err.capsize > 0:
            p.drawLine(8, 4, 12, 4)
            p.drawLine(8, 16, 12, 16)


class MarkerErrorItemSample(QtItemSampleBase[_leg.MarkerErrorLegendItem]):
    def paint(self, p: QtGui.QPainter, *args):
        p.translate(10, 10)
        drawSymbol(
            p,
            to_qt_symbol(self._item.markers.symbol),
            self._item.markers.size,
            _edge_to_qpen(self._item.markers.edge),
            _face_to_qbrush(self._item.markers.face),
        )
        if self._item.xerr is not None:
            p.setPen(_edge_to_qpen(self._item.xerr))
            p.drawLine(4, 10, 16, 10)
            if self._item.xerr.capsize > 0:
                p.drawLine(4, 8, 4, 12)
                p.drawLine(16, 8, 16, 12)
        if self._item.yerr is not None:
            p.setPen(_edge_to_qpen(self._item.yerr))
            p.drawLine(10, 4, 10, 16)
            if self._item.yerr.capsize > 0:
                p.drawLine(8, 4, 12, 4)
                p.drawLine(8, 16, 12, 16)


class PlotErrorItemSample(QtItemSampleBase[_leg.PlotErrorLegendItem]):
    def paint(self, p: QtGui.QPainter, *args):
        p.setPen(_edge_to_qpen(self._item.plot.line))
        p.drawLine(0, 11, 20, 11)
        p.translate(10, 10)
        drawSymbol(
            p,
            to_qt_symbol(self._item.plot.markers.symbol),
            self._item.plot.markers.size,
            _edge_to_qpen(self._item.plot.markers.edge),
            _face_to_qbrush(self._item.plot.markers.face),
        )
        if self._item.xerr is not None:
            p.setPen(_edge_to_qpen(self._item.xerr))
            p.drawLine(4, 10, 16, 10)
            if self._item.xerr.capsize > 0:
                p.drawLine(4, 8, 4, 12)
                p.drawLine(16, 8, 16, 12)
        if self._item.yerr is not None:
            p.setPen(_edge_to_qpen(self._item.yerr))
            p.drawLine(10, 4, 10, 16)
            if self._item.yerr.capsize > 0:
                p.drawLine(8, 4, 12, 4)
                p.drawLine(8, 16, 12, 16)


class StemItemSample(QtItemSampleBase[_leg.StemLegendItem]):
    def paint(self, p: QtGui.QPainter, *args):
        p.setPen(_edge_to_qpen(self._item.line))
        p.drawLine(10, 0, 10, 12)
        p.translate(10, 12)
        drawSymbol(
            p,
            to_qt_symbol(self._item.markers.symbol),
            self._item.markers.size,
            _edge_to_qpen(self._item.markers.edge),
            _face_to_qbrush(self._item.markers.face),
        )


class TitleItemSample(QtItemSampleBase[_leg.TitleItem]):
    def paint(self, p: QtGui.QPainter, *args):
        pass


def _face_to_qbrush(face: _leg.FaceInfo) -> QtGui.QBrush:
    brush = QtGui.QBrush()
    brush.setColor(array_to_qcolor(face.color))
    brush.setStyle(to_qt_brush_style(face.hatch))
    return brush


def _edge_to_qpen(edge: _leg.EdgeInfo) -> QtGui.QPen:
    pen = QtGui.QPen()
    pen.setCosmetic(True)
    pen.setColor(array_to_qcolor(edge.color))
    pen.setStyle(to_qt_line_style(edge.style))
    pen.setWidthF(edge.width)
    return pen
