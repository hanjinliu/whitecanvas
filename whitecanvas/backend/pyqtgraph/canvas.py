from __future__ import annotations

from typing import Callable, cast

import numpy as np
import pyqtgraph as pg
import qtpy
from pyqtgraph.GraphicsScene.mouseEvents import HoverEvent as pgHoverEvent
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent as pgMouseClickEvent
from qtpy import QtCore, QtGui
from qtpy.QtCore import Signal

from whitecanvas import protocols
from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from whitecanvas.backend.pyqtgraph._labels import Axis, AxisLabel, Ticks, Title
from whitecanvas.backend.pyqtgraph._legend import QtItemSampleBase, make_sample_item
from whitecanvas.backend.pyqtgraph._qt_utils import from_qt_button, from_qt_modifiers
from whitecanvas.layers._legend import LegendItem, LegendItemCollection
from whitecanvas.types import LegendLocation, MouseButton, MouseEvent, MouseEventType


@protocols.check_protocol(protocols.CanvasProtocol)
class Canvas:
    def __init__(
        self,
        item: pg.PlotItem | None = None,
        *,
        xaxis: str = "bottom",
        yaxis: str = "left",
    ):
        # prepare widget
        if item is None:
            viewbox = pg.ViewBox()
            item = pg.PlotItem(viewBox=viewbox)
        item.vb.disableAutoRange()  # auto range is done in the whitecanvas side
        self._signals = SignalListener()
        item.vb.sigRangeChanged.connect(self._signals._set_rect)
        item.addItem(self._signals)
        self._plot_item = item
        self._xaxis = Axis(self, axis=xaxis)
        self._yaxis = Axis(self, axis=yaxis)
        self._xticks = Ticks(self, axis=xaxis)
        self._yticks = Ticks(self, axis=yaxis)
        self._title = Title(self)
        self._xlabel = AxisLabel(self, axis=xaxis)
        self._ylabel = AxisLabel(self, axis=yaxis)
        self._last_event: MouseEvent = None

    def _plt_get_native(self):
        return self._plot_item

    def _plt_get_title(self):
        return self._title

    def _plt_get_xaxis(self):
        return self._xaxis

    def _plt_get_yaxis(self):
        return self._yaxis

    def _plt_get_xlabel(self):
        return self._xlabel

    def _plt_get_xticks(self):
        return self._xticks

    def _plt_get_yticks(self):
        return self._yticks

    def _plt_get_ylabel(self):
        return self._ylabel

    def _plt_reorder_layers(self, layers: list[PyQtLayer]):
        for i, layer in enumerate(layers):
            layer._plt_set_zorder(i)

    def _plt_get_aspect_ratio(self) -> float | None:
        """Get aspect ratio of canvas"""
        locked = self._viewbox().state["aspectLocked"]
        if not locked:
            return None
        return float(locked)

    def _plt_set_aspect_ratio(self, ratio: float | None):
        """Set aspect ratio of canvas"""
        if ratio is None:
            self._viewbox().setAspectLocked(lock=False)
        else:
            self._viewbox().setAspectLocked(lock=True, ratio=ratio)

    def _viewbox(self) -> pg.ViewBox:
        return self._plot_item.vb

    def _plt_add_layer(self, layer: protocols.BaseProtocol):
        self._plot_item.addItem(layer)

    def _plt_remove_layer(self, layer):
        """Remove layer from the canvas"""
        self._plot_item.removeItem(layer)

    def _plt_get_visible(self) -> bool:
        """Get visibility of canvas"""
        return self._plot_item.isVisible()

    def _plt_set_visible(self, visible: bool):
        """Set visibility of canvas"""
        self._plot_item.setVisible(visible)

    def _plt_twinx(self) -> Canvas:
        """Create a twinx canvas"""
        plotitem = self._plot_item
        vb1 = plotitem.vb
        vb2 = pg.ViewBox()
        canvas = Canvas(pg.PlotItem(viewBox=vb2), yaxis="right")

        self._get_scene().addItem(vb2)
        plotitem.getAxis("right").linkToView(vb2)
        vb2.setXLink(plotitem)

        def _update_views():
            vb2.setGeometry(vb1.sceneBoundingRect())
            vb2.linkedViewChanged(vb1, vb2.XAxis)

        _update_views()
        vb1.sigResized.connect(_update_views)

        return canvas

    def _plt_twiny(self) -> Canvas:
        plotitem = self._plot_item
        vb1 = plotitem.vb
        vb2 = pg.ViewBox()
        canvas = Canvas(pg.PlotItem(viewBox=vb2), xaxis="top")

        self._get_scene().addItem(vb2)
        plotitem.getAxis("bottom").linkToView(vb2)
        vb2.setYLink(plotitem)

        def _update_views():
            vb2.setGeometry(vb1.sceneBoundingRect())
            vb2.linkedViewChanged(vb1, vb2.YAxis)

        _update_views()
        vb1.sigResized.connect(_update_views)

        return canvas

    # def _plt_inset(self, rect: Rect) -> Canvas:
    #     ...

    def _get_scene(self) -> pg.GraphicsScene:
        return self._plot_item.scene()

    def _plt_connect_mouse_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(ev):
            mev = self._translate_mouse_event(ev, MouseEventType.CLICK)
            callback(mev)

        self._signals.pressed.connect(_cb)

    def _plt_connect_mouse_drag(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(qpoint: QtCore.QPointF):
            mev = self._translate_mouse_event(qpoint, MouseEventType.MOVE)
            callback(mev)

        self._signals.moved.connect(_cb)

    def _plt_connect_mouse_double_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(ev):
            mev = self._translate_mouse_event(ev, MouseEventType.DOUBLE_CLICK)
            callback(mev)

        self._signals.double_clicked.connect(_cb)

    def _plt_connect_mouse_release(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(ev):
            mev = self._translate_mouse_event(ev, MouseEventType.RELEASE)
            callback(mev)

        self._signals.released.connect(_cb)

    def _plt_connect_xlim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        """Connect callback to x-limits changed event"""
        self._plot_item.sigXRangeChanged.connect(lambda _, x: callback(x))

    def _plt_connect_ylim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        """Connect callback to y-limits changed event"""
        self._plot_item.sigYRangeChanged.connect(lambda _, y: callback(y))

    def _plt_draw(self):
        pass  # pyqtgraph has its own draw mechanism

    def _plt_make_legend(
        self,
        items: list[tuple[str, LegendItem]],
        anchor: LegendLocation,
    ):
        pos, offset = _LEGEND_POS[anchor]
        legend = self._plot_item.addLegend(sampleType=QtItemSampleBase)
        legend.anchor(pos, pos, offset=offset)
        for label, item in items:
            if item is None:
                continue
            if isinstance(item, LegendItemCollection):
                for sub_label, sub_item in item.items:
                    sample = make_sample_item(sub_item)
                    if sample is not None:
                        legend.addItem(sample, sub_label)
            else:
                sample = make_sample_item(item)
                if sample is not None:
                    legend.addItem(sample, label)

    def _translate_mouse_event(
        self,
        ev: QtGui.QMouseEvent | QtCore.QPointF,
        typ: MouseEventType,
    ) -> MouseEvent:
        """Translate a mouse event from pyqtgraph to whitecanvas."""
        if isinstance(ev, QtCore.QPointF):
            # Hover events only have positions.
            qpoint = ev
            scene = self._get_scene()
            btns = scene.dragButtons
            evs: list[pgMouseClickEvent] = scene.clickEvents
            if len(btns) == 0:
                button = MouseButton.NONE
            else:
                button = from_qt_button(btns[0])
            if len(evs) == 0:
                modifiers = ()
            else:
                modifiers = from_qt_modifiers(evs[0].modifiers())
            mev = MouseEvent(
                button=button,
                modifiers=modifiers,
                pos=(qpoint.x(), qpoint.y()),
                type=typ,
            )
        else:
            qpoint = cast(QtCore.QPointF, ev.pos())

            modifiers = from_qt_modifiers(ev.modifiers())

            mev = MouseEvent(
                button=from_qt_button(ev.button()),
                modifiers=modifiers,
                pos=(qpoint.x(), qpoint.y()),
                type=typ,
            )
        return mev


_LEGEND_POS = {
    LegendLocation.TOP_LEFT: ((0.0, 0.0), (10, 10)),
    LegendLocation.TOP_CENTER: ((0.5, 0.0), (0, 10)),
    LegendLocation.TOP_RIGHT: ((1.0, 0.0), (-10, 10)),
    LegendLocation.CENTER_LEFT: ((0.0, 0.5), (10, 0)),
    LegendLocation.CENTER: ((0.5, 0.5), (0, 0)),
    LegendLocation.CENTER_RIGHT: ((1.0, 0.5), (-10, 0)),
    LegendLocation.BOTTOM_LEFT: ((0.0, 1.0), (10, -10)),
    LegendLocation.BOTTOM_CENTER: ((0.5, 1.0), (0, -10)),
    LegendLocation.BOTTOM_RIGHT: ((1.0, 1.0), (-10, -10)),
    # These are not supported in pyqtgraph. Use the closest one.
    LegendLocation.LEFT_SIDE_TOP: ((0.0, 0.0), (10, 10)),
    LegendLocation.LEFT_SIDE_CENTER: ((0.0, 0.5), (10, 0)),
    LegendLocation.LEFT_SIDE_BOTTOM: ((0.0, 1.0), (10, -10)),
    LegendLocation.RIGHT_SIDE_TOP: ((1.0, 0.0), (-10, 10)),
    LegendLocation.RIGHT_SIDE_CENTER: ((1.0, 0.5), (-10, 0)),
    LegendLocation.RIGHT_SIDE_BOTTOM: ((1.0, 1.0), (-10, -10)),
    LegendLocation.TOP_SIDE_LEFT: ((0.0, 0.0), (10, 10)),
    LegendLocation.TOP_SIDE_CENTER: ((0.5, 0.0), (0, 10)),
    LegendLocation.TOP_SIDE_RIGHT: ((1.0, 0.0), (-10, 10)),
    LegendLocation.BOTTOM_SIDE_LEFT: ((0.0, 1.0), (10, -10)),
    LegendLocation.BOTTOM_SIDE_CENTER: ((0.5, 1.0), (0, -10)),
    LegendLocation.BOTTOM_SIDE_RIGHT: ((1.0, 1.0), (-10, -10)),
}


@protocols.check_protocol(protocols.CanvasGridProtocol)
class CanvasGrid:
    def __init__(self, heights: list[float], widths: list[float], app: str = "default"):
        if app == "notebook":
            from pyqtgraph.jupyter import GraphicsLayoutWidget
        elif app in ("default", "qt"):
            from pyqtgraph import GraphicsLayoutWidget
        else:
            raise ValueError(f"pyqtgraph does not support {app!r}")
        self._layoutwidget = GraphicsLayoutWidget()
        self._heights = heights  # TODO: not used
        self._widths = widths

    def _plt_add_canvas(self, row: int, col: int, rowspan: int, colspan: int) -> Canvas:
        vb = pg.ViewBox()
        item = pg.PlotItem(viewBox=vb)
        self._layoutwidget.addItem(item, row, col)
        if rowspan != 1:
            self._layoutwidget.ci.layout.setRowStretchFactor(row, rowspan)
        if colspan != 1:
            self._layoutwidget.ci.layout.setColumnStretchFactor(col, colspan)
        canvas = Canvas(item)
        return canvas

    def _plt_show(self) -> bool:
        self._layoutwidget.setVisible(True)

    def _get_background_brush(self) -> QtGui.QBrush:
        return self._layoutwidget.backgroundBrush()

    def _plt_get_background_color(self):
        brush = self._get_background_brush()
        return np.array(brush.color().getRgbF())

    def _plt_set_background_color(self, color):
        brush = self._get_background_brush()
        brush.setColor(QtGui.QColor.fromRgbF(*color))
        self._layoutwidget.setBackgroundBrush(brush)

    def _plt_screenshot(self):
        img: QtGui.QImage = self._layoutwidget.grab().toImage()
        bits = img.constBits()
        h, w, c = img.height(), img.width(), 4
        if qtpy.API_NAME.startswith("PySide"):
            arr = np.asarray(bits).reshape(h, w, c)
        else:
            bits.setsize(h * w * c)
            arr = np.frombuffer(bits, np.uint8).reshape(h, w, c)

        return arr[:, :, [2, 1, 0, 3]]

    def _plt_set_figsize(self, width: int, height: int):
        self._layoutwidget.resize(width, height)

    def _plt_set_spacings(self, wspace: float, hspace: float):
        self._layoutwidget.ci.layout.setHorizontalSpacing(wspace)
        self._layoutwidget.ci.layout.setVerticalSpacing(hspace)


class SignalListener(pg.GraphicsObject):
    # Mouse events in pyqtgraph is very complicated.
    # Adding this graphics object to the scene will make it easier.
    pressed = Signal(object)
    moved = Signal(object)
    released = Signal(object)
    double_clicked = Signal(object)

    def __init__(self):
        super().__init__()
        self._rect = QtCore.QRectF(-10000, -10000, 10000, 10000)
        self._had_button = False

    def _set_rect(self, vb, rng: tuple[tuple[float, float], tuple[float, float]], _):
        (x0, x1), (y0, y1) = rng
        self._rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

    def boundingRect(self):
        return self._rect

    def mousePressEvent(self, ev: pgMouseClickEvent):
        self.pressed.emit(ev)
        self._had_button = bool(ev.buttons() ^ QtCore.Qt.MouseButton.NoButton)
        super().mousePressEvent(ev)

    def hoverEvent(self, ev: pgHoverEvent):
        if ev.isExit():
            return
        btns: QtCore.Qt.MouseButtons = ev.buttons()

        has_button = bool(btns ^ QtCore.Qt.MouseButton.NoButton)
        if (not has_button) and self._had_button:
            self.released.emit(ev.pos())
        else:
            self.moved.emit(ev.pos())
        self._had_button = has_button

    def mouseDoubleClickEvent(self, ev):
        self.double_clicked.emit(ev)
        super().mouseDoubleClickEvent(ev)

    def paint(self, *args):
        pass
