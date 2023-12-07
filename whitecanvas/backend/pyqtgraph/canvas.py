from __future__ import annotations
from typing import Callable, cast

import qtpy
from qtpy import QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import (
    MouseClickEvent as pgMouseClickEvent,
    MouseDragEvent as pgMouseDragEvent,
)
import numpy as np

from whitecanvas import protocols
from whitecanvas.types import MouseButton, Modifier, MouseEventType, MouseEvent
from whitecanvas.backend.pyqtgraph._base import PyQtLayer
from ._labels import Title, AxisLabel, Axis, Ticks


@protocols.check_protocol(protocols.CanvasProtocol)
class Canvas:
    def __init__(self, item: pg.PlotItem | None = None):
        # prepare widget
        if item is None:
            viewbox = pg.ViewBox()
            item = pg.PlotItem(viewBox=viewbox)
        item.vb.disableAutoRange()  # auto range is done in the whitecanvas side
        self._plot_item = item
        self._xaxis = Axis(self, axis="bottom")
        self._yaxis = Axis(self, axis="left")
        self._xticks = Ticks(self, axis="bottom")
        self._yticks = Ticks(self, axis="left")
        self._title = Title(self)
        self._xlabel = AxisLabel(self, axis="bottom")
        self._ylabel = AxisLabel(self, axis="left")
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
        locked = self._viewbox().state['aspectLocked']
        if locked == False:
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

    def _get_scene(self) -> pg.GraphicsScene:
        return self._plot_item.scene()

    def _plt_connect_mouse_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(ev: pgMouseClickEvent):
            if ev.double():
                return
            callback(self._translate_mouse_event(ev, MouseEventType.CLICK))

        self._get_scene().sigMouseClicked.connect(_cb)

    def _plt_connect_mouse_drag(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(qpoint: QtCore.QPointF):
            scene = self._get_scene()
            btns = scene.dragButtons
            evs: list[pgMouseClickEvent] = scene.clickEvents
            if len(btns) == 0:
                button = MouseButton.NONE
            else:
                button = _QT_BUTTON_MAP.get(btns[0], MouseButton.NONE)
            if len(evs) == 0:
                modifiers = ()
            else:
                modifiers = _from_qt_modifiers(evs[0].modifiers())
            MouseEvent(
                button=button,
                modifiers=modifiers,
                pos=(qpoint.x(), qpoint.y()),
                type=MouseEventType.MOVE,
            )

        self._get_scene().sigMouseMoved.connect(_cb)

    def _plt_connect_mouse_double_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(ev: pgMouseClickEvent):
            if not ev.double():
                return
            callback(self._translate_mouse_event(ev, MouseEventType.DOUBLE_CLICK))

        self._get_scene().sigMouseClicked.connect(_cb)

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

    def _translate_mouse_event(
        self,
        ev: pgMouseClickEvent | pgMouseDragEvent,
        typ: MouseEventType,
    ) -> MouseEvent:
        """Translate a mouse event from pyqtgraph to whitecanvas."""
        ev.currentItem = self._viewbox().childGroup  # as fiducial
        qpoint = cast(QtCore.QPointF, ev.pos())

        modifiers = _from_qt_modifiers(ev.modifiers())

        return MouseEvent(
            button=_QT_BUTTON_MAP.get(ev.button(), MouseButton.NONE),
            modifiers=modifiers,
            pos=(qpoint.x(), qpoint.y()),
            type=typ,
        )


def _from_qt_modifiers(
    qt_modifiers: QtCore.Qt.KeyboardModifier,
) -> tuple[Modifier, ...]:
    if (modifiers := _QT_MODIFIERS_MAP.get(qt_modifiers, None)) is None:
        # NOTE: some OS have default modifiers
        _lst = []
        if QtCore.Qt.KeyboardModifier.ShiftModifier & qt_modifiers:
            _lst.append(Modifier.SHIFT)
        if QtCore.Qt.KeyboardModifier.ControlModifier & qt_modifiers:
            _lst.append(Modifier.CTRL)
        if QtCore.Qt.KeyboardModifier.AltModifier & qt_modifiers:
            _lst.append(Modifier.ALT)
        if QtCore.Qt.KeyboardModifier.MetaModifier & qt_modifiers:
            _lst.append(Modifier.META)
        modifiers = tuple(_lst)
        _QT_MODIFIERS_MAP[qt_modifiers] = modifiers
    return modifiers


_QT_MODIFIERS_MAP = {
    QtCore.Qt.KeyboardModifier.NoModifier: (),
    QtCore.Qt.KeyboardModifier.ShiftModifier: (Modifier.SHIFT,),
    QtCore.Qt.KeyboardModifier.ControlModifier: (Modifier.CTRL,),
    QtCore.Qt.KeyboardModifier.AltModifier: (Modifier.ALT,),
    QtCore.Qt.KeyboardModifier.MetaModifier: (Modifier.META,),
    QtCore.Qt.KeyboardModifier.ShiftModifier
    | QtCore.Qt.KeyboardModifier.ControlModifier: (Modifier.SHIFT, Modifier.CTRL),
    QtCore.Qt.KeyboardModifier.ShiftModifier
    | QtCore.Qt.KeyboardModifier.AltModifier: (Modifier.SHIFT, Modifier.ALT),
    QtCore.Qt.KeyboardModifier.ShiftModifier
    | QtCore.Qt.KeyboardModifier.MetaModifier: (Modifier.SHIFT, Modifier.META),
    QtCore.Qt.KeyboardModifier.ControlModifier
    | QtCore.Qt.KeyboardModifier.AltModifier: (Modifier.CTRL, Modifier.ALT),
    QtCore.Qt.KeyboardModifier.ControlModifier
    | QtCore.Qt.KeyboardModifier.MetaModifier: (Modifier.CTRL, Modifier.META),
    QtCore.Qt.KeyboardModifier.AltModifier
    | QtCore.Qt.KeyboardModifier.MetaModifier: (Modifier.ALT, Modifier.META),
}
_QT_BUTTON_MAP = {
    QtCore.Qt.MouseButton.LeftButton: MouseButton.LEFT,
    QtCore.Qt.MouseButton.RightButton: MouseButton.RIGHT,
    QtCore.Qt.MouseButton.MiddleButton: MouseButton.MIDDLE,
    QtCore.Qt.MouseButton.BackButton: MouseButton.BACK,
    QtCore.Qt.MouseButton.ForwardButton: MouseButton.FORWARD,
}


@protocols.check_protocol(protocols.CanvasGridProtocol)
class CanvasGrid:
    def __init__(self, heights: list[int], widths: list[int], app: str = "default"):
        if app == "notebook":
            from pyqtgraph.jupyter import GraphicsLayoutWidget
        elif app in ("default", "qt"):
            from pyqtgraph import GraphicsLayoutWidget
        else:
            raise ValueError(f"pyqtgraph does not support {app!r}. Must be ")
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
        return Canvas(item)

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

    def _plt_set_figsize(self, width: float, height: float):
        self._layoutwidget.resize(int(width), int(height))
