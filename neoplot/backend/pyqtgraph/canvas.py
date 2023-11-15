from __future__ import annotations
from typing import Callable, cast

from qtpy import QtWidgets as QtW, QtCore
import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import (
    MouseClickEvent as pgMouseClickEvent,
    MouseDragEvent as pgMouseDragEvent,
)
import numpy as np

from neoplot import protocols
from neoplot.types import MouseButton, Modifier, MouseEventType, MouseEvent
from .app import run_app, get_app
from ._labels import Title, XLabel, YLabel, XAxis, YAxis


@protocols.check_protocol(protocols.CanvasProtocol)
class Canvas(QtW.QWidget):
    """A 1-D data viewer that have similar API as napari Viewer."""

    def __init__(self, parent: QtW.QWidget | None = None):
        # prepare widget
        viewbox = pg.ViewBox()
        self._plot_item = pg.PlotItem(viewBox=viewbox)

        # This ROI is not editable. Mouse click event will use it to determine
        # the origin of the coordinate system.
        self._coordinate_fiducial = pg.ROI((0, 0))
        self._coordinate_fiducial.setVisible(False)
        self._viewbox().addItem(self._coordinate_fiducial, ignoreBounds=True)

        _layoutwidget = pg.GraphicsLayoutWidget()
        _layoutwidget.addItem(self._plot_item)

        super().__init__(parent)
        layout = QtW.QVBoxLayout()
        layout.addWidget(_layoutwidget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self._xaxis = XAxis(self)
        self._yaxis = YAxis(self)
        self._title = Title(self)
        self._xlabel = XLabel(self)
        self._ylabel = YLabel(self)

    def _plt_get_title(self):
        return self._title

    def _plt_get_xaxis(self):
        return self._xaxis

    def _plt_get_yaxis(self):
        return self._yaxis

    def _plt_get_xlabel(self):
        return self._xlabel

    def _plt_get_ylabel(self):
        return self._ylabel

    def _viewbox(self) -> pg.ViewBox:
        return self._plot_item.vb

    def _plt_insert_layer(self, idx: int, layer: protocols.BaseProtocol):
        self._plot_item.addItem(layer)

    def _plt_remove_layer(self, layer):
        """Remove layer from the canvas"""
        self._plot_item.removeItem(layer)

    def _plt_get_visible(self) -> bool:
        """Get visibility of canvas"""
        return self.isVisible()

    def _plt_set_visible(self, visible: bool):
        """Set visibility of canvas"""
        self.setVisible(visible)

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

        def _cb(ev: pgMouseDragEvent):
            callback(self._translate_mouse_event(ev, MouseEventType.MOVE))

        self._get_scene().sigMouseMoved.connect(_cb)

    def _plt_connect_mouse_double_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

        def _cb(ev: pgMouseClickEvent):
            if not ev.double():
                return
            callback(self._translate_mouse_event(ev, MouseEventType.DOUBLE_CLICK))

        self._get_scene().sigMouseClicked.connect(_cb)

    def _plt_connect_xlim_changed(self, callback: Callable[[tuple[float, float]], None]):
        """Connect callback to x-limits changed event"""
        self._plot_item.sigXRangeChanged.connect(lambda _, x: callback(x))

    def _plt_connect_ylim_changed(self, callback: Callable[[tuple[float, float]], None]):
        """Connect callback to y-limits changed event"""
        self._plot_item.sigYRangeChanged.connect(lambda _, y: callback(y))

    def _translate_mouse_event(
        self,
        ev: pgMouseClickEvent | pgMouseDragEvent,
        typ: MouseEventType,
    ) -> MouseEvent:
        """Translate a mouse event from pyqtgraph to neoplot."""
        ev.currentItem = self._viewbox().childGroup  # as fiducial
        qpoint = cast(QtCore.QPointF, ev.pos())

        qt_modifiers = ev.modifiers()
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

        return MouseEvent(
            button=_QT_BUTTON_MAP.get(ev.button(), MouseButton.NONE),
            modifiers=modifiers,
            pos=(qpoint.x(), qpoint.y()),
            type=typ,
        )

    # def to_clipboard(self):
    #     """Copy the image to clipboard."""
    #     app = get_app()
    #     img = self.native.grab().toImage()
    #     app.clipboard().setImage(img)


_QT_MODIFIERS_MAP = {
    QtCore.Qt.KeyboardModifier.NoModifier: (),
    QtCore.Qt.KeyboardModifier.ShiftModifier: (Modifier.SHIFT,),
    QtCore.Qt.KeyboardModifier.ControlModifier: (Modifier.CTRL,),
    QtCore.Qt.KeyboardModifier.AltModifier: (Modifier.ALT,),
    QtCore.Qt.KeyboardModifier.MetaModifier: (Modifier.META,),
    QtCore.Qt.KeyboardModifier.ShiftModifier
    | QtCore.Qt.KeyboardModifier.ControlModifier: (Modifier.SHIFT, Modifier.CTRL),
    QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.AltModifier: (Modifier.SHIFT, Modifier.ALT),
    QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.MetaModifier: (Modifier.SHIFT, Modifier.META),
    QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.AltModifier: (Modifier.CTRL, Modifier.ALT),
    QtCore.Qt.KeyboardModifier.ControlModifier
    | QtCore.Qt.KeyboardModifier.MetaModifier: (Modifier.CTRL, Modifier.META),
    QtCore.Qt.KeyboardModifier.AltModifier | QtCore.Qt.KeyboardModifier.MetaModifier: (Modifier.ALT, Modifier.META),
}
_QT_BUTTON_MAP = {
    QtCore.Qt.MouseButton.LeftButton: MouseButton.LEFT,
    QtCore.Qt.MouseButton.RightButton: MouseButton.RIGHT,
    QtCore.Qt.MouseButton.MiddleButton: MouseButton.MIDDLE,
    QtCore.Qt.MouseButton.BackButton: MouseButton.BACK,
    QtCore.Qt.MouseButton.ForwardButton: MouseButton.FORWARD,
}


@protocols.check_protocol(protocols.MainWindowProtocol)
class MainCanvas(QtW.QMainWindow):
    def __init__(self):
        app = get_app()  # type: ignore
        super().__init__()
        self._canvas = Canvas(self)
        self.setCentralWidget(self._canvas)

    def _plt_get_canvas(self) -> protocols.CanvasProtocol:
        """Get canvas of main window"""
        return self._canvas

    def _plt_get_visible(self) -> bool:
        """Get visibility of canvas"""
        return self.isVisible()

    def _plt_set_visible(self, visible: bool):
        """Set visibility of canvas"""
        if visible:
            self.show()
            run_app()
        else:
            self.hide()
