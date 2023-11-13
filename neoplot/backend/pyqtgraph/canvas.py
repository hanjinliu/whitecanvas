from __future__ import annotations

from qtpy import QtWidgets as QtW
import pyqtgraph as pg
import numpy as np

from psygnal import Signal

# from .mouse_event import MouseClickEvent

from neoplot import protocols
from .app import run_app, get_app
from ._labels import Title, XLabel, YLabel, XAxis, YAxis


class Canvas(QtW.QWidget):
    """A 1-D data viewer that have similar API as napari Viewer."""
    
    # range_changed = Signal(object)
    # mouse_clicked = Signal(MouseClickEvent)

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
        self._update_scene()

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
        # TODO: zorder
    
    def _plt_remove_layer(self, layer):
        """Remove layer from the canvas"""
        self._plot_item.removeItem(layer)

    def _plt_get_visible(self) -> bool:
        """Get visibility of canvas"""
        return self.isVisible()
    
    def _plt_set_visible(self, visible: bool):
        """Set visibility of canvas"""
        self.setVisible(visible)
    
    def _update_scene(self):
        # Since plot item does not have graphics scene before being added to
        # a graphical layout, mouse event should be connected afterward.
        self._plot_item.scene().sigMouseClicked.connect(self._mouse_clicked)
        self._plot_item.sigRangeChanged.connect(self._range_changed)
        
    def _range_changed(self, *_):
        ...
    
    def _mouse_clicked(self, *_):
        ...

    # def to_clipboard(self):
    #     """Copy the image to clipboard."""
    #     app = get_app()
    #     img = self.native.grab().toImage()
    #     app.clipboard().setImage(img)

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

assert isinstance(Canvas, protocols.CanvasProtocol)
assert isinstance(MainCanvas, protocols.MainWindowProtocol)
