from __future__ import annotations

import pyqtgraph as pg
from qtpy import QtCore
from qtpy import QtWidgets as QtW

from whitecanvas.types import Rect


class PyQtLayer:
    def _plt_get_visible(self) -> bool:
        return self.isVisible()

    def _plt_set_visible(self, visible: bool):
        self.setVisible(visible)

    def _plt_set_zorder(self, zorder: int):
        self.setZValue(zorder)


class InsetPlotItem(pg.GraphicsWidget):
    def __init__(self, item: pg.PlotItem, rect: Rect):
        pg.GraphicsWidget.__init__(self)
        self._layout = QtW.QGraphicsGridLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.setLayout(self._layout)
        self._layout.addItem(item, 0, 0)
        self._item = item
        self._rect = rect

    def mouseDragEvent(self, ev):
        self._item.vb.mouseDragEvent(ev)

    def setParentItem(self, parent: pg.ViewBox):
        super().setParentItem(parent)
        parent.geometryChanged.connect(self._update_geometry)
        parent.sigResized.connect(self._update_geometry)

    def _update_geometry(self, *_):
        vb_rect: QtCore.QRect = self.parentItem().boundingRect()
        width = vb_rect.width()
        height = vb_rect.height()
        self.setGeometry(
            int(width * self._rect.left),
            int(height * (1 - self._rect.top)),
            int(width * self._rect.width),
            int(height * self._rect.height),
        )
