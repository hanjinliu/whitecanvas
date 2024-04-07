from __future__ import annotations

import pyqtgraph as pg
from qtpy import QtCore, QtGui
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


class PyQtAxis(pg.AxisItem):
    def __init__(self, orientation, **kwargs):
        super().__init__(orientation, **kwargs)
        self.style["tickRotation"] = 0.0

    def drawPicture(self, p: QtGui.QPainter, axisSpec, tickSpecs, textSpecs):
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        p.setRenderHint(p.RenderHint.TextAntialiasing, True)

        ## draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        # p.translate(0.5,0)  ## resolves some damn pixel ambiguity

        ## draw ticks
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)

        # Draw all text
        if self.style["tickFont"] is not None:
            p.setFont(self.style["tickFont"])
        p.setPen(self.textPen())
        bounding = self.boundingRect().toAlignedRect()
        p.setClipRect(bounding)
        rot = self.style["tickRotation"]
        if abs(rot) < 1e-3:
            for rect, flags, text in textSpecs:
                p.drawText(rect, int(flags), text)
        else:
            for rect, flags, text in textSpecs:
                rect: QtCore.QRect
                p.save()
                p.translate(rect.center())
                p.rotate(rot)
                p.drawText(rect.translated(-rect.center()), int(flags), text)
                p.rotate(-rot)
                p.restore()

    def tickRotation(self) -> float:
        return -self.style["tickRotation"]

    def setTickRotation(self, rotation: float) -> None:
        self.style["tickRotation"] = -rotation
        self.picture = None
        self.update()
