from __future__ import annotations

import numpy as np
from vispy.scene import visuals

from whitecanvas.backend.vispy.line import MonoLine as MonoLine2D


class MonoLine3D(MonoLine2D):
    def __init__(self, xdata, ydata, zdata):
        data = np.stack([xdata, ydata, zdata], axis=1)
        visuals.Line.__init__(self, data, antialias=True)
        self.unfreeze()

    ##### BaseProtocol #####
    def _plt_get_visible(self) -> bool:
        return self.visible

    def _plt_set_visible(self, visible: bool):
        self.visible = visible

    ##### XYDataProtocol #####
    def _plt_get_data(self):
        pos = self.pos
        return pos[:, 0], pos[:, 1], pos[:, 2]

    def _plt_set_data(self, xdata, ydata, zdata):
        self.set_data(np.stack([xdata, ydata, zdata], axis=1))
