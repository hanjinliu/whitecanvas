from __future__ import annotations

import pyqtgraph as pg
import numpy as np
from cmap import Colormap
from whitecanvas.protocols import ImageProtocol, check_protocol
from ._base import PlotlyLayer


@check_protocol(ImageProtocol)
class Image(PlotlyLayer):
    def __init__(self, data: np.ndarray):
        self._props = {
            "z": data,
            "visible": True,
            "type": "heatmap",
            "colorscale": "gray",
            "zmin": np.min(data),
            "zmax": np.max(data),
            "showlegend": False,
        }
        self._cmap = Colormap("gray")

    def _plt_get_data(self) -> np.ndarray:
        return self._props["z"]

    def _plt_set_data(self, data: np.ndarray):
        self._props["z"] = data

    def _plt_get_colormap(self) -> Colormap:
        return self._cmap

    def _plt_set_colormap(self, cmap: Colormap):
        self._props["colorscale"] = cmap.to_plotly()
        self._cmap = cmap

    def _plt_get_clim(self) -> tuple[float, float]:
        return self._props["zmin"], self._props["zmax"]

    def _plt_set_clim(self, clim: tuple[float, float]):
        self._props["zmin"], self._props["zmax"] = clim

    def _plt_get_visible(self) -> bool:
        return self._props["visible"]
