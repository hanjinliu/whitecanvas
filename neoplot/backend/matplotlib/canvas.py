from __future__ import annotations

import weakref
from matplotlib import pyplot as plt
from neoplot import protocols
from .line import Line
from .scatter import Scatter
from ._labels import Title, XAxis, YAxis, XLabel, YLabel

class Canvas:
    def __init__(self, *, ax: plt.Axes | None = None):
        if ax is None:
            ax = plt.gca()
        self._axes = ax
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

    def _plt_insert_layer(self, idx: int, layer: protocols.BaseProtocol):
        if isinstance(layer, Line):
            self._axes.add_line(layer)
        elif isinstance(layer, Scatter):
            self._axes.add_collection(layer)
        else:
            raise NotImplementedError
        self._axes.autoscale_view()
        # TODO: zorder
    
    def _plt_remove_layer(self, layer):
        """Remove layer from the canvas"""
        raise NotImplementedError

    def _plt_get_visible(self) -> bool:
        """Get visibility of canvas"""
        return self._axes.get_visible()
    
    def _plt_set_visible(self, visible: bool):
        """Set visibility of canvas"""
        self._axes.set_visible(visible)
    

class MainCanvas:
    def __init__(self):
        self._canvas = Canvas()
    
    def _plt_get_canvas(self) -> protocols.CanvasProtocol:
        return self._canvas
    
    def _plt_get_visible(self) -> bool:
        return True
    
    def _plt_set_visible(self, visible: bool):
        pass
