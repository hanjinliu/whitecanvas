from __future__ import annotations

from typing import Callable

import numpy as np
from mpl_toolkits.mplot3d import Axes3D, art3d

from whitecanvas.backend.matplotlib._base import MplLayer, MplMouseEventsMixin
from whitecanvas.backend.matplotlib._labels import MplAxis, MplLabel, MplTicks, Title


class Canvas3D:
    def __init__(self, ax: Axes3D):
        self._axes = ax
        self._xaxis = MplAxis(self, "x")
        self._yaxis = MplAxis(self, "y")
        self._zaxis = MplAxis(self, "z")
        self._title = Title(self)
        self._xlabel = MplLabel(self, "x")
        self._ylabel = MplLabel(self, "y")
        self._zlabel = MplLabel(self, "z")
        self._xticks = MplTicks(self, "x")
        self._yticks = MplTicks(self, "y")
        self._zticks = MplTicks(self, "z")

    def _plt_add_layer(self, layer):
        if isinstance(layer, art3d.Line3D):
            self._axes.add_line(layer)
        elif isinstance(layer, art3d.Path3DCollection):
            self._axes.add_collection3d(layer)
        else:
            raise NotImplementedError(f"{layer}")
        if hasattr(layer, "post_add"):
            layer.post_add(self)

    def _plt_get_native(self):
        return self._axes

    def _plt_get_title(self):
        return self._title

    def _plt_get_xaxis(self):
        return self._xaxis

    def _plt_get_yaxis(self):
        return self._yaxis

    def _plt_get_zaxis(self):
        return self._zaxis

    def _plt_get_xlabel(self):
        return self._xlabel

    def _plt_get_ylabel(self):
        return self._ylabel

    def _plt_get_zlabel(self):
        return self._zlabel

    def _plt_get_xticks(self):
        return self._xticks

    def _plt_get_yticks(self):
        return self._yticks

    def _plt_get_zticks(self):
        return self._zticks

    def _plt_reorder_layers(self, layers: list[MplLayer]):
        for i, layer in enumerate(layers):
            layer._plt_set_zorder(i)

    def _plt_connect_xlim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        """Connect callback to x-limits changed event"""
        self._axes.callbacks.connect("xlim_changed", lambda ax: callback(ax.get_xlim()))

    def _plt_connect_ylim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        """Connect callback to y-limits changed event"""
        self._axes.callbacks.connect("ylim_changed", lambda ax: callback(ax.get_ylim()))

    def _plt_connect_zlim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        """Connect callback to z-limits changed event"""
        self._axes.callbacks.connect("zlim_changed", lambda ax: callback(ax.get_zlim()))

    def _plt_draw(self):
        if fig := self._axes.get_figure():
            fig.canvas.draw_idle()
