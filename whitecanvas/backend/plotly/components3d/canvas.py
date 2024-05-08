from __future__ import annotations

import sys
import warnings
import weakref
from typing import TYPE_CHECKING, Callable

import numpy as np
from plotly import graph_objects as go

from whitecanvas.backend.plotly._labels import Axis3D, AxisLabel3D, Ticks3D
from whitecanvas.backend.plotly.canvas import Canvas


class Canvas3D(Canvas):
    def __init__(
        self,
        fig: go.Figure,
        *,
        row: int = 0,
        col: int = 0,
        secondary_y: bool = False,
        app: str = "default",
    ):
        super().__init__(fig, row=row, col=col, secondary_y=secondary_y, app=app)
        self._xaxis = Axis3D(self, axis="xaxis")
        self._yaxis = Axis3D(self, axis="yaxis")
        self._zaxis = Axis3D(self, axis="zaxis")
        self._xticks = Ticks3D(self, axis="xaxis")
        self._yticks = Ticks3D(self, axis="yaxis")
        self._zticks = Ticks3D(self, axis="zaxis")
        self._xlabel = AxisLabel3D(self, axis="xaxis")
        self._ylabel = AxisLabel3D(self, axis="yaxis")
        self._zlabel = AxisLabel3D(self, axis="zaxis")
        self._is_aspect_locked = False

    def _plt_get_zaxis(self):
        return self._zaxis

    def _plt_get_zlabel(self):
        return self._zlabel

    def _plt_get_zticks(self):
        return self._zticks

    def _plt_get_aspect_locked(self) -> bool:
        return self._is_aspect_locked

    def _plt_set_aspect_locked(self, locked: bool):
        if locked:
            self._fig.update_layout(scene_aspectmode="auto")
        else:
            self._fig.update_layout(scene_aspectmode="data")

    def _plt_connect_xlim_changed(self, callback):
        propname = f"scene.{self._xaxis.name}.range"
        self._fig.layout.on_change(lambda _, lim: callback(lim), propname)

    def _plt_connect_ylim_changed(self, callback):
        propname = f"scene.{self._yaxis.name}.range"
        self._fig.layout.on_change(lambda _, lim: callback(lim), propname)

    def _plt_connect_zlim_changed(self, callback):
        propname = f"scene.{self._zaxis.name}.range"
        self._fig.layout.on_change(lambda _, lim: callback(lim), propname)
