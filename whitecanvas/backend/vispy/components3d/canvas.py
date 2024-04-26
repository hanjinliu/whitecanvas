from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from vispy.scene import ArcballCamera, SceneCanvas, ViewBox, visuals
from vispy.util.quaternion import Quaternion

from whitecanvas import protocols
from whitecanvas.backend.vispy._label import TextLabel
from whitecanvas.backend.vispy.components3d.axis import Axis3D, AxisLabel3D

if TYPE_CHECKING:
    from vispy.app.canvas import MouseEvent as vispyMouseEvent
    from vispy.scene import Grid


class Camera(ArcballCamera):
    def __init__(
        self,
        fov: float = 0,
        distance: None | float = None,
        translate_speed: float = 1,
        **kwargs,
    ):
        super().__init__(fov, distance, translate_speed, **kwargs)


class Canvas3D:
    def __init__(self, viewbox: ViewBox):
        self._outer_viewbox = viewbox
        grid = cast("Grid", viewbox.add_grid())
        grid.spacing = 0
        camera = Camera()
        _viewbox = grid.add_view(row=1, col=0, camera=camera)
        self._viewbox: ViewBox = _viewbox

        title = TextLabel("")
        title.height_max = 40
        self._xaxis = Axis3D(self, 0)
        self._yaxis = Axis3D(self, 1)
        self._zaxis = Axis3D(self, 2)
        self._xlabel = AxisLabel3D(self._xaxis)
        self._ylabel = AxisLabel3D(self._yaxis)
        self._zlabel = AxisLabel3D(self._zaxis)
        grid.add_widget(title, row=0, col=0)
        self._title = title
        camera._quaternion = Quaternion(-0.4, 0.1, -0.9, 0.2)

    def _plt_add_layer(self, layer: visuals.visuals.Visual):
        layer.set_gl_state(
            depth_test=False,
            cull_face=False,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha", "one", "one"),
            blend_equation="func_add",
        )
        layer.parent = self._viewbox.scene

    def _plt_get_native(self):
        return self._viewbox

    def _plt_get_title(self):
        return self._title

    def _plt_reorder_layers(self, layers: list[protocols.BaseProtocol]):
        """Reorder layers in the canvas"""
        vb = self._viewbox
        for idx, layer in enumerate(layers):
            layer.order = idx
        if hasattr(vb, "_scene_ref"):
            scene: SceneCanvas = vb._scene_ref()
            scene._draw_order.clear()
            scene.update()

    def _plt_draw(self):
        pass

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
