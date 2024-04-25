from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from vispy.scene import PanZoomCamera, SceneCanvas, ViewBox, visuals

from whitecanvas import protocols
from whitecanvas.backend.vispy._label import TextLabel

if TYPE_CHECKING:
    from vispy.app.canvas import MouseEvent as vispyMouseEvent
    from vispy.scene import Grid
    from vispy.scene.subscene import SubScene
    from vispy.visuals import Visual


class Canvas3D:
    def __init__(self, viewbox: ViewBox):
        self._outer_viewbox = viewbox
        grid = cast("Grid", viewbox.add_grid())
        grid.spacing = 0
        _viewbox = grid.add_view(row=1, col=0, camera=PanZoomCamera())
        self._viewbox: ViewBox = _viewbox

        title = TextLabel("")
        title.height_max = 40
        grid.add_widget(title, row=0, col=0, col_span=2)
        self._title = title

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
