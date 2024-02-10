from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from numpy.typing import NDArray
from psygnal import Signal
from vispy import use as vispy_use
from vispy.scene import PanZoomCamera, SceneCanvas, ViewBox, visuals
from vispy.util import keys

from whitecanvas import protocols
from whitecanvas.backend.vispy._label import Axis, TextLabel, Ticks
from whitecanvas.types import Modifier, MouseButton, MouseEvent, MouseEventType

if TYPE_CHECKING:
    from vispy.app.canvas import MouseEvent as vispyMouseEvent
    from vispy.scene import Grid
    from vispy.scene.subscene import SubScene
    from vispy.visuals import Visual


class Camera(PanZoomCamera):
    changed = Signal()

    def view_changed(self):
        super().view_changed()
        self.changed.emit()


@protocols.check_protocol(protocols.CanvasProtocol)
class Canvas:
    def __init__(self, viewbox: ViewBox):
        self._outer_viewbox = viewbox
        grid = cast("Grid", viewbox.add_grid())
        grid.spacing = 0
        _viewbox = grid.add_view(row=1, col=1, camera=Camera())
        self._viewbox: ViewBox = _viewbox

        title = TextLabel("")
        title.height_max = 40
        grid.add_widget(title, row=0, col=0, col_span=2)
        self._title = title
        x_axis = Axis(
            self,
            dim=1,
            orientation="bottom",
            anchors=("center", "bottom"),
            font_size=6,
            axis_label_margin=40,
            tick_label_margin=5,
            axis_label="",
        )
        x_axis.stretch = (1, 0.1)
        x_axis.height_min = x_axis.height_max = 40
        grid.add_widget(x_axis, row=2, col=1)
        x_axis.link_view(self._viewbox)
        y_axis = Axis(
            self,
            dim=0,
            orientation="left",
            anchors=("right", "middle"),
            font_size=6,
            axis_label_margin=50,
            tick_label_margin=5,
            axis_label="",
        )

        y_axis.stretch = (0.1, 1)
        grid.add_widget(y_axis, row=1, col=0)
        y_axis.link_view(self._viewbox)
        self._xaxis = x_axis
        self._yaxis = y_axis
        self._xticks = Ticks(x_axis)
        self._yticks = Ticks(y_axis)
        self._title = TextLabel("")
        self._xlabel = TextLabel("")
        self._ylabel = TextLabel("")
        self._grid = grid
        self._mouse_click_callbacks: list[Callable[[MouseEvent], None]] = []
        self._mouse_move_callbacks: list[Callable[[MouseEvent], None]] = []
        self._mouse_double_click_callbacks: list[Callable[[MouseEvent], None]] = []
        self._mouse_release_callbacks: list[Callable[[MouseEvent], None]] = []

    def _set_scene_ref(self, scene):
        self._viewbox.unfreeze()
        self._viewbox._canvas_ref = weakref.ref(self)
        self._viewbox._scene_ref = weakref.ref(scene)
        self._viewbox.freeze()

    def _plt_get_native(self):
        return self._viewbox.scene

    def _plt_get_title(self):
        return self._title

    def _plt_get_xaxis(self):
        return self._xaxis

    def _plt_get_yaxis(self):
        return self._yaxis

    def _plt_get_xlabel(self):
        return self._xlabel

    def _plt_get_xticks(self):
        return self._xticks

    def _plt_get_yticks(self):
        return self._yticks

    def _plt_get_ylabel(self):
        return self._ylabel

    def _plt_reorder_layers(self, layers: list[protocols.BaseProtocol]):
        """Reorder layers in the canvas"""
        vb = self._viewbox
        for idx, layer in enumerate(layers):
            layer.order = idx
        if hasattr(vb, "_scene_ref"):
            scene: SceneCanvas = vb._scene_ref()
            scene._draw_order.clear()
            scene.update()

    @property
    def _camera(self) -> Camera:
        return self._viewbox.camera

    def _plt_get_aspect_ratio(self) -> float | None:
        """Get aspect ratio of canvas"""
        return self._camera.aspect

    def _plt_set_aspect_ratio(self, ratio: float | None):
        """Set aspect ratio of canvas"""
        self._camera.aspect = ratio

    def _plt_add_layer(self, layer: visuals.visuals.Visual):
        layer.set_gl_state("opaque", depth_test=False)
        layer.parent = self._viewbox.scene

    def _plt_remove_layer(self, layer):
        """Remove layer from the canvas"""
        layer.parent = None

    def _plt_get_visible(self) -> bool:
        """Get visibility of canvas"""
        return self._grid.visible

    def _plt_set_visible(self, visible: bool):
        """Set visibility of canvas"""
        self._grid.visible = visible

    @property
    def _scene(self) -> SceneCanvas:
        return self._viewbox.scene

    def _plt_connect_mouse_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""
        self._mouse_click_callbacks.append(callback)

    def _plt_connect_mouse_drag(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""
        self._mouse_move_callbacks.append(callback)

    def _plt_connect_mouse_double_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""
        self._mouse_double_click_callbacks.append(callback)

    def _plt_connect_mouse_release(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""
        self._mouse_release_callbacks.append(callback)

    def _plt_connect_xlim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        @self._camera.changed.connect
        def _cb():
            with self._camera.changed.blocked():
                callback(self._xaxis._plt_get_limits())

    def _plt_connect_ylim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        @self._camera.changed.connect
        def _cb():
            with self._camera.changed.blocked():
                callback(self._yaxis._plt_get_limits())

    def _plt_draw(self):
        pass  # vispy has its own draw mechanism


@protocols.check_protocol(protocols.CanvasGridProtocol)
class CanvasGrid:
    def __init__(self, heights: list[int], widths: list[int], app: str = "default"):
        if app != "default":
            vispy_use(_APP_NAMES.get(app, app))
        self._scene = SceneCanvasExt(keys="interactive")
        self._grid: Grid = self._scene.central_widget.add_grid()
        self._scene.create_native()
        self._heights = heights  # TODO: not used
        self._widths = widths

    def _plt_add_canvas(self, row: int, col: int, rowspan: int, colspan: int):
        rspan = sum(self._heights[row : row + rowspan])
        cspan = sum(self._widths[col : col + colspan])
        r = sum(self._heights[:row])
        c = sum(self._widths[:col])
        viewbox: ViewBox = self._grid.add_view(r, c, rspan, cspan)
        canvas = Canvas(viewbox)
        canvas._set_scene_ref(self._scene)
        return canvas

    def _plt_get_background_color(self):
        return self._scene.bgcolor

    def _plt_set_background_color(self, color):
        self._scene.bgcolor = color

    def _plt_screenshot(self) -> NDArray[np.uint8]:
        return self._scene.render()

    def _plt_show(self):
        self._scene.show()

    def _plt_set_figsize(self, width: int, height: int):
        self._scene.size = (width, height)


_APP_NAMES = {
    "qt4": "pyqt4",
    "qt5": "pyqt5",
    "qt6": "pyqt6",
    "qt": "pyqt5",
    "tk": "tkinter",
    "notebook": "jupyter_rfb",
}


class SceneCanvasExt(SceneCanvas):
    scene: SubScene

    def on_mouse_press(self, event: vispyMouseEvent):
        visual = self.visual_at(event.pos)
        if isinstance(visual, ViewBox) and hasattr(visual, "_canvas_ref"):
            canvas: Canvas = visual._canvas_ref()
            tr = self.scene.node_transform(visual.scene)
            pos = tr.map(event.pos)[:2] - 0.5
            ev = MouseEvent(
                button=_VISPY_BUTTON_MAP.get(event.button, MouseButton.NONE),
                modifiers=tuple(_VISPY_KEY_MAP[mod] for mod in event.modifiers),
                pos=pos,
                type=MouseEventType.CLICK,
            )
            for callback in canvas._mouse_click_callbacks:
                callback(ev)

    def on_mouse_move(self, event: vispyMouseEvent):
        visual = self.visual_at(event.pos)
        if isinstance(visual, ViewBox) and hasattr(visual, "_canvas_ref"):
            canvas: Canvas = visual._canvas_ref()
            tr = self.scene.node_transform(visual.scene)
            pos = tr.map(event.pos)[:2] - 0.5
            ev = MouseEvent(
                button=_VISPY_BUTTON_MAP.get(event.button, MouseButton.NONE),
                modifiers=tuple(_VISPY_KEY_MAP[mod] for mod in event.modifiers),
                pos=pos,
                type=MouseEventType.MOVE,
            )

            for callback in canvas._mouse_move_callbacks:
                callback(ev)

    def on_mouse_double_click(self, event: vispyMouseEvent):
        visual = self.visual_at(event.pos)
        if isinstance(visual, ViewBox) and hasattr(visual, "_canvas_ref"):
            canvas: Canvas = visual._canvas_ref()
            tr = self.scene.node_transform(visual.scene)
            pos = tr.map(event.pos)[:2] - 0.5
            ev = MouseEvent(
                button=_VISPY_BUTTON_MAP.get(event.button, MouseButton.NONE),
                modifiers=tuple(_VISPY_KEY_MAP[mod] for mod in event.modifiers),
                pos=pos,
                type=MouseEventType.DOUBLE_CLICK,
            )

            for callback in canvas._mouse_double_click_callbacks:
                callback(ev)

    def on_mouse_release(self, event: vispyMouseEvent):
        visual = self.visual_at(event.pos)
        if isinstance(visual, ViewBox) and hasattr(visual, "_canvas_ref"):
            canvas: Canvas = visual._canvas_ref()
            tr = self.scene.node_transform(visual.scene)
            pos = tr.map(event.pos)[:2] - 0.5
            ev = MouseEvent(
                button=_VISPY_BUTTON_MAP.get(event.button, MouseButton.NONE),
                modifiers=tuple(_VISPY_KEY_MAP[mod] for mod in event.modifiers),
                pos=pos,
                type=MouseEventType.RELEASE,
            )

            for callback in canvas._mouse_release_callbacks:
                callback(ev)


def as_overlay(layer: Visual, canvas: Canvas):
    layer.parent = canvas._outer_viewbox.scene
    layer.order = 10000


_VISPY_KEY_MAP = {
    keys.SHIFT: Modifier.SHIFT,
    keys.CONTROL: Modifier.CTRL,
    keys.ALT: Modifier.ALT,
    keys.META: Modifier.META,
}

_VISPY_BUTTON_MAP = {
    0: MouseButton.LEFT,
    1: MouseButton.RIGHT,
    2: MouseButton.MIDDLE,
}
