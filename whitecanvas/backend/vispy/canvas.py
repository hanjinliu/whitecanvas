from __future__ import annotations
from typing import Callable, TYPE_CHECKING
import weakref

from psygnal import Signal
from vispy.scene import ViewBox, SceneCanvas, PanZoomCamera
import numpy as np
from numpy.typing import NDArray

from whitecanvas import protocols
from whitecanvas.types import MouseButton, Modifier, MouseEventType, MouseEvent
from ._label import TextLabel, Axis, Ticks

if TYPE_CHECKING:
    from vispy.scene import Grid
    from vispy.app.canvas import MouseEvent as vispyMouseEvent


class Camera(PanZoomCamera):
    resized = Signal()

    def viewbox_resize_event(self, event):
        super().viewbox_resize_event(event)
        self.resized.emit()


@protocols.check_protocol(protocols.CanvasProtocol)
class Canvas:
    def __init__(self, viewbox: ViewBox):
        grid: Grid = viewbox.add_grid()
        grid.spacing = 0
        _viewbox = grid.add_view(row=1, col=1, camera=Camera())
        self._viewbox: ViewBox = _viewbox

        title = TextLabel("")
        title.height_max = 40
        grid.add_widget(title, row=0, col=0, col_span=2)
        self._title = title
        x_axis = Axis(
            dim=1,
            orientation="bottom",
            anchors=("center", "bottom"),
            font_size=6,
            axis_label_margin=40,
            tick_label_margin=5,
            axis_label="",
        )
        x_axis.height_min = 65
        x_axis.height_max = 80
        x_axis.stretch = (1, 0.1)
        grid.add_widget(x_axis, row=2, col=1)
        x_axis.link_view(self._viewbox)
        y_axis = Axis(
            dim=0,
            orientation="left",
            anchors=("right", "middle"),
            font_size=6,
            axis_label_margin=50,
            tick_label_margin=5,
            axis_label="",
        )
        y_axis.width_max = 80
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

    @property
    def _camera(self) -> Camera:
        return self._viewbox.camera

    def _plt_get_aspect_ratio(self) -> float | None:
        """Get aspect ratio of canvas"""
        return self._camera.aspect

    def _plt_set_aspect_ratio(self, ratio: float | None):
        """Set aspect ratio of canvas"""
        self._camera.aspect = ratio

    def _plt_add_layer(self, layer: protocols.BaseProtocol):
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

    def _plt_connect_xlim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        self._camera.resized.connect(lambda: callback(self._xaxis._plt_get_limits()))

    def _plt_connect_ylim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        self._camera.resized.connect(lambda: callback(self._yaxis._plt_get_limits()))


_VISPY_BUTTON_MAP = {
    0: MouseButton.LEFT,
    1: MouseButton.RIGHT,
    2: MouseButton.MIDDLE,
}


@protocols.check_protocol(protocols.CanvasGridProtocol)
class CanvasGrid:
    def __init__(self, heights: list[int], widths: list[int]):
        self._scene = SceneCanvasExt(keys="interactive")
        self._grid: Grid = self._scene.central_widget.add_grid()
        self._scene.create_native()

    def _plt_add_canvas(self, row: int, col: int, rowspan: int, colspan: int):
        viewbox = self._grid.add_view(row, col, rowspan, colspan)
        canvas = Canvas(viewbox)
        viewbox.unfreeze()
        viewbox._canvas_ref = weakref.ref(canvas)
        viewbox.freeze()
        return canvas

    def _plt_get_background_color(self):
        return self._scene.bgcolor

    def _plt_set_background_color(self, color):
        self._scene.bgcolor = color

    def _plt_screenshot(self) -> NDArray[np.uint8]:
        return self._scene.render()

    def _plt_get_visible(self) -> bool:
        """Get visibility of canvas"""
        return True

    def _plt_set_visible(self, visible: bool):
        """Set visibility of canvas"""
        if visible:
            self._scene.show()
        else:
            self._scene.close()


class SceneCanvasExt(SceneCanvas):
    def on_mouse_press(self, event: vispyMouseEvent):
        visual = self.visual_at(event.pos)
        if isinstance(visual, ViewBox) and hasattr(visual, "_canvas_ref"):
            canvas: Canvas = visual._canvas_ref()
            tr = self.scene.node_transform(visual.scene)
            pos = tr.map(event.pos)[:2] - 0.5
            ev = MouseEvent(
                button=_VISPY_BUTTON_MAP.get(event.button, MouseButton.NONE),
                modifiers=tuple(Modifier(mod) for mod in event.modifiers),
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
                modifiers=tuple(Modifier(mod) for mod in event.modifiers),
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
                modifiers=tuple(Modifier(mod) for mod in event.modifiers),
                pos=pos,
                type=MouseEventType.DOUBLE_CLICK,
            )

            for callback in canvas._mouse_double_click_callbacks:
                callback(ev)
