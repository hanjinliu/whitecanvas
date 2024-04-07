from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Literal, NamedTuple, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray
from psygnal import Signal

from whitecanvas.canvas import CanvasBase
from whitecanvas.layers import Layer, Line, Rects, Spans
from whitecanvas.tools._polygon_utils import is_in_polygon
from whitecanvas.types import (
    ColorType,
    LineStyle,
    Modifier,
    MouseButton,
    MouseEvent,
    MouseEventType,
    Point,
    Rect,
    XYData,
    _Void,
)

if TYPE_CHECKING:
    from whitecanvas.layers._mixin import ConstEdge, ConstFace

    _MouseButton = Literal["left", "middle", "right", "back", "forward"] | MouseButton
    _Modifier = Literal["shift", "ctrl", "alt", "meta"] | Modifier

_L = TypeVar("_L", bound=Layer)
_void = _Void()


class SelectionToolBase(ABC, Generic[_L]):
    changed = Signal(object)
    """Emitted when the selection is changed."""

    cleared = Signal()
    """Emitted when the selection is cleared by user."""

    def __init__(
        self,
        canvas: CanvasBase,
        buttons: list[MouseButton],
        modifiers: list[Modifier],
        tracking: bool = False,
    ):
        self._canvas_ref = weakref.ref(canvas)
        self._valid_buttons = set(buttons)
        self._modifiers = set(modifiers)
        self._layer = self._create_layer()
        self._tracking = tracking
        self._persist = True
        self._enabled = True
        canvas.mouse.moved.connect(self.callback)

    def _canvas(self) -> CanvasBase:
        canvas = self._canvas_ref()
        if canvas is None:
            raise RuntimeError("Canvas has been deleted")
        return canvas

    @abstractmethod
    def _create_layer(self) -> _L: ...

    @abstractmethod
    def _update_layer(
        self,
        start: tuple[float, float],
        now: tuple[float, float],
    ): ...

    def _on_press(self, start: tuple[float, float]):
        pass

    @property
    @abstractmethod
    def selection(self):
        """This method returns the values that represents current selections."""

    @property
    def persist(self) -> bool:
        """Whether the selection persists after mouse release."""
        return self._persist

    @persist.setter
    def persist(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("persist must be a boolean value.")
        self._persist = value

    def callback(self, e: MouseEvent):
        """The callback function that is called when mouse is moved."""
        if not self._enabled:
            return
        if e.button not in self._valid_buttons or set(e.modifiers) != self._modifiers:
            return
        canvas = self._canvas()
        pos_start = e.pos
        self._on_press(pos_start)
        yield
        if self._layer in self._canvas().layers:
            self.clear_selection()
        with canvas.autoscale_context(enabled=False):
            canvas.add_layer(self._layer)
        dragged = False
        while e.type is not MouseEventType.RELEASE:
            self._update_layer(pos_start, e.pos)
            yield
            if self._tracking:
                self.changed.emit(self.selection)
            dragged = True

        if dragged:
            if not self._tracking:
                self.changed.emit(self.selection)
            if not self._persist:
                with self.cleared.blocked():
                    self.clear_selection()
        else:
            # clicked
            if self._persist:
                self.clear_selection()

    def clear_selection(self):
        """Clear the current selection."""
        self._canvas().layers.remove(self._layer)
        self.cleared.emit()

    @property
    def enabled(self) -> bool:
        """Whether the tool is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = bool(value)
        if not self._enabled:
            self.clear_selection()


class RectSelectionTool(SelectionToolBase[Rects]):
    def _create_layer(self) -> Rects:
        layer = Rects([[0, 1, 0, 1]], color="blue", alpha=0.4).with_edge(width=2)
        layer.visible = False
        return layer

    def _update_layer(
        self,
        start: tuple[float, float],
        now: tuple[float, float],
    ):
        x0, y0 = start
        x1, y1 = now
        self._layer.data = np.array([[x0, x1, y0, y1]])
        self._layer.visible = True

    @property
    def selection(self) -> Rect:
        return self._layer.rects[0]

    @property
    def face(self) -> ConstFace:
        """Face color of the selection span."""
        return self._layer.face

    @property
    def edge(self) -> ConstEdge:
        """Edge color of the selection span."""
        return self._layer.edge

    def contains_point(self, point: tuple[float, float]) -> bool:
        x, y = point
        sel = self.selection
        return sel.left <= x <= sel.right and sel.bottom <= y <= sel.top

    def contains_points(self, points: XYData | NDArray[np.number]) -> NDArray[np.bool_]:
        points = _atleast_2d(points)
        sel = self.selection
        if points.ndim == 2 and points.shape[1] == 2:
            xs = points[:, 0]
            ys = points[:, 1]
            return (
                (sel.left <= xs)
                & (xs <= sel.right)
                & (sel.bottom <= ys)
                & (ys <= sel.top)
            )
        else:
            raise ValueError("points must be (2,) or (N, 2) array.")


class LineSelection(NamedTuple):
    start: Point
    end: Point


class LineSelectionTool(SelectionToolBase[Line]):
    def _create_layer(self) -> Line:
        return Line(np.array([]), np.array([]), width=2, color="blue", alpha=0.4)

    def _update_layer(
        self,
        start: tuple[float, float],
        now: tuple[float, float],
    ):
        x0, y0 = start
        x1, y1 = now
        self._layer.data = np.array([x0, x1]), np.array([y0, y1])

    @property
    def selection(self) -> LineSelection:
        xs, ys = self._layer.data
        return LineSelection(Point(xs[0], ys[0]), Point(xs[1], ys[1]))

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | LineStyle | _Void = _void,
        alpha: float | _Void = _void,
    ):
        self._layer.update(color=color, width=width, style=style, alpha=alpha)

    @property
    def color(self) -> NDArray[np.float32]:
        """Color of the selection line."""
        return self._layer.color

    @color.setter
    def color(self, color: ColorType):
        self._layer.color = color

    @property
    def width(self) -> float:
        """Width of the selection line."""
        return self._layer.width

    @width.setter
    def width(self, width: float):
        self._layer.width = width

    @property
    def style(self) -> LineStyle:
        """Style of the selection line."""
        return self._layer.style

    @style.setter
    def style(self, style: str | LineStyle):
        self._layer.style = style

    @property
    def alpha(self) -> float:
        """Alpha channel of the selection line."""
        return self._layer.alpha

    @alpha.setter
    def alpha(self, alpha: float):
        self._layer.alpha = alpha


class SpanSelection(NamedTuple):
    start: float
    end: float


class _SpanSelectionTool(SelectionToolBase[Spans]):
    @property
    def selection(self) -> SpanSelection:
        span = self._layer.data[0]
        return SpanSelection(*sorted(span))

    @property
    def face(self) -> ConstFace:
        """Face color of the selection span."""
        return self._layer.face

    @property
    def edge(self) -> ConstEdge:
        """Edge color of the selection span."""
        return self._layer.edge


class XSpanSelectionTool(_SpanSelectionTool):
    def _create_layer(self) -> Spans:
        layer = Spans([[0, 1]], orient="vertical", color="red", alpha=0.4)
        layer.visible = False
        return layer

    def _update_layer(
        self,
        start: tuple[float, float],
        now: tuple[float, float],
    ):
        x0, _ = start
        x1, _ = now
        self._layer.data = np.array([[x0, x1]], dtype=np.float32)
        self._layer.visible = True

    def contains_point(self, point: tuple[float, float]) -> bool:
        x, _ = point
        sel = self.selection
        return sel.start <= x <= sel.end

    def contains_points(self, points: XYData | NDArray[np.number]) -> NDArray[np.bool_]:
        points = _atleast_2d(points)
        sel = self.selection
        if points.ndim == 2 and points.shape[1] == 2:
            xs = points[:, 0]
            return (sel.start <= xs) & (xs <= sel.end)
        else:
            raise ValueError("points must be (2,) or (N, 2) array.")


class YSpanSelectionTool(_SpanSelectionTool):
    def _create_layer(self) -> Spans:
        layer = Spans([[0, 1]], orient="horizontal", color="red", alpha=0.4)
        layer.visible = False
        return layer

    def _update_layer(
        self,
        start: tuple[float, float],
        now: tuple[float, float],
    ):
        _, y0 = start
        _, y1 = now
        self._layer.data = np.array([[y0, y1]], dtype=np.float32)
        self._layer.visible = True

    def contains_point(self, point: tuple[float, float]) -> bool:
        _, y = point
        sel = self.selection
        return sel.start <= y <= sel.end

    def contains_points(self, points: XYData | NDArray[np.number]) -> NDArray[np.bool_]:
        points = _atleast_2d(points)
        sel = self.selection
        if points.ndim == 2 and points.shape[1] == 2:
            ys = points[:, 1]
            return (sel.start <= ys) & (ys <= sel.end)
        else:
            raise ValueError("points must be (2,) or (N, 2) array.")


class LassoSelectionTool(LineSelectionTool):
    def _update_layer(
        self,
        start: tuple[float, float],
        now: tuple[float, float],
    ):
        x1, y1 = now
        current = self._layer.data
        xs = np.concatenate([current.x, [x1]])
        ys = np.concatenate([current.y, [y1]])
        self._layer.data = xs, ys

    @property
    def selection(self) -> XYData:
        return self._layer.data

    def _on_press(self, start: tuple[float, float]):
        self._layer.data = np.array([start[0]]), np.array([start[1]])

    def close_path(self, emit: bool = False):
        """
        Close the path by connecting the last point to the first point.

        Parameters
        ----------
        emit : bool, default False
            If True, the changed signal is emitted after closing the path.
        """
        current = self._layer.data
        if len(current.x) > 2:
            xs = np.concatenate([current.x, [current.x[0]]])
            ys = np.concatenate([current.y, [current.y[0]]])
            self._layer.data = xs, ys
        if emit:
            self.changed.emit(self.selection)
        return

    def contains_point(self, point: tuple[float, float]) -> bool:
        x, y = point
        poly = self._layer.data
        return is_in_polygon(np.array([[x, y]]), poly.stack())[0]

    def contains_points(self, points: XYData | NDArray[np.number]) -> NDArray[np.bool_]:
        points = _atleast_2d(points)
        poly = self._layer.data
        if points.ndim == 2 and points.shape[1] == 2:
            return is_in_polygon(points, poly.stack())
        else:
            raise ValueError("points must be (2,) or (N, 2) array.")


def _norm_input(
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
):
    if isinstance(buttons, (str, MouseButton)):
        buttons = [buttons]
    _buttons = [MouseButton(btn) for btn in buttons]
    if MouseButton.NONE in _buttons:
        raise ValueError("MouseButton.NONE is not allowed.")
    if modifiers is None:
        modifiers = []
    elif isinstance(modifiers, (str, Modifier)):
        modifiers = [modifiers]
    _modifiers = [Modifier(mod) for mod in modifiers]
    return _buttons, _modifiers


def _atleast_2d(points: NDArray[np.number]) -> NDArray[np.number]:
    if isinstance(points, XYData):
        return points.stack()
    return np.atleast_2d(points)


def line_selector(
    canvas: CanvasBase,
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
    *,
    tracking: bool = False,
) -> LineSelectionTool:
    """
    Create a line selector tool with given settings.

    A line selector emits a LineSelection object when a line is drawn.
    A selection tool is constructed by specifying the canvas to attach the tool.

    >>> canvas = new_canvas("matplotlib:qt")
    >>> tool = line_selector(canvas)

    Use `buttons` and `modifiers` to specify how to trigger the tool.

    >>> tool = line_selector(canvas, buttons="right", modifiers="ctrl")

    Parameters
    ----------
    canvas : CanvasBase
        The canvas to which the tool is attached.
    buttons : MouseButton or Sequence[MouseButton], default "left"
        The mouse buttons that can trigger the tool.
    modifiers : Modifier or Sequence[Modifier], optional
        The modifier keys that must be pressed to trigger the tool.
    tracking : bool, default False
        If True, the tool emits the changed signal while dragging. Otherwise, it emits
        the signal only when dragging is finished.
    """
    _buttons, _modifiers = _norm_input(buttons, modifiers)
    return LineSelectionTool(canvas, _buttons, _modifiers, tracking=tracking)


def rect_selector(
    canvas: CanvasBase,
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
    *,
    tracking: bool = False,
) -> RectSelectionTool:
    """
    Create a rectangle selector tool with given settings.

    A rectangle selector emits a Rect object when a rectangle is drawn.
    A selection tool is constructed by specifying the canvas to attach the tool.

    >>> canvas = new_canvas("matplotlib:qt")
    >>> tool = rect_selector(canvas)

    Use `buttons` and `modifiers` to specify how to trigger the tool.

    >>> tool = rect_selector(canvas, buttons="right", modifiers="ctrl")

    Parameters
    ----------
    canvas : CanvasBase
        The canvas to which the tool is attached.
    buttons : MouseButton or Sequence[MouseButton], default "left"
        The mouse buttons that can trigger the tool.
    modifiers : Modifier or Sequence[Modifier], optional
        The modifier keys that must be pressed to trigger the tool.
    tracking : bool, default False
        If True, the tool emits the changed signal while dragging. Otherwise, it emits
        the signal only when dragging is finished.
    """
    _buttons, _modifiers = _norm_input(buttons, modifiers)
    return RectSelectionTool(canvas, _buttons, _modifiers, tracking=tracking)


def xspan_selector(
    canvas: CanvasBase,
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
    *,
    tracking: bool = False,
) -> XSpanSelectionTool:
    """
    Create a x-span selector tool with given settings.

    A x-span selector emits a SpanSelection object (tuple of start and end) when a span
    is drawn.
    A selection tool is constructed by specifying the canvas to attach the tool.

    >>> canvas = new_canvas("matplotlib:qt")
    >>> tool = xspan_selector(canvas)

    Use `buttons` and `modifiers` to specify how to trigger the tool.

    >>> tool = line_selector(canvas, buttons="right", modifiers="ctrl")

    Parameters
    ----------
    canvas : CanvasBase
        The canvas to which the tool is attached.
    buttons : MouseButton or Sequence[MouseButton], default "left"
        The mouse buttons that can trigger the tool.
    modifiers : Modifier or Sequence[Modifier], optional
        The modifier keys that must be pressed to trigger the tool.
    tracking : bool, default False
        If True, the tool emits the changed signal while dragging. Otherwise, it emits
        the signal only when dragging is finished.
    """
    _buttons, _modifiers = _norm_input(buttons, modifiers)
    return XSpanSelectionTool(canvas, _buttons, _modifiers, tracking=tracking)


def yspan_selector(
    canvas: CanvasBase,
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
    *,
    tracking: bool = False,
) -> YSpanSelectionTool:
    """
    Create a line selector tool with given settings.

    A y-span selector emits a SpanSelection object (tuple of start and end) when a span
    is drawn.
    A selection tool is constructed by specifying the canvas to attach the tool.

    >>> canvas = new_canvas("matplotlib:qt")
    >>> tool = yspan_selector(canvas)

    Use `buttons` and `modifiers` to specify how to trigger the tool.

    >>> tool = line_selector(canvas, buttons="right", modifiers="ctrl")

    Parameters
    ----------
    canvas : CanvasBase
        The canvas to which the tool is attached.
    buttons : MouseButton or Sequence[MouseButton], default "left"
        The mouse buttons that can trigger the tool.
    modifiers : Modifier or Sequence[Modifier], optional
        The modifier keys that must be pressed to trigger the tool.
    tracking : bool, default False
        If True, the tool emits the changed signal while dragging. Otherwise, it emits
        the signal only when dragging is finished.
    """
    _buttons, _modifiers = _norm_input(buttons, modifiers)
    return YSpanSelectionTool(canvas, _buttons, _modifiers, tracking=tracking)


def lasso_selector(
    canvas: CanvasBase,
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
    *,
    tracking: bool = False,
) -> LassoSelectionTool:
    """
    Create a Lasso selector tool with given settings.

    A Lasso selector emits a XYData object by freehand drawing.
    A selection tool is constructed by specifying the canvas to attach the tool.

    >>> canvas = new_canvas("matplotlib:qt")
    >>> tool = lasso_selector(canvas)

    Use `buttons` and `modifiers` to specify how to trigger the tool.

    >>> tool = lasso_selector(canvas, buttons="right", modifiers="ctrl")

    Parameters
    ----------
    canvas : CanvasBase
        The canvas to which the tool is attached.
    buttons : MouseButton or Sequence[MouseButton], default "left"
        The mouse buttons that can trigger the tool.
    modifiers : Modifier or Sequence[Modifier], optional
        The modifier keys that must be pressed to trigger the tool.
    tracking : bool, default False
        If True, the tool emits the changed signal while dragging. Otherwise, it emits
        the signal only when dragging is finished.
    """
    _buttons, _modifiers = _norm_input(buttons, modifiers)
    return LassoSelectionTool(canvas, _buttons, _modifiers, tracking=tracking)
