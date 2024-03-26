from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Literal, NamedTuple, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray
from psygnal import Signal

from whitecanvas.canvas import CanvasBase
from whitecanvas.layers import Layer, Line, Rects, Spans
from whitecanvas.types import (
    ColorType,
    LineStyle,
    Modifier,
    MouseButton,
    MouseEvent,
    MouseEventType,
    Point,
    _Void,
)

if TYPE_CHECKING:
    _MouseButton = Literal["left", "middle", "right", "back", "forward"] | MouseButton
    _Modifier = Literal["shift", "ctrl", "alt", "meta"] | Modifier

_L = TypeVar("_L", bound=Layer)
_void = _Void()


class SelectionToolBase(ABC, Generic[_L]):
    changed = Signal(object)

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
        canvas.events.mouse_moved.connect(self.callback)

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

    @abstractmethod
    def selection(self):
        """This method returns the values that represents current selections."""

    def callback(self, e: MouseEvent):
        """The callback function that is called when mouse is moved."""
        if e.button not in self._valid_buttons or set(e.modifiers) != self._modifiers:
            return
        canvas = self._canvas()
        pos_start = e.pos
        if self._layer in self._canvas().layers:
            self.clear_selection()
        canvas.add_layer(self._layer)
        dragged = False
        while e.type is not MouseEventType.RELEASE:
            self._update_layer(pos_start, e.pos)
            yield
            if self._tracking:
                self.changed.emit(self.selection())
            dragged = True

        if dragged:
            self.changed.emit(self.selection())

    def clear_selection(self, e: MouseEvent | None = None):
        self._canvas().layers.remove(self._layer)


class RectSelectionTool(SelectionToolBase[Rects]):
    def _create_layer(self) -> Rects:
        layer = Rects([[0, 0, 0, 0]], color="blue", alpha=0.4)
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

    def selection(self):
        return self._layer.rects[0]


class LineSelection(NamedTuple):
    start: Point
    end: Point


class LineSelectionTool(SelectionToolBase[Line]):
    def _create_layer(self) -> Line:
        return Line(np.array([0, 1]), np.array([0, 1]), color="blue", alpha=0.4)

    def _update_layer(
        self,
        start: tuple[float, float],
        now: tuple[float, float],
    ):
        x0, y0 = start
        x1, y1 = now
        self._layer.data = np.array([x0, x1]), np.array([y0, y1])

    def selection(self):
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
        return self._layer.color

    @color.setter
    def color(self, color: ColorType):
        self._layer.color = color

    @property
    def width(self) -> float:
        return self._layer.width

    @width.setter
    def width(self, width: float):
        self._layer.width = width

    @property
    def style(self) -> LineStyle:
        return self._layer.style

    @style.setter
    def style(self, style: str | LineStyle):
        self._layer.style = style

    @property
    def alpha(self) -> float:
        return self._layer.alpha

    @alpha.setter
    def alpha(self, alpha: float):
        self._layer.alpha = alpha


class SpanSelection(NamedTuple):
    start: float
    end: float


class _SpanSelectionTool(SelectionToolBase[Spans]):
    def selection(self):
        span = self._layer.data[0]
        return SpanSelection(span[0], span[1])

    @property
    def face(self):
        return self._layer.face

    @property
    def edge(self):
        return self._layer.edge


class XSpanSelectionTool(_SpanSelectionTool):
    def _create_layer(self) -> Spans:
        layer = Spans([[0, 0]], orient="vertical", color="red", alpha=0.4)
        layer.visible = False
        return layer

    def _update_layer(
        self,
        start: tuple[float, float],
        now: tuple[float, float],
    ):
        x0, _ = start
        x1, _ = now
        self._layer.data = [[x0, x1]]
        self._layer.visible = True


class YSpanSelectionTool(_SpanSelectionTool):
    def _create_layer(self) -> Spans:
        layer = Spans([[0, 0]], orient="horizontal", color="green", alpha=0.4)
        layer.visible = False
        return layer

    def _update_layer(
        self,
        start: tuple[float, float],
        now: tuple[float, float],
    ):
        _, y0 = start
        _, y1 = now
        self._layer.data = [[y0, y1]]


# class Lasso


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


def line_selector(
    canvas: CanvasBase,
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
    *,
    tracking: bool = False,
) -> LineSelectionTool:
    _buttons, _modifiers = _norm_input(buttons, modifiers)
    return LineSelectionTool(canvas, _buttons, _modifiers, tracking=tracking)


def rect_selector(
    canvas: CanvasBase,
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
    *,
    tracking: bool = False,
) -> RectSelectionTool:
    _buttons, _modifiers = _norm_input(buttons, modifiers)
    return RectSelectionTool(canvas, _buttons, _modifiers, tracking=tracking)


def xspan_selector(
    canvas: CanvasBase,
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
    *,
    tracking: bool = False,
) -> XSpanSelectionTool:
    _buttons, _modifiers = _norm_input(buttons, modifiers)
    return XSpanSelectionTool(canvas, _buttons, _modifiers, tracking=tracking)


def yspan_selector(
    canvas: CanvasBase,
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
    *,
    tracking: bool = False,
) -> YSpanSelectionTool:
    _buttons, _modifiers = _norm_input(buttons, modifiers)
    return YSpanSelectionTool(canvas, _buttons, _modifiers, tracking=tracking)
