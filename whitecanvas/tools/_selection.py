from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Sequence,
    TypeVar,
    overload,
)

import numpy as np
from numpy.typing import NDArray
from psygnal import Signal

from whitecanvas.canvas import CanvasBase
from whitecanvas.layers import Layer, Line, Markers, Rects, Spans
from whitecanvas.tools._selection_types import (
    LineSelection,
    PointSelection,
    PolygonSelection,
    RectSelection,
    SelectionMode,
    SpanSelection,
    XSpanSelection,
    YSpanSelection,
)
from whitecanvas.types import (
    ColorType,
    Hatch,
    LineStyle,
    Modifier,
    MouseButton,
    MouseEvent,
    MouseEventType,
    Point,
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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._canvas_ref() is not None:
            self.disconnect()

    def _canvas(self) -> CanvasBase:
        canvas = self._canvas_ref()
        if canvas is None:
            raise RuntimeError(f"Canvas is not connected to {self!r}.")
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

    def disconnect(self):
        """Disconnect the tool from the canvas."""
        self._canvas().mouse.moved.disconnect(self.callback)
        if self._layer in self._canvas().layers:
            self.clear_selection()
        return None

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
        layer = Rects([[0, 1, 0, 1]], color="blue", alpha=0.25).with_edge(
            width=2, alpha=0.4
        )
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
    def selection(self) -> RectSelection:
        return RectSelection(*self._layer.rects[0])

    @property
    def face(self) -> ConstFace:
        """Face color of the selection span."""
        return self._layer.face

    @property
    def edge(self) -> ConstEdge:
        """Edge color of the selection span."""
        return self._layer.edge

    @overload
    def contains_point(self, point: tuple[float, float], /) -> bool: ...
    @overload
    def contains_point(self, x: float, y: float, /) -> bool: ...

    def contains_point(self, *args) -> bool:
        return self.selection.contains_point(*args)

    def contains_points(self, points: XYData | NDArray[np.number]) -> NDArray[np.bool_]:
        return self.selection.contains_points(points)


class PointSelectionTool(SelectionToolBase[Markers]):
    def _create_layer(self) -> Markers:
        return Markers(np.array([]), np.array([]), size=10, color="blue", alpha=0.4)

    def _update_layer(
        self,
        start: tuple[float, float],
        now: tuple[float, float],
    ):
        x, y = now
        self._layer.data = np.array([x]), np.array([y])

    @property
    def selection(self) -> Point:
        xs, ys = self._layer.data
        return PointSelection(xs[0], ys[0])

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        symbol: str | _Void = _void,
        size: float | _Void = _void,
        alpha: float | _Void = _void,
        hatch: str | Hatch | _Void = _void,
    ):
        self._layer.update(
            color=color, symbol=symbol, size=size, alpha=alpha, hatch=hatch
        )
        return self


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
        return self

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


class _SpanSelectionTool(SelectionToolBase[Spans]):
    @property
    @abstractmethod
    def selection(self) -> SpanSelection: ...

    @property
    def face(self) -> ConstFace:
        """Face color of the selection span."""
        return self._layer.face

    @property
    def edge(self) -> ConstEdge:
        """Edge color of the selection span."""
        return self._layer.edge

    @overload
    def contains_point(self, point: tuple[float, float], /) -> bool: ...
    @overload
    def contains_point(self, x: float, y: float, /) -> bool: ...

    def contains_point(self, *args) -> bool:
        return self.selection.contains_point(*args)

    def contains_points(self, points: XYData | NDArray[np.number]) -> NDArray[np.bool_]:
        return self.selection.contains_points(points)


class XSpanSelectionTool(_SpanSelectionTool):
    @property
    def selection(self) -> XSpanSelection:
        span = self._layer.data[0]
        return XSpanSelection(*sorted(span))

    def _create_layer(self) -> Spans:
        layer = Spans([[0, 1]], orient="vertical", color="red", alpha=0.25)
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


class YSpanSelectionTool(_SpanSelectionTool):
    @property
    def selection(self) -> YSpanSelection:
        span = self._layer.data[0]
        return YSpanSelection(*sorted(span))

    def _create_layer(self) -> Spans:
        layer = Spans([[0, 1]], orient="horizontal", color="red", alpha=0.25)
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
    def selection(self) -> PolygonSelection:
        return PolygonSelection(*self._layer.data)

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

    @overload
    def contains_point(self, point: tuple[float, float], /) -> bool: ...
    @overload
    def contains_point(self, x: float, y: float, /) -> bool: ...

    def contains_point(self, *args) -> bool:
        return self.selection.contains_point(*args)

    def contains_points(self, points: XYData | NDArray[np.number]) -> NDArray[np.bool_]:
        return self.selection.contains_points(points)


class PolygonSelectionTool(LassoSelectionTool):
    def __init__(
        self,
        canvas: CanvasBase,
        buttons: list[MouseButton],
        modifiers: list[Modifier],
        tracking: bool = False,
        auto_close: bool = False,
    ):
        super().__init__(canvas, buttons, modifiers, tracking)
        self._auto_close = auto_close

    def _redraw_layer(self, now: tuple[float, float]):
        cur_data = self._layer.data
        x0, y0 = now
        xs = np.concatenate([cur_data.x[:-1], [x0]])
        ys = np.concatenate([cur_data.y[:-1], [y0]])
        self._layer.data = xs, ys

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
            return
        with canvas.autoscale_context(enabled=False):
            canvas.add_layer(self._layer)

        while e.type is not MouseEventType.RELEASE:
            yield  # dragging
        self._update_layer(pos_start, e.pos)
        while True:
            while e.type is not MouseEventType.RELEASE:
                yield  # dragging
            self._redraw_layer(e.pos)
            yield
            while e.button is MouseButton.NONE:
                self._redraw_layer(e.pos)
                yield
            if e.type is MouseEventType.DOUBLE_CLICK:
                self._remove_duplicates()
                if self._auto_close:
                    self.close_path(emit=True)
                break
            elif e.button in self._valid_buttons:
                if e.type is MouseEventType.PRESS:
                    self._update_layer(pos_start, e.pos)
                    if self._tracking:
                        self.changed.emit(self.selection)
            elif e.type is MouseEventType.PRESS:
                break
            yield
        yield
        if not self._tracking:
            self.changed.emit(self.selection)
        if not self._persist:
            with self.cleared.blocked():
                self.clear_selection()

    def _remove_duplicates(self):
        xs, ys = self._layer.data
        if xs.size >= 2 and xs[-2] == xs[-1] and ys[-2] == ys[-1]:
            self._layer.data = xs[:-1], ys[:-1]
        return


def _norm_input(
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
):
    return _norm_button(buttons), _norm_modifier(modifiers)


def _norm_button(
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
) -> list[MouseButton]:
    if isinstance(buttons, (str, MouseButton)):
        buttons = [buttons]
    _buttons = [MouseButton(btn) for btn in buttons]
    if MouseButton.NONE in _buttons:
        raise ValueError("MouseButton.NONE is not allowed.")
    return _buttons


def _norm_modifier(
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
) -> list[Modifier]:
    if modifiers is None:
        modifiers = []
    elif isinstance(modifiers, (str, Modifier)):
        modifiers = [modifiers]
    _modifiers = [Modifier(mod) for mod in modifiers]
    return _modifiers


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


def point_selector(
    canvas: CanvasBase,
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
) -> PointSelectionTool:
    """
    Create a point selector tool with given settings.

    A point selector emits a Point object when a point is drawn.
    A selection tool is constructed by specifying the canvas to attach the tool.

    >>> canvas = new_canvas("matplotlib:qt")
    >>> tool = point_selector(canvas)

    Use `buttons` and `modifiers` to specify how to trigger the tool.

    >>> tool = point_selector(canvas, buttons="right", modifiers="ctrl")

    Parameters
    ----------
    canvas : CanvasBase
        The canvas to which the tool is attached.
    buttons : MouseButton or Sequence[MouseButton], default "left"
        The mouse buttons that can trigger the tool.
    modifiers : Modifier or Sequence[Modifier], optional
        The modifier keys that must be pressed to trigger the tool.
    """
    _buttons, _modifiers = _norm_input(buttons, modifiers)
    return PointSelectionTool(canvas, _buttons, _modifiers)


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


def polygon_selector(
    canvas: CanvasBase,
    buttons: _MouseButton | Sequence[_MouseButton] = "left",
    modifiers: _Modifier | Sequence[_Modifier] | None = None,
    *,
    tracking: bool = False,
) -> LassoSelectionTool:
    """
    Create a polygon selector tool with given settings.

    A polygon selector emits a XYData object by freehand drawing.
    A selection tool is constructed by specifying the canvas to attach the tool.

    >>> canvas = new_canvas("matplotlib:qt")
    >>> tool = polygon_selector(canvas)

    Use `buttons` and `modifiers` to specify how to trigger the tool.

    >>> tool = polygon_selector(canvas, buttons="right", modifiers="ctrl")

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
    return PolygonSelectionTool(canvas, _buttons, _modifiers, tracking=tracking)


_S = TypeVar("_S", bound=SelectionToolBase)


class SelectionToolConstructor(ABC, Generic[_S]):
    def __init__(
        self,
        buttons: _MouseButton | list[_MouseButton] = "left",
        modifiers: _Modifier | list[_Modifier] | None = None,
        tracking: bool = False,
    ):
        self._buttons = _norm_button(buttons)
        self._modifiers = _norm_modifier(modifiers)
        self._tracking = tracking

    @property
    def buttons(self) -> list[MouseButton]:
        return list(self._buttons)

    @buttons.setter
    def buttons(self, value: list[MouseButton]):
        self._buttons = _norm_button(value)

    @property
    def modifiers(self) -> list[Modifier]:
        return list(self._modifiers)

    @modifiers.setter
    def modifiers(self, value: list[Modifier]):
        self._modifiers = _norm_modifier(value)

    @property
    def tracking(self) -> bool:
        return self._tracking

    @tracking.setter
    def tracking(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("tracking must be a boolean value.")
        self._tracking = value

    def _prep_kwargs(self) -> dict[str, Any]:
        return {
            "buttons": self._buttons,
            "modifiers": self._modifiers,
            "tracking": self._tracking,
        }

    def install(self, canvas: CanvasBase) -> _S:
        return self._install(canvas, **self._prep_kwargs())

    @abstractmethod
    def _install(self, canvas: CanvasBase, **kwargs) -> _S: ...


class LineSelectorConstructor(SelectionToolConstructor[LineSelectionTool]):
    def _install(self, canvas: CanvasBase, buttons, modifiers, tracking):
        return LineSelectionTool(canvas, buttons, modifiers, tracking)


class PointSelectorConstructor(SelectionToolConstructor[PointSelectionTool]):
    def _prep_kwargs(self) -> dict[str, Any]:
        return {
            "buttons": self._buttons,
            "modifiers": self._modifiers,
        }

    def _install(self, canvas: CanvasBase, buttons, modifiers):
        return PointSelectionTool(canvas, buttons, modifiers)


class RectSelectorConstructor(SelectionToolConstructor[RectSelectionTool]):
    def _install(self, canvas: CanvasBase, buttons, modifiers, tracking):
        return RectSelectionTool(canvas, buttons, modifiers, tracking)


class XSpanSelectorConstructor(SelectionToolConstructor[XSpanSelectionTool]):
    def _install(self, canvas: CanvasBase, buttons, modifiers, tracking):
        return XSpanSelectionTool(canvas, buttons, modifiers, tracking)


class YSpanSelectorConstructor(SelectionToolConstructor[YSpanSelectionTool]):
    def _install(self, canvas: CanvasBase, buttons, modifiers, tracking):
        return YSpanSelectionTool(canvas, buttons, modifiers, tracking)


class LassoSelectorConstructor(SelectionToolConstructor[LassoSelectionTool]):
    def _install(self, canvas: CanvasBase, buttons, modifiers, tracking):
        return LassoSelectionTool(canvas, buttons, modifiers, tracking)


class PolygonSelectorConstructor(SelectionToolConstructor[PolygonSelectionTool]):
    def __init__(
        self,
        buttons: _MouseButton | list[_MouseButton] = "left",
        modifiers: _Modifier | list[_Modifier] | None = None,
        tracking: bool = False,
        auto_close: bool = False,
    ):
        super().__init__(buttons, modifiers, tracking)
        self._auto_close = auto_close

    @property
    def auto_close(self) -> bool:
        return self._auto_close

    @auto_close.setter
    def auto_close(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("auto_close must be a boolean value.")
        self._auto_close = value

    def _prep_kwargs(self) -> dict[str, Any]:
        kwargs = super()._prep_kwargs()
        kwargs["auto_close"] = self._auto_close
        return kwargs

    def _install(self, canvas: CanvasBase, buttons, modifiers, tracking, auto_close):
        return PolygonSelectionTool(canvas, buttons, modifiers, tracking, auto_close)


class SelectionManager:
    def __init__(self, canvas: CanvasBase):
        self._canvas_ref = weakref.ref(canvas)
        self._current_tool: SelectionToolBase | None = None
        self._line_constructor = LineSelectorConstructor()
        self._point_constructor = PointSelectorConstructor()
        self._rect_constructor = RectSelectorConstructor()
        self._xspan_constructor = XSpanSelectorConstructor()
        self._yspan_constructor = YSpanSelectorConstructor()
        self._lasso_constructor = LassoSelectorConstructor()
        self._polygon_constructor = PolygonSelectorConstructor()

    @property
    def line(self) -> LineSelectorConstructor:
        return self._line_constructor

    @property
    def points(self) -> PointSelectorConstructor:
        return self._point_constructor

    @property
    def rect(self) -> RectSelectorConstructor:
        return self._rect_constructor

    @property
    def xspan(self) -> XSpanSelectorConstructor:
        return self._xspan_constructor

    @property
    def yspan(self) -> YSpanSelectorConstructor:
        return self._yspan_constructor

    @property
    def lasso(self) -> LassoSelectorConstructor:
        return self._lasso_constructor

    @property
    def polygon(self) -> PolygonSelectorConstructor:
        return self._polygon_constructor

    @property
    def current_tool(self) -> SelectionToolBase | None:
        return self._current_tool

    @property
    def mode(self) -> SelectionMode:
        if self._current_tool is None:
            return SelectionMode.NONE
        elif isinstance(self._current_tool, LineSelectionTool):
            return SelectionMode.LINE
        elif isinstance(self._current_tool, PointSelectionTool):
            return SelectionMode.POINT
        elif isinstance(self._current_tool, RectSelectionTool):
            return SelectionMode.RECT
        elif isinstance(self._current_tool, XSpanSelectionTool):
            return SelectionMode.XSPAN
        elif isinstance(self._current_tool, YSpanSelectionTool):
            return SelectionMode.YSPAN
        elif isinstance(self._current_tool, LassoSelectionTool):
            return SelectionMode.LASSO
        elif isinstance(self._current_tool, PolygonSelectionTool):
            return SelectionMode.POLYGON
        else:
            raise RuntimeError("Invalid tool mode.")

    @mode.setter
    def mode(self, value: str | SelectionMode):
        value = SelectionMode(value)
        if self._current_tool is not None:
            self._current_tool.disconnect()
            self._current_tool = None
        canvas = self._canvas_ref()
        if canvas is None:
            raise RuntimeError("Canvas is not connected.")
        if value is SelectionMode.NONE:
            self._current_tool = None
        elif value is SelectionMode.LINE:
            self._current_tool = self._line_constructor.install(canvas)
        elif value is SelectionMode.POINT:
            self._current_tool = self._point_constructor.install(canvas)
        elif value is SelectionMode.RECT:
            self._current_tool = self._rect_constructor.install(canvas)
        elif value is SelectionMode.XSPAN:
            self._current_tool = self._xspan_constructor.install(canvas)
        elif value is SelectionMode.YSPAN:
            self._current_tool = self._yspan_constructor.install(canvas)
        elif value is SelectionMode.LASSO:
            self._current_tool = self._lasso_constructor.install(canvas)
        elif value is SelectionMode.POLYGON:
            self._current_tool = self._polygon_constructor.install(canvas)
        else:
            raise ValueError("Invalid mode value.")
