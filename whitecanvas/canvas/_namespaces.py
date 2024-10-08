from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Sequence,
    TypeVar,
)

import numpy as np
from cmap import Color
from numpy.typing import NDArray
from psygnal import Signal, SignalGroup

from whitecanvas import protocols
from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas._signal import MouseMoveSignal, MouseSignal
from whitecanvas.types import (
    ColorType,
    LineStyle,
    Modifier,
    MouseButton,
    MouseEvent,
    MouseEventType,
    Point,
)
from whitecanvas.utils.normalize import arr_color

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas._base import CanvasBase

_no_canvas = object()
_T = TypeVar("_T")


class AxisSignals(SignalGroup):
    """Signals emitted by an axis."""

    lim = Signal(tuple)


class _StrongRef(Generic[_T]):
    """Strong reference to an object."""

    def __init__(self, obj: _T):
        self._obj = obj

    def __call__(self) -> _T:
        return self._obj


class Namespace:
    _attrs: tuple[str, ...] = ()

    def __init__(self, canvas: CanvasBase | None = None):
        if canvas is not None:
            # This line *should* be an weak reference, but canvas is sometimes deleted
            # for some reason. Just use a strong reference for now.
            self._canvas_ref = _StrongRef(canvas)
        else:
            self._canvas_ref = _StrongRef(_no_canvas)
        self._instances: dict[int, Self] = {}

    def __get__(self, canvas, owner=None) -> Self:
        if canvas is None:
            return self
        while isinstance(canvas, Namespace):
            canvas = canvas._canvas_ref()
        id_ = id(canvas)
        if (ns := self._instances.get(id_)) is None:
            ns = self._instances[id_] = type(self)(canvas)
        return ns

    def _get_canvas(self) -> protocols.CanvasProtocol:
        l = self._canvas_ref()
        if l is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        elif l is _no_canvas:
            raise TypeError("No canvas is associated with the class itself.")
        return l._canvas()

    def _draw_canvas(self):
        if canvas := self._canvas_ref():
            canvas._draw_canvas()

    def __repr__(self) -> str:
        cname = type(self).__name__
        l = self._canvas_ref()
        if l is None:
            return f"<{cname} of deleted canvas>"
        elif l is _no_canvas:
            return f"<{cname}>"
        props = [f"canvas={l!r}"]
        for k in self._attrs:
            v = getattr(self, k)
            props.append(f"{k}={v!r}")
        return f"{cname}({', '.join(props)})"

    def update(self, d: dict[str, Any] = {}, **kwargs):
        values = dict(d, **kwargs)
        invalid_args = set(values) - set(self._attrs)
        if invalid_args:
            raise TypeError(f"Cannot set {invalid_args!r} on {type(self).__name__}")
        for k, v in values.items():
            if isinstance(ns := getattr(self, k), Namespace):
                ns.update(v)
            else:
                setattr(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the namespace."""
        out = {}
        for k in self._attrs:
            val = getattr(self, k)
            if isinstance(val, Namespace):
                out[k] = val.to_dict()
            else:
                out[k] = val
        return out


class _TextBoundNamespace(Namespace):
    _attrs = ("color", "size", "family", "visible")

    def _get_object(self) -> protocols.TextLabelProtocol:
        raise NotImplementedError

    @property
    def color(self):
        """Text color"""
        return self._get_object()._plt_get_color()

    @color.setter
    def color(self, color):
        self._get_object()._plt_set_color(np.fromiter(Color(color), dtype=np.float32))

    @property
    def size(self) -> float:
        """Text font size"""
        return self._get_object()._plt_get_size()

    @size.setter
    def size(self, size: float):
        self._get_object()._plt_set_size(size)

    @property
    def family(self) -> str:
        """Text font family."""
        return self._get_object()._plt_get_fontfamily()

    @family.setter
    def family(self, font):
        self._get_object()._plt_set_fontfamily(font)

    @property
    def visible(self) -> bool:
        """Text visibility."""
        return self._get_object()._plt_get_visible()

    @visible.setter
    def visible(self, visible: bool):
        self._get_object()._plt_set_visible(visible)


class _TextLabelNamespace(_TextBoundNamespace):
    def __repr__(self) -> str:
        cname = type(self).__name__
        l = self._canvas_ref()
        if l is None:
            return f"<{cname} of deleted canvas>"
        elif l is _no_canvas:
            return f"<{cname}>"
        text = self.text
        color = self.color
        size = self.size
        family = self.family
        return f"{cname}({text=!r}, {color=!r}, {size=!r}, {family=!r})"

    @property
    def text(self) -> str:
        """Text content."""
        return self._get_object()._plt_get_text()

    @text.setter
    def text(self, text: str):
        self._get_object()._plt_set_text(text)

    def __set__(self, instance, value):
        # allow canvas.x.label = "X" as a shortcut for canvas.x.label.text = "X"
        obj = self.__get__(instance)
        if isinstance(value, str):
            obj.text = value
        elif isinstance(value, dict):
            obj.update(value)
        else:
            raise TypeError(f"Cannot set {value!r} to {type(obj)}.")


class _TicksNamespace(_TextBoundNamespace):
    def __repr__(self) -> str:
        cname = type(self).__name__
        l = self._canvas_ref()
        if l is None:
            return f"<{cname} of deleted canvas>"
        elif l is _no_canvas:
            return f"<{cname}>"
        pos, labels = self._get_object()._plt_get_tick_labels()
        pos = list(pos)
        color = self.color
        size = self.size
        family = self.family
        return f"{cname}({pos=!r}, {labels=}, {color=!r}, {size=!r}, {family=!r})"

    def _get_object(self) -> protocols.TicksProtocol:
        raise NotImplementedError

    @property
    def pos(self) -> NDArray[np.floating]:
        pos, _ = self._get_object()._plt_get_tick_labels()
        return np.asarray(pos)

    @property
    def labels(self) -> list[str]:
        _, labels = self._get_object()._plt_get_tick_labels()
        return labels

    def set_labels(self, pos: Iterable[float], labels: Iterable[str] | None = None):
        """
        Override tick labels.

        >>> canvas.x.ticks.set_labels([0, 1, 2], ["a", "b", "c"])

        Parameters
        ----------
        pos : iterable of float
            The positions of the ticks.
        labels : iterable of str, optional
            The label strings of the ticks. If None, the labels are set to the
            `pos` values.
        """
        _pos = list(pos)
        # test sorted
        if len(_pos) > 0 and np.any(np.diff(_pos) <= 0):
            raise ValueError(f"pos must be strictly increasing, got {pos}.")
        if labels is not None:
            _labels = [str(l) for l in labels]
        else:
            ndigits = int(np.log10(_pos[-1] - _pos[0])) + 1
            _labels = [str(round(p, ndigits)) for p in _pos]
        if len(_pos) != len(_labels):
            raise ValueError("pos and labels must have the same length.")
        self._get_object()._plt_override_labels(_pos, _labels)
        self._get_canvas()._plt_draw()

    def reset_labels(self) -> None:
        """Reset the tick labels to the default."""
        self._get_object()._plt_reset_override()
        self._get_canvas()._plt_draw()

    @property
    def rotation(self) -> float:
        """Tick label rotation in degrees."""
        return self._get_object()._plt_get_text_rotation()

    @rotation.setter
    def rotation(self, rotation: float):
        self._get_object()._plt_set_text_rotation(rotation)


class XTickNamespace(_TicksNamespace):
    def _get_object(self):
        return self._get_canvas()._plt_get_xticks()


class YTickNamespace(_TicksNamespace):
    def _get_object(self):
        return self._get_canvas()._plt_get_yticks()


class ZTickNamespace(_TicksNamespace):
    def _get_object(self):
        return self._get_canvas()._plt_get_zticks()


class TitleNamespace(_TextLabelNamespace):
    def _get_object(self):
        return self._get_canvas()._plt_get_title()


class XLabelNamespace(_TextLabelNamespace):
    def _get_object(self):
        return self._get_canvas()._plt_get_xlabel()


class YLabelNamespace(_TextLabelNamespace):
    def _get_object(self):
        return self._get_canvas()._plt_get_ylabel()


class ZLabelNamespace(_TextLabelNamespace):
    def _get_object(self):
        return self._get_canvas()._plt_get_zlabel()


class AxisNamespace(Namespace):
    events: AxisSignals
    _attrs = ("lim", "color", "flipped")

    def __init__(self, canvas: CanvasBase | None = None):
        super().__init__(canvas)
        self.events = AxisSignals()
        self._flipped = False
        self._lim_updated_by_user = False

    def _get_object(self) -> protocols.AxisProtocol:
        raise NotImplementedError

    @property
    def lim(self) -> tuple[float, float]:
        """Limits of the axis."""
        return self._get_object()._plt_get_limits()

    @lim.setter
    def lim(self, lim: tuple[float, float]):
        low, high = lim
        if low is None or high is None:
            _low, _high = self._get_object()._plt_get_limits()
            low = low if low is not None else _low
            high = high if high is not None else _high
        elif low >= high:
            a = type(self).__name__[0].lower()
            raise ValueError(
                f"low must be less than high, but got {lim!r}. If you "
                f"want to flip the axis, use `canvas.{a}.flipped = True`."
            )
        self._unsafe_set_lim(low, high)
        self._draw_canvas()
        return None

    def _unsafe_set_lim(self, low: float, high: float):
        # Manually emit signal. This is needed when the plot backend is
        # implemented in JS (such as bokeh) and the python callback is not
        # enabled. Otherwise axis linking fails.
        with self.events.blocked():
            self._get_object()._plt_set_limits((low, high))
        self.events.lim.emit((low, high))
        self._lim_updated_by_user = True
        return None

    @property
    def color(self):
        """Color of the axis."""
        return self._get_object()._plt_get_color()

    @color.setter
    def color(self, color):
        self._get_object()._plt_set_color(np.fromiter(Color(color), dtype=np.float32))
        self._draw_canvas()

    @property
    def flipped(self) -> bool:
        """Return true if the axis is flipped."""
        return self._flipped

    @flipped.setter
    def flipped(self, flipped: bool):
        """Set the axis to be flipped."""
        if flipped != self._flipped:
            self._get_object()._plt_flip()
            self._flipped = flipped

    def set_gridlines(
        self,
        visible: bool = True,
        color: ColorType = "gray",
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
    ):
        color = arr_color(color)
        style = LineStyle(style)
        if width < 0:
            raise ValueError("width must be non-negative.")
        self._get_object()._plt_set_grid_state(visible, color, width, style)


class XAxisNamespace(AxisNamespace):
    label = XLabelNamespace()
    ticks = XTickNamespace()

    def _get_object(self):
        return self._get_canvas()._plt_get_xaxis()


class YAxisNamespace(AxisNamespace):
    label = YLabelNamespace()
    ticks = YTickNamespace()

    def _get_object(self):
        return self._get_canvas()._plt_get_yaxis()


class ZAxisNamespace(AxisNamespace):
    label = ZLabelNamespace()
    ticks = ZTickNamespace()

    def _get_object(self):
        return self._get_canvas()._plt_get_zaxis()


class MouseNamespace(Namespace):
    """Namespace that contains the mouse events."""

    clicked = MouseSignal(object)
    """Signal emitted when a mouse button is clicked."""

    moved = MouseMoveSignal()
    """Signal emitted when the mouse is moved."""

    double_clicked = MouseSignal(object)
    """Signal emitted when a mouse button is double-clicked."""

    @property
    def enabled(self) -> bool:
        """Return whether pan/zoom is enabled."""
        return self._get_canvas()._plt_get_mouse_enabled()

    @enabled.setter
    def enabled(self, enabled: bool):
        self._get_canvas()._plt_set_mouse_enabled(enabled)

    def emulate_click(
        self,
        position: tuple[float, float],
        *,
        button: str | MouseButton = MouseButton.LEFT,
        modifiers: str | Modifier | Sequence[str | Modifier] = (),
    ) -> None:
        """Emulate a mouse press event."""
        ev = MouseEvent(
            MouseButton(button),
            _norm_modifiers(modifiers),
            Point(*position),
            MouseEventType.PRESS,
        )
        self.clicked.emit(ev)
        return None

    def emulate_double_click(
        self,
        position: tuple[float, float],
        *,
        button: str | MouseButton = MouseButton.LEFT,
        modifiers: str | Modifier | Sequence[str | Modifier] = (),
    ) -> None:
        """Emulate a mouse double-click event."""
        ev = MouseEvent(
            MouseButton(button),
            _norm_modifiers(modifiers),
            Point(*position),
            MouseEventType.DOUBLE_CLICK,
        )
        self.double_clicked.emit(ev)

    def emulate_hover(
        self,
        positions: Sequence[tuple[float, float]],
        *,
        modifiers: str | Modifier | Sequence[str | Modifier] = (),
    ) -> None:
        """Emulate a mouse move event."""
        _modifiers = _norm_modifiers(modifiers)

        for pos in positions:
            ev = MouseEvent(
                MouseButton.NONE,
                _modifiers,
                Point(*pos),
                MouseEventType.MOVE,
            )
            self.moved.emit(ev)
        return None

    def emulate_drag(
        self,
        positions: Sequence[tuple[float, float]],
        *,
        button: str | MouseButton = MouseButton.LEFT,
        modifiers: str | Modifier | Sequence[str | Modifier] = (),
    ):
        """Emulate a mouse press-move-release event."""
        _modifiers = _norm_modifiers(modifiers)

        ev = MouseEvent(
            MouseButton(button),
            _modifiers,
            Point(*positions[0]),
            MouseEventType.PRESS,
        )
        self.moved.emit(ev)

        for pos in positions[1:]:
            ev = MouseEvent(
                MouseButton(button),
                _modifiers,
                Point(*pos),
                MouseEventType.MOVE,
            )
            self.moved.emit(ev)

        ev = MouseEvent(
            MouseButton(button),
            _modifiers,
            Point(*positions[-1]),
            MouseEventType.RELEASE,
        )
        self.moved.emit(ev)
        return None


def _norm_modifiers(modifiers) -> tuple[Modifier, ...]:
    if isinstance(modifiers, str):
        _modifiers = (Modifier(modifiers),)
    elif isinstance(modifiers, Modifier):
        _modifiers = [modifiers]
    else:
        _modifiers = tuple(Modifier(m) for m in modifiers)
    return _modifiers
