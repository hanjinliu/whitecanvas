from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from cmap import Color
from numpy.typing import NDArray
from psygnal import Signal, SignalGroup

from whitecanvas import protocols
from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas.types import ColorType, LineStyle
from whitecanvas.utils.normalize import arr_color

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas._base import CanvasBase


class AxisSignals(SignalGroup):
    lim = Signal(tuple)


class Namespace:
    _attrs: tuple[str, ...] = ()

    def __init__(self, canvas: CanvasBase | None = None):
        if canvas is not None:
            while isinstance(canvas, Namespace):
                canvas = canvas._canvas_ref()
            self._canvas_ref = weakref.ref(canvas)
        else:
            self._canvas_ref = lambda: None
        self._instances: dict[int, Self] = {}

    def __get__(self, canvas, owner) -> Self:
        if canvas is None:
            return self
        _id = id(canvas)
        if (ns := self._instances.get(_id)) is None:
            ns = self._instances[_id] = type(self)(canvas)
        return ns

    def _get_canvas(self) -> protocols.CanvasProtocol:
        l = self._canvas_ref()
        if l is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        return l._canvas()

    def __repr__(self) -> str:
        cname = type(self).__name__
        try:
            props = [f"canvas={self._get_canvas()!r}"]
            for k in self._attrs:
                v = getattr(self, k)
                props.append(f"{k}={v!r}")
            return f"{cname}({', '.join(props)})"

        except ReferenceDeletedError:
            return f"<{cname} of deleted canvas>"

    def update(self, d: dict[str, Any] = {}, **kwargs):
        values = dict(d, **kwargs)
        invalid_args = set(values) - set(self._attrs)
        if invalid_args:
            raise TypeError(f"Cannot set {invalid_args!r} on {type(self).__name__}")
        for k, v in values.items():
            setattr(self, k, v)


class _TextBoundNamespace(Namespace):
    def _get_object(self) -> protocols.TextLabelProtocol:
        raise NotImplementedError

    @property
    def color(self):
        """Text color"""
        return self._get_object()._plt_get_color()

    @color.setter
    def color(self, color):
        self._get_object()._plt_set_color(np.array(Color(color)))

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
        text = self.text
        color = self.color
        size = self.size
        fontfamily = self.family
        name = type(self).__name__
        return f"{name}({text=!r}, {color=!r}, {size=!r}, {fontfamily=!r})"

    @property
    def text(self) -> str:
        return self._get_object()._plt_get_text()

    @text.setter
    def text(self, text: str):
        self._get_object()._plt_set_text(text)

    def __set__(self, instance, value):
        # allow canvas.x.label = "X" as a shortcut for canvas.x.label.text = "X"
        if isinstance(value, str):
            self.text = value
        elif isinstance(value, dict):
            self.update(value)
        else:
            raise TypeError(f"Cannot set {type(self)} to {value!r}.")


class _TicksNamespace(_TextBoundNamespace):
    def __repr__(self) -> str:
        pos, labels = self._get_object()._plt_get_tick_labels()
        pos = list(pos)
        color = self.color
        size = self.size
        family = self.family
        name = type(self).__name__
        return f"{name}({pos=!r}, {labels=}, {color=!r}, {size=!r}, {family=!r})"

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
        self._get_object()._plt_override_labels((_pos, _labels))

    def reset_labels(self) -> None:
        """Reset the tick labels to the default."""
        self._get_object()._plt_reset_override()

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


class TitleNamespace(_TextLabelNamespace):
    def _get_object(self):
        return self._get_canvas()._plt_get_title()


class XLabelNamespace(_TextLabelNamespace):
    def _get_object(self):
        return self._get_canvas()._plt_get_xlabel()


class YLabelNamespace(_TextLabelNamespace):
    def _get_object(self):
        return self._get_canvas()._plt_get_ylabel()


class _AxisNamespace(Namespace):
    events: AxisSignals

    def __init__(self, canvas: CanvasBase | None = None):
        super().__init__(canvas)
        self.events = AxisSignals()
        self._flipped = False

    def _get_object(self) -> protocols.AxisProtocol:
        raise NotImplementedError

    @property
    def lim(self) -> tuple[float, float]:
        """Limits of the axis."""
        return self._get_object()._plt_get_limits()

    @lim.setter
    def lim(self, lim: tuple[float, float]):
        low, high = lim
        if low >= high:
            a = type(self).__name__[0].lower()
            raise ValueError(
                f"low must be less than high, but got {lim!r}. If you "
                f"want to flip the axis, use `canvas.{a}.flipped = True`."
            )
        # Manually emit signal. This is needed when the plot backend is
        # implemented in JS (such as bokeh) and the python callback is not
        # enabled. Otherwise axis linking fails.
        with self.events.blocked():
            self._get_object()._plt_set_limits(lim)
        self.events.lim.emit(lim)
        return None

    @property
    def color(self):
        """Color of the axis."""
        return self._get_object()._plt_get_color()

    @color.setter
    def color(self, color):
        self._get_object()._plt_set_color(np.array(Color(color)))

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


class XAxisNamespace(_AxisNamespace):
    label = XLabelNamespace()
    ticks = XTickNamespace()

    def _get_object(self):
        return self._get_canvas()._plt_get_xaxis()


class YAxisNamespace(_AxisNamespace):
    label = YLabelNamespace()
    ticks = YTickNamespace()

    def _get_object(self):
        return self._get_canvas()._plt_get_yaxis()
