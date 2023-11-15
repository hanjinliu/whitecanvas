from __future__ import annotations

from typing import TYPE_CHECKING, Any
import weakref

from psygnal import Signal
from cmap import Color
from neoplot import protocols
from neoplot._exceptions import ReferenceDeletedError
from ._signal import GeneratorSignal

if TYPE_CHECKING:
    from typing_extensions import Self
    from neoplot.canvas._base import CanvasBase


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


class _TextLabelNamespace(Namespace):
    def _get_object(self) -> protocols.TextLabelProtocol:
        raise NotImplementedError

    def __repr__(self) -> str:
        text = self.text
        color = self.color
        size = self.size
        fontfamily = self.fontfamily
        name = type(self).__name__
        return f"{name}({text=!r}, {color=!r}, {size=!r}, {fontfamily=!r})"

    @property
    def text(self) -> protocols.TextLabelProtocol:
        return self._get_object()._plt_get_text()

    @text.setter
    def text(self, text: str):
        self._get_object()._plt_set_text(text)

    @property
    def color(self):
        return self._get_object()._plt_get_color()

    @color.setter
    def color(self, color):
        self._get_object()._plt_set_color(Color(color).name)

    @property
    def size(self) -> float:
        return self._get_object()._plt_get_size()

    @size.setter
    def size(self, size: float):
        self._get_object()._plt_set_size(size)

    @property
    def fontfamily(self) -> str:
        return self._get_object()._plt_get_fontfamily()

    @fontfamily.setter
    def fontfamily(self, font):
        self._get_object()._plt_set_fontfamily(font)


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
    lim_changed = Signal(tuple)

    def _get_object(self) -> protocols.AxisProtocol:
        raise NotImplementedError

    @property
    def lim(self) -> tuple[float, float]:
        return self._get_object()._plt_get_limits()

    @lim.setter
    def lim(self, lim: tuple[float, float]):
        return self._get_object()._plt_set_limits(lim)


class XAxisNamespace(_AxisNamespace):
    label = XLabelNamespace()

    def _get_object(self):
        return self._get_canvas()._plt_get_xaxis()


class YAxisNamespace(_AxisNamespace):
    label = YLabelNamespace()

    def _get_object(self):
        return self._get_canvas()._plt_get_yaxis()


class MouseNamespace(Namespace):
    clicked = Signal(object)
    double_clicked = Signal(object)

    def __init__(self, canvas: CanvasBase | None = None):
        super().__init__(canvas)
        self.moved = GeneratorSignal()
