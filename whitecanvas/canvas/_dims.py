from __future__ import annotations
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Any, TypeVar, overload
import weakref
import numpy as np
from numpy.typing import NDArray

from psygnal import Signal, SignalGroup
from whitecanvas.types import (
    LineStyle,
    FacePattern,
    ColorType,
    Orientation,
    ColormapType,
)
from whitecanvas import layers as _l
from whitecanvas.layers import _ndim as _ndl
from whitecanvas._exceptions import ReferenceDeletedError

if TYPE_CHECKING:
    from typing_extensions import Self
    from whitecanvas.canvas._base import CanvasBase

_T = TypeVar("_T")


class DimsEvents(SignalGroup):
    """Event signals for :class:`Dims`."""

    indices = Signal(dict)


class DimAxis(ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """Name of the axis."""
        return self._name

    @abstractmethod
    def value(self) -> int:
        """Value of the axis."""

    @abstractmethod
    def set_value(self, value: Any):
        """Set the value of the axis."""


class RangeAxis(DimAxis):
    def __init__(self, name: str, size: int):
        super().__init__(name)
        self._size = size
        self._value = 0

    def value(self) -> int:
        return self._value

    def set_value(self, value: Any):
        v = int(value)
        if not 0 <= v < self._size:
            raise ValueError(
                f"Size of axis {self!r} is {self._size} but got index {value!r}."
            )
        self._value = value

    def set_size(self, size: int):
        self._size = size
        self._value = min(self._value, size - 1)


class CategoricalAxis(DimAxis):
    def __init__(self, name: str, categories: list[str]):
        super().__init__(name)
        self._mapper = {c: i for i, c in enumerate(categories)}
        self._value = categories[0]

    def value(self) -> int:
        return self._mapper[self._value]

    def set_value(self, value: Any):
        if value not in self._mapper:
            raise ValueError(
                f"Value must be one of {list(self._mapper)} but got {value!r}."
            )
        self._value = value


_default = object()


class Dims:
    """
    Multi-dimensional plotting interface.
    """

    def __init__(self, canvas: CanvasBase | None = None):
        self._instances: dict[int, Self] = {}
        self._axes: list[DimAxis] = []
        self.events = DimsEvents()
        if canvas is not None:
            self._canvas_ref = weakref.ref(canvas)
            self.events.indices.connect(canvas._draw_canvas, unique=True)
        else:
            self._canvas_ref = lambda: None

    def __get__(self, canvas, owner) -> Self:
        if canvas is None:
            return self
        _id = id(canvas)
        if (ns := self._instances.get(_id)) is None:
            ns = self._instances[_id] = type(self)(canvas)
        return ns

    def _get_canvas(self) -> CanvasBase:
        l = self._canvas_ref()
        if l is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        return l

    def _default_axes_names(self, ndim: int) -> list[str]:
        names = self.names
        n_new = ndim - len(names)
        if n_new <= 0:
            return names[-ndim:]
        else:
            new = [f"axis_{i}" for i in range(ndim, ndim + n_new)]
            return new[::-1] + names

    @property
    def names(self) -> list[str]:
        """Names of the dimensions."""
        return [a.name for a in self._axes]

    @overload
    def axis(self, name: str) -> DimAxis:
        ...

    @overload
    def axis(self, name: str, *, default: _T) -> DimAxis | _T:
        ...

    def axis(self, name, *, default=_default):
        """Get the axis by name."""
        for a in self._axes:
            if a.name == name:
                return a
        if default is _default:
            raise ValueError(f"Axis {name!r} not found.")
        return default

    @property
    def indices(self) -> dict[str, int]:
        """Current indices for each dimension."""
        return {a.name: a.value() for a in self._axes}

    def set_indices(self, **kwargs: Any):
        for k, v in kwargs.items():
            self.axis(k).set_value(v)
        self.events.indices.emit(self.indices)

    def in_axes(self, *names: str) -> InAxes:
        n_names = len(names)
        if n_names == 0:
            raise ValueError("At least one axis name must be provided.")
        if n_names == 1 and isinstance(names[0], (list, tuple)):
            names = names[0]
        existing_names = self.names
        new_axes: list[str] = []
        for name in names:
            if not isinstance(name, str):
                raise TypeError(f"Axis name must be str but got {name!r}.")
            if name not in existing_names:
                new_axes.append(RangeAxis(name, 0))
        self._axes.extend(new_axes)
        return InAxes(self, names)

    def add_line(
        self,
        xdata: Any,
        ydata: Any,
        name=None,
        color=None,
        width=1.0,
        style=LineStyle.SOLID,
        alpha=1.0,
        antialias=True,
    ):
        return InAxes(self).add_line(
            xdata,
            ydata,
            name=name,
            color=color,
            width=width,
            style=style,
            alpha=alpha,
            antialias=antialias,
        )


class InAxes:
    def __init__(self, dims: Dims, axis_names: list[str] | None = None):
        self._dims = dims
        self._axis_names = axis_names

    def _get_canvas(self) -> CanvasBase:
        return self._dims._get_canvas()

    def add_line(
        self,
        xdata: Any,
        ydata: Any,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        alpha: float = 1.0,
        antialias: bool = True,
    ):
        canvas = self._get_canvas()
        name = canvas._coerce_name(_l.Line, name)
        color = canvas._generate_colors(color)
        stack = _ndl.XYLayerStack.from_layer_class(
            _l.Line,
            xdata,
            ydata,
            name=name,
            color=color,
            width=width,
            style=style,
            alpha=alpha,
            antialias=antialias,
            backend=canvas._get_backend(),
        )
        return self._add_layer(stack)

    def add_band(
        self,
        xdata: Any,
        ylow: Any,
        yhigh: Any,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        alpha: float = 1.0,
        pattern: str | FacePattern = FacePattern.SOLID,
    ) -> _ndl.XYYLayerStack:
        canvas = self._get_canvas()
        name = canvas._coerce_name(_l.Band, name)
        color = canvas._generate_colors(color)
        stack = _ndl.XYYLayerStack.from_layer_class(
            _l.Band,
            xdata,
            ylow,
            yhigh,
            name=name,
            color=color,
            pattern=pattern,
            alpha=alpha,
            backend=canvas._get_backend(),
        )
        return self._add_layer(stack)

    def add_errorbars(
        self,
        xdata: Any,
        ylow: Any,
        yhigh: Any,
        *,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        color: ColorType = "blue",
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = False,
        capsize: float = 0.0,
    ) -> _ndl.XYYLayerStack:
        canvas = self._get_canvas()
        name = canvas._coerce_name(_l.Errorbars, name)
        color = canvas._generate_colors(color)
        stack = _ndl.XYYLayerStack.from_layer_class(
            _l.Errorbars,
            xdata,
            ylow,
            yhigh,
            name=name,
            orient=orient,
            color=color,
            width=width,
            style=style,
            antialias=antialias,
            capsize=capsize,
            backend=canvas._get_backend(),
        )
        return self._add_layer(stack)

    def add_rug(
        self,
        events: Any,
        *,
        low: float = 0.0,
        high: float = 1.0,
        name: str | None = None,
        orient: str | Orientation = Orientation.VERTICAL,
        color: ColorType = "black",
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        alpha: float = 1.0,
    ) -> _ndl.YLayerStack:
        canvas = self._get_canvas()
        name = canvas._coerce_name(_l.Rug, name)
        color = canvas._generate_colors(color)
        stack = _ndl.YLayerStack.from_layer_class(
            _l.Rug,
            events,
            low=low,
            high=high,
            name=name,
            orient=orient,
            color=color,
            width=width,
            style=style,
            antialias=antialias,
            backend=canvas._get_backend(),
        )
        return self._add_layer(stack)

    def add_image(
        self,
        image: Any,
        *,
        name: str | None = None,
        cmap: ColormapType = "gray",
        clim: tuple[float | None, float | None] | None = None,
        rgb: bool = False,
        flip_canvas: bool = True,
        lock_aspect: bool = True,
    ) -> _ndl.ImageLayerStack:
        canvas = self._get_canvas()
        name = canvas._coerce_name(_l.Image, name)
        stack = _ndl.ImageLayerStack.from_layer_class(
            _l.Image,
            image,
            name=name,
            cmap=cmap,
            clim=clim,
            rgb=rgb,
            backend=canvas._get_backend(),
        )
        if self._axis_names is not None:
            stack._with_axis_names(self._axis_names)
        self._add_layer(stack)

        if flip_canvas and not canvas.y.flipped:
            canvas.y.flipped = True
        if lock_aspect:
            canvas.aspect_ratio = 1.0
        return stack

    def _add_layer(self, stack: _l.LayerStack):
        canvas = self._get_canvas()
        if self._axis_names is not None:
            stack._with_axis_names(self._axis_names)
        new_axes = []
        for a, s in zip(stack.axis_names, stack._data_stack.shape):
            axis = self._dims.axis(a, default=None)
            if axis is None:
                new_axes.append(RangeAxis(a, s))
            elif axis.value() < s:
                if isinstance(axis, RangeAxis):
                    axis.set_size(s)
                else:
                    raise NotImplementedError
        self._dims._axes.extend(new_axes)
        return canvas.add_layer(stack)
