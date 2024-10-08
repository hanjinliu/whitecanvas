from __future__ import annotations

import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Mapping,
    Sequence,
    SupportsIndex,
    TypeVar,
    overload,
)

from psygnal import Signal, SignalGroup

from whitecanvas import layers as _l
from whitecanvas import theme
from whitecanvas._axis import DimAxis, RangeAxis
from whitecanvas._exceptions import ReferenceDeletedError
from whitecanvas.layers import _ndim as _ndl
from whitecanvas.types import (
    Alignment,
    ColormapType,
    ColorType,
    Hatch,
    LineStyle,
    OrientationLike,
    Symbol,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas._base import CanvasBase

_T = TypeVar("_T")


class DimsEvents(SignalGroup):
    """Event signals for `Dims`."""

    indices = Signal(dict)
    axis_names = Signal(list)


_default = object()


class Dims:
    """
    Multi-dimensional plotting interface.
    """

    def __init__(self, canvas: CanvasBase | None = None):
        self._instances: dict[int, Self] = {}
        self._axes: list[DimAxis] = []
        self._dim_indices = DimIndices(self)
        self._dim_values = DimValues(self)
        self.events = DimsEvents()
        if canvas is not None:
            self._canvas_ref = weakref.ref(canvas)
            self.events.indices.connect(canvas._draw_canvas, unique=True, max_args=0)
        else:
            self._canvas_ref = lambda: None

    def __repr__(self) -> str:
        return f"Dims(values={self.values!r}, axes={self._axes!r})"

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

    @property
    def names(self) -> list[str]:
        """Names of the dimensions."""
        return [a.name for a in self._axes]

    @overload
    def axis(self, name: str) -> DimAxis: ...

    @overload
    def axis(self, name: str, *, default: _T) -> DimAxis | _T: ...

    def axis(self, name, *, default=_default):
        """Get the axis by name."""
        for a in self._axes:
            if a.name == name:
                return a
        if default is _default:
            raise ValueError(f"Axis {name!r} not found.")
        return default

    @property
    def axes(self) -> list[DimAxis]:
        return self._axes.copy()

    @property
    def indices(self) -> DimIndices:
        """Current indices for each dimension."""
        return self._dim_indices

    @property
    def values(self) -> DimValues:
        """Current values of each dimension."""
        return self._dim_values

    @overload
    def set_indices(self, arg: dict[str, SupportsIndex]): ...

    @overload
    def set_indices(self, arg: SupportsIndex | Sequence[SupportsIndex]): ...

    @overload
    def set_indices(self, **kwargs: SupportsIndex): ...

    def set_indices(self, arg=None, **kwargs: Any):
        """
        Set current indices and update the canvas.

        Either of the following forms are allowed:
        >>> canvas.dims.set_indices(axis_0=3, axis_1=4)
        >>> canvas.dims.set_indices({"axis_0": 3, "axis_1": 4})
        >>> canvas.dims.set_indices([3, 4])
        >>> canvas.dims.set_indices(1)
        """
        kwargs = self._norm_kwargs(arg, kwargs)
        for k, v in kwargs.items():
            self.axis(k).set_index(v)
        self.events.indices.emit(self.indices)

    @overload
    def set_values(self, arg: dict[str, Any]): ...

    @overload
    def set_values(self, arg: Any | Sequence[Any]): ...

    @overload
    def set_values(self, **kwargs: Any): ...

    def set_values(self, arg=None, **kwargs) -> None:
        kwargs = self._norm_kwargs(arg, kwargs)
        for k, v in kwargs.items():
            self.axis(k).set_value(v)
        self.events.indices.emit(self.indices)

    def in_axes(self, *names: str) -> InAxes:
        """
        Add multi-dimensional layers in the given axis names.

        >>> canvas.dims.in_axes("time").add_line(
        ...     x, [np.sin(x + x0) for x0 in [0, 1, 2]]
        >>> )
        """
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

    def create_widget(self):
        from whitecanvas.backend._window import make_dim_slider

        canvas = self._get_canvas()
        sl = make_dim_slider(canvas, canvas._get_backend().app)
        return sl

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
        return InAxes(self).add_line(
            xdata, ydata, name=name, color=color, width=width,
            style=style, alpha=alpha, antialias=antialias,
        )  # fmt: skip

    def add_markers(
        self,
        xdata: Any,
        ydata: Any,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        hatch: str | Hatch = Hatch.SOLID,
        symbol: str | Symbol = Symbol.CIRCLE,
        size: float = 5.0,
    ):
        return InAxes(self).add_markers(
            xdata, ydata, name=name, color=color, hatch=hatch,
            symbol=symbol, size=size,
        )  # fmt: skip

    def add_band(
        self,
        xdata: Any,
        ylow: Any,
        yhigh: Any,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
    ):
        return InAxes(self).add_band(
            xdata,
            ylow,
            yhigh,
            name=name,
            color=color,
            alpha=alpha,
            hatch=hatch,
        )

    def add_errorbars(
        self,
        xdata: Any,
        ylow: Any,
        yhigh: Any,
        *,
        name: str | None = None,
        orient: OrientationLike = "vertical",
        color: ColorType = "blue",
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = False,
        capsize: float = 0.0,
    ):
        return InAxes(self).add_errorbars(
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
        )

    def add_rug(
        self,
        events: Any,
        *,
        low: float = 0.0,
        high: float = 1.0,
        name: str | None = None,
        orient: OrientationLike = "vertical",
        color: ColorType = "black",
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        alpha: float = 1.0,
    ):
        return InAxes(self).add_rug(
            events, low=low, high=high, name=name, orient=orient, color=color,
            width=width, style=style, antialias=antialias, alpha=alpha
        )  # fmt: skip

    def add_text(
        self,
        xdata: Any,
        ydata: Any,
        string: Any,
        *,
        name: str | None = None,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        family: str | None = None,
    ):
        return InAxes(self).add_text(
            xdata, ydata, string, name=name, color=color, size=size,
            rotation=rotation, anchor=anchor, family=family,
        )  # fmt: skip

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
    ):
        return InAxes(self).add_image(
            image, name=name, cmap=cmap, clim=clim, rgb=rgb,
            flip_canvas=flip_canvas, lock_aspect=lock_aspect,
        )  # fmt: skip

    def _norm_kwargs(self, arg, kwargs) -> dict[str, Any]:
        if isinstance(arg, dict):
            if kwargs:
                raise TypeError("Cannot specify both positional and keyword arguments.")
            kwargs = arg
        elif isinstance(arg, (list, tuple)):
            names = self.names
            if len(names) < len(arg):
                raise ValueError(
                    f"Number of indices ({len(arg)}) exceeds number of dimensions "
                    f"({len(names)})."
                )
            if kwargs:
                raise TypeError("Cannot specify both positional and keyword arguments.")
            kwargs = dict(zip(names, arg))
        elif arg is None:
            pass
        elif hasattr(arg, "__index__"):
            if kwargs:
                raise TypeError("Cannot specify both positional and keyword arguments.")
            kwargs = {self.names[0]: arg}
        else:
            raise TypeError(
                f"Argument must be dict, list, tuple, or None but got {arg!r}."
            )
        return kwargs


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
        """
        Add multi-dimensional line layer.

        Parameters
        ----------
        xdata : Any
            X data of multi-dimensional structure.
        ydata : Any
            Y data of multi-dimensional structure.

        Returns
        -------
        XYLayerStack
            Layer stack of a line layer.
        """
        canvas = self._get_canvas()
        name = canvas._coerce_name(name)
        color = canvas._generate_colors(color)
        stack = _ndl.XYLayerStack.from_layer_class(
            _l.Line, xdata, ydata, name=name, color=color, width=width, style=style,
            alpha=alpha, antialias=antialias, backend=canvas._get_backend(),
        )  # fmt: skip
        return self._add_layer(stack)

    def add_markers(
        self,
        xdata: Any,
        ydata: Any,
        *,
        name: str | None = None,
        color: ColorType | None = None,
        hatch: str | Hatch = Hatch.SOLID,
        symbol: str | Symbol = Symbol.CIRCLE,
        size: float = 5.0,
    ) -> _ndl.XYLayerStack:
        canvas = self._get_canvas()
        name = canvas._coerce_name(name)
        color = canvas._generate_colors(color)
        stack = _ndl.XYLayerStack.from_layer_class(
            _l.Markers, xdata, ydata, name=name, color=color, hatch=hatch,
            symbol=symbol, size=size, backend=canvas._get_backend(),
        )  # fmt: skip
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
        hatch: str | Hatch = Hatch.SOLID,
    ) -> _ndl.XYYLayerStack:
        """
        Add multi-dimensional band layer.

        Parameters
        ----------
        xdata : Any
            X data of multi-dimensional structure.
        ylow : Any
            Lower y data of multi-dimensional structure.
        yhigh : Any
            Higher y data of multi-dimensional structure.

        Returns
        -------
        XYYLayerStack
            Layer stack of a band layer.
        """
        canvas = self._get_canvas()
        name = canvas._coerce_name(name)
        color = canvas._generate_colors(color)
        stack = _ndl.XYYLayerStack.from_layer_class(
            _l.Band, xdata, ylow, yhigh, name=name, color=color, hatch=hatch,
            alpha=alpha, backend=canvas._get_backend(),
        )  # fmt: skip
        return self._add_layer(stack)

    def add_errorbars(
        self,
        xdata: Any,
        ylow: Any,
        yhigh: Any,
        *,
        name: str | None = None,
        orient: OrientationLike = "vertical",
        color: ColorType = "blue",
        width: float = 1,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = False,
        capsize: float = 0.0,
    ) -> _ndl.XYYLayerStack:
        """
        Add multi-dimensional errorbars layer.

        Parameters
        ----------
        xdata : Any
            X data of multi-dimensional structure.
        ylow : Any
            Lower y data of multi-dimensional structure.
        yhigh : Any
            Higher y data of multi-dimensional structure.

        Returns
        -------
        XYYLayerStack
            Layer stack of an errorbars layer.
        """
        canvas = self._get_canvas()
        name = canvas._coerce_name(name)
        color = canvas._generate_colors(color)
        stack = _ndl.XYYLayerStack.from_layer_class(
            _l.Errorbars, xdata, ylow, yhigh, name=name, orient=orient, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
            backend=canvas._get_backend(),
        )  # fmt: skip
        return self._add_layer(stack)

    def add_rug(
        self,
        events: Any,
        *,
        low: float = 0.0,
        high: float = 1.0,
        name: str | None = None,
        orient: OrientationLike = "vertical",
        color: ColorType = "black",
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        alpha: float = 1.0,
    ) -> _ndl.YLayerStack:
        """
        Add multi-dimensional rug layer.

        Parameters
        ----------
        events : Any
            Multi-dimensional event data.

        Returns
        -------
        YLayerStack
            Layer stack of a rug layer.
        """
        canvas = self._get_canvas()
        name = canvas._coerce_name(name)
        color = canvas._generate_colors(color)
        stack = _ndl.YLayerStack.from_layer_class(
            _l.Rug, events, low=low, high=high, name=name, orient=orient,
            color=color,  width=width,  style=style, antialias=antialias,
            alpha=alpha, backend=canvas._get_backend(),
        )  # fmt: skip
        return self._add_layer(stack)

    def add_text(
        self,
        xdata: Any,
        ydata: Any,
        string: Any,
        *,
        name: str | None = None,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        family: str | None = None,
    ) -> _ndl.TextLayerStack:
        """
        Add multi-dimensional text layer.

        Parameters
        ----------
        xdata : Any
            _description_
        ydata : Any
            _description_
        string : Any
            _description_

        Returns
        -------
        TextLayerStack
            Layer stack of text layers.
        """
        canvas = self._get_canvas()
        name = canvas._coerce_name(name)
        stack = _ndl.TextLayerStack.from_layer_class(
            _l.Texts, xdata, ydata, string, name=name, color=color, size=size,
            rotation=rotation, anchor=anchor, family=family,
            backend=canvas._get_backend(),
        )  # fmt: skip
        return self._add_layer(stack)

    def add_image(
        self,
        image: Any,
        *,
        name: str | None = None,
        cmap: ColormapType | None = None,
        clim: tuple[float | None, float | None] | None = None,
        rgb: bool = False,
        flip_canvas: bool = True,
        lock_aspect: bool = True,
    ) -> _ndl.ImageLayerStack:
        """
        Add multi-dimensional image layer.

        Parameters
        ----------
        image : Any
            Image stack of multi-dimensional structure.
        rgb : bool, default False
            If input image is an RGB(A) image of shape (..., 3) or (..., 4),
            then set this to True to display the image as RGB(A).

        Returns
        -------
        ImageLayerStack
            Layer stack of image layers.
        """
        canvas = self._get_canvas()
        name = canvas._coerce_name(name)
        cmap = theme._default("colormap_image", cmap)
        stack = _ndl.ImageLayerStack.from_layer_class(
            _l.Image, image, name=name, cmap=cmap, clim=clim, rgb=rgb,
            backend=canvas._get_backend(),
        )  # fmt: skip
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
            elif axis.current_index() < s:
                if isinstance(axis, RangeAxis):
                    axis.set_size(s)
                else:
                    raise NotImplementedError
        self._dims._axes.extend(new_axes)
        if len(new_axes) > 0:
            self._dims.events.axis_names.emit(self._dims.names)
        return canvas.add_layer(stack)


for meth in dir(InAxes):
    if meth.startswith("add_"):
        doc = getattr(InAxes, meth).__doc__
        getattr(Dims, meth).__doc__ = doc


class _DimInterface(Mapping[str, _T]):
    def __init__(self, dims: Dims):
        self._dims = weakref.ref(dims)

    @property
    def dims(self) -> Dims:
        dims = self._dims()
        if dims is None:
            raise ReferenceDeletedError(f"Dims of {self!r} is deleted.")
        return dims

    def __iter__(self) -> Iterator[str]:
        return iter(axis.name for axis in self.dims.axes)

    def __len__(self) -> int:
        return len(self.dims.axes)

    def __repr__(self) -> str:
        content = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"{type(self).__name__}({content})"


class DimIndices(_DimInterface[SupportsIndex]):
    def __getitem__(self, key: str) -> SupportsIndex:
        for axis in self.dims.axes:
            if axis.name == key:
                return axis.current_index()
        raise KeyError(f"{key!r}. Existing axis names are {self.dims.names!r}.")

    def __setitem__(self, key: str, value: SupportsIndex) -> None:
        return self.dims.set_indices({key: value})


class DimValues(_DimInterface[Any]):
    def __getitem__(self, key: str) -> Any:
        for axis in self.dims.axes:
            if axis.name == key:
                return axis.current_value()
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        return self.dims.set_values({key: value})
