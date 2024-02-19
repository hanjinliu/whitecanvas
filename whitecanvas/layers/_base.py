from __future__ import annotations

import warnings
import weakref
from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, Generic, Iterable, Iterator, TypeVar

import numpy as np
from numpy.typing import NDArray
from psygnal import Signal, SignalGroup

from whitecanvas.backend import Backend
from whitecanvas.layers._legend import EmptyLegendItem, LegendItem
from whitecanvas.protocols import BaseProtocol

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.canvas import CanvasBase

_P = TypeVar("_P", bound=BaseProtocol)
_L = TypeVar("_L", bound="Layer")
_T = TypeVar("_T")


class LayerEvents(SignalGroup):
    data = Signal(check_nargs_on_connect=False)  # (data)
    name = Signal(str)
    visible = Signal(bool)
    _layer_grouped = Signal(object)  # (group)


class Layer(ABC):
    events: LayerEvents
    _events_class: type[LayerEvents]
    _ATTACH_TO_AXIS = False

    def __init__(self, name: str | None = None):
        if not hasattr(self.__class__, "_events_class"):
            self.events = LayerEvents()
        else:
            self.events = self.__class__._events_class()
        self._name = name if name is not None else self.__class__.__name__
        self._x_hint = self._y_hint = None
        self._is_grouped = False
        self._canvas_ref = lambda: None
        _set_deprecated_aliases(self)

    @abstractproperty
    def visible(self) -> bool:
        """Return true if the layer is visible"""

    @visible.setter
    def visible(self, visible: bool):
        """Set the visibility of the layer"""

    @property
    def name(self) -> str:
        """Name of this layer."""
        return self._name

    @name.setter
    def name(self, name: str):
        """Set the name of this layer."""
        self._name = str(name)

    def __repr__(self):
        return f"{self.__class__.__name__}<{self.name!r}>"

    def _connect_canvas(self, canvas: CanvasBase):
        """If needed, do something when layer is added to a canvas."""
        self.events._layer_grouped.connect(canvas._cb_layer_grouped, unique=True)
        self.events.connect(canvas._draw_canvas, unique=True)
        self._canvas_ref = weakref.ref(canvas)

    def _disconnect_canvas(self, canvas: CanvasBase):
        """If needed, do something when layer is removed from a canvas."""
        self.events._layer_grouped.disconnect(canvas._cb_layer_grouped)
        self.events.disconnect(canvas._draw_canvas)
        self._canvas_ref = lambda: None

    def _canvas(self) -> CanvasBase:
        canvas = self._canvas_ref()
        if canvas is None:
            raise ValueError("Layer is not in any canvas.")
        return canvas

    @abstractmethod
    def bbox_hint(self) -> NDArray[np.float64]:
        """Return the bounding box hint (xmin, xmax, ymin, ymax) of this layer."""

    def as_overlay(self) -> Self:
        """Move this layer to the overlay level."""
        canvas = self._canvas()
        canvas.layers.remove(self)
        canvas.overlays.append(self)
        return self

    def _as_legend_item(self) -> LegendItem:
        """Return the legend item for this layer."""
        return EmptyLegendItem()


class PrimitiveLayer(Layer, Generic[_P]):
    """Layers that are composed of a single component."""

    _backend: _P
    _backend_name: str
    _backend_class_name = None

    @property
    def visible(self) -> bool:
        """Return true if the layer is visible"""
        return self._backend._plt_get_visible()

    @visible.setter
    def visible(self, visible: bool):
        """Set the visibility of the layer"""
        self._backend._plt_set_visible(visible)
        self.events.visible.emit(visible)

    def _create_backend(self, backend: Backend, *args) -> _P:
        """Create a backend object."""
        if self._backend_class_name is not None:
            self._backend_name = backend.name
            return backend.get(self._backend_class_name)(*args)
        for mro in reversed(type(self).__mro__):
            name = mro.__name__
            if (
                issubclass(mro, PrimitiveLayer)
                and mro is not PrimitiveLayer
                and backend.has(name)
            ):
                self._backend_name = backend.name
                return backend.get(name)(*args)
        raise TypeError(
            f"Cannot create a {backend.name} backend for {type(self).__name__}"
        )

    def bbox_hint(self) -> NDArray[np.float64]:
        """Return the bounding box hint (xmin, xmax, ymin, ymax) of this layer."""
        if self._x_hint is None:
            _x = (np.nan, np.nan)
        else:
            _x = self._x_hint
        if self._y_hint is None:
            _y = (np.nan, np.nan)
        else:
            _y = self._y_hint
        return np.array(_x + _y, dtype=np.float64)


class DataBoundLayer(PrimitiveLayer[_P], Generic[_P, _T]):
    @abstractmethod
    def _get_layer_data(self) -> _T:
        """Get the data for this layer."""

    @abstractmethod
    def _set_layer_data(self, data: _T):
        """Set the data for this layer."""

    def _norm_layer_data(self, data: Any) -> _T:
        """Normalize the data for this layer."""
        return data

    @property
    def data(self) -> _T:
        """Data for this layer."""
        return self._get_layer_data()

    @data.setter
    def data(self, data):
        """Set the data for this layer."""
        self._set_layer_data(self._norm_layer_data(data))
        self.events.data.emit(data)


class HoverableDataBoundLayer(DataBoundLayer[_P, _T]):
    def with_hover_text(self, text: str | Iterable[Any]) -> Self:
        """Add hover text to the data points."""
        if isinstance(text, str):
            texts = [text] * self.ndata
        else:
            texts = [str(t) for t in text]
        if len(texts) != self.ndata:
            raise ValueError(
                "Expected text to have the same size as the data, "
                f"got {len(texts)} and {self.ndata}"
            )
        self._backend._plt_set_hover_text(texts)
        return self

    @abstractproperty
    def ndata(self) -> int:
        """Number of data points."""


class LayerGroup(Layer):
    """
    A group of layers that will be treated as a single layer in the canvas.
    """

    def __init__(self, name: str | None = None):
        super().__init__(name)
        self._visible = True

    @abstractmethod
    def iter_children(self) -> Iterator[Layer]:
        """Iterate over all children."""

    def iter_children_recursive(self) -> Iterator[PrimitiveLayer[BaseProtocol]]:
        for child in self.iter_children():
            if isinstance(child, LayerGroup):
                yield from child.iter_children_recursive()
            elif isinstance(child, LayerWrapper):
                if isinstance(child._base_layer, LayerGroup):
                    yield from child._base_layer.iter_children_recursive()
                else:
                    yield child._base_layer
            else:
                yield child

    def _emit_layer_grouped(self):
        """Emit all the grouped signal."""
        for c in self.iter_children():
            c.events._layer_grouped.emit(self)

    @property
    def visible(self) -> bool:
        """Return true if the layer is visible"""
        return self._visible

    @visible.setter
    def visible(self, visible: bool):
        """Set the visibility of the layer"""
        self._visible = visible
        for child in self.iter_children():
            child.visible = visible
        self.events.visible.emit(visible)

    @property
    def _backend_name(self) -> str:
        """The backend name of this layer group."""
        for child in self.iter_children():
            return child._backend_name
        raise RuntimeError(f"No backend name found for {self!r}")

    def bbox_hint(self) -> NDArray[np.float64]:
        """
        Return the bounding box hint (xmin, xmax, ymin, ymax) of this group.

        Note that unless children notifies the change of their bounding box hint, bbox
        hint needs recalculation.
        """
        hints = [child.bbox_hint() for child in self.iter_children()]
        if len(hints) == 0:
            return np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        ar = np.stack(hints, axis=1)
        # Any of the four corners could be all-nan. In that case, we should return nan.
        # Otherwise, we should return the known min/max.
        allnan = np.isnan(ar).all(axis=1)
        xmin = np.nan if allnan[0] else np.nanmin(ar[0, :])
        xmax = np.nan if allnan[1] else np.nanmax(ar[1, :])
        ymin = np.nan if allnan[2] else np.nanmin(ar[2, :])
        ymax = np.nan if allnan[3] else np.nanmax(ar[3, :])
        return np.array([xmin, xmax, ymin, ymax], dtype=np.float64)


class LayerWrapper(Layer, Generic[_L]):
    def __init__(
        self,
        base_layer: _L,
    ):
        self._base_layer = base_layer
        super().__init__(base_layer.name)

    @property
    def visible(self) -> bool:
        """Whether the layer is visible."""
        return self._base_layer.visible

    @visible.setter
    def visible(self, visible: bool):
        self._base_layer.visible = visible

    @property
    def name(self) -> str:
        """Name of the layer."""
        return self._base_layer.name

    @name.setter
    def name(self, name: str):
        self._base_layer.name = name

    @property
    def base(self) -> _L:
        """The base layer."""
        return self._base_layer

    def bbox_hint(self) -> NDArray[np.floating]:
        """Return the bounding box hint using the base layer."""
        return self._base_layer.bbox_hint()

    def _connect_canvas(self, canvas: CanvasBase):
        self._base_layer._connect_canvas(canvas)
        return super()._connect_canvas(canvas)

    def _disconnect_canvas(self, canvas: CanvasBase):
        self._base_layer._disconnect_canvas(canvas)
        return super()._disconnect_canvas(canvas)

    @property
    def _ATTACH_TO_AXIS(self) -> bool:
        return self._base_layer._ATTACH_TO_AXIS

    def _as_legend_item(self) -> LegendItem:
        """Return the legend item for this layer."""
        return self._base_layer._as_legend_item()


# deprecated, new
_DEPRECATED = [
    ("with_shift", "move"),
    ("with_color", "update_color"),
    ("with_color_palette", "update_color_palette"),
    ("with_colormap", "update_colormap"),
    ("with_style", "update_style"),
    ("with_width", "update_width"),
    ("with_size", "update_size"),
    ("with_hatch", "update_hatch"),
    ("with_symbol", "update_symbol"),
    ("with_edge_color", "update_edge_color"),
    ("with_edge_colormap", "update_edge_colormap"),
]


def _wrap_deprecation(method, old: str):
    def wrapper(self, *args, **kwargs):
        warnings.warn(
            f"{old} is deprecated and will be removed in the future. "
            f"Use {method.__name__} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return method(self, *args, **kwargs)

    wrapper.__doc__ = f"Deprecated. Use {method.__name__} instead."
    return wrapper


def _set_deprecated_aliases(layer: Layer):
    for deprecated, new in _DEPRECATED:
        if hasattr(layer, new):
            setattr(
                layer, deprecated, _wrap_deprecation(getattr(layer, new), deprecated)
            )
