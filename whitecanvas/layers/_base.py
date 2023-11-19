from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import Generic, Iterator, TypeVar, NamedTuple, TYPE_CHECKING
from psygnal import Signal, SignalGroup
import numpy as np
from numpy.typing import ArrayLike
from whitecanvas.protocols import BaseProtocol
from whitecanvas.backend import Backend

if TYPE_CHECKING:
    from whitecanvas.canvas import Canvas

_P = TypeVar("_P", bound=BaseProtocol)
_L = TypeVar("_L", bound="Layer")


class Layer(ABC):
    _layer_grouped = Signal(object)  # (group)
    _name: str

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

    def expect(self, layer_type: _L, /) -> _L:
        """
        A type guard for layers.

        >>> canvas.layers["scatter-layer-name"].expect(Scatter).face_color
        """
        if not isinstance(layer_type, type) or issubclass(layer_type, PrimitiveLayer):
            raise TypeError(
                "Argument of `expect` must be a layer class, "
                f"got {layer_type!r} (type: {type(layer_type).__name__}))"
            )
        if not isinstance(self, layer_type):
            raise TypeError(
                f"Expected {layer_type.__name__}, got {type(self).__name__}"
            )
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}<{self.name!r}>"

    def _connect_canvas(self, canvas: Canvas):
        """If needed, do something when layer is added to a canvas."""
        self._layer_grouped.connect(canvas._group_layers, unique=True)

    def _disconnect_canvas(self, canvas: Canvas):
        """If needed, do something when layer is removed from a canvas."""
        self._layer_grouped.disconnect(canvas._group_layers)


class PrimitiveLayer(Layer, Generic[_P]):
    """Layers that are composed of a single component."""

    _backend: _P
    _backend_name: str

    @property
    def visible(self) -> bool:
        """Return true if the layer is visible"""
        return self._backend._plt_get_visible()

    @visible.setter
    def visible(self, visible: bool):
        """Set the visibility of the layer"""
        self._backend._plt_set_visible(visible)

    def _create_backend(self, backend: Backend, *args) -> _P:
        """Create a backend object."""
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


class XYDataLayer(PrimitiveLayer[_P]):
    @abstractproperty
    def data(self) -> XYData:
        """Current data of the layer."""

    @abstractmethod
    def set_data(
        self,
        xdata: ArrayLike | None = None,
        ydata: ArrayLike | None = None,
    ):
        """Set the x and y data of the layer."""


class LayerGroup(Layer):
    """
    A group of layers that will be treated as a single layer in the canvas.
    """

    def __init__(self, name: str | None = None):
        self._name = name if name is not None else "LayerGroup"
        self._visible = True

    @abstractmethod
    def iter_children(self) -> Iterator[Layer]:
        """Iterate over all children."""

    def iter_children_recursive(self) -> Iterator[PrimitiveLayer[BaseProtocol]]:
        for child in self.iter_children():
            if isinstance(child, LayerGroup):
                yield from child.iter_children_recursive()
            else:
                yield child

    def _emit_layer_grouped(self):
        """Emit all the grouped signal."""
        for c in self.iter_children():
            c._layer_grouped.emit(self)

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

    @property
    def _backend_name(self) -> str:
        """The backend name of this layer group."""
        for child in self.iter_children():
            return child._backend_name
        raise RuntimeError(f"No backend name found for {self!r}")


class XYData(NamedTuple):
    x: np.ndarray
    y: np.ndarray


class XYYData(NamedTuple):
    x: np.ndarray
    y0: np.ndarray
    y1: np.ndarray

    @property
    def ycenter(self) -> np.ndarray:
        return (self.y0 + self.y1) / 2

    @property
    def ydiff(self) -> np.ndarray:
        return self.y1 - self.y0
