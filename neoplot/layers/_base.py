from __future__ import annotations

from typing import Generic, TypeVar, NamedTuple, TYPE_CHECKING

import numpy as np
from neoplot.protocols import BaseProtocol
from neoplot.backend import Backend

if TYPE_CHECKING:
    from neoplot.canvas import Canvas

_P = TypeVar("_P", bound=BaseProtocol)
_L = TypeVar("_L", bound="Layer")


class Layer(Generic[_P]):
    _backend: _P
    _name: str

    @property
    def visible(self) -> bool:
        """Return true if the layer is visible"""
        return self._backend._plt_get_visible()

    @visible.setter
    def visible(self, visible: bool):
        """Set the visibility of the layer"""
        self._backend._plt_set_visible(visible)

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
        if not isinstance(layer_type, type) or issubclass(layer_type, Layer):
            raise TypeError(
                "Argument of `expect` must be a layer class, "
                f"got {layer_type!r} (type: {type(layer_type).__name__}))"
            )
        if not isinstance(self, layer_type):
            raise TypeError(f"Expected {layer_type.__name__}, got {type(self).__name__}")
        return self

    def _create_backend(self, backend: Backend, *args) -> _P:
        """Create a backend object."""
        for mro in reversed(type(self).__mro__):
            name = mro.__name__
            if issubclass(mro, Layer) and mro is not Layer and backend.has(name):
                return backend.get(name)(*args)
        raise TypeError(f"Cannot create a backend for {type(self).__name__}")

    def __repr__(self):
        return f"{self.__class__.__name__}<{self.name!r}>"

    def _connect_canvas(self, canvas: Canvas):
        """If needed, do something when layer is added to a canvas."""

    def _disconnect_canvas(self, canvas: Canvas):
        """If needed, do something when layer is removed from a canvas."""


AnyLayer = Layer[BaseProtocol]


class XYData(NamedTuple):
    x: np.ndarray
    y: np.ndarray

    @property
    def concat(self) -> np.ndarray:
        """Concatenate x and y data into a single (N, 2) array."""
        return np.stack([self.x, self.y], axis=1)
