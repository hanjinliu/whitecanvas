from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from whitecanvas.layers._base import Layer, PrimitiveLayer
from whitecanvas.protocols import BaseProtocol

_P = TypeVar("_P", bound=BaseProtocol)
_T = TypeVar("_T")


class Layer3D(Layer):
    def __init__(self, name: str | None = None):
        super().__init__(name=name)
        self._z_hint = None

    def bbox_hint(self) -> NDArray[np.float64]:
        return np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)


class PrimitiveLayer3D(PrimitiveLayer[_P], Layer3D):
    def bbox_hint(self) -> NDArray[np.float64]:
        """Return the bounding box hint (xmin, xmax, ymin, ymax, zmin, zmax)."""
        if self._x_hint is None:
            _x = (np.nan, np.nan)
        else:
            _x = self._x_hint
        if self._y_hint is None:
            _y = (np.nan, np.nan)
        else:
            _y = self._y_hint
        if self._z_hint is None:
            _z = (np.nan, np.nan)
        else:
            _z = self._z_hint
        return np.array(_x + _y + _z, dtype=np.float64)


class DataBoundLayer3D(PrimitiveLayer3D[_P], Generic[_P, _T]):
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
        data_normed = self._norm_layer_data(data)
        self._set_layer_data(data_normed)
        self.events.data.emit(data_normed)
