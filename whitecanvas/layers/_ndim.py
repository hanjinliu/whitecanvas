"""LayerStack represents a stack of layers for n-D visualization."""

from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty

from typing import TYPE_CHECKING, Any, Callable, TypeVar, Generic
from typing_extensions import ParamSpec, Concatenate

import numpy as np
from numpy.typing import NDArray
from whitecanvas.layers._base import DataBoundLayer, Layer
from whitecanvas.layers import _primitive
from whitecanvas.types import XYData, XYYData, XYTextData
from whitecanvas.utils.normalize import as_array_1d

if TYPE_CHECKING:
    from whitecanvas.canvas import Canvas

_T = TypeVar("_T")
_P = ParamSpec("_P")


class LayerStack(Layer, Generic[_T]):
    def __init__(
        self,
        base_layer: DataBoundLayer[_T],
        data: Slicable[_T],
        axis_names: list[str] | None = None,
    ):
        self._base_layer = base_layer
        self._data_stack = data
        if axis_names is None:
            axis_names = [f"axis_{i}" for i in reversed(range(data.ndim))]
        self._axis_names = axis_names
        super().__init__(base_layer.name)

    @property
    def visible(self) -> bool:
        return self._base_layer.visible

    @visible.setter
    def visible(self, visible: bool):
        self._base_layer.visible = visible

    @property
    def name(self) -> str:
        return self._base_layer.name

    @name.setter
    def name(self, name: str):
        self._base_layer.name = name

    def bbox_hint(self) -> NDArray[np.floating]:
        """Return the bounding box hint using the base layer."""
        return self._base_layer.bbox_hint()

    @property
    def axis_names(self) -> list[str]:
        return self._axis_names.copy()

    def _update_layer_data(self, index: dict[str, int]) -> None:
        sl = tuple(index[a] for a in self._axis_names)
        self._base_layer._set_layer_data(self._get_slice(sl))

    def _get_slice(self, index) -> _T:
        return self._base_layer._norm_layer_data(self._data_stack.slice_at(index))

    def _connect_canvas(self, canvas: Canvas):
        canvas.dims.events.indices.connect(self._update_layer_data, unique=True)
        return super()._connect_canvas(canvas)

    def _disconnect_canvas(self, canvas: Canvas):
        canvas.dims.events.indices.disconnect(self._update_layer_data)
        return super()._disconnect_canvas(canvas)

    def _with_axis_names(self, axis_names: list[str]):
        stack = self._data_stack
        if len(axis_names) != stack.ndim:
            raise ValueError(
                f"Got {stack.ndim}D dataset but {len(axis_names)} axes names "
                "were given."
            )
        self._axis_names = axis_names


class YLayerStack(LayerStack[NDArray[np.number]]):
    @classmethod
    def from_layer_class(
        cls,
        constructor: Callable[Concatenate[Any, Any, _P], DataBoundLayer[XYData]],
        x: Any,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ):
        """Create a layer stack from the given layer constructor and data."""
        stack = _norm_one(x)
        sl = (0,) * stack.ndim
        data = stack.slice_at(sl)
        return cls(constructor(*data, *args, **kwargs), stack)


class XYLayerStack(LayerStack[XYData]):
    @classmethod
    def from_layer_class(
        cls,
        constructor: Callable[Concatenate[Any, Any, _P], DataBoundLayer[XYData]],
        x: Any,
        y: Any,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ):
        """Create a layer stack from the given layer constructor and data."""
        x0 = _norm_one(x)
        y0 = _norm_one(y)
        stack = StackedArray([x0, y0])
        sl = (0,) * stack.ndim
        data = stack.slice_at(sl)
        return cls(constructor(*data, *args, **kwargs), stack)


class XYYLayerStack(LayerStack[XYYData]):
    @classmethod
    def from_layer_class(
        cls,
        constructor: Callable[Concatenate[Any, Any, _P], DataBoundLayer[XYYData]],
        x: Any,
        y0: Any,
        y1: Any,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ):
        """Create a layer stack from the given layer constructor and data."""
        xdata = _norm_one(x)
        ydata0 = _norm_one(y0)
        ydata1 = _norm_one(y1)
        stack = StackedArray([xdata, ydata0, ydata1])
        sl = (0,) * stack.ndim
        data = stack.slice_at(sl)
        return cls(constructor(*data, *args, **kwargs), stack)


class ImageLayerStack(LayerStack[NDArray[np.number]]):
    @classmethod
    def from_layer_class(
        cls,
        constructor: Callable[Concatenate[Any, Any, _P], _primitive.Image],
        img: Any,
        rgb: bool = False,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ):
        """Create a layer stack for multi-dimension image."""
        if rgb:
            stack = _norm_one(img, dim=3)
        else:
            stack = _norm_one(img, dim=2)
        sl = (0,) * stack.ndim
        data = stack.slice_at(sl)
        return cls(constructor(*data, *args, **kwargs), stack)


class SpansLayerStack(LayerStack[NDArray[np.number]]):
    @classmethod
    def from_layer_class(
        cls,
        constructor: Callable[Concatenate[Any, Any, _P], _primitive.Spans],
        spans: Any,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ):
        stack = _norm_one(spans, dim=2)
        sl = (0,) * stack.ndim
        data = stack.slice_at(sl)
        return cls(constructor(*data, *args, **kwargs), stack)


def _norm_one(data, dim: int = 1) -> Slicable[NDArray[np.number]]:
    try:
        arr = np.asarray(data)
    except ValueError:
        arr = np.asarray(data, dtype=np.object_)

    if arr.ndim == dim:
        if arr.dtype.kind not in "iuf":
            raise ValueError("xdata must be numerical.")
        return ConstArray(arr)
    if arr.dtype == np.object_:
        # convert all the elements to 1D array
        for sl in zip(*[range(s) for s in arr.shape]):
            arr1d = np.asarray(arr[sl])
            if arr1d.ndim != dim:
                raise ValueError(
                    f"Data at index {sl} has {arr1d.ndim} dimensions but "
                    f"{dim} is required."
                )
            if arr1d.dtype.kind not in "iuf":
                raise ValueError(f"Data at index {sl} is not numerical.")
            arr[sl] = arr1d
        return NonuniformArray(arr)
    else:
        return GridArray(arr)


class Slicable(ABC, Generic[_T]):
    """A class that can generate sliced data."""

    @abstractmethod
    def slice_at(self, index: tuple[int, ...]) -> _T:
        """Get the sliced data at the given index."""

    @abstractproperty
    def shape(self) -> tuple[int, ...]:
        """The shape of the data."""

    @property
    def ndim(self) -> int:
        """The number of dimensions of the data."""
        return len(self.shape)


class ConstArray(Slicable[_T]):
    def __init__(self, obj: _T):
        self._obj = obj

    def slice_at(self, index: tuple[int, ...]) -> _T:
        return self._obj

    @property
    def shape(self):
        return ()


class GridArray(Slicable[NDArray[np.number]]):
    def __init__(self, obj: NDArray[np.number], dim: int = 1):
        self._obj = obj
        self._dim = dim  # the ndim of data

    def slice_at(self, index: tuple[int, ...]) -> NDArray[np.number]:
        return self._obj[index]

    @property
    def shape(self):
        return self._obj.shape[: -self._dim]


class NonuniformArray(Slicable[_T]):
    def __init__(self, obj: NDArray[np.object_]):
        self._obj = obj

    def slice_at(self, index: tuple[int, ...]) -> NDArray[np.number]:
        return self._obj[index]

    @property
    def shape(self):
        return self._obj.shape


class StackedArray(Slicable[_T]):
    """A stacked Slicable objects."""

    def __init__(
        self,
        obj: list[Slicable[_T]],
    ):
        self._objs = obj
        self._shape = ()
        for o in obj:
            if o.ndim > 0:
                if self._shape and self._shape != o.shape:
                    raise ValueError("All the objects must have the same shape or 1D.")
                self._shape = o.shape

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._objs!r})"

    def slice_at(self, index: tuple[int, ...]) -> list[_T]:
        return [obj.slice_at(index) for obj in self._objs]

    @property
    def shape(self):
        return self._shape
