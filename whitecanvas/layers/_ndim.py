"""LayerStack represents a stack of layers for n-D visualization."""

from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Concatenate, ParamSpec

from whitecanvas.layers import _primitive
from whitecanvas.layers._base import DataBoundLayer, LayerWrapper
from whitecanvas.types import XYData, XYTextData, XYYData

if TYPE_CHECKING:
    from whitecanvas.canvas import Canvas

_T = TypeVar("_T")
_P = ParamSpec("_P")


# TODO: Generic type wrong.
class LayerStack(LayerWrapper["DataBoundLayer[_T]"], Generic[_T]):
    def __init__(
        self,
        base_layer: DataBoundLayer[_T],
        data: Slicable[_T],
        axis_names: list[str] | None = None,
    ):
        super().__init__(base_layer)
        self._data_stack = data
        if axis_names is None:
            axis_names = [f"axis_{i}" for i in reversed(range(data.ndim))]
        else:
            axis_names = list(axis_names)
        self._axis_names = axis_names

    @property
    def axis_names(self) -> list[str]:
        """List of axis names."""
        return self._axis_names.copy()

    def _update_layer_data(self, index: dict[str, int]) -> None:
        sl = tuple(index[a] for a in self._axis_names)
        data = self._get_slice(sl)
        self._base_layer._set_layer_data(data)
        self._base_layer.events.data.emit(data)

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
        self._axis_names = list(axis_names)


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


class TextLayerStack(LayerStack[XYTextData]):
    @classmethod
    def from_layer_class(
        cls,
        constructor: Callable[Concatenate[Any, Any, _P], DataBoundLayer[XYTextData]],
        x: Any,
        y: Any,
        text: Any,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ):
        """Create a layer stack from the given layer constructor and data."""
        xdata = _norm_one(x)
        ydata = _norm_one(y)
        textdata = _norm_one(text, dtype=np.str_)
        stack = StackedArray([xdata, ydata, textdata])
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
        im = stack.slice_at(sl)
        return cls(constructor(im, *args, **kwargs), stack)


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
        spans = stack.slice_at(sl)
        return cls(constructor(spans, *args, **kwargs), stack)


def _norm_one(data, dim: int = 1, dtype=np.float32) -> Slicable[NDArray[np.number]]:
    try:
        arr = np.asarray(data, dtype=dtype)
    except ValueError:
        arr = np.asarray(data, dtype=np.object_)

    if arr.ndim == dim:
        return ConstArray(arr)
    if arr.dtype == np.object_:
        # convert all the elements to 1D array
        for sl in zip(*[range(s) for s in arr.shape]):
            arr1d = np.asarray(arr[sl], dtype=dtype)
            if arr1d.ndim != dim:
                raise ValueError(
                    f"Data at index {sl} has {arr1d.ndim} dimensions but "
                    f"{dim} is required."
                )
            arr[sl] = arr1d
        return NonuniformArray(arr)
    else:
        return GridArray(arr, dim=dim)


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


# TODO: multidimensional stripplot etc.
class TableArray(Slicable[NDArray[np.number]]):
    def __init__(self, obj: dict[Any, dict[str, NDArray[np.number]]]):
        self._obj = obj

    def slice_at(self, index: tuple[int, ...]):
        ...


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
