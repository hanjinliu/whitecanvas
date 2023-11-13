from __future__ import annotations
from typing import Generic, overload, TypeVar
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from neoplot import protocols
from neoplot.layers import Layer, Line, Scatter
from neoplot.types import LineDict, MarkerDict
from neoplot.canvas import canvas_namespace as _ns
from neoplot.backend import Backend

_T = TypeVar("_T", bound=protocols.HasVisibility)

class CanvasBase(ABC, Generic[_T]):
    title = _ns.TitleNamespace()
    x = _ns.XAxisNamespace()
    y = _ns.YAxisNamespace()

    def __init__(self, backend: str | None = None):
        self._backend_installer = Backend(backend)
        self._backend = self._create_backend()
        self.layers = []
    
    @abstractmethod
    def _create_backend(self) -> _T:
        """Create a backend object."""
    
    @abstractmethod
    def _canvas(self) -> protocols.CanvasProtocol:
        """Return the canvas object."""
    
    @property
    def native(self) -> _T:
        return self._backend

    def show(self):
        self._backend._plt_set_visible(True)
    
    def close(self):
        self._backend._plt_set_visible(False)
    
    @overload
    def add_line(
        self,
        ydata: ArrayLike,
        *,
        name: str | None = None,
        line: LineDict = {},
        marker: MarkerDict = {},
        **kwargs,
    ) -> Line:
        ...

    @overload
    def add_line(
        self,
        xdata: ArrayLike,
        ydata: ArrayLike,
        *,
        name: str | None = None,
        line: LineDict = {},
        marker: MarkerDict = {},
        **kwargs,
    ) -> Line:
        ...
    

    def add_line(
        self,
        *args,
        name=None,
        line={},
        marker={},
        **kwargs,
    ):
        xdata, ydata = _normalize_xy(*args)
        name = self._coerce_name(Line, name)
        # TODO: extract info from kwargs
        layer = Line(
            xdata,
            ydata,
            name=name, 
            line=line,
            marker=marker,
            backend=self._backend_installer,
        )
        self._canvas()._plt_insert_layer(len(self.layers), layer._backend)
        self.layers.append(layer)
        return layer
    
    def add_scatter(
        self,
        *args,
        name: str | None = None,
        marker: MarkerDict = {},
        **kwargs,
    ):
        xdata, ydata = _normalize_xy(*args)
        name = self._coerce_name(Scatter, name)
        layer = Scatter(
            xdata,
            ydata,
            name=name, 
            marker=marker,
            backend=self._backend_installer,
        )
        self._canvas()._plt_insert_layer(len(self.layers), layer._backend)
        self.layers.append(layer)
        return layer
    
    def add_lines(
        self,
        *args,
        name: str | None = None,
        line: LineDict = {},
        marker: MarkerDict = {},
        **kwargs,
    ):
        if len(args) == 1:
            ydatas = [_as_array_1d(args[0])]
            xdata = np.arange(ydatas[0].size)
        else :
            xdata = _as_array_1d(args[0])
            ydatas = [_as_array_1d(a) for a in args[1:]]
        layers = []
        for ydata in ydatas:
            name = self._coerce_name(Line, name)
            layer = Line(
                xdata,
                ydata,
                name=name,
                line=line,
                marker=marker,
                backend=self._backend_installer,
            )
            self._canvas()._plt_insert_layer(len(self.layers), layer._backend)
            self.layers.append(layer)
            layers.append(layer)
        return layers

    def _coerce_name(self, layer_type: type[Layer], name: str | None) -> str:
        if name is None:
            name = layer_type.__name__
        basename = name
        i = 0
        while name in self.layers:
            name = f"{basename}-{i}"
            i += 1
        return name

def _as_array_1d(x: ArrayLike) -> NDArray[np.number]:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got {x.ndim}D array")
    if x.dtype.kind not in "iuf":
        raise ValueError(f"Input {x!r} did not return a numeric array")
    return x

def _normalize_xy(*args) -> tuple[NDArray[np.number], NDArray[np.number]]:
    if len(args) == 1:
        ydata = _as_array_1d(args[0])
        xdata = np.arange(ydata.size)
    elif len(args) == 2:
        xdata = _as_array_1d(args[0])
        ydata = _as_array_1d(args[1])
    else:
        raise TypeError(
            f"Expected 1 or 2 positional arguments, got {len(args)}"
        )
    return xdata, ydata
    
