from __future__ import annotations

import weakref
from typing import TypeVar, Generic, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from whitecanvas.types import ArrayLike1D
from whitecanvas.utils.normalize import as_array_1d
from whitecanvas._exceptions import ReferenceDeletedError

if TYPE_CHECKING:
    from whitecanvas.canvas._base import CanvasBase

_C = TypeVar("_C", bound="CanvasBase")


class StackedDataPlotter(Generic[_C]):
    def __init__(self, canvas: _C, xdata: ArrayLike1D):
        self._canvas_ref = weakref.ref(canvas)
        self._xdata = as_array_1d(xdata)

    def _canvas(self) -> _C:
        canvas = self._canvas_ref()
        if canvas is None:
            raise ReferenceDeletedError("Canvas has been deleted.")
        return canvas
