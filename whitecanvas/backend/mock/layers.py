from __future__ import annotations

import numpy as np
from cmap import Colormap

from whitecanvas import protocols
from whitecanvas.backend.mock._base import (
    BaseMockLayer,
    MockHasData,
    MockHasEdges,
    MockHasFaces,
    MockHasMouseEvents,
    MockHasMultiEdges,
    MockHasMultiFaces,
)
from whitecanvas.types import Alignment, Orientation, Symbol
from whitecanvas.utils.normalize import as_color_array
from whitecanvas.utils.type_check import is_real_number


@protocols.check_protocol(protocols.BandProtocol)
class Band(MockHasData, MockHasFaces, MockHasEdges, MockHasMouseEvents):
    def __init__(
        self,
        t: np.ndarray,
        ydata0: np.ndarray,
        ydata1: np.ndarray,
        orient: Orientation,
    ):
        super().__init__((t, ydata0, ydata1))

    ##### XYYDataProtocol #####
    def _plt_get_vertical_data(self):
        return self._plt_get_data()

    def _plt_get_horizontal_data(self):
        return self._plt_get_data()

    def _plt_set_vertical_data(self, t, ydata0, ydata1):
        self._plt_set_data((t, ydata0, ydata1))

    def _plt_set_horizontal_data(self, t, ydata0, ydata1):
        self._plt_set_data((t, ydata0, ydata1))


@protocols.check_protocol(protocols.BarProtocol)
class Bars(MockHasData, MockHasMultiFaces, MockHasMultiEdges, MockHasMouseEvents):
    def __init__(self, xlow, xhigh, ylow, yhigh):
        super().__init__((xlow, xhigh, ylow, yhigh))

    def _plt_get_ndata(self) -> int:
        return self._plt_get_data()[0].size


@protocols.check_protocol(protocols.ImageProtocol)
class Image(BaseMockLayer):
    def __init__(self, data: np.ndarray):
        super().__init__()
        self._data = data
        self._clim = (0, 1)
        self._cmap = Colormap("gray")
        self._scale = (1, 1)
        self._translation = (0, 0)

    def _plt_get_data(self) -> np.ndarray:
        return self._data

    def _plt_set_data(self, data: np.ndarray):
        self._data = data

    def _plt_get_colormap(self) -> Colormap:
        return self._cmap

    def _plt_set_colormap(self, cmap: Colormap):
        self._cmap = cmap

    def _plt_get_clim(self) -> tuple[float, float]:
        return self._clim

    def _plt_set_clim(self, clim: tuple[float, float]):
        self._clim = clim

    def _plt_get_scale(self) -> tuple[float, float]:
        return self._scale

    def _plt_set_scale(self, scale: tuple[float, float]):
        self._scale = scale

    def _plt_get_translation(self) -> tuple[float, float]:
        return self._translation

    def _plt_set_translation(self, translation: tuple[float, float]):
        self._translation = translation


@protocols.check_protocol(protocols.LineProtocol)
class MonoLine(MockHasData, MockHasEdges, MockHasMouseEvents):
    def __init__(self, xdata, ydata):
        super().__init__((xdata, ydata))
        self._antialias = True

    def _plt_get_antialias(self) -> bool:
        return self._antialias

    def _plt_set_antialias(self, antialias: bool):
        self._antialias = antialias


@protocols.check_protocol(protocols.MultiLineProtocol)
class MultiLine(BaseMockLayer, MockHasMultiEdges, MockHasMouseEvents):
    def __init__(self, data: list[np.ndarray]):
        super().__init__()
        self._data = data
        self._antialias = True

    def _plt_get_data(self) -> list[np.ndarray]:
        return self._data

    def _plt_set_data(self, data: list[np.ndarray]):
        self._data = data

    def _plt_get_antialias(self) -> bool:
        return self._antialias

    def _plt_set_antialias(self, antialias: bool):
        self._antialias = antialias

    def _plt_get_ndata(self) -> int:
        return len(self._plt_get_data())


@protocols.check_protocol(protocols.MarkersProtocol)
class Markers(MockHasData, MockHasMultiFaces, MockHasMultiEdges, MockHasMouseEvents):
    def __init__(self, xdata, ydata):
        super().__init__((xdata, ydata))
        self._symbol = Symbol.CIRCLE
        self._size = np.full(xdata.shape, 10.0, dtype=np.float32)

    def _plt_get_ndata(self) -> int:
        return self._plt_get_data()[0].size

    def _plt_get_symbol(self) -> Symbol:
        return self._symbol

    def _plt_set_symbol(self, symbol: Symbol):
        self._symbol = symbol

    def _plt_get_symbol_size(self):
        return self._size

    def _plt_set_symbol_size(self, size):
        if is_real_number(size):
            size = np.full(self._size.shape, size, dtype=np.float32)
        self._size = size


@protocols.check_protocol(protocols.TextProtocol)
class Texts(MockHasData, MockHasMultiFaces, MockHasMultiEdges):
    def __init__(self, x, y, text):
        super().__init__((x, y))
        self._text = text
        ndata = len(text)
        self._fontfamily = ["sans-serif"] * ndata
        self._fontsize = np.full(ndata, 10.0, dtype=np.float32)
        self._text_color = np.zeros((ndata, 4), dtype=np.float32)
        self._rotation = np.zeros(ndata, dtype=np.float32)
        self._anchors = [Alignment.LEFT] * ndata

    def _plt_get_ndata(self) -> int:
        return len(self._text)

    def _plt_get_text(self) -> list[str]:
        return self._text

    def _plt_set_text(self, text: list[str]):
        self._text = text

    def _plt_get_text_color(self):
        return self._text_color

    def _plt_set_text_color(self, color):
        self._text_color = as_color_array(color, self._plt_get_ndata())

    def _plt_get_text_size(self):
        return self._fontsize

    def _plt_set_text_size(self, size):
        if is_real_number(size):
            size = np.full(self._fontsize.shape, size, dtype=np.float32)
        self._fontsize = size

    def _plt_get_text_position(self):
        return self._plt_get_data()

    def _plt_set_text_position(self, position):
        self._plt_set_data(position)

    def _plt_get_text_anchor(self) -> list[Alignment]:
        return self._anchors

    def _plt_set_text_anchor(self, anc: list[Alignment]):
        if isinstance(anc, (str, Alignment)):
            anc = [Alignment(anc)] * self._plt_get_ndata()
        self._anchors = anc

    def _plt_get_text_rotation(self):
        return self._rotation

    def _plt_set_text_rotation(self, rotation):
        if is_real_number(rotation):
            rotation = np.full(self._rotation.shape, rotation, dtype=np.float32)
        self._rotation = rotation

    def _plt_get_text_fontfamily(self) -> list[str]:
        return self._fontfamily

    def _plt_set_text_fontfamily(self, family: list[str]):
        if isinstance(family, str):
            family = [family] * self._plt_get_ndata()
        self._fontfamily = family
