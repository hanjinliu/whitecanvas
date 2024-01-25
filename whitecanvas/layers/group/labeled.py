from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from whitecanvas.layers._base import PrimitiveLayer
from whitecanvas.layers._primitive import Bars, Errorbars, Line, Markers, Texts
from whitecanvas.layers.group._collections import LayerContainer
from whitecanvas.layers.group._offsets import NoOffset, TextOffset
from whitecanvas.types import Alignment, ColorType, LineStyle, XYData

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.layers.group.line_markers import Plot


class _LabeledLayerBase(LayerContainer):
    def __init__(
        self,
        layer: PrimitiveLayer,
        xerr: Errorbars,
        yerr: Errorbars,
        texts: Texts | None = None,
        name: str | None = None,
        offset: TextOffset | None = None,
    ):
        if offset is None:
            offset = NoOffset()
        if texts is None:
            px, py = self._get_data_xy(layer)
            texts = Texts(
                px,
                py,
                [""] * px.size,
                backend=layer._backend_name,
            )
        super().__init__([layer, xerr, yerr, texts], name=name)
        self._text_offset = offset

    def _get_data_xy(
        self, layer: PrimitiveLayer | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if layer is None:
            layer = self._children[0]
        return layer.data

    def _default_ordering(self, n: int) -> list[int]:
        assert n == 4
        return [2, 0, 1, 3]

    @property
    def xerr(self) -> Errorbars:
        """The errorbars layer for x."""
        return self._children[1]

    @property
    def yerr(self) -> Errorbars:
        """The errorbars layer for y."""
        return self._children[2]

    @property
    def texts(self) -> Texts:
        """The text group layer."""
        return self._children[3]

    @property
    def data(self) -> XYData:
        """The internal (x, y) data of this layer."""
        return self._children[0].data

    def set_data(self, xdata=None, ydata=None):
        """Set the (x, y) data of this layer."""
        px, py = self._get_data_xy()
        if xdata is None:
            dx = 0
        else:
            dx = xdata - px
        if ydata is None:
            dy = 0
        else:
            dy = ydata - py
        self._children[0].set_data(xdata, ydata)
        if self.xerr.ndata > 0:
            y, x0, x1 = self.xerr.data
            self.xerr.set_data(y + dy, x0 + dx, x1 + dx)
        if self.yerr.ndata > 0:
            x, y0, y1 = self.yerr.data
            self.yerr.set_data(x + dx, y0 + dy, y1 + dy)
        if self.texts.ntexts > 0:
            dx, dy = self._text_offset._asarray()
            self.texts.set_pos(px + dx, py + dy)

    @property
    def text_offset(self) -> TextOffset:
        """Return the text offset."""
        return self._text_offset

    def add_text_offset(self, dx: Any, dy: Any):
        """Add offset to text positions."""
        _offset = self._text_offset._add(dx, dy)
        if self.texts.ntexts > 0:
            px, py = self._get_data_xy()
            xoff, yoff = _offset._asarray()
            self.texts.set_pos(px + xoff, py + yoff)
        self._text_offset = _offset

    def with_xerr(
        self,
        len_lower: float,
        len_higher: float | None = None,
        *,
        color: ColorType = "black",
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        antialias: bool = True,
        capsize: float = 0,
    ) -> Self:
        """
        Set the x error bar data.

        Parameters
        ----------
        len_lower : float
            Length of lower error.
        len_higher : float, optional
            Length of higher error. If not given, set to the same as `len_lower`.
        """
        if len_higher is None:
            len_higher = len_lower
        x, y = self.data
        self.xerr.set_data(y, x - len_lower, x + len_higher)
        self.xerr.update(
            color=color, width=width, style=style, antialias=antialias,
            capsize=capsize,
        )  # fmt: skip
        return self

    def with_yerr(
        self,
        len_lower: float,
        len_higher: float | None = None,
        *,
        color: ColorType = "black",
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
        antialias: bool = True,
        capsize: float = 0,
    ) -> Self:
        """
        Set the y error bar data.

        Parameters
        ----------
        len_lower : float
            Length of lower error.
        len_higher : float, optional
            Length of higher error. If not given, set to the same as `len_lower`.
        """
        if len_higher is None:
            len_higher = len_lower
        x, y = self.data
        self.yerr.set_data(x, y - len_lower, y + len_higher)
        self.yerr.update(
            color=color, width=width, style=style, antialias=antialias,
            capsize=capsize
        )  # fmt: skip
        return self

    def with_text(
        self,
        strings: list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str | None = None,
        offset: tuple[Any, Any] | None = None,
    ) -> Self:
        """
        Add texts to the layer.

        Parameters
        ----------
        strings : str or list of str
            The text strings. If a single string is given, it will be used for all
            the data points.
        color : ColorType, default "black"
            Text color.
        size : float, default 12
            Font point size of the text.
        rotation : float, default 0.0
            Rotation of the text in degrees.
        anchor : str or Alignment, default Alignment.BOTTOM_LEFT
            Text anchoring position.
        fontfamily : str, optional
            The font family of the text.
        offset : tuple, default None
            The offset of the text from the data point.

        Returns
        -------
        Self
            Same layer with texts added.
        """
        if isinstance(strings, str):
            strings = [strings] * self.data.x.size
        if offset is None:
            _offset = self._text_offset
        else:
            _offset = NoOffset()._add(*offset)

        xdata, ydata = self.data
        dx, dy = _offset._asarray()
        self.texts.string = strings
        self.texts.set_pos(xdata + dx, ydata + dy)
        self.texts.update(
            color=color,
            size=size,
            rotation=rotation,
            anchor=anchor,
            family=fontfamily,
        )
        return self


class LabeledLine(_LabeledLayerBase):
    @property
    def line(self) -> Line:
        """The line layer."""
        return self._children[0]


class LabeledMarkers(_LabeledLayerBase):
    @property
    def markers(self) -> Markers:
        return self._children[0]


class LabeledBars(_LabeledLayerBase):
    @property
    def bars(self) -> Bars:
        """The bars layer."""
        return self._children[0]

    def _get_data_xy(self, layer: Bars | None = None) -> tuple[np.ndarray, np.ndarray]:
        if layer is None:
            layer = self.bars
        return layer.data.x, layer.top

    def _default_ordering(self, n: int) -> list[int]:
        assert n == 4
        return [0, 1, 2, 3]


class LabeledPlot(_LabeledLayerBase):
    @property
    def plot(self) -> Plot:
        """The plot (line + markers) layer."""
        return self._children[0]

    @property
    def line(self) -> Line:
        """The line layer."""
        return self.plot.line

    @property
    def markers(self) -> Markers:
        """The markers layer."""
        return self.plot.markers
