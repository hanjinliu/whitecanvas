from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from whitecanvas.types import ColorType, _Void, Symbol, LineStyle, Alignment
from whitecanvas.layers.primitive import Line, Markers, Bars, Errorbars
from whitecanvas.layers._base import XYData, PrimitiveLayer
from whitecanvas.layers.group._collections import ListLayerGroup
from whitecanvas.layers.group.text_group import TextGroup
from whitecanvas.utils.normalize import as_array_1d

_void = _Void()

if TYPE_CHECKING:
    from typing_extensions import Self
    from whitecanvas.layers.group.line_markers import Plot


class _TextOffset:
    def _asarray(self) -> tuple[Any, Any]:
        raise NotImplementedError

    def _add(self, dx, dy) -> _TextOffset:
        raise NotImplementedError


class NoOffset(_TextOffset):
    def _asarray(self):
        return (0, 0)

    def _add(self, dx, dy) -> _TextOffset:
        if dx == 0 and dy == 0:
            return self
        if np.isscalar(dx) and np.isscalar(dy):
            return ConstantOffset(dx, dy)
        else:
            return CustomOffset(as_array_1d(dx), as_array_1d(dy))

    def __repr__(self) -> str:
        return "<NoOffset>"


class ConstantOffset(_TextOffset):
    def __init__(self, x: float, y: float):
        self._x, self._y = x, y

    def _asarray(self) -> tuple[float, float]:
        return (self._x, self._y)

    def _add(self, dx, dy) -> _TextOffset:
        if np.isscalar(dx) and np.isscalar(dy):
            return ConstantOffset(self._x + dx, self._y + dy)
        else:
            return CustomOffset(as_array_1d(dx) + self._x, as_array_1d(dy) + self._y)

    def __repr__(self) -> str:
        return f"<ConstantOffset({self._x}, {self._y})>"


class CustomOffset(_TextOffset):
    def __init__(self, x: Any, y: Any):
        self._x, self._y = x, y

    def _asarray(self) -> tuple[Any, Any]:
        return (self._x, self._y)

    def _add(self, dx, dy) -> _TextOffset:
        return CustomOffset(as_array_1d(dx + self._x), as_array_1d(dy + self._y))

    def __repr__(self) -> str:
        return f"<CustomOffset({self._x}, {self._y})>"


class _AnnotatedLayerBase(ListLayerGroup):
    def __init__(
        self,
        layer: PrimitiveLayer,
        xerr: Errorbars,
        yerr: Errorbars,
        texts: TextGroup | None = None,
        name: str | None = None,
        offset: _TextOffset = NoOffset(),
    ):
        if texts is None:
            texts = TextGroup([])

        super().__init__([layer, xerr, yerr, texts], name=name)
        self._text_offset = offset

    def _default_ordering(self, n: int) -> list[int]:
        assert n == 4
        return [2, 0, 1, 3]

    def _set_data_to_first_layer(self, xdata=None, ydata=None):
        self._children[0].set_data(xdata, ydata)

    @property
    def xerr(self) -> Errorbars:
        """The errorbars layer for x."""
        return self._children[1]

    @property
    def yerr(self) -> Errorbars:
        """The errorbars layer for y."""
        return self._children[2]

    @property
    def texts(self) -> TextGroup:
        """The text group layer."""
        return self._children[3]

    @property
    def data(self) -> XYData:
        """The internal (x, y) data of this layer."""
        return self._children[0].data

    def set_data(self, xdata=None, ydata=None):
        """Set the (x, y) data of this layer."""
        data = self.data
        if xdata is None:
            dx = 0
        else:
            dx = xdata - data.x
        if ydata is None:
            dy = 0
        else:
            dy = ydata - data.y
        self._set_data_to_first_layer(xdata, ydata)
        if self.xerr.ndata > 0:
            y, x0, x1 = self.xerr.data
            self.xerr.set_data(y + dy, x0 + dx, x1 + dx)
        if self.yerr.ndata > 0:
            x, y0, y1 = self.yerr.data
            self.yerr.set_data(x + dx, y0 + dy, y1 + dy)
        if self.texts.ntexts > 0:
            dx, dy = self._text_offset._asarray()
            self.texts.set_pos(data.x + dx, data.y + dy)

    @property
    def text_offset(self) -> _TextOffset:
        """Return the text offset."""
        return self._text_offset

    def add_text_offset(self, dx: Any, dy: Any):
        """Add offset to text positions."""
        _offset = self._text_offset._add(dx, dy)
        if self.texts.ntexts > 0:
            data = self.data
            xoff, yoff = _offset._asarray()
            self.texts.set_pos(data.x + xoff, data.y + yoff)
        self._text_offset = _offset

    def setup_xerr(
        self,
        *,
        color: ColorType | _Void = _void,
        line_width: float | _Void = _void,
        line_style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float | _Void = _void,
    ) -> Self:
        self.xerr.setup(
            color=color, line_width=line_width, line_style=line_style,
            antialias=antialias, capsize=capsize
        )  # fmt: skip
        return self

    def setup_yerr(
        self,
        *,
        color: ColorType | _Void = _void,
        line_width: float | _Void = _void,
        line_style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float | _Void = _void,
    ) -> Self:
        self.yerr.setup(
            color=color, line_width=line_width, line_style=line_style,
            antialias=antialias, capsize=capsize
        )  # fmt: skip
        return self

    def setup_texts(
        self,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str = "sans-serif",
    ) -> Self:
        self.texts.setup(
            color=color, size=size, rotation=rotation, anchor=anchor,
            fontfamily=fontfamily
        )  # fmt: skip
        return self

    def with_xerr(
        self,
        len_lower: float,
        len_higher: float | None = None,
        *,
        color: ColorType = "black",
        line_width: float = 1.0,
        line_style: str | LineStyle = LineStyle.SOLID,
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
        return self.setup_xerr(
            color=color,
            line_width=line_width,
            line_style=line_style,
            antialias=antialias,
            capsize=capsize,
        )

    def with_yerr(
        self,
        len_lower: float,
        len_higher: float | None = None,
        *,
        color: ColorType = "black",
        line_width: float = 1.0,
        line_style: str | LineStyle = LineStyle.SOLID,
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
        return self.setup_yerr(
            color=color, line_width=line_width, line_style=line_style,
            antialias=antialias, capsize=capsize
        )  # fmt: skip

    def with_text(
        self,
        strings: list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str = "sans-serif",
        offset: tuple[Any, Any] | None = None,
    ) -> Self:
        """
        Add texts to the layer.

        Parameters
        ----------
        strings : str or list of str
            The text strings. If a single string is given, it will be used for all
            the data points.
        color : ColorType, default is "black"
            Text color.
        size : float, default 12
            Font point size of the text.
        rotation : float, default 0.0
            Rotation of the text in degrees.
        anchor : str or Alignment, default is Alignment.BOTTOM_LEFT
            Text anchoring position.
        fontfamily : str, default is "sans-serif"
            The font family of the text.
        offset : tuple, default is None
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
        texts = TextGroup.from_strings(
            xdata + dx, ydata + dy, strings, color=color, size=size, rotation=rotation,
            anchor=anchor, fontfamily=fontfamily, backend=self._backend_name,
        )  # fmt: skip
        return self.__class__(
            self._children[0],
            self.xerr,
            self.yerr,
            texts=texts,
            name=self.name,
            offset=_offset,
        )


class AnnotatedLine(_AnnotatedLayerBase):
    @property
    def line(self) -> Line:
        """The line layer."""
        return self._children[0]

    def setup_line(
        self,
        *,
        color: ColorType | _Void = _void,
        line_width: float | _Void = _void,
        line_style: str | _Void = _void,
        antialias: bool | _Void = _void,
    ):
        self.line.setup(
            color=color, line_width=line_width, style=line_style,
            antialias=antialias
        )  # fmt: skip
        return self


class AnnotatedMarkers(_AnnotatedLayerBase):
    @property
    def markers(self) -> Markers:
        return self._children[0]

    def setup_markers(
        self,
        *,
        symbol: Symbol | str | _Void = _void,
        size: float | _Void = _void,
        face_color: ColorType | _Void = _void,
        edge_color: ColorType | _Void = _void,
        edge_width: float | _Void = _void,
        edge_style: LineStyle | str | _Void = _void,
    ):
        self.markers.setup(
            symbol=symbol,
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=edge_width,
            edge_style=edge_style,
        )
        return self


class AnnotatedBars(_AnnotatedLayerBase):
    @property
    def bars(self) -> Bars:
        """The bars layer."""
        return self._children[0]

    @property
    def data(self) -> XYData:
        x, top, _ = self.bars.data
        return XYData(x, top)

    def _set_data_to_first_layer(self, xdata=None, ydata=None):
        _, _, bottom = self.bars.data
        self.bars.set_data(xdata, ydata, bottom)

    def setup_bars(
        self,
        *,
        face_color: ColorType | _Void = _void,
        edge_color: ColorType | _Void = _void,
        edge_width: float | _Void = _void,
        edge_style: LineStyle | str | _Void = _void,
    ):
        self.bars.setup(
            face_color=face_color,
            edge_color=edge_color,
            edge_width=edge_width,
            edge_style=edge_style,
        )
        return self


class AnnotatedPlot(_AnnotatedLayerBase):
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
