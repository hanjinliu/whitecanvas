from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, Iterable, TypeVar, overload

import numpy as np
from cmap import Colormap
from numpy.typing import NDArray
from typing_extensions import deprecated

from whitecanvas.backend import Backend
from whitecanvas.layers import _legend, _mixin, _text_utils
from whitecanvas.layers._base import PrimitiveLayer
from whitecanvas.layers._primitive import Bars, Errorbars, Image, Line, Markers, Texts
from whitecanvas.layers.group._cat_utils import check_array_input
from whitecanvas.layers.group._collections import LayerContainer, RichContainerEvents
from whitecanvas.layers.group._offsets import NoOffset, TextOffset
from whitecanvas.layers.group.colorbar import Colorbar
from whitecanvas.layers.group.line_markers import Plot
from whitecanvas.types import (
    Alignment,
    ArrayLike1D,
    ColormapType,
    ColorType,
    Hatch,
    LineStyle,
    Orientation,
    OrientationLike,
    Origin,
    Rect,
    XYData,
    _Void,
)
from whitecanvas.utils.normalize import as_any_1d_array, as_color_array

if TYPE_CHECKING:
    from typing_extensions import Self

_void = _Void()
_NFace = TypeVar("_NFace", bound="_mixin.FaceNamespace")
_NEdge = TypeVar("_NEdge", bound="_mixin.EdgeNamespace")
_Size = TypeVar("_Size")


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
        if self.texts.ndata > 0:
            dx, dy = self._text_offset._asarray()
            self.texts.set_pos(px + dx, py + dy)

    @property
    def text_offset(self) -> TextOffset:
        """Return the text offset."""
        return self._text_offset

    def with_text_offset(self, dx: Any, dy: Any):
        """Add offset to text positions."""
        _offset = self._text_offset._add(dx, dy)
        if self.texts.ndata > 0:
            px, py = self._get_data_xy()
            xoff, yoff = _offset._asarray()
            self.texts.set_pos(px + xoff, py + yoff)
        self._text_offset = _offset

    @deprecated("add_text_offset is deprecated. Please use with_text_offset instead.")
    def add_text_offset(self, *args, **kwargs):
        return self.with_text_offset(*args, **kwargs)

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
        if canvas := self._canvas_ref():
            canvas._autoscale_for_layer(self.xerr, maybe_empty=False)
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
        if canvas := self._canvas_ref():
            canvas._autoscale_for_layer(self.yerr, maybe_empty=False)
        return self

    def with_text(
        self,
        strings: str | list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        family: str | None = None,
        offset: tuple[Any, Any] | None = None,
    ) -> Self:
        """
        Add texts to the layer.

        Parameters
        ----------
        strings : str or list of str
            The text strings. If a single string is given, it will be used for all
            the data points. You can also use format strings with "{x}", "{y}", and
            "{i}" to format the text with the data point values and index.
        color : ColorType, default "black"
            Text color.
        size : float, default 12
            Font point size of the text.
        rotation : float, default 0.0
            Rotation of the text in degrees.
        anchor : str or Alignment, default Alignment.BOTTOM_LEFT
            Text anchoring position.
        family : str, optional
            The font family of the text.
        offset : tuple, default None
            The offset of the text from the data point.
        """
        strings = _text_utils.norm_label_text(strings, self.data)
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
            family=family,
        )
        return self


class LabeledLine(_LabeledLayerBase):
    @property
    def line(self) -> Line:
        """The line layer."""
        return self._children[0]

    def _as_legend_item(self) -> _legend.LineErrorLegendItem:
        line = self.line._as_legend_item()
        if self.xerr.nlines == 0:
            xerr = None
        else:
            xerr = self.xerr._as_legend_item()
        if self.yerr.nlines == 0:
            yerr = None
        else:
            yerr = self.yerr._as_legend_item()
        return _legend.LineErrorLegendItem(line, xerr, yerr)


class LabeledMarkers(_LabeledLayerBase, Generic[_NFace, _NEdge, _Size]):
    @property
    def markers(self) -> Markers[_NFace, _NEdge, _Size]:
        return self._children[0]

    def _as_legend_item(self) -> _legend.MarkerErrorLegendItem:
        markers = self.markers._as_legend_item()
        if self.xerr.nlines == 0:
            xerr = None
        else:
            xerr = self.xerr._as_legend_item()
        if self.yerr.nlines == 0:
            yerr = None
        else:
            yerr = self.yerr._as_legend_item()
        return _legend.MarkerErrorLegendItem(markers, xerr, yerr)


def _init_mean_sd(x, data, color):
    x, data = check_array_input(x, data)
    color = as_color_array(color, len(x))

    est_data = []
    err_data = []

    for sub_data in data:
        _mean = np.mean(sub_data)
        _sd = np.std(sub_data, ddof=1)
        est_data.append(_mean)
        err_data.append(_sd)

    est_data = np.array(est_data)
    err_data = np.array(err_data)
    return x, est_data, err_data


def _init_error_bars(
    x,
    est,
    err,
    orient,
    capsize,
    backend,
) -> tuple[Errorbars, Errorbars]:
    ori = Orientation.parse(orient)
    errorbar = Errorbars(
        x, est - err, est + err, orient=ori, backend=backend, capsize=capsize,
    )  # fmt: skip
    if ori.is_vertical:
        xerr = Errorbars.empty_h(backend=backend)
        yerr = errorbar
    else:
        xerr = errorbar
        yerr = Errorbars.empty_v(backend=backend)
    return xerr, yerr


class LabeledBars(
    _LabeledLayerBase,
    _mixin.AbstractFaceEdgeMixin["PlotFace", "PlotEdge"],
    Generic[_NFace, _NEdge],
):
    _ATTACH_TO_AXIS = True
    events: RichContainerEvents
    _events_class = RichContainerEvents

    def __init__(
        self,
        layer: Bars[_NFace, _NEdge],
        xerr: Errorbars,
        yerr: Errorbars,
        texts: Texts | None = None,
        name: str | None = None,
        offset: TextOffset | None = None,
    ):
        _LabeledLayerBase.__init__(self, layer, xerr, yerr, texts, name, offset)
        _mixin.AbstractFaceEdgeMixin.__init__(self, PlotFace(self), PlotEdge(self))
        self._init_events()

    @property
    def bars(self) -> Bars[_NFace, _NEdge]:
        """The bars layer."""
        return self._children[0]

    @property
    def orient(self) -> Orientation:
        """The orientation of the bars."""
        return self.bars.orient

    def _main_object_layer(self):
        return self.bars

    def _get_data_xy(self, layer: Bars | None = None) -> tuple[np.ndarray, np.ndarray]:
        if layer is None:
            layer = self.bars
        return layer.data.x, layer.top

    def _default_ordering(self, n: int) -> list[int]:
        assert n == 4
        return [0, 1, 2, 3]

    @classmethod
    def from_arrays(
        cls,
        x: list[float],
        data: list[ArrayLike1D],
        *,
        name: str | None = None,
        orient: OrientationLike = "vertical",
        capsize: float = 0.15,
        color: ColorType | list[ColorType] = "blue",
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
        extent: float = 0.8,
        backend: str | Backend | None = None,
    ) -> LabeledBars[_mixin.MultiFace, _mixin.MonoEdge]:
        x, height, err_data = _init_mean_sd(x, data, color)
        bars = Bars(x, height, extent=extent, backend=backend).with_face_multi(
            color=color, hatch=hatch, alpha=alpha
        )
        xerr, yerr = _init_error_bars(x, height, err_data, orient, capsize, backend)
        return cls(bars, xerr=xerr, yerr=yerr, name=name)

    def _as_legend_item(self):
        return self.bars._as_legend_item()


class LabeledPlot(
    _LabeledLayerBase,
    _mixin.AbstractFaceEdgeMixin["PlotFace", "PlotEdge"],
    Generic[_NFace, _NEdge, _Size],
):
    evens: RichContainerEvents
    _events_class = RichContainerEvents

    def __init__(
        self,
        layer: Plot,
        xerr: Errorbars,
        yerr: Errorbars,
        texts: Texts | None = None,
        name: str | None = None,
        offset: TextOffset | None = None,
    ):
        _LabeledLayerBase.__init__(self, layer, xerr, yerr, texts, name, offset)
        _mixin.AbstractFaceEdgeMixin.__init__(self, PlotFace(self), PlotEdge(self))
        self._init_events()

    @property
    def plot(self) -> Plot:
        """The plot (line + markers) layer."""
        return self._children[0]

    @property
    def line(self) -> Line:
        """The line layer."""
        return self.plot.line

    @property
    def markers(self) -> Markers[_NFace, _NEdge, _Size]:
        """The markers layer."""
        return self.plot.markers

    def _main_object_layer(self):
        """The main layer with face that will be used in PlotFace/PlotEdge."""
        return self.markers

    @classmethod
    def from_arrays(
        cls,
        x: list[float],
        data: list[ArrayLike1D],
        *,
        name: str | None = None,
        orient: OrientationLike = "vertical",
        capsize: float = 0.15,
        color: ColorType | list[ColorType] = "blue",
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
        backend: str | Backend | None = None,
    ) -> LabeledPlot[_mixin.MultiFace, _mixin.MultiEdge, float]:
        x, y, err_data = _init_mean_sd(x, data, color)
        xerr, yerr = _init_error_bars(x, y, err_data, orient, capsize, backend)
        if not orient.is_vertical:
            x, y = y, x
        markers = Markers(
            x, y, backend=backend,
        ).with_face_multi(
            color=color, hatch=hatch, alpha=alpha,
        )  # fmt: skip
        lines = Line(x, y, backend=backend)
        plot = Plot(lines, markers)
        lines.visible = False
        return cls(plot, xerr=xerr, yerr=yerr, name=name)

    @classmethod
    def from_arrays_2d(
        cls,
        xdata: list[ArrayLike1D],
        ydata: list[ArrayLike1D],
        *,
        name: str | None = None,
        capsize: float = 0.15,
        color: ColorType | list[ColorType] = "blue",
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
        backend: str | Backend | None = None,
    ) -> LabeledPlot[_mixin.MultiFace, _mixin.MultiEdge, float]:
        def _estimate(arrs: list[NDArray[np.number]]):
            _mean = []
            _sd = []
            for arr in arrs:
                _mean.append(np.mean(arr))
                _sd.append(np.std(arr, ddof=1))
            return np.array(_mean), np.array(_sd)

        xmean, xsd = _estimate(xdata)
        ymean, ysd = _estimate(ydata)
        markers = Markers(
            xmean,
            ymean,
            backend=backend,
        ).with_face_multi(
            color=color,
            hatch=hatch,
            alpha=alpha,
        )
        lines = Line(xmean, ymean, backend=backend)
        plot = Plot(lines, markers)
        lines.visible = False
        xerr = Errorbars(
            ymean,
            xmean - xsd,
            xmean + xsd,
            orient=Orientation.HORIZONTAL,
            capsize=capsize,
            backend=backend,
        )
        yerr = Errorbars(
            xmean,
            ymean - ysd,
            ymean + ysd,
            orient=Orientation.VERTICAL,
            capsize=capsize,
            backend=backend,
        )
        return cls(plot, xerr=xerr, yerr=yerr, name=name)

    def _as_legend_item(self) -> _legend.LegendItem:
        if self.xerr.nlines == 0:
            xerr = None
        else:
            xerr = self.xerr._as_legend_item()
        if self.yerr.nlines == 0:
            yerr = None
        else:
            yerr = self.yerr._as_legend_item()
        return _legend.PlotErrorLegendItem(self.plot._as_legend_item(), xerr, yerr)


class PlotFace(_mixin.MultiPropertyFaceBase):
    _layer: LabeledPlot[_mixin.MultiFace, _mixin.MultiEdge, float]

    @property
    def color(self) -> NDArray[np.floating]:
        """Face color of the bar."""
        return self._layer._main_object_layer().face.color

    @color.setter
    def color(self, color):
        ndata = self._layer._main_object_layer().ndata
        col = as_color_array(color, ndata)
        self._layer._main_object_layer().with_face_multi(color=col)
        self.events.color.emit(col)

    @property
    def hatch(self) -> _mixin.EnumArray[Hatch]:
        """Face fill hatch."""
        return self._layer._main_object_layer().face.hatch

    @hatch.setter
    def hatch(self, hatch: str | Hatch | Iterable[str | Hatch]):
        ndata = self._layer._main_object_layer().ndata
        hatches = as_any_1d_array(hatch, ndata, dtype=object)
        self._layer._main_object_layer().with_face_multi(hatch=hatches)
        self.events.hatch.emit(hatches)

    @property
    def alpha(self) -> float:
        """Alpha value of the face."""
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, value):
        color = self.color.copy()
        color[:, 3] = value
        self.color = color


class PlotEdge(_mixin.MultiPropertyEdgeBase):
    _layer: LabeledPlot[_NFace, _NEdge, float] | LabeledBars[_NFace, _NEdge]

    @property
    def color(self) -> NDArray[np.floating]:
        """Edge color of the plot."""
        return self._layer._main_object_layer().edge.color

    @color.setter
    def color(self, color: ColorType):
        self._layer._main_object_layer().with_edge_multi(color=color)
        if self._layer.xerr.ndata > 0:
            self._layer.xerr.color = color
        if self._layer.yerr.ndata > 0:
            self._layer.yerr.color = color
        self.events.color.emit(color)

    @property
    def width(self) -> NDArray[np.float32]:
        """Edge widths."""
        return self._layer._main_object_layer().edge.width

    @width.setter
    def width(self, width: float):
        self._layer._main_object_layer().edge.width = width
        if self._layer.xerr.ndata > 0:
            self._layer.xerr.width = width
        if self._layer.yerr.ndata > 0:
            self._layer.yerr.width = width
        self.events.width.emit(width)

    @property
    def style(self) -> _mixin.EnumArray[LineStyle]:
        """Edge styles."""
        return self._layer._main_object_layer().edge.style

    @style.setter
    def style(self, style: str | LineStyle):
        style = LineStyle(style)
        self._layer._main_object_layer().edge.style = style
        if self._layer.xerr.ndata > 0:
            self._layer.xerr.style = style
        if self._layer.yerr.ndata > 0:
            self._layer.yerr.style = style
        self.events.style.emit(style)

    @property
    def alpha(self) -> float:
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, value):
        color = self.color.copy()
        color[:, 3] = value
        self.color = color


class LabeledImage(LayerContainer):
    """
    Layer group for an image with texts and colorbar.
    """

    def __init__(
        self,
        layer: Image,
        texts: Texts | None = None,
        colorbar: Colorbar | None = None,
        name: str | None = None,
    ):
        if texts is None:
            texts = Texts(
                [], [], [], name="texts", backend=layer._backend_name
            ).with_font_multi()
        if colorbar is None:
            colorbar = Colorbar(layer.cmap)
            colorbar.visible = False
        layer.events.cmap.connect_setattr(colorbar, "cmap", maxargs=1)
        super().__init__([layer, texts, colorbar], name=name)

    @property
    def image(self) -> Image:
        """The image layer."""
        return self._children[0]

    @property
    def texts(self) -> Texts:
        """The text layer for the overlay texts."""
        return self._children[1]

    @property
    def colorbar(self) -> Colorbar:
        """The colorbar layer."""
        return self._children[2]

    @property
    def cmap(self) -> Colormap:
        """Current colormap."""
        return self.image.cmap

    @cmap.setter
    def cmap(self, cmap: ColormapType):
        self.image.cmap = cmap

    @property
    def clim(self) -> tuple[float, float]:
        """Current contrast limits."""
        return self.image.clim

    @clim.setter
    def clim(self, clim: tuple[float | None, float | None] | None):
        self.image.clim = clim

    @property
    def shape(self) -> tuple[int, int]:
        """The visual shape of the image (shape without the color axis)."""
        return self.image.shape

    @property
    def data(self) -> NDArray[np.number]:
        """The colored image data (N, M, 4)."""
        return self.image.data

    @data.setter
    def data(self, data: NDArray[np.number]):
        self.image.data = data

    @property
    def data_mapped(self) -> NDArray[np.number]:
        """The colored image data (N, M, 4) mapped by the colormap."""
        return self.image.data_mapped

    @property
    def shift(self) -> tuple[float, float]:
        """Current shift from the origin."""
        return self.image.shift

    @shift.setter
    def shift(self, shift: tuple[float, float]):
        self.image.shift = shift

    @property
    def shift_raw(self) -> tuple[float, float]:
        """Current shift from the origin as a raw data."""
        return self.image.shift

    @property
    def scale(self) -> tuple[float, float]:
        """Current scale."""
        return self.image.scale

    @scale.setter
    def scale(self, scale: float | tuple[float, float]):
        self.image.scale = scale

    @property
    def origin(self) -> Origin:
        """Current origin of the image."""
        return self.image.origin

    @origin.setter
    def origin(self, origin: Origin | str):
        self.image.origin = origin

    @overload
    def fit_to(self, bbox: Rect | tuple[float, float, float, float], /) -> Image:
        ...

    @overload
    def fit_to(self, left: float, right: float, bottom: float, top: float, /) -> Image:
        ...

    def fit_to(self, *args) -> Image:
        """Fit the image to the given bounding box."""
        self.image.fit_to(*args)

    def with_text(
        self,
        *,
        size: int = 8,
        color_rule: ColorType | Callable[[np.ndarray], ColorType] | None = None,
        fmt: str = "",
        text_invalid: str | None = None,
        mask: NDArray[np.bool_] | None = None,
    ) -> LabeledImage:
        """
        Add text annotation to each pixel of the image.

        Parameters
        ----------
        size : int, default 8
            Font size of the text.
        color_rule : color-like, callable, optional
            Rule to define the color for each text based on the color-mapped image
            intensity.
        fmt : str, optional
            Format string for the text.
        mask : array-like, optional
            Mask to specify the valid pixels for the text annotation.
        """
        texts = self.image._make_text_layer(
            size=size, color_rule=color_rule, fmt=fmt,
            text_invalid=text_invalid, mask=mask,
        )  # fmt: skip
        self.texts.data = texts.data
        self.texts.update(color=texts.color, size=texts.size, anchor=texts.anchor)
        return self

    def with_colorbar(
        self,
        bbox: Rect | None = None,
        *,
        orient: OrientationLike = "vertical",
    ) -> LabeledImage:
        """
        Add a colorbar to the image.

        Parameters
        ----------
        bbox : four float, optional
            Bounding box of the colorbar. If `None`, the colorbar will be placed at the
            bottom-right corner of the image.
        orient : str or Orientation, default "vertical"
            Orientation of the colorbar.
        """
        cbar = self.image._make_colorbar(bbox, orient=orient)
        self.colorbar.visible = True
        self.colorbar.fit_to(cbar.lut.bbox)
        return self
