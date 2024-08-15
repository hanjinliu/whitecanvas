from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray
from psygnal import Signal

from whitecanvas import theme
from whitecanvas.backend import Backend
from whitecanvas.layers import _legend, _text_utils
from whitecanvas.layers._base import (
    HoverableDataBoundLayer,
    LayerEvents,
    PrimitiveLayer,
)
from whitecanvas.layers._mixin import EnumArray
from whitecanvas.layers._primitive.text import Texts
from whitecanvas.layers._sizehint import xy_size_hint
from whitecanvas.protocols import LineProtocol, MultiLineProtocol
from whitecanvas.types import (
    Alignment,
    ArrayLike1D,
    ColorType,
    Hatch,
    LineStyle,
    Orientation,
    OrientationLike,
    StepStyle,
    Symbol,
    XYData,
    _Void,
)
from whitecanvas.utils.normalize import (
    arr_color,
    as_array_1d,
    as_color_array,
    normalize_xy,
    parse_texts,
)
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.layers import group as _lg

_void = _Void()
_Line = TypeVar("_Line", bound=LineProtocol)


class LineLayerEvents(LayerEvents):
    color = Signal(np.ndarray)
    width = Signal(float)
    style = Signal(str)
    antialias = Signal(bool)
    clicked = Signal(int)


class LineMixin(PrimitiveLayer[_Line]):
    events: LineLayerEvents
    _events_class = LineLayerEvents

    @property
    def color(self) -> NDArray[np.floating]:
        """Color of the line."""
        return self._backend._plt_get_edge_color()

    @color.setter
    def color(self, color: ColorType):
        col = arr_color(color)
        self._backend._plt_set_edge_color(col)
        self.events.color.emit(col)

    @property
    def width(self) -> float:
        """Width of the line."""
        return self._backend._plt_get_edge_width()

    @width.setter
    def width(self, width: float):
        if not is_real_number(width):
            raise TypeError(f"Width must be a number, got {type(width)}")
        if width < 0:
            raise ValueError(f"Width must be non-negative, got {width!r}")
        w = float(width)
        self._backend._plt_set_edge_width(w)
        self.events.width.emit(w)

    @property
    def style(self) -> LineStyle:
        """Style of the line."""
        return LineStyle(self._backend._plt_get_edge_style())

    @style.setter
    def style(self, style: str | LineStyle):
        s = LineStyle(style)
        self._backend._plt_set_edge_style(s)
        self.events.style.emit(s.value)

    @property
    def alpha(self) -> float:
        return float(self.color[3])

    @alpha.setter
    def alpha(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {value!r}")
        self.color = (*self.color[:3], value)

    @property
    def antialias(self) -> bool:
        """Whether to use antialiasing."""
        return self._backend._plt_get_antialias()

    @antialias.setter
    def antialias(self, antialias: bool) -> None:
        if not isinstance(antialias, bool):
            raise TypeError(f"Expected antialias to be bool, got {type(antialias)}")
        self._backend._plt_set_antialias(antialias)
        self.events.antialias.emit(antialias)

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        alpha: float | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
    ):
        if color is not _void:
            self.color = color
        if width is not _void:
            self.width = width
        if style is not _void:
            self.style = style
        if alpha is not _void:
            self.alpha = alpha
        if antialias is not _void:
            self.antialias = antialias
        return self

    def _as_legend_item(self) -> _legend.LineLegendItem:
        return _legend.LineLegendItem(self.color, self.width, self.style)


class _SingleLine(
    LineMixin[LineProtocol], HoverableDataBoundLayer[LineProtocol, XYData]
):
    _backend_class_name = "MonoLine"
    events: LineLayerEvents
    _events_class = LineLayerEvents

    def _norm_layer_data(self, data: Any) -> XYData:
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError(f"Expected data to be (N, 2), got {data.shape}")
            xdata, ydata = data[:, 0], data[:, 1]
        else:
            xdata, ydata = data
            if xdata is None:
                xdata = self.data.x
            else:
                xdata = as_array_1d(xdata)
            if ydata is None:
                ydata = self.data.y
            else:
                ydata = as_array_1d(ydata)
        if xdata.size != ydata.size:
            raise ValueError(
                "Expected xdata and ydata to have the same size, "
                f"got {xdata.size} and {ydata.size}"
            )
        return XYData(xdata, ydata)

    def set_data(
        self,
        xdata: ArrayLike1D | None = None,
        ydata: ArrayLike1D | None = None,
    ):
        """Set x and y data of the line."""
        self.data = xdata, ydata

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return self.data.x.size

    def _data_to_backend_data(self, data: XYData) -> XYData:
        return data

    def with_markers(
        self,
        symbol: Symbol | str = Symbol.CIRCLE,
        size: float | None = None,
        color: ColorType | _Void = _void,
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
    ) -> _lg.Plot[Self]:
        """
        Add markers at each data point.

        >>> canvas.add_line(x, y).with_markers(color="yellow")

        Parameters
        ----------
        symbol : str or Symbol, default Symbol.CIRCLE
            Marker symbols.
        size : float, optional
            Marker size. Use theme default if not given.
        color : color-like, optional
            Marker face colors. Set to the line color by default.
        alpha : float, default 1.0
            The alpha channel value of the markers.
        hatch : str or FacePattern, default FacePattern.SOLID
            The marker face hatch.

        Returns
        -------
        Plot
            The plot layer.
        """
        from whitecanvas.layers._primitive import Markers
        from whitecanvas.layers.group import Plot

        if color is _void:
            color = self.color
        size = theme._default("markers.size", size)
        markers = Markers(
            *self.data, symbol=symbol, size=size, color=color, alpha=alpha,
            hatch=hatch, name=f"markers-of-{self.name}", backend=self._backend_name,
        )  # fmt: skip
        old_name = self.name
        self.name = f"line-of-{self.name}"
        return Plot(self, markers, name=old_name)

    def with_xerr(
        self,
        err: ArrayLike1D,
        err_high: ArrayLike1D | None = None,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float = 0,
    ) -> _lg.LabeledLine[Self]:
        from whitecanvas.layers._primitive import Errorbars
        from whitecanvas.layers.group import LabeledLine

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.color
        if width is _void:
            width = self.width
        if style is _void:
            style = self.style
        if antialias is _void:
            antialias = self.antialias
        xerr = Errorbars(
            self.data.y, self.data.x - err, self.data.x + err_high, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
            name=f"xerr-of-{self.name}", orient=Orientation.HORIZONTAL,
            backend=self._backend_name
        )  # fmt: skip
        yerr = Errorbars.empty_v(f"yerr-of-{self.name}", backend=self._backend_name)
        old_name = self.name
        self.name = f"line-of-{self.name}"
        return LabeledLine(self, xerr, yerr, name=old_name)

    def with_yerr(
        self,
        err: ArrayLike1D,
        err_high: ArrayLike1D | None = None,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float = 0,
    ) -> _lg.LabeledLine[Self]:
        from whitecanvas.layers._primitive import Errorbars
        from whitecanvas.layers.group import LabeledLine

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.color
        if width is _void:
            width = self.width
        if style is _void:
            style = self.style
        if antialias is _void:
            antialias = self.antialias
        yerr = Errorbars(
            self.data.x, self.data.y - err, self.data.y + err_high, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
            name=f"yerr-of-{self.name}", backend=self._backend_name
        )  # fmt: skip
        xerr = Errorbars.empty_h(f"xerr-of-{self.name}", backend=self._backend_name)
        old_name = self.name
        self.name = f"line-of-{self.name}"
        return LabeledLine(self, xerr, yerr, name=old_name)

    def with_xband(
        self,
        err: ArrayLike1D,
        err_high: ArrayLike1D | None = None,
        *,
        color: ColorType | _Void = _void,
        alpha: float = 0.3,
        hatch: str | Hatch = Hatch.SOLID,
    ) -> _lg.LineBand[Self]:
        from whitecanvas.layers._primitive import Band
        from whitecanvas.layers.group import LineBand

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.color
        data = self.data
        _x, _y0 = self._data_to_backend_data(XYData(data.y, data.x - err))
        _, _y1 = self._data_to_backend_data(XYData(data.y, data.x + err_high))
        band = Band(
            _x, _y0, _y1, orient=Orientation.HORIZONTAL, color=color, alpha=alpha,
            hatch=hatch, name=f"xband-of-{self.name}", backend=self._backend_name,
        )  # fmt: skip
        old_name = self.name
        self.name = f"line-of-{self.name}"
        return LineBand(self, band, name=old_name)

    def with_yband(
        self,
        err: ArrayLike1D,
        err_high: ArrayLike1D | None = None,
        *,
        color: ColorType | _Void = _void,
        alpha: float = 0.3,
        hatch: str | Hatch = Hatch.SOLID,
    ) -> _lg.LineBand[Self]:
        from whitecanvas.layers._primitive import Band
        from whitecanvas.layers.group import LineBand

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.color
        data = self.data
        _x, _y0 = self._data_to_backend_data(XYData(data.x, data.y - err))
        _, _y1 = self._data_to_backend_data(XYData(data.x, data.y + err_high))
        band = Band(
            _x, _y0, _y1, orient=Orientation.VERTICAL, color=color, alpha=alpha,
            hatch=hatch, name=f"yband-of-{self.name}", backend=self._backend_name,
        )  # fmt: skip
        old_name = self.name
        self.name = f"line-of-{self.name}"
        return LineBand(self, band, name=old_name)

    def with_xfill(
        self,
        bottom: float = 0.0,
        *,
        color: ColorType | _Void = _void,
        alpha: float = 0.3,
        hatch: str | Hatch = Hatch.SOLID,
    ) -> _lg.LineBand[Self]:
        from whitecanvas.layers._primitive import Band
        from whitecanvas.layers.group import LineBand

        if color is _void:
            color = self.color
        data = self.data
        x0 = np.full((data.x.size,), bottom)
        band = Band(
            data.y, x0, data.x, orient=Orientation.HORIZONTAL, color=color, alpha=alpha,
            hatch=hatch, name=f"xfill-of-{self.name}", backend=self._backend_name,
        )  # fmt: skip
        old_name = self.name
        self.name = f"line-of-{self.name}"
        return LineBand(self, band, name=old_name)

    def with_yfill(
        self,
        bottom: float = 0.0,
        *,
        color: ColorType | _Void = _void,
        alpha: float = 0.3,
        hatch: str | Hatch = Hatch.SOLID,
    ) -> _lg.LineBand[Self]:
        from whitecanvas.layers._primitive import Band
        from whitecanvas.layers.group import LineBand

        if color is _void:
            color = self.color
        data = self.data
        y0 = np.full((data.y.size,), bottom)
        band = Band(
            data.x, y0, data.y, orient=Orientation.VERTICAL, color=color, alpha=alpha,
            hatch=hatch, name=f"yfill-of-{self.name}", backend=self._backend_name,
        )  # fmt: skip
        old_name = self.name
        self.name = f"line-of-{self.name}"
        return LineBand(self, band, name=old_name)


class Line(_SingleLine):
    _backend_class_name = "MonoLine"
    events: LineLayerEvents
    _events_class = LineLayerEvents

    def __init__(
        self,
        xdata: ArrayLike1D,
        ydata: ArrayLike1D,
        *,
        name: str | None = None,
        color: ColorType = "blue",
        width: float = 1.0,
        alpha: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        backend: Backend | str | None = None,
    ):
        xdata, ydata = normalize_xy(xdata, ydata)
        super().__init__(name=name)
        self._backend = self._create_backend(Backend(backend), xdata, ydata)
        self.update(
            color=color, width=width, style=style, alpha=alpha, antialias=antialias
        )
        self._x_hint, self._y_hint = xy_size_hint(xdata, ydata)
        self._backend._plt_connect_pick_event(self.events.clicked.emit)

    def _data_to_backend_data(self, data: XYData) -> XYData:
        return data

    def _get_layer_data(self) -> XYData:
        return XYData(*self._backend._plt_get_data())

    def _set_layer_data(self, data: XYData):
        x0, y0 = data
        self._backend._plt_set_data(x0, y0)
        self._x_hint, self._y_hint = xy_size_hint(x0, y0)

    def with_text(
        self,
        strings: list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        family: str | None = None,
    ) -> _lg.LabeledLine:
        from whitecanvas.layers import Errorbars
        from whitecanvas.layers.group import LabeledLine

        strings = _text_utils.norm_label_text(strings, self.data)
        texts = Texts(
            *self.data, strings, color=color, size=size, rotation=rotation,
            anchor=anchor, family=family, backend=self._backend_name,
        )  # fmt: skip
        old_name = self.name
        self.name = f"line-of-{self.name}"
        return LabeledLine(
            self,
            Errorbars.empty_h(name=f"xerr-of-{self.name}", backend=self._backend_name),
            Errorbars.empty_v(name=f"yerr-of-{self.name}", backend=self._backend_name),
            texts=texts,
            name=old_name,
        )

    def with_hover_template(
        self,
        template: str,
        extra: Any | None = None,
    ) -> Self:
        """Add hover template to the markers."""
        xs, ys = self.data
        if self._backend_name in ("plotly", "bokeh"):  # conversion for HTML
            template = template.replace("\n", "<br>")
        params = parse_texts(template, xs.size, extra)
        # set default format keys
        params.setdefault("x", xs)
        params.setdefault("y", ys)
        if "i" not in params:
            params["i"] = np.arange(xs.size)
        texts = [
            template.format(**{k: v[i] for k, v in params.items()})
            for i in range(xs.size)
        ]
        self._backend._plt_set_hover_text(texts)
        return self

    @classmethod
    def build_cdf(
        cls,
        data: ArrayLike1D,
        *,
        sorted: bool = False,
        orient: OrientationLike = "vertical",
        name: str | None = None,
        color: ColorType = "blue",
        alpha: float = 1.0,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        backend: Backend | str | None = None,
    ):
        """Construct a line from a cumulative histogram."""
        xdata = as_array_1d(data)
        if not sorted:
            xdata = np.sort(xdata)
        ydata = np.linspace(0, 1, xdata.size)
        if not Orientation.parse(orient).is_vertical:
            xdata, ydata = ydata, xdata
        return Line(
            xdata, ydata, name=name, color=color, alpha=alpha, antialias=antialias,
            width=width, style=style, backend=backend,
        )  # fmt: skip


class LineStep(_SingleLine):
    _backend_class_name = "MonoLine"
    events: LineLayerEvents
    _events_class = LineLayerEvents

    def __init__(
        self,
        xdata: ArrayLike1D,
        ydata: ArrayLike1D,
        *,
        name: str | None = None,
        where: str | StepStyle = StepStyle.MID,
        color: ColorType = "blue",
        width: float = 1,
        alpha: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        backend: Backend | str | None = None,
    ):
        xdata, ydata = normalize_xy(xdata, ydata)
        self._where = StepStyle(where)
        super().__init__(name=name)
        xback, yback = self._data_to_backend_data(XYData(xdata, ydata))
        self._backend = self._create_backend(Backend(backend), xback, yback)
        self.update(
            color=color, width=width, style=style, alpha=alpha, antialias=antialias
        )
        self._x_hint, self._y_hint = xy_size_hint(xback, yback)
        self._backend._plt_connect_pick_event(self.events.clicked.emit)

    def _data_to_backend_data(self, data: XYData) -> XYData:
        if data.x.size < 2:
            return data
        if self._where is StepStyle.PRE or self._where is StepStyle.POST_T:
            xdata = np.repeat(data.x, 2)[:-1]
            ydata = np.repeat(data.y, 2)[1:]
        elif self._where is StepStyle.POST or self._where is StepStyle.PRE_T:
            xdata = np.repeat(data.x, 2)[1:]
            ydata = np.repeat(data.y, 2)[:-1]
        elif self._where is StepStyle.MID:
            xrep = np.repeat((data.x[1:] + data.x[:-1]) / 2, 2)
            xdata = np.concatenate([data.x[:1], xrep, data.x[-1:]])
            ydata = np.repeat(data.y, 2)
        elif self._where is StepStyle.MID_T:
            yrep = np.repeat((data.y[1:] + data.y[:-1]) / 2, 2)
            xdata = np.repeat(data.x, 2)
            ydata = np.concatenate([data.y[:1], yrep, data.y[-1:]])
        else:  # pragma: no cover
            raise ValueError(f"Invalid step style: {self._where}")
        return XYData(xdata, ydata)

    def _get_layer_data(self) -> XYData:
        xback, yback = self._backend._plt_get_data()
        if (
            self._where is StepStyle.PRE
            or self._where is StepStyle.POST_T
            or self._where is StepStyle.POST
            or self._where is StepStyle.PRE_T
        ):
            xdata = xback[::2]
            ydata = yback[::2]
        elif self._where is StepStyle.MID:
            xmids = xback[1:-1:2]
            xmid = (xmids[1:] + xmids[:-1]) / 2
            xdata = np.concatenate([xback[:1], xmid, xback[-1:]])
            ydata = yback[1::2]
        elif self._where is StepStyle.MID_T:
            ymids = yback[1:-1:2]
            ymid = (ymids[1:] + ymids[:-1]) / 2
            xdata = xback[::2]
            ydata = np.concatenate([yback[:1], ymid, yback[-1:]])
        else:  # pragma: no cover
            raise ValueError(f"Invalid step style: {self._where}")
        return XYData(xdata, ydata)

    def _norm_layer_data(self, data: Any) -> XYData:
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError(f"Expected data to be (N, 2), got {data.shape}")
            xdata, ydata = data[:, 0], data[:, 1]
        else:
            xdata, ydata = data
            if xdata is None:
                xdata = self.data.x
            else:
                xdata = as_array_1d(xdata)
            if ydata is None:
                ydata = self.data.y
            else:
                ydata = as_array_1d(ydata)
        if xdata.size != ydata.size:
            raise ValueError(
                "Expected xdata and ydata to have the same size, "
                f"got {xdata.size} and {ydata.size}"
            )
        return XYData(xdata, ydata)

    def _set_layer_data(self, data: XYData):
        x0, y0 = self._data_to_backend_data(data)
        self._backend._plt_set_data(x0, y0)
        self._x_hint, self._y_hint = xy_size_hint(x0, y0)

    @property
    def where(self) -> StepStyle:
        return self._where

    @where.setter
    def where(self, where: str | StepStyle):
        data = self.data
        self._where = StepStyle(where)
        self._set_layer_data(data)


class MultiLineEvents(LayerEvents):
    color = Signal(np.ndarray)
    width = Signal(float)
    style = Signal(str)
    antialias = Signal(bool)
    clicked = Signal(int)


class MultiLine(HoverableDataBoundLayer[MultiLineProtocol, "list[NDArray[np.number]]"]):
    events: MultiLineEvents
    _events_class = MultiLineEvents

    def __init__(
        self,
        data: list[ArrayLike1D],
        *,
        name: str | None = None,
        color: ColorType = "blue",
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        antialias: bool = True,
        alpha: float = 1.0,
        backend: Backend | str | None = None,
    ):
        data_normed, xhint, yhint = _norm_data(data)
        super().__init__(name=name)
        self._x_hint, self._y_hint = xhint, yhint
        self._backend = self._create_backend(Backend(backend), data_normed)
        self.update(
            color=color, width=width, style=style, alpha=alpha, antialias=antialias
        )
        self._backend._plt_connect_pick_event(self.events.clicked.emit)

    def _get_layer_data(self) -> list[NDArray[np.number]]:
        """Current data of the layer."""
        return self._backend._plt_get_data()

    def _set_layer_data(self, data: list[NDArray[np.number]]):
        data_norm, x_hint, y_hint = _norm_data(data)
        self._backend._plt_set_data(data)
        self._backend._plt_set_data(data_norm)
        self._x_hint, self._y_hint = x_hint, y_hint

    @property
    def nlines(self) -> int:
        """Number of lines."""
        return len(self._backend._plt_get_data())

    ndata = nlines

    @property
    def color(self) -> NDArray[np.floating]:
        """Color of the line."""
        if self.nlines == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return self._backend._plt_get_edge_color()

    @color.setter
    def color(self, color: ColorType):
        col = as_color_array(color, self.nlines)
        self._backend._plt_set_edge_color(col)
        self.events.color.emit(col)

    @property
    def width(self) -> float:
        """Width of the line."""
        if self.nlines == 0:
            return np.zeros(0, dtype=np.float32)
        return self._backend._plt_get_edge_width()

    @width.setter
    def width(self, width: float | Sequence[float]):
        self._backend._plt_set_edge_width(width)
        self.events.width.emit(width)

    @property
    def style(self) -> EnumArray[LineStyle]:
        """Style of the line."""
        if self.nlines == 0:
            return np.zeros(0, dtype=object)
        return np.array(self._backend._plt_get_edge_style(), dtype=object)

    @style.setter
    def style(self, style: str | LineStyle | Sequence[str | LineStyle]):
        if isinstance(style, (str, LineStyle)):
            s = LineStyle(style)
            self._backend._plt_set_edge_style(s)
            self.events.style.emit(s)
        else:
            styles = [LineStyle(s) for s in style]
            self._backend._plt_set_edge_style(styles)
            self.events.style.emit(styles)

    @property
    def alpha(self) -> NDArray[np.float32]:
        """Alpha channel of each line."""
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, value):
        if self.nlines == 0:
            return
        col = self.color.copy()
        col[:, 3] = value
        self.color = col

    @property
    def antialias(self) -> bool:
        """Whether to use antialiasing."""
        return self._backend._plt_get_antialias()

    @antialias.setter
    def antialias(self, antialias: bool) -> None:
        if not isinstance(antialias, bool):
            raise TypeError(f"Expected antialias to be bool, got {type(antialias)}")
        self._backend._plt_set_antialias(antialias)
        self.events.antialias.emit(antialias)

    def update(
        self,
        *,
        color: ColorType | Sequence[ColorType] | _Void = _void,
        width: float | Sequence[float] | _Void = _void,
        style: str | LineStyle | Sequence[str | LineStyle] | _Void = _void,
        alpha: float | Sequence[float] | _Void = _void,
        antialias: bool | _Void = _void,
    ):
        if color is not _void:
            self.color = color
        if width is not _void:
            self.width = width
        if style is not _void:
            self.style = style
        if alpha is not _void:
            self.alpha = alpha
        if antialias is not _void:
            self.antialias = antialias
        return self

    def with_hover_template(
        self,
        template: str,
        extra: Any | None = None,
    ) -> Self:
        """Add hover template to the markers."""
        if self._backend_name in ("plotly", "bokeh"):  # conversion for HTML
            template = template.replace("\n", "<br>")
        params = parse_texts(template, self.nlines, extra)
        # set default format keys
        if "i" not in params:
            params["i"] = np.arange(self.nlines)
        texts = [
            template.format(**{k: v[i] for k, v in params.items()})
            for i in range(self.nlines)
        ]
        self._backend._plt_set_hover_text(texts)
        return self

    def _as_legend_item(self) -> _legend.LineLegendItem:
        if self.nlines == 0:
            return _legend.EmptyLegendItem()
        return _legend.LineLegendItem(self.color[0], self.width[0], self.style[0])


def _norm_data(data: list[NDArray[np.number]]) -> NDArray[np.number]:
    data_normed: list[NDArray[np.number]] = []
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for each in data:
        arr = np.asarray(each)
        if arr.dtype.kind not in "uif":
            raise ValueError(f"Expected data to be numeric, got {arr.dtype}")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected data to be (N, 2), got {arr.shape}")
        data_normed.append(arr)
        xmins.append(arr[:, 0].min())
        xmaxs.append(arr[:, 0].max())
        ymins.append(arr[:, 1].min())
        ymaxs.append(arr[:, 1].max())
    if len(data) > 0:
        xhint = min(xmins), max(xmaxs)
        yhint = min(ymins), max(ymaxs)
    else:
        xhint = yhint = None
    return data_normed, xhint, yhint
