from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike
from whitecanvas.types import ColorType, _Void, Alignment
from whitecanvas.layers.primitive import Line, Markers, Errorbars
from whitecanvas.layers._base import LayerGroup

if TYPE_CHECKING:
    from .text_group import TextGroup
    from .annotated import AnnotatedPlot

_void = _Void()


class Plot(LayerGroup):
    """A Plot layer is composed of a line and a markers layer."""

    def __init__(
        self,
        line: Line,
        markers: Markers,
        name: str | None = None,
    ):
        super().__init__([line, markers], name=name)

    @property
    def line(self) -> Line:
        """The line layer."""
        return self._children[0]

    @property
    def markers(self) -> Markers:
        """The markers layer."""
        return self._children[1]

    def setup_line(
        self,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
    ):
        self.line.setup(color=color, line_width=width, style=style, antialias=antialias)
        return self

    def setup_markers(
        self,
        symbol: str | _Void = _void,
        size: float | _Void = _void,
        face_color: ColorType | _Void = _void,
        edge_color: ColorType | _Void = _void,
        edge_width: float | _Void = _void,
        edge_style: str | _Void = _void,
    ):
        if symbol is not _void:
            self.markers.symbol = symbol
        if size is not _void:
            self.markers.size = size
        if face_color is not _void:
            self.markers.face_color = face_color
        if edge_color is not _void:
            self.markers.edge_color = edge_color
        if edge_width is not _void:
            self.markers.edge_width = edge_width
        if edge_style is not _void:
            self.markers.edge_style = edge_style
        return self

    def with_text(
        self,
        strings: list[str],
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        fontfamily: str = "sans-serif",
    ) -> Plot:
        from whitecanvas.layers.group.annotated import AnnotatedPlot
        from .text_group import TextGroup

        texts = TextGroup.from_strings(
            *self.line.data,
            strings,
            color=color,
            size=size,
            rotation=rotation,
            anchor=anchor,
            fontfamily=fontfamily,
            backend=self._backend_name,
        )
        xerr = Errorbars([], [], [], orient="vertical", backend=self._backend_name)
        yerr = Errorbars([], [], [], orient="horizontal", backend=self._backend_name)
        return AnnotatedPlot(self, xerr, yerr, texts, name=self.name)

    def with_xerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        color: ColorType | _Void = _void,
        line_width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float = 0,
    ) -> AnnotatedPlot:
        from whitecanvas.layers.group.annotated import AnnotatedPlot
        from whitecanvas.layers.primitive import Errorbars

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.line.color
        if line_width is _void:
            line_width = self.markers.edge_width
        if style is _void:
            style = self.markers.edge_style
        if antialias is _void:
            antialias = self.line.antialias
        xerr = Errorbars(
            self.data.y, self.data.x - err, self.data.x + err_high, color=color,
            line_width=line_width, line_style=style, antialias=antialias, capsize=capsize,
            backend=self._backend_name
        )  # fmt: skip
        yerr = Errorbars([], [], [], orient="horizontal", backend=self._backend_name)
        return AnnotatedPlot(self, xerr, yerr, name=self.name)

    def with_yerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        color: ColorType | _Void = _void,
        line_width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float = 0,
    ) -> AnnotatedPlot:
        from whitecanvas.layers.group.annotated import AnnotatedPlot
        from whitecanvas.layers.primitive import Errorbars

        if err_high is None:
            err_high = err
        if color is _void:
            color = self.line.color
        if line_width is _void:
            line_width = self.markers.edge_width
        if style is _void:
            style = self.markers.edge_style
        if antialias is _void:
            antialias = self.line.antialias
        yerr = Errorbars(
            self.data.x, self.data.y - err, self.data.y + err_high, color=color,
            line_width=line_width, line_style=style, antialias=antialias, capsize=capsize,
            backend=self._backend_name
        )  # fmt: skip
        xerr = Errorbars([], [], [], orient="vertical", backend=self._backend_name)
        return AnnotatedPlot(self, xerr, yerr, name=self.name)

    @property
    def data(self):
        return self.line.data

    def set_data(self, xdata=None, ydata=None):
        self.line.set_data(xdata, ydata)
        self.markers.set_data(xdata, ydata)
