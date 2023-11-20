from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike
from whitecanvas.types import ColorType, _Void, Alignment, LineStyle
from whitecanvas.layers.primitive import Line, Markers, Errorbars
from whitecanvas.layers.group._collections import ListLayerGroup

if TYPE_CHECKING:
    from .annotated import AnnotatedPlot

_void = _Void()


class Plot(ListLayerGroup):
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

    def with_edge(
        self,
        color: ColorType | _Void = _void,
        alpha: float = 1.0,
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
    ) -> Plot:
        self.markers.with_edge(color=color, alpha=alpha, width=width, style=style)
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
        width: float | _Void = _void,
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
        if width is _void:
            width = self.markers.edge.width
        if style is _void:
            style = self.markers.edge.style
        if antialias is _void:
            antialias = self.line.antialias
        xerr = Errorbars(
            self.data.y, self.data.x - err, self.data.x + err_high, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
            backend=self._backend_name
        )  # fmt: skip
        yerr = Errorbars([], [], [], orient="horizontal", backend=self._backend_name)
        return AnnotatedPlot(self, xerr, yerr, name=self.name)

    def with_yerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
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
        if width is _void:
            width = self.markers.edge.width
        if style is _void:
            style = self.markers.edge.style
        if antialias is _void:
            antialias = self.line.antialias
        yerr = Errorbars(
            self.data.x, self.data.y - err, self.data.y + err_high, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
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
