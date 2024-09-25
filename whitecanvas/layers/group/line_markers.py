from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from numpy.typing import ArrayLike

from whitecanvas.layers import _legend, _text_utils
from whitecanvas.layers._primitive import Errorbars, Markers, Texts
from whitecanvas.layers._primitive.line import _SingleLine
from whitecanvas.layers.group._collections import LayerContainer
from whitecanvas.types import Alignment, ColorType, LineStyle, Orientation, _Void

if TYPE_CHECKING:
    from whitecanvas.layers._primitive.markers import _Edge, _Face, _Size
    from whitecanvas.layers.group.labeled import LabeledPlot

_void = _Void()
_L = TypeVar("_L", bound=_SingleLine)


class Plot(LayerContainer, Generic[_L]):
    """A Plot layer is composed of a line and a markers layer."""

    def __init__(
        self,
        line: _L,
        markers: Markers[_Face, _Edge, _Size],
        name: str | None = None,
    ):
        super().__init__([line, markers], name=name)

    @property
    def line(self) -> _L:
        """The line layer."""
        return self._children[0]

    @property
    def markers(self) -> Markers[_Face, _Edge, _Size]:
        """The markers layer."""
        return self._children[1]

    def with_edge(
        self,
        color: ColorType | None = None,
        alpha: float = 1.0,
        width: float = 1.0,
        style: str | LineStyle = LineStyle.SOLID,
    ) -> Plot:
        """
        Set the edge properties of the markers.

        Parameters
        ----------
        color : ColorType, optional
            The color of the edge. If not given, use the color defined by the theme.
        alpha : float, default 1.0
            The transparency of the edge.
        width : float, default 1.0
            The width of the edge.
        style : str or LineStyle, default LineStyle.SOLID
            The style of the edge.

        Returns
        -------
        Plot
            The current plot with the edge properties set.
        """
        self.markers.with_edge(color=color, alpha=alpha, width=width, style=style)
        return self

    def with_text(
        self,
        strings: list[str],
        *,
        color: ColorType = "black",
        size: float = 12,
        rotation: float = 0.0,
        anchor: str | Alignment = Alignment.BOTTOM_LEFT,
        family: str | None = None,
    ) -> LabeledPlot:
        """
        Add texts at the positions of markers.

        Parameters
        ----------
        strings : list of str
            The text to add to the markers.
        color : ColorType, default "black"
            The color of the text.
        size : float, default 12
            The size of the text.
        rotation : float, default 0.0
            The rotation of the text.
        anchor : str or Alignment, default Alignment.BOTTOM_LEFT
            The anchor position of the text.
        family : str or None, optional
            The font family of the text.

        Returns
        -------
        LabeledPlot
            The current plot with the text added to the markers.
        """
        from whitecanvas.layers.group.labeled import LabeledPlot

        strings = _text_utils.norm_label_text(strings, self.line.data)
        texts = Texts(
            *self.line.data,
            strings,
            color=color,
            size=size,
            rotation=rotation,
            anchor=anchor,
            family=family,
            backend=self._backend_name,
        )
        old_name = self.name
        self.name = f"plot-of-{old_name}"
        xerr = Errorbars._empty_v(f"xerr-of-{old_name}", backend=self._backend_name)
        yerr = Errorbars._empty_h(f"yerr-of-{old_name}", backend=self._backend_name)
        return LabeledPlot(self, xerr, yerr, texts, name=old_name)

    def with_xerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float = 0,
    ) -> LabeledPlot:
        """
        Add horizontal error bars from the markers.

        Parameters
        ----------
        err : array-like
            The error values.
        err_high : array-like or None, optional
            The high error values. If not given, use the same values as `err`.
        color : ColorType or None, optional
            The color of the error bars. If not given, use the color of the line.
        width : float or None, optional
            The width of the error bars. If not given, use the width of the markers.
        style : str or None, optional
            The style of the error bars. If not given, use the style of the markers.
        antialias : bool or None, optional
            Whether to use antialiasing. If not given, use the antialiasing of the line.
        capsize : float, default 0
            The size of the caps at the end of the error bars.

        Returns
        -------
        LabeledPlot
            The current plot with the horizontal error bars added.
        """
        from whitecanvas.layers._primitive import Errorbars
        from whitecanvas.layers.group.labeled import LabeledPlot

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
        old_name = self.name
        self.name = f"plot-of-{old_name}"
        xerr = Errorbars(
            self.data.y, self.data.x - err, self.data.x + err_high, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
            orient=Orientation.HORIZONTAL, name=f"xerr-of-{old_name}",
            backend=self._backend_name,
        )  # fmt: skip
        yerr = Errorbars._empty_v(f"yerr-of-{old_name}", backend=self._backend_name)
        return LabeledPlot(self, xerr, yerr, name=old_name)

    def with_yerr(
        self,
        err: ArrayLike,
        err_high: ArrayLike | None = None,
        *,
        color: ColorType | _Void = _void,
        width: float | _Void = _void,
        style: str | _Void = _void,
        antialias: bool | _Void = _void,
        capsize: float = 0,
    ) -> LabeledPlot:
        """
        Add vertical error bars from the markers.

        Parameters
        ----------
        err : array-like
            The error values.
        err_high : array-like or None, optional
            The high error values. If not given, use the same values as `err`.
        color : ColorType or None, optional
            The color of the error bars. If not given, use the color of the line.
        width : float or None, optional
            The width of the error bars. If not given, use the width of the markers.
        style : str or None, optional
            The style of the error bars. If not given, use the style of the markers.
        antialias : bool or None, optional
            Whether to use antialiasing. If not given, use the antialiasing of the line.
        capsize : float, default 0
            The size of the caps at the end of the error bars.

        Returns
        -------
        LabeledPlot
            The current plot with the vertical error bars added.
        """
        from whitecanvas.layers._primitive import Errorbars
        from whitecanvas.layers.group.labeled import LabeledPlot

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
        old_name = self.name
        self.name = f"plot-of-{old_name}"
        yerr = Errorbars(
            self.data.x, self.data.y - err, self.data.y + err_high, color=color,
            width=width, style=style, antialias=antialias, capsize=capsize,
            name=f"yerr-of-{old_name}", backend=self._backend_name,
        )  # fmt: skip
        xerr = Errorbars._empty_h(f"xerr-of-{old_name}", backend=self._backend_name)
        return LabeledPlot(self, xerr, yerr, name=old_name)

    @property
    def data(self):
        """The internal data of the line and markers."""
        return self.line.data

    def set_data(self, xdata=None, ydata=None):
        self.line.set_data(xdata, ydata)
        self.markers.set_data(xdata, ydata)

    def _as_legend_item(self):
        line = self.line._as_legend_item()
        markers = self.markers._as_legend_item()
        return _legend.PlotLegendItem(line, markers)
