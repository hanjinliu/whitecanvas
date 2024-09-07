from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator

from whitecanvas.types import LineStyle

if TYPE_CHECKING:
    from matplotlib import axis as mpl_axis
    from matplotlib import text as mpl_text

    from whitecanvas.backend.matplotlib.canvas import Canvas


class SupportsText:
    def __init__(self, canvas: Canvas):
        self._canvas = weakref.ref(canvas)
        self._text = ""
        self._fontdict = {
            "color": plt.rcParams["text.color"],
            "fontsize": plt.rcParams["font.size"],
            "family": plt.rcParams["font.family"],
        }

    def _plt_get_color(self):
        return self._fontdict["color"]

    def _plt_get_size(self) -> str:
        return self._fontdict["fontsize"]

    def _plt_get_fontfamily(self) -> str:
        return self._fontdict["family"]


class Title(SupportsText):
    def __init__(self, canvas: Canvas):
        self._canvas = weakref.ref(canvas)
        self._text = ""
        self._fontdict = {
            "color": plt.rcParams["text.color"],
            "fontsize": plt.rcParams["font.size"],
            "family": plt.rcParams["font.family"],
        }

    def _plt_get_text(self) -> str:
        return self._canvas()._axes.get_title()

    def _plt_set_text(self, text: str):
        self._canvas()._axes.set_title(text, self._fontdict)
        self._text = text

    def _plt_set_color(self, color):
        d = self._fontdict.copy()
        d["color"] = color
        self._canvas()._axes.set_title(self._text, fontdict=d)
        self._fontdict = d

    def _plt_get_visible(self) -> bool:
        return bool(self._canvas()._axes.get_title())

    def _plt_set_visible(self, visible: bool):
        if visible:
            self._canvas()._axes.set_title(self._text, fontdict=self._fontdict)
        else:
            self._canvas()._axes.set_title("")

    def _plt_set_size(self, size: str):
        d = self._fontdict.copy()
        d["fontsize"] = size
        self._canvas()._axes.set_title(self._text, fontdict=d)
        self._fontdict = d

    def _plt_set_fontfamily(self, font):
        d = self._fontdict.copy()
        d["family"] = font
        self._canvas()._axes.set_title(self._text, fontdict=d)
        self._fontdict = d


class MplLabel(SupportsText):
    def __init__(self, canvas: Canvas, name: str):
        super().__init__(canvas)
        self._axis_name = name

    def _set_label(self, text: str, fontdict: dict[str, str] = {}):
        fname = f"set_{self._axis_name}label"
        getattr(self._canvas()._axes, fname)(text, fontdict)

    def _plt_get_text(self) -> str:
        return getattr(self._canvas()._axes, f"get_{self._axis_name}label")()

    def _plt_set_text(self, text: str):
        self._set_label(text, self._fontdict)
        self._text = text

    def _plt_set_color(self, color):
        d = self._fontdict.copy()
        d["color"] = color
        self._set_label(self._text, fontdict=d)
        self._fontdict = d

    def _plt_get_visible(self) -> bool:
        return bool(self._canvas()._axes.get_xlabel())

    def _plt_set_visible(self, visible: bool):
        if visible:
            self._set_label(self._text, fontdict=self._fontdict)
        else:
            self._set_label("")

    def _plt_set_size(self, size: str):
        d = self._fontdict.copy()
        d["fontsize"] = size
        self._set_label(self._text, fontdict=d)
        self._fontdict = d

    def _plt_set_fontfamily(self, font):
        d = self._fontdict.copy()
        d["family"] = font
        self._set_label(self._text, fontdict=d)
        self._fontdict = d


class MplTicks(SupportsText):
    def __init__(self, canvas: Canvas, name: str):
        super().__init__(canvas)
        self._axis_name = name

    def _get_tick_labels(self) -> list[mpl_text.Text]:
        axes = self._canvas()._axes
        return getattr(axes, f"get_{self._axis_name}ticklabels")()

    def _get_ticks(self) -> list[float]:
        axes = self._canvas()._axes
        return getattr(axes, f"get_{self._axis_name}ticks")()

    def _set_ticks(self, ticks: list[float]):
        axes = self._canvas()._axes
        getattr(axes, f"set_{self._axis_name}ticks")(ticks)

    def _plt_get_tick_labels(self) -> tuple[list[float], list[str]]:
        return self._get_ticks(), [x.get_text() for x in self._get_tick_labels()]

    def _plt_override_labels(self, pos: list[float], labels: list[str]):
        axes = self._canvas()._axes
        getattr(axes, f"set_{self._axis_name}ticks")(pos, labels)

    def _plt_reset_override(self):
        # FIXME: This is not a perfect solution. It will update the x-axis scale.
        axis = getattr(self._canvas()._axes, f"get_{self._axis_name}axis")()
        axis.clear()

    def _plt_set_color(self, color):
        d = self._fontdict.copy()
        d["color"] = color
        for x in self._get_tick_labels():
            x.set_color(color)
        self._fontdict = d

    def _plt_get_visible(self) -> bool:
        ticklines = getattr(self._canvas()._axes, f"get_{self._axis_name}ticklines")()
        return ticklines[0].get_visible()

    def _plt_set_visible(self, visible: bool):
        axes = self._canvas()._axes
        for tick in getattr(axes, f"get_{self._axis_name}ticklines")():
            tick.set_visible(visible)
        for text in self._get_tick_labels():
            text.set_visible(visible)

    def _plt_set_size(self, size: str):
        d = self._fontdict.copy()
        d["fontsize"] = size
        for x in self._get_tick_labels():
            x.set_fontsize(size)
        self._fontdict = d

    def _plt_set_fontfamily(self, font):
        d = self._fontdict.copy()
        d["family"] = font
        for x in self._get_tick_labels():
            x.set_fontfamily(font)
        self._fontdict = d

    def _plt_get_text_rotation(self) -> float:
        return self._get_tick_labels()[0].get_rotation()

    def _plt_set_text_rotation(self, rotation: float):
        for x in self._get_tick_labels():
            x.set_rotation(rotation)


class MplAxis:
    def __init__(self, canvas: Canvas, name: str):
        self._canvas = weakref.ref(canvas)
        self._axis_name = name

    def _plt_get_limits(self) -> tuple[float, float]:
        axes = self._canvas()._axes
        x0, x1 = getattr(axes, f"get_{self._axis_name}lim")()
        if getattr(axes, f"{self._axis_name}axis_inverted")():
            return float(x1), float(x0)
        else:
            return float(x0), float(x1)

    def _plt_set_limits(self, limits: tuple[float, float]):
        axes = self._canvas()._axes
        if getattr(axes, f"{self._axis_name}axis_inverted")():
            limits = limits[::-1]
        getattr(axes, f"set_{self._axis_name}lim")(*limits)

    def _get_mpl_axis(self) -> mpl_axis.Axis:
        return getattr(self._canvas()._axes, f"{self._axis_name}axis")

    def _plt_get_color(self):
        return self._get_mpl_axis().get_tick_params()["color"]

    def _plt_set_color(self, color):
        ax = self._get_mpl_axis()
        color = tuple(color)
        ax.set_tick_params(color=color, labelcolor=color)
        if hasattr(ax, "line"):  # 3D
            ax.line.set_color(color)
        else:
            if self._axis_name == "x":
                self._canvas()._axes.spines["bottom"].set_color(color)
            else:
                self._canvas()._axes.spines["left"].set_color(color)

    def _plt_flip(self):
        axis = self._get_mpl_axis()
        axis.set_inverted(not axis.get_inverted())

    def _plt_set_grid_state(self, visible: bool, color, width: float, style: LineStyle):
        axis = self._get_mpl_axis()
        axis.grid(
            visible,
            which="major",
            color=color,
            linestyle=style.value,
            linewidth=width,
        )
