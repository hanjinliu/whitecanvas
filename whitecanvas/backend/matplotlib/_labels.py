from __future__ import annotations

import weakref
from typing import TYPE_CHECKING
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from .canvas import Canvas


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

    def _plt_set_fontfamily(self, font):
        d = self._fontdict.copy()
        d["family"] = font
        self._canvas()._axes.set_title(self._text, fontdict=d)


class XLabel(SupportsText):
    def _plt_get_text(self) -> str:
        return self._canvas()._axes.get_xlabel()

    def _plt_set_text(self, text: str):
        self._canvas()._axes.set_xlabel(text, self._fontdict)
        self._text = text

    def _plt_set_color(self, color):
        d = self._fontdict.copy()
        d["color"] = color
        self._canvas()._axes.set_xlabel(self._text, fontdict=d)

    def _plt_get_visible(self) -> bool:
        return bool(self._canvas()._axes.get_xlabel())

    def _plt_set_visible(self, visible: bool):
        if visible:
            self._canvas()._axes.set_xlabel(self._text, fontdict=self._fontdict)
        else:
            self._canvas()._axes.set_xlabel("")

    def _plt_set_size(self, size: str):
        d = self._fontdict.copy()
        d["fontsize"] = size
        self._canvas()._axes.set_xlabel(self._text, fontdict=d)

    def _plt_set_fontfamily(self, font):
        d = self._fontdict.copy()
        d["family"] = font
        self._canvas()._axes.set_xlabel(self._text, fontdict=d)


class YLabel(SupportsText):
    def _plt_get_text(self) -> str:
        return self._canvas()._axes.get_ylabel()

    def _plt_set_text(self, text: str):
        self._canvas()._axes.set_ylabel(text, self._fontdict)
        self._text = text

    def _plt_set_color(self, color):
        d = self._fontdict.copy()
        d["color"] = color
        self._canvas()._axes.set_ylabel(self._text, fontdict=d)

    def _plt_get_visible(self) -> bool:
        return bool(self._canvas()._axes.get_ylabel())

    def _plt_set_visible(self, visible: bool):
        if visible:
            self._canvas()._axes.set_ylabel(self._text, fontdict=self._fontdict)
        else:
            self._canvas()._axes.set_ylabel("")

    def _plt_set_size(self, size: str):
        d = self._fontdict.copy()
        d["fontsize"] = size
        self._canvas()._axes.set_ylabel(self._text, fontdict=d)

    def _plt_set_fontfamily(self, font):
        d = self._fontdict.copy()
        d["family"] = font
        self._canvas()._axes.set_ylabel(self._text, fontdict=d)


class XTicks(SupportsText):
    def _plt_get_text(self) -> tuple[list[float], list[str]]:
        axes = self._canvas()._axes
        return axes.get_xticks(), [x.get_text() for x in axes.get_xticklabels()]

    def _plt_set_text(self, text: tuple[list[float], list[str]]):
        pos, texts = text
        self._canvas()._axes.set_xticks(pos, texts)

    def _plt_set_color(self, color):
        d = self._fontdict.copy()
        d["color"] = color
        for x in self._canvas()._axes.get_xticklabels():
            x.set_color(color)

    def _plt_get_visible(self) -> bool:
        return self._canvas()._axes.get_xticklines()[0].get_visible()

    def _plt_set_visible(self, visible: bool):
        axes = self._canvas()._axes
        for tick in axes.get_xticklines():
            tick.set_visible(visible)
        for text in axes.get_xticklabels():
            text.set_visible(visible)

    def _plt_set_size(self, size: str):
        d = self._fontdict.copy()
        d["fontsize"] = size
        for x in self._canvas()._axes.get_xticklabels():
            x.set_fontsize(size)

    def _plt_set_fontfamily(self, font):
        d = self._fontdict.copy()
        d["family"] = font
        for x in self._canvas()._axes.get_xticklabels():
            x.set_fontfamily(font)


class YTicks(SupportsText):
    def _plt_get_text(self) -> tuple[list[float], list[str]]:
        axes = self._canvas()._axes
        return axes.get_yticks(), [x.get_text() for x in axes.get_yticklabels()]

    def _plt_set_text(self, text: tuple[list[float], list[str]]):
        pos, texts = text
        self._canvas()._axes.set_yticks(pos, texts)

    def _plt_set_color(self, color):
        d = self._fontdict.copy()
        d["color"] = color
        for x in self._canvas()._axes.get_yticklabels():
            x.set_color(color)

    def _plt_get_visible(self) -> bool:
        return self._canvas()._axes.get_yticklines()[0].get_visible()

    def _plt_set_visible(self, visible: bool):
        axes = self._canvas()._axes
        for tick in axes.get_yticklines():
            tick.set_visible(visible)
        for text in axes.get_yticklabels():
            text.set_visible(visible)

    def _plt_set_size(self, size: str):
        d = self._fontdict.copy()
        d["fontsize"] = size
        for x in self._canvas()._axes.get_yticklabels():
            x.set_fontsize(size)

    def _plt_set_fontfamily(self, font):
        d = self._fontdict.copy()
        d["family"] = font
        for x in self._canvas()._axes.get_yticklabels():
            x.set_fontfamily(font)


class AxisBase:
    def __init__(self, canvas: Canvas):
        self._canvas = weakref.ref(canvas)


class XAxis(AxisBase):
    def __init__(self, canvas: Canvas):
        self._canvas = weakref.ref(canvas)

    def _plt_get_limits(self) -> tuple[float, float]:
        return self._canvas()._axes.get_xlim()

    def _plt_set_limits(self, limits: tuple[float, float]):
        self._canvas()._axes.set_xlim(*limits)

    def _plt_get_color(self):
        return self._canvas()._axes.xaxis.get_tick_params()["color"]

    def _plt_set_color(self, color):
        ax = self._canvas()._axes
        color = tuple(color)
        ax.xaxis.set_tick_params(color=color, labelcolor=color)
        ax.spines["bottom"].set_color(color)

    def _plt_flip(self):
        self._canvas()._axes.invert_xaxis()


class YAxis(AxisBase):
    def __init__(self, canvas: Canvas):
        self._canvas = weakref.ref(canvas)

    def _plt_get_limits(self) -> tuple[float, float]:
        return self._canvas()._axes.get_ylim()

    def _plt_set_limits(self, limits: tuple[float, float]):
        self._canvas()._axes.set_ylim(*limits)

    def _plt_get_color(self):
        return self._canvas()._axes.yaxis.get_tick_params()["color"]

    def _plt_set_color(self, color):
        ax = self._canvas()._axes
        color = tuple(color)
        ax.yaxis.set_tick_params(color=color, labelcolor=color)
        ax.spines["left"].set_color(color)

    def _plt_flip(self):
        self._canvas()._axes.invert_yaxis()