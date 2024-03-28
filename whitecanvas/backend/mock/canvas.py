from __future__ import annotations

from typing import Callable

import numpy as np
from psygnal import Signal

from whitecanvas import protocols
from whitecanvas.types import MouseEvent, Rect


class MockObject:
    def __repr__(self) -> str:
        return f"<Mock backend object at {hex(id(self))}>"


@protocols.check_protocol(protocols.CanvasProtocol)
class Canvas:
    def __init__(self):
        self._xaxis = Axis()
        self._yaxis = Axis()
        self._title = Title()
        self._xticks = Ticks()
        self._yticks = Ticks()
        self._aspect_ratio = None
        self._visible = True
        self._mouse_enabled = True
        self._obj = MockObject()

    def _plt_get_native(self):
        return self._obj

    def _plt_get_title(self):
        return self._title

    def _plt_get_xaxis(self):
        return self._xaxis

    def _plt_get_yaxis(self):
        return self._yaxis

    def _plt_get_xlabel(self):
        return self._xaxis._label

    def _plt_get_ylabel(self):
        return self._yaxis._label

    def _plt_get_xticks(self):
        return self._xticks

    def _plt_get_yticks(self):
        return self._yticks

    def _plt_reorder_layers(self, layers: list):
        pass

    def _plt_get_aspect_ratio(self) -> float | None:
        return self._aspect_ratio

    def _plt_set_aspect_ratio(self, ratio: float | None):
        self._aspect_ratio = ratio

    def _plt_add_layer(self, layer):
        pass

    def _plt_remove_layer(self, layer):
        pass

    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        self._visible = visible

    def _plt_twinx(self) -> Canvas:
        new = Canvas()

        @new._xaxis.lim_changed.connect
        def _(lims):
            with self._xaxis.lim_changed.blocked():
                self._xaxis._plt_set_limits(lims)

        @self._xaxis.lim_changed.connect
        def _(lims):
            with new._xaxis.lim_changed.blocked():
                new._xaxis._plt_set_limits(lims)

        return new

    def _plt_twiny(self) -> Canvas:
        new = Canvas()

        @new._yaxis.lim_changed.connect
        def _(lims):
            with self._yaxis.lim_changed.blocked():
                self._yaxis._plt_set_limits(lims)

        @self._yaxis.lim_changed.connect
        def _(lims):
            with new._yaxis.lim_changed.blocked():
                new._yaxis._plt_set_limits(lims)

        return new

    def _plt_inset(self, rect: Rect) -> Canvas:
        return Canvas()

    def _plt_connect_mouse_click(self, callback: Callable[[MouseEvent], None]):
        pass

    def _plt_connect_mouse_drag(self, callback: Callable[[MouseEvent], None]):
        pass

    def _plt_connect_mouse_double_click(self, callback: Callable[[MouseEvent], None]):
        pass

    def _plt_connect_mouse_release(self, callback: Callable[[MouseEvent], None]):
        pass

    def _plt_draw(self):
        pass

    def _plt_connect_xlim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        self._xaxis.lim_changed.connect(callback, max_args=1)

    def _plt_connect_ylim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        self._yaxis.lim_changed.connect(callback, max_args=1)

    def _plt_make_legend(self, *args, **kwargs):
        pass

    def _plt_get_mouse_enable(self):
        return self._mouse_enabled

    def _plt_set_mouse_enable(self, enabled: bool):
        self._mouse_enabled = enabled


@protocols.check_protocol(protocols.CanvasGridProtocol)
class CanvasGrid:
    def __init__(self, heights: list[float], widths: list[float], app: str = "default"):
        self._background_color = np.array([1, 1, 1, 1], dtype=np.float32)
        self._figsize = (100, 100)

    def _plt_add_canvas(self, row: int, col: int, rowspan: int, colspan: int) -> Canvas:
        return Canvas()

    def _plt_show(self):
        pass

    def _plt_get_background_color(self):
        return self._background_color

    def _plt_set_background_color(self, color):
        self._background_color = color

    def _plt_screenshot(self):
        raise RuntimeError("Mock backend does not support screenshots")

    def _plt_set_figsize(self, width: int, height: int):
        self._figsize = (width, height)

    def _plt_set_spacings(self, wspace: float, hspace: float):
        pass


class _SupportsText:
    def __init__(self):
        self._visible = True
        self._text = ""
        self._color = np.array([0, 0, 0, 1], dtype=np.float32)
        self._size = 10
        self._fontfamily = "Arial"

    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        self._visible = visible

    def _plt_get_text(self) -> str:
        return self._text

    def _plt_set_text(self, text: str):
        self._text = text

    def _plt_get_color(self):
        return self._color

    def _plt_set_color(self, color):
        self._color = color

    def _plt_get_size(self) -> int:
        return self._size

    def _plt_set_size(self, size: int):
        self._size = size

    def _plt_get_fontfamily(self) -> str:
        return self._fontfamily

    def _plt_set_fontfamily(self, family: str):
        self._fontfamily = family


class Title(_SupportsText):
    pass


class Label(_SupportsText):
    pass


class Axis:
    lim_changed = Signal(tuple)

    def __init__(self):
        self._visible = True
        self._limits = (0, 1)
        self._flipped = False
        self._color = np.array([0, 0, 0, 1], dtype=np.float32)
        self._label = Label()

    def _plt_get_visible(self) -> bool:
        return self._visible

    def _plt_set_visible(self, visible: bool):
        self._visible = visible

    def _plt_get_color(self):
        return self._color

    def _plt_set_color(self, color):
        self._color = color

    def _plt_get_label(self) -> Label:
        return self._label

    def _plt_flip(self) -> None:
        self._flipped = not self._flipped

    def _plt_get_limits(self) -> tuple[float, float]:
        return self._limits

    def _plt_set_limits(self, limits: tuple[float, float]):
        self._limits = limits
        self.lim_changed.emit(limits)

    def _plt_set_grid_state(self, *args, **kwargs):
        pass


class Ticks(_SupportsText):
    def __init__(self):
        super().__init__()
        self._pos = []
        self._labels = []
        self._rotation = 0.0

    def _plt_get_tick_labels(self) -> tuple[list[float], list[str]]:
        return self._pos, self._labels

    def _plt_override_labels(self, pos: list[float], labels: list[str]):
        self._pos = pos
        self._labels = labels

    def _plt_reset_override(self):
        self._pos = []
        self._labels = []

    def _plt_get_text_rotation(self) -> float:
        return self._rotation

    def _plt_set_text_rotation(self, rotation: float):
        self._rotation = rotation
