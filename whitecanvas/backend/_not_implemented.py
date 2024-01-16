from __future__ import annotations

from whitecanvas.types import Hatch, LineStyle


def edge_style():
    def _getter(self):
        return getattr(self, "__edge_style_value", LineStyle.SOLID)

    def _setter(self, value: LineStyle):
        setattr(self, "__edge_style_value", value)

    return _getter, _setter


def edge_styles():
    def _getter(self):
        return getattr(
            self, "__edge_style_value", [LineStyle.SOLID] * self._plt_get_ndata()
        )

    def _setter(self, value: LineStyle | list[LineStyle]):
        if isinstance(value, LineStyle):
            value = [value] * self._plt_get_ndata()
        setattr(self, "__edge_style_value", value)

    return _getter, _setter


def face_pattern():
    def _getter(self):
        return getattr(self, "__face_pattern_value", Hatch.SOLID)

    def _setter(self, value: Hatch):
        setattr(self, "__face_pattern_value", value)

    return _getter, _setter


def face_patterns():
    def _getter(self):
        return getattr(
            self, "__face_pattern_value", [Hatch.SOLID] * self._plt_get_ndata()
        )

    def _setter(self, value: Hatch | list[Hatch]):
        if isinstance(value, Hatch):
            value = [value] * self._plt_get_ndata()
        setattr(self, "__face_pattern_value", value)

    return _getter, _setter
