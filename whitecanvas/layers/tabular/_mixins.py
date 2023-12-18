from __future__ import annotations

from whitecanvas.layers._mixin import LayerEvents, LayerNamespace
from ._dataframe import TabularLayer


class TabularLayerNamespace(LayerNamespace[TabularLayer]):
    def __init__(self, layer: TabularLayer | None = None) -> None:
        super().__init__(layer)


# markers --> setting color column = change color
# lines, boxplot, violinplot etc --> setting color needs resetting data.


class NamedFace(TabularLayerNamespace):
    def __init__(self, layer: TabularLayer | None = None) -> None:
        super().__init__(layer)
        self._color = None

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value: str | tuple[str, ...]):
        self._layer._source
        self._layer._backend._plt_set_face_color(col)
        self.events.color.emit(col)

    @property
    def pattern(self) -> FacePattern:
        return self._layer._backend._plt_get_face_pattern()

    @pattern.setter
    def pattern(self, value: str | FacePattern):
        pattern = FacePattern(value)
        self._layer._backend._plt_set_face_pattern(pattern)
        self.events.pattern.emit(pattern)
