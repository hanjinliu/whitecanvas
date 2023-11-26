from __future__ import annotations

from cmap import Color, Colormap


class ColorPalette:
    def __init__(
        self,
        cmap: str | Colormap | ColorPalette,
    ):
        if isinstance(cmap, ColorPalette):
            _cmap = cmap._cmap
            pos = cmap._pos
        else:
            _cmap = Colormap(cmap)
            if isinstance(cmap, list):
                num = len(cmap)
                pos = [i / (num - 1) for i in range(num)] + [1.0]
            elif _cmap.category in ("qualitative", "categorical"):
                num = len(_cmap.color_stops)
                pos = [(i + 0.5) / num for i in range(num)]
            else:
                pos = [0, 0.3, 0.6, 0.9, 0.2, 0.5, 0.8, 0.1, 0.4, 0.7, 1.0]
        self._cmap: Colormap = _cmap
        self._pos: list[float] = pos
        self._n_generated = 0

    def next(self, update: bool = True) -> Color:
        """Generate the next color."""
        pos = self._pos[self._n_generated % len(self._pos)]
        if update:
            self._n_generated += 1
        return self._cmap(pos)

    def init(self):
        """Initialize the palette."""
        self._n_generated = 0

    def nextn(self, num: int, update: bool = True) -> list[Color]:
        """Generate the next n colors."""
        current = self._n_generated
        out = [self.next(update=True) for _ in range(num)]
        if not update:
            self._n_generated = current
        return out

    def copy(self) -> ColorPalette:
        """Make a copy of the palette."""
        return ColorPalette(self._cmap)
