from __future__ import annotations

from cmap import Color, Colormap


class ColorPalette:
    def __init__(
        self,
        cmap: str | Colormap | ColorPalette,
        num: int | None = None,
    ):
        if isinstance(cmap, ColorPalette):
            _cmap = cmap._cmap
        else:
            _cmap = Colormap(cmap)
        self._cmap: Colormap = _cmap
        self._is_categorical = self._cmap.category in ("qualitative", "categorical")
        if num is None:
            if self._is_categorical:
                self._num = len(self._cmap.color_stops)
            else:
                self._num = 10
        self._n_generated = 0

    def next(self, update: bool = True) -> Color:
        """Generate the next color."""
        pos = (self._n_generated / self._num + 1e-6) % 1.0
        if update:
            self._n_generated += 1
        return self._cmap(pos)

    def init(self):
        """Initialize the palette."""
        self._n_generated = 0
