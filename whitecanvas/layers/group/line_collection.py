from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from whitecanvas import theme
from whitecanvas.backend import Backend
from whitecanvas.layers._primitive import Line, Markers
from whitecanvas.layers.group._collections import LayerCollection
from whitecanvas.types import Hatch, LineStyle, Symbol, XYData
from whitecanvas.utils.normalize import as_any_1d_array, as_color_array, parse_texts
from whitecanvas.utils.type_check import is_real_number

if TYPE_CHECKING:
    from typing_extensions import Self


class LineCollection(LayerCollection[Line]):
    """Collection of lines."""

    @classmethod
    def from_segments(
        cls,
        segments: list[Any],
        *,
        name: str | None = None,
        backend: str | Backend | None = None,
    ) -> Self:
        lines = [Line([], [], backend=backend) for _ in segments]
        for line, seg in zip(lines, segments):
            line.data = seg
        return cls(lines, name=name)

    @property
    def data(self) -> list[XYData]:
        return [line.data for line in self]

    @data.setter
    def data(self, data: list[XYData]):
        ndata_in = len(data)
        ndata_now = len(self)
        if ndata_in > ndata_now:
            for _ in range(ndata_now, ndata_in):
                self.append(Line([], []))
        elif ndata_in < ndata_now:
            for _ in range(ndata_in, ndata_now):
                del self[-1]
        for line, d in zip(self, data):
            line.data = d

    @property
    def width(self) -> NDArray[np.float32]:
        """Array of line widths."""
        return np.array([line.width for line in self], dtype=np.float32)

    @width.setter
    def width(self, width: float | Sequence[float]):
        if is_real_number(width):
            _width = [width] * len(self)
        else:
            _width = np.asarray(width, dtype=np.float32)
        if len(_width) != len(self):
            raise ValueError(
                f"width must be a float or a sequence of length {len(self)}"
            )
        for line, w in zip(self, _width):
            line.width = w

    @property
    def color(self) -> NDArray[np.float32]:
        """2D array of RGBA color values."""
        return np.array([line.color for line in self], dtype=np.float32)

    @color.setter
    def color(self, color: str | Sequence[str]):
        col = as_color_array(color, len(self))
        for line, c in zip(self, col):
            line.color = c

    @property
    def style(self) -> Sequence[LineStyle]:
        """Array of line styles."""
        return np.array([line.style for line in self], dtype=np.float32)

    @style.setter
    def style(self, style: str | Sequence[str]):
        styles = as_any_1d_array(style, len(self))
        for line, s in zip(self, styles):
            line.style = s

    def with_hover_texts(self, text: str | Iterable[Any]) -> Self:
        if isinstance(text, str):
            texts = [text] * len(self)
        else:
            texts = [str(t) for t in text]
            if len(texts) != len(self):
                raise ValueError("Length of texts must match the number of lines.")
        for line, txt in zip(self, texts):
            line.with_hover_text(txt)
        return self

    def with_hover_template(
        self,
        template: str,
        extra: Any | None = None,
    ) -> Self:
        """Define hover template to the layer."""
        if self._backend_name in ("plotly", "bokeh"):  # conversion for HTML
            template = template.replace("\n", "<br>")
        params = parse_texts(template, len(self), extra)
        # set default format keys
        if "i" not in params:
            params["i"] = np.arange(len(self))
        texts = [
            template.format(**{k: v[i] for k, v in params.items()})
            for i in range(len(self))
        ]
        return self.with_hover_texts(texts)

    def _prep_markers(
        self,
        *,
        symbol: Symbol,
        size: float | None = None,
        alpha: float = 1.0,
        hatch: str | Hatch = Hatch.SOLID,
    ):
        from whitecanvas.layers.group import MarkerCollection

        markers = []
        size = theme._default("markers.size", size)
        for layer in self:
            color = layer.color
            mk = Markers(
                *layer.data, symbol=symbol, size=size, color=color, alpha=alpha,
                hatch=hatch, name=f"markers-of-{layer.name}", backend=self._backend_name
            )  # fmt: skip
            markers.append(mk)
        mcol = MarkerCollection(markers, name=f"markers-of-{self.name}")
        return mcol
