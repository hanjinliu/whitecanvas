"""
Plot API for whitecanvas.

>>> import numpy as np
>>> from whitecanvas import plot as plt
>>> plt.line(np.sin([1, 2, 3])).with_markers()
>>> plt.show()
"""

from ._methods import (
    line, markers, bars, band, hist, spans, infcurve, infline, errorbars,
    kde, rug, text, cat, new_canvas, show,
)  # fmt: skip
from ._current import current_canvas, current_grid

figure = new_canvas

__all__ = [
    "line",
    "markers",
    "bars",
    "band",
    "hist",
    "spans",
    "infcurve",
    "infline",
    "errorbars",
    "kde",
    "rug",
    "text",
    "cat",
    "figure",
    "show",
    "current_grid",
    "current_canvas",
]
