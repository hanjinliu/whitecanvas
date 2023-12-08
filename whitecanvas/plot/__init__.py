"""
Plot API for whitecanvas.

>>> import numpy as np
>>> from whitecanvas import plot as plt
>>> plt.line(np.sin([1, 2, 3])).with_markers()
>>> plt.show()
"""

from whitecanvas.plot._methods import (
    line, markers, bars, band, hist, spans, infcurve, infline, errorbars,
    kde, rug, text, cat,
)  # fmt: skip
from whitecanvas.plot._canvases import current_canvas, current_grid, show, subplots
from whitecanvas.core import new_canvas

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
    "subplots",
    "current_grid",
    "current_canvas",
]
