"""
Plot API for whitecanvas.

>>> import numpy as np
>>> from whitecanvas import plot as plt
>>> plt.line(np.sin([1, 2, 3])).with_markers()
>>> plt.show()
"""

from whitecanvas.core import new_canvas
from whitecanvas.plot._canvases import current_canvas, current_grid, show, subplots
from whitecanvas.plot._methods import (
    band,
    bars,
    cat,
    errorbars,
    hist,
    infcurve,
    infline,
    kde,
    line,
    markers,
    rug,
    spans,
    text,
)

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
