"""
Plot API for whitecanvas.

>>> import numpy as np
>>> from whitecanvas import plot as plt
>>> plt.line(np.sin([1, 2, 3])).with_markers()
>>> plt.show()
"""

from whitecanvas.plot._canvases import (
    current_canvas,
    current_grid,
    figure,
    show,
    subplots,
    use,
)
from whitecanvas.plot._methods import (
    band,
    bars,
    cat,
    cat_x,
    cat_xy,
    cat_y,
    errorbars,
    hist,
    hline,
    infcurve,
    infline,
    kde,
    legend,
    line,
    markers,
    rug,
    spans,
    text,
    update_axes,
    update_font,
    update_labels,
    vline,
)

__all__ = [
    "line",
    "hline",
    "vline",
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
    "cat_x",
    "cat_y",
    "cat_xy",
    "update_axes",
    "update_labels",
    "update_font",
    "legend",
    "figure",
    "show",
    "subplots",
    "current_grid",
    "current_canvas",
    "use",
]
