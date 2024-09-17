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
    plot,
    rug,
    scatter,
    spans,
    step,
    text,
    title,
    update_axes,
    update_font,
    update_labels,
    vline,
    xlabel,
    xlim,
    xticks,
    ylabel,
    ylim,
    yticks,
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
    "step",
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
    "plot",
    "scatter",
    "show",
    "subplots",
    "current_grid",
    "current_canvas",
    "use",
    "xlim",
    "ylim",
    "xlabel",
    "ylabel",
    "xticks",
    "yticks",
    "title",
]
