__version__ = "0.3.2"

from whitecanvas import theme
from whitecanvas.canvas import link_axes
from whitecanvas.core import (
    new_canvas,
    new_canvas_3d,
    new_col,
    new_grid,
    new_jointgrid,
    new_row,
    wrap_canvas,
)

__all__ = [
    "new_canvas",
    "new_canvas_3d",
    "new_col",
    "new_grid",
    "new_row",
    "new_jointgrid",
    "wrap_canvas",
    "theme",
    "link_axes",
]
