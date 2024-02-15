__version__ = "0.2.2.dev0"

from whitecanvas import theme
from whitecanvas.canvas import Canvas, CanvasGrid
from whitecanvas.core import (
    new_canvas,
    new_col,
    new_grid,
    new_jointcanvas,
    new_row,
    wrap_canvas,
)

__all__ = [
    "Canvas",
    "CanvasGrid",
    "new_canvas",
    "new_col",
    "new_grid",
    "new_row",
    "new_jointcanvas",
    "wrap_canvas",
    "theme",
]


def __getattr__(name: str):
    import warnings

    if name in ("grid", "grid_nonuniform"):
        warnings.warn(
            f"{name!r} is deprecated. Use `new_grid` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_grid
    elif name in ("vgrid", "vgrid_nonuniform"):
        warnings.warn(
            f"{name!r} is deprecated. Use `new_col` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_col
    elif name in ("hgrid", "hgrid_nonuniform"):
        warnings.warn(
            f"{name!r} is deprecated. Use `new_row` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_row
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
