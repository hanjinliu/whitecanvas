from __future__ import annotations

from whitecanvas.backend import Backend
from whitecanvas.canvas import Canvas, CanvasGrid
from whitecanvas.core import grid, new_canvas


def current_grid() -> CanvasGrid:
    """Return the current canvas grid."""
    grid = CanvasGrid._CURRENT_INSTANCE
    if grid is None:
        grid = new_canvas()._grid
    return grid


def current_canvas() -> Canvas:
    """Return the current canvas."""
    canvas = Canvas._CURRENT_INSTANCE
    if canvas is None:
        canvas = new_canvas()
    return canvas


def show(block: bool = False):
    """Show the current canvas."""
    current_grid().show(block=block)


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    backend: Backend | str | None = None,
) -> CanvasGrid:
    """Create a new grid of subplots."""
    return grid(nrows, ncols, backend=backend).fill()
