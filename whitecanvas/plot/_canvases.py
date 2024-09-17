from __future__ import annotations

from whitecanvas.backend import Backend
from whitecanvas.canvas import Canvas, CanvasGrid, SingleCanvas
from whitecanvas.core import new_canvas, new_grid


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


def show(block: bool = False, flush: bool = True) -> None:
    """Show the current canvas."""
    current_grid().show(block=block)
    if flush:
        Canvas._CURRENT_INSTANCE = None
    return None


def subplots(nrows: int = 1, ncols: int = 1) -> CanvasGrid:
    """Create a new grid of subplots."""
    return new_grid(nrows, ncols).fill()


def figure(
    *,
    size: tuple[float, float] | None = None,
    palette: str | None = None,
) -> SingleCanvas:
    """Create a new grid of subplots."""
    return new_canvas(size=size, palette=palette)


def use(backend: str):
    """Set the backend to use."""
    return Backend(backend)  # update default
