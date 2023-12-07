from whitecanvas.canvas import Canvas, CanvasGrid
from whitecanvas.core import new_canvas


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
