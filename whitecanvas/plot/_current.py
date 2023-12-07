from whitecanvas.canvas import Canvas, SingleCanvas, CanvasGrid


def current_grid() -> CanvasGrid:
    """Return the current canvas grid."""
    grid = CanvasGrid._CURRENT_INSTANCE
    if grid is None:
        grid = SingleCanvas()._grid
    return grid


def current_canvas() -> Canvas:
    """Return the current canvas."""
    canvas = Canvas._CURRENT_INSTANCE
    if canvas is None:
        canvas = SingleCanvas()
    return canvas
