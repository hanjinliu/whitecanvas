from whitecanvas.canvas import CanvasGrid, CanvasVGrid, CanvasHGrid, SingleCanvas
from whitecanvas.backend import Backend


def grid(
    nrows: int = 1,
    ncols: int = 1,
    *,
    backend: Backend | str | None = None,
) -> CanvasGrid:
    return CanvasGrid.uniform(nrows, ncols, backend=backend)


def grid_nonuniform(
    heights: list[int],
    widths: list[int],
    *,
    backend: Backend | str | None = None,
) -> CanvasGrid:
    return CanvasGrid(heights, widths, backend=backend)


def vgrid(
    nrows: int = 1,
    *,
    backend: Backend | str | None = None,
) -> CanvasVGrid:
    return CanvasVGrid.uniform(nrows, backend=backend)


def vgrid_nonuniform(
    heights: list[int],
    *,
    backend: Backend | str | None = None,
) -> CanvasVGrid:
    return CanvasVGrid(heights, backend=backend)


def hgrid(
    ncols: int = 1,
    *,
    backend: Backend | str | None = None,
) -> CanvasHGrid:
    return CanvasHGrid.uniform(ncols, backend=backend)


def hgrid_nonuniform(
    widths: list[int],
    *,
    backend: Backend | str | None = None,
) -> CanvasHGrid:
    return CanvasHGrid(widths, backend=backend)


def new_canvas(
    backend: Backend | str | None = None,
    **kwargs,
) -> SingleCanvas:
    """Create a new canvas with a single cell."""
    _grid = grid(backend=backend)
    _grid.add_canvas(0, 0, **kwargs)
    return SingleCanvas(_grid)
