from whitecanvas.canvas import CanvasGrid, CanvasVGrid, CanvasHGrid, SingleCanvas
from whitecanvas.backend import Backend


def grid(
    nrows: int = 1,
    ncols: int = 1,
    *,
    link_x: bool = False,
    link_y: bool = False,
    backend: Backend | str | None = None,
) -> CanvasGrid:
    return CanvasGrid.uniform(
        nrows, ncols, link_x=link_x, link_y=link_y, backend=backend
    )


def grid_nonuniform(
    heights: list[int],
    widths: list[int],
    *,
    link_x: bool = False,
    link_y: bool = False,
    backend: Backend | str | None = None,
) -> CanvasGrid:
    return CanvasGrid(heights, widths, link_x=link_x, link_y=link_y, backend=backend)


def vgrid(
    nrows: int = 1,
    *,
    link_x: bool = False,
    link_y: bool = False,
    backend: Backend | str | None = None,
) -> CanvasVGrid:
    return CanvasVGrid.uniform(nrows, link_x=link_x, link_y=link_y, backend=backend)


def vgrid_nonuniform(
    heights: list[int],
    *,
    link_x: bool = False,
    link_y: bool = False,
    backend: Backend | str | None = None,
) -> CanvasVGrid:
    return CanvasVGrid(heights, link_x=link_x, link_y=link_y, backend=backend)


def hgrid(
    ncols: int = 1,
    *,
    link_x: bool = False,
    link_y: bool = False,
    backend: Backend | str | None = None,
) -> CanvasHGrid:
    return CanvasHGrid.uniform(ncols, link_x=link_x, link_y=link_y, backend=backend)


def hgrid_nonuniform(
    widths: list[int],
    *,
    link_x: bool = False,
    link_y: bool = False,
    backend: Backend | str | None = None,
) -> CanvasHGrid:
    return CanvasHGrid(widths, link_x=link_x, link_y=link_y, backend=backend)


def new_canvas(
    backend: Backend | str | None = None,
    **kwargs,
) -> SingleCanvas:
    """Create a new canvas with a single cell."""
    _grid = grid(backend=backend)
    _grid.add_canvas(0, 0, **kwargs)
    return SingleCanvas(_grid)
