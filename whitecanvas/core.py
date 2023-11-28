import sys
from typing import Any

from whitecanvas.canvas import (
    CanvasGrid,
    CanvasVGrid,
    CanvasHGrid,
    SingleCanvas,
    Canvas,
)
from whitecanvas.backend import Backend
from whitecanvas.types import ColormapType


def grid(
    nrows: int = 1,
    ncols: int = 1,
    *,
    link_x: bool = False,
    link_y: bool = False,
    backend: Backend | str | None = None,
) -> CanvasGrid:
    """
    Create a canvas grid with uniform cell sizes.

    Parameters
    ----------
    nrows : int, optional
        Number of rows, by default 1
    ncols : int, optional
        Number of columns, by default 1
    link_x : bool, optional
        Whether to link x axes, by default False
    link_y : bool, optional
        Whether to link y axes, by default False
    backend : Backend or str, optional
        Backend name.

    Returns
    -------
    CanvasGrid
        Grid of empty canvases.
    """
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
    *,
    size: tuple[int, int] | None = None,
    palette: str | ColormapType | None = None,
) -> SingleCanvas:
    """Create a new canvas with a single cell."""
    _grid = grid(backend=backend)
    _grid.add_canvas(0, 0, palette=palette)
    cvs = SingleCanvas(_grid)
    if size is not None:
        cvs.size = size
    return cvs


def wrap_canvas(obj: Any, palette=None) -> Canvas:
    """
    Wrap a backend object into a whitecanvas Canvas.

    >>> import matplotlib.pyplot as plt
    >>> canvas = wrap_canvas(plt.gca())
    """
    mod = type(obj).__module__.split(".")[0]
    typ = type(obj).__name__

    if _is_in_module(typ, "matplotlib", "Axes"):
        from matplotlib.axes import Axes
        from whitecanvas.backend.matplotlib import Canvas as BackendCanvas

        if not isinstance(obj, Axes):
            raise TypeError(f"Expected matplotlib Axes, got {typ}")
        backend = "matplotlib"

    elif _is_in_module(typ, "plotly", "FigureWidget"):
        from plotly.graph_objs import FigureWidget
        from whitecanvas.backend.plotly import Canvas as BackendCanvas

        if not isinstance(obj, FigureWidget):
            raise TypeError(f"Expected plotly FigureWidget, got {typ}")
        backend = "plotly"
    elif _is_in_module(typ, "bokeh", "Plot"):
        from bokeh.models import Plot
        from whitecanvas.backend.bokeh import Canvas as BackendCanvas

        if not isinstance(obj, Plot):
            raise TypeError(f"Expected bokeh Plot, got {typ}")
        backend = "bokeh"
    elif _is_in_module(typ, "vispy", "ViewBox"):
        from vispy.scene import ViewBox
        from whitecanvas.backend.vispy import Canvas as BackendCanvas

        if not isinstance(obj, ViewBox):
            raise TypeError(f"Expected vispy ViewBox, got {typ}")
        backend = "vispy"
    elif _is_in_module(typ, "pyqtgraph", "ViewBox"):
        from pyqtgraph import ViewBox
        from whitecanvas.backend.pyqtgraph import Canvas as BackendCanvas

        if not isinstance(obj, ViewBox):
            raise TypeError(f"Expected pyqtgraph ViewBox, got {typ}")
        backend = "pyqtgraph"
    else:
        raise TypeError(f"Cannot convert {typ} to Canvas")
    return Canvas.from_backend(BackendCanvas(obj), palette=palette, backend=backend)


def _is_in_module(typ_str: str, mod_name: str, cls_name: str) -> bool:
    return mod_name in sys.modules and typ_str.split(".")[-1] == cls_name
