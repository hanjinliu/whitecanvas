from __future__ import annotations

import sys
from typing import Any

from whitecanvas.backend import Backend
from whitecanvas.canvas import (
    Canvas,
    CanvasGrid,
    CanvasHGrid,
    CanvasVGrid,
    SingleCanvas,
)
from whitecanvas.types import ColormapType


def grid(
    nrows: int = 1,
    ncols: int = 1,
    *,
    size: tuple[int, int] | None = None,
    backend: Backend | str | None = None,
) -> CanvasGrid:
    """
    Create a canvas grid with uniform cell sizes.

    Parameters
    ----------
    nrows : int, default 1
        Number of rows.
    ncols : int, default 1
        Number of columns.
    size : (int, int), optional
        Displaying size of the grid (in pixels).
    backend : Backend or str, optional
        Backend name.

    Returns
    -------
    CanvasGrid
        Grid of empty canvases.
    """
    g = CanvasGrid.uniform(nrows, ncols, backend=backend)
    if size is not None:
        g.size = size
    return g


def grid_nonuniform(
    heights: list[int],
    widths: list[int],
    *,
    size: tuple[int, int] | None = None,
    backend: Backend | str | None = None,
) -> CanvasGrid:
    """
    Create a canvas grid with non-uniform cell sizes.

    Parameters
    ----------
    heights : list of int
        Height ratio of the rows.
    widths : list of int
        Width ratio the columns.
    size : (int, int), optional
        Displaying size of the grid (in pixels).
    backend : Backend or str, optional
        Backend name.

    Returns
    -------
    CanvasGrid
        Grid of empty canvases.
    """
    g = CanvasGrid(heights, widths, backend=backend)
    if size is not None:
        g.size = size
    return g


def vgrid(
    nrows: int = 1,
    *,
    size: tuple[int, int] | None = None,
    backend: Backend | str | None = None,
) -> CanvasVGrid:
    """
    Create a vertical canvas grid with uniform cell sizes.

    Parameters
    ----------
    nrows : int, default 1
        Number of rows.
    size : (int, int), optional
        Displaying size of the grid (in pixels).
    backend : Backend or str, optional
        Backend name.

    Returns
    -------
    CanvasVGrid
        1D Grid of empty canvases.
    """
    g = CanvasVGrid.uniform(nrows, backend=backend)
    if size is not None:
        g.size = size
    return g


def vgrid_nonuniform(
    heights: list[int],
    *,
    size: tuple[int, int] | None = None,
    backend: Backend | str | None = None,
) -> CanvasVGrid:
    """
    Create a vertical canvas grid with non-uniform cell sizes.

    Parameters
    ----------
    heights : list of int
        Height ratios of rows.
    size : (int, int), optional
        Displaying size of the grid (in pixels).
    backend : Backend or str, optional
        Backend name.

    Returns
    -------
    CanvasVGrid
        1D Grid of empty canvases.
    """
    g = CanvasVGrid(heights, backend=backend)
    if size is not None:
        g.size = size
    return g


def hgrid(
    ncols: int = 1,
    *,
    size: tuple[int, int] | None = None,
    backend: Backend | str | None = None,
) -> CanvasHGrid:
    """
    Create a horizontal canvas grid with uniform cell sizes.

    Parameters
    ----------
    ncols : int, default 1
        Number of columns.
    size : (int, int), optional
        Displaying size of the grid (in pixels).
    backend : Backend or str, optional
        Backend name.

    Returns
    -------
    CanvasHGrid
        1D Grid of empty canvases.
    """
    g = CanvasHGrid.uniform(ncols, backend=backend)
    if size is not None:
        g.size = size
    return g


def hgrid_nonuniform(
    widths: list[int],
    *,
    size: tuple[int, int] | None = None,
    backend: Backend | str | None = None,
) -> CanvasHGrid:
    """
    Create a horizontal canvas grid with non-uniform cell sizes.

    Parameters
    ----------
    widths : list of int
        Width ratios of columns.
    size : (int, int), optional
        Displaying size of the grid (in pixels).
    backend : Backend or str, optional
        Backend name.

    Returns
    -------
    CanvasHGrid
        1D Grid of empty canvases.
    """
    g = CanvasHGrid(widths, backend=backend)
    if size is not None:
        g.size = size
    return g


def new_canvas(
    backend: Backend | str | None = None,
    *,
    size: tuple[int, int] | None = None,
    palette: str | ColormapType | None = None,
) -> SingleCanvas:
    """
    Create a new canvas with a single cell.

    Parameters
    ----------
    backend : Backend or str, optional
        Backend name.
    size : (int, int), optional
        Displaying size of the canvas (in pixels).
    palette : str or ColormapType, optional
        Color palette of the canvas. This color palette will be used to generate colors
        for the plots.
    """
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
    typ = type(obj).__name__

    if _is_in_module(typ, "matplotlib", "Axes"):
        from matplotlib.axes import Axes

        from whitecanvas.backend.matplotlib import Canvas as BackendCanvas

        if not isinstance(obj, Axes):
            raise TypeError(f"Expected matplotlib Axes, got {typ}")
        backend = "matplotlib"

    elif _is_in_module(typ, "plotly", "Figure"):
        from plotly.graph_objs import Figure

        from whitecanvas.backend.plotly import Canvas as BackendCanvas

        if not isinstance(obj, Figure):
            raise TypeError(f"Expected plotly Figure, got {typ}")
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
