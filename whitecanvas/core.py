from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, overload

from whitecanvas.backend import Backend
from whitecanvas.canvas import (
    Canvas,
    CanvasGrid,
    CanvasHGrid,
    CanvasVGrid,
    JointGrid,
    SingleCanvas,
)
from whitecanvas.types import ColormapType

if TYPE_CHECKING:
    from typing import Literal

    import pandas as pd
    import polars as pl

    _0_or_1 = Literal[0, 1]


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
    cvs = SingleCanvas._new(palette=palette, backend=backend)
    if size is not None:
        cvs.size = size
    return cvs


def new_canvas_3d(
    backend: Backend | str | None = None,
    *,
    size: tuple[int, int] | None = None,
    palette: str | ColormapType | None = None,
):
    from whitecanvas.canvas.canvas3d._base import SingleCanvas3D

    _grid = CanvasGrid([1], [1], backend=backend)
    _grid.add_canvas_3d(0, 0, palette=palette)
    cvs = SingleCanvas3D(_grid)
    if size is not None:
        cvs.size = size
    return cvs


def new_grid(
    rows: int | Sequence[int] = 1,
    cols: int | Sequence[int] = 1,
    *,
    size: tuple[int, int] | None = None,
    backend: Backend | str | None = None,
) -> CanvasGrid:
    """
    Create a new canvas grid with uniform or non-uniform cell sizes.

    >>> grid = new_grid(2, 3)  # 2x3 grid
    >>> grid = new_grid(2, 3, size=(800, 600))  # 2x3 grid with size 800x600
    >>> grid = new_grid([1, 2], [2, 1])  # 2x2 grid with non-uniform sizes

    If you want to create a 1D grid, use `new_row` or `new_col` instead.

    Parameters
    ----------
    rows : int or sequence of int, default 1
        Number of rows (if an integer is given) or height ratio of the rows (if a
        sequence of intergers is given).
    cols : int or sequence of int, default 1
        Number of columns (if an integer is given) or width ratio of the columns (if a
        sequence of intergers is given).
    size : (int, int), optional
        Displaying size of the grid (in pixels).
    backend : Backend or str, optional
        Backend name, such as "matplotlib:qt".

    Returns
    -------
    CanvasGrid
        Grid of empty canvases.
    """
    heights = _norm_ratio(rows)
    widths = _norm_ratio(cols)
    grid = CanvasGrid(heights, widths, backend=backend)
    if size is not None:
        grid.size = size
    return grid


def new_row(
    cols: int | Sequence[int] = 1,
    *,
    size: tuple[int, int] | None = None,
    backend: Backend | str | None = None,
) -> CanvasHGrid:
    """Create a new horizontal canvas grid with uniform or non-uniform cell sizes."""
    widths = _norm_ratio(cols)
    grid = CanvasHGrid._from_widths(widths, backend=backend)
    if size is not None:
        grid.size = size
    return grid


def new_col(
    rows: int | Sequence[int] = 1,
    *,
    size: tuple[int, int] | None = None,
    backend: Backend | str | None = None,
) -> CanvasVGrid:
    """Create a new vertical canvas grid with uniform or non-uniform cell sizes."""
    heights = _norm_ratio(rows)
    grid = CanvasVGrid._from_heights(heights, backend=backend)
    if size is not None:
        grid.size = size
    return grid


def new_jointgrid(
    backend: Backend | str | None = None,
    *,
    loc: tuple[_0_or_1, _0_or_1] = (1, 0),
    size: tuple[int, int] | None = None,
    palette: str | ColormapType | None = None,
) -> JointGrid:
    """
    Create a new joint grid.

    Parameters
    ----------
    backend : Backend or str, optional
        Backend of the canvas.
    loc : (int, int), default (1, 0)
        Location of the main canvas. Each integer must be 0 or 1.
    size : (int, int), optional
        Size of the canvas in pixel.
    palette : colormap type, optional
        Color palette used for the canvases.

    Returns
    -------
    JointGrid
        Joint grid object.
    """
    joint = JointGrid(loc, palette=palette, backend=backend)
    if size is not None:
        joint.size = size
    return joint


@overload
def load_dataset(name: str, type: None = None, cache: bool = True) -> Any: ...
@overload
def load_dataset(
    name: str, type: Literal["pandas"], cache: bool = True
) -> pd.DataFrame: ...
@overload
def load_dataset(
    name: str, type: Literal["polars"], cache: bool = True
) -> pl.DataFrame: ...


def load_dataset(name, type=None, cache=True) -> Any:
    from urllib.request import urlopen, urlretrieve

    from platformdirs import user_cache_dir

    dataset_names_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/dataset_names.txt"
    url = f"https://raw.githubusercontent.com/mwaskom/seaborn-data/master/{name}.csv"

    if cache:
        cache_dir = Path(user_cache_dir("whitecanvas"))
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        cache_path = Path(user_cache_dir("whitecanvas")) / f"{name}.csv"

        with urlopen(dataset_names_url) as resp:  # noqa: S310
            txt = resp.read()
        assert isinstance(txt, bytes)

        dataset_names = [name.strip() for name in txt.decode().split("\n")]
        dataset_names = list(filter(None, dataset_names))
        if not cache_path.exists():
            if name not in dataset_names:
                raise ValueError(f"'{name}' is not one of the example datasets.")
            urlretrieve(url, cache_path)  # noqa: S310
        full_path = cache_path
    else:
        full_path = url

    return _get_reader(type)(full_path)


def _get_reader(typ) -> Callable[[str], Any]:
    if typ is None:
        if "pandas" in sys.modules:
            typ = "pandas"
        elif "polars" in sys.modules:
            typ = "polars"
        else:
            try:
                import polars as pl

                typ = "polars"
            except ImportError:
                try:
                    import pandas as pd

                    typ = "pandas"
                except ImportError:
                    raise ImportError(
                        "No backend found. Please install either pandas or polars."
                    )

    if typ == "pandas":
        import pandas as pd

        return pd.read_csv
    elif typ == "polars":
        import polars as pl

        return pl.read_csv
    else:
        raise ValueError(f"Unknown type: {typ}")


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
    elif _is_in_module(typ, "bokeh", "figure"):
        from bokeh.plotting import figure

        from whitecanvas.backend.bokeh import Canvas as BackendCanvas

        if not isinstance(obj, figure):
            raise TypeError(f"Expected bokeh figure, got {typ}")
        backend = "bokeh"
    elif _is_in_module(typ, "vispy", "ViewBox"):
        from vispy.scene import ViewBox

        from whitecanvas.backend.vispy import Canvas as BackendCanvas

        if not isinstance(obj, ViewBox):
            raise TypeError(f"Expected vispy ViewBox, got {typ}")
        backend = "vispy"
    elif _is_in_module(typ, "pyqtgraph", "PlotItem"):
        from pyqtgraph import PlotItem

        from whitecanvas.backend.pyqtgraph import Canvas as BackendCanvas

        if not isinstance(obj, PlotItem):
            raise TypeError(f"Expected pyqtgraph PlotItem, got {typ}")
        backend = "pyqtgraph"
    else:
        raise TypeError(f"Cannot convert {typ} to Canvas")
    return Canvas.from_backend(BackendCanvas(obj), palette=palette, backend=backend)


def _is_in_module(typ_str: str, mod_name: str, cls_name: str) -> bool:
    return mod_name in sys.modules and typ_str.split(".")[-1] == cls_name


def _norm_ratio(r: int | Sequence[int]) -> list[int]:
    if hasattr(r, "__int__"):
        out = [1] * int(r)
    else:
        out: list[int] = []
        for x in r:
            if not hasattr(x, "__int__"):
                raise ValueError(f"Invalid value for size ratio: {r!r}.")
            out.append(int(x))
        if len(out) == 0:
            raise ValueError("Size ratio must not be empty.")
    return out
