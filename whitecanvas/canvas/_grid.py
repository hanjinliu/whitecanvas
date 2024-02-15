from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal

import numpy as np
from numpy.typing import NDArray
from psygnal import Signal, SignalGroup
from typing_extensions import override

from whitecanvas import protocols
from whitecanvas.backend import Backend
from whitecanvas.canvas import Canvas, CanvasBase
from whitecanvas.canvas._linker import link_axes
from whitecanvas.theme import get_theme
from whitecanvas.utils.normalize import arr_color

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.types import ColormapType


class GridEvents(SignalGroup):
    drawn = Signal()


class CanvasGrid:
    _CURRENT_INSTANCE: CanvasGrid | None = None
    events: GridEvents

    def __init__(
        self,
        heights: list[int],
        widths: list[int],
        *,
        backend: Backend | str | None = None,
    ) -> None:
        self._heights = heights
        self._widths = widths
        self._backend = Backend(backend)
        self._backend_object = self._create_backend()
        self._canvas_array = np.empty((len(heights), len(widths)), dtype=object)
        self._canvas_array.fill(None)

        # link axes
        self._x_linked = False
        self._y_linked = False
        self._x_linker_ref = None
        self._y_linker_ref = None

        # update settings
        theme = get_theme()
        self.background_color = theme.background_color
        self.size = theme.canvas_size
        self.events = GridEvents()
        self.__class__._CURRENT_INSTANCE = self

    @classmethod
    def uniform(
        cls,
        nrows: int = 1,
        ncols: int = 1,
        *,
        backend: Backend | str | None = None,
    ) -> CanvasGrid:
        """
        Create a canvas grid with uniform row and column sizes.

        Parameters
        ----------
        nrows : int
            The number of rows in the grid.
        ncols : int
            The number of columns in the grid.
        backend : backend-like, optional
            The backend to use for the grid.
        """
        return CanvasGrid([10] * nrows, [10] * ncols, backend=backend)

    @property
    def shape(self) -> tuple[int, int]:
        """The (row, col) shape of the grid"""
        return self._canvas_array.shape

    def link_x(self, *, future: bool = True, hide_ticks: bool = True) -> Self:
        """
        Link all the x-axes of the canvases in the grid.

        >>> from whitecanvas import new_grid
        >>> g = new_grid(2, 2).link_x()  # link x-axes of all canvases

        Parameters
        ----------
        future : bool, default True
            If Ture, all the canvases added in the future will also be linked. Only link
            the existing canvases if False.
        """
        if self._x_linker_ref is not None:
            self._x_linker_ref.unlink_all()  # initialize linker
        to_link = []
        for (_r, _), _canvas in self.iter_canvas():
            to_link.append(_canvas)
            if hide_ticks and _r != self.shape[0] - 1:
                _canvas.x.ticks.visible = False
        self._x_linker_ref = link_axes(to_link)
        if future:
            self._x_linked = True
        return self

    def link_y(self, *, future: bool = True, hide_ticks: bool = True) -> Self:
        """
        Link all the y-axes of the canvases in the grid.

        >>> from whitecanvas import new_grid
        >>> g = new_grid(2, 2).link_y()  # link y-axes of all canvases

        Parameters
        ----------
        future : bool, default True
            If Ture, all the canvases added in the future will also be linked. Only link
            the existing canvases if False.
        """
        if self._y_linker_ref is not None:
            self._y_linker_ref.unlink_all()
        to_link = []
        for (_, _c), _canvas in self.iter_canvas():
            to_link.append(_canvas)
            if hide_ticks and _c != self.shape[1] - 1:
                _canvas.y.ticks.visible = False
        self._y_linker_ref = link_axes(to_link)
        if future:
            self._y_linked = True
        return self

    def __repr__(self) -> str:
        cname = type(self).__name__
        w, h = self._size
        hex_id = hex(id(self))
        return f"<{cname} ({w:.1f} x {h:.1f}) at {hex_id}>"

    def __getitem__(self, key: tuple[int, int]) -> Canvas:
        canvas = self._canvas_array[key]
        if canvas is None:
            raise ValueError(f"Canvas at {key} is not set")
        elif isinstance(canvas, np.ndarray):
            raise ValueError(f"Cannot index by {key}.")
        return canvas

    def _create_backend(self) -> protocols.CanvasGridProtocol:
        return self._backend.get("CanvasGrid")(
            self._heights, self._widths, self._backend._app
        )

    def fill(self, palette: ColormapType | None = None) -> Self:
        """Fill the grid with canvases."""
        for _ in self.iter_add_canvas(palette=palette):
            pass
        return self

    def add_canvas(
        self,
        row: int,
        col: int,
        rowspan: int = 1,
        colspan: int = 1,
        *,
        palette: str | None = None,
    ) -> Canvas:
        """Add a canvas to the grid at the given position"""
        for idx, item in np.ndenumerate(self._canvas_array[row, col]):
            if item is not None:
                raise ValueError(f"Canvas already exists at {idx}")
        backend_canvas = self._backend_object._plt_add_canvas(
            row, col, rowspan, colspan
        )
        canvas = self._canvas_array[row, col] = Canvas.from_backend(
            backend_canvas,
            backend=self._backend,
            palette=palette,
        )
        # Now backend axes/viewbox are created, we can install mouse events
        canvas._install_mouse_events()

        # link axes if needed
        if self._x_linked:
            self._x_linker_ref.link(canvas.x)
        if self._y_linked:
            self._y_linker_ref.link(canvas.y)
        canvas.events.drawn.connect(self.events.drawn.emit, unique=True)
        return canvas

    def iter_add_canvas(self, **kwargs) -> Iterator[Canvas]:
        for row in range(len(self._heights)):
            for col in range(len(self._widths)):
                yield self.add_canvas(row, col, **kwargs)

    def iter_canvas(self) -> Iterator[tuple[tuple[int, int], Canvas]]:
        yielded: set[int] = set()
        for idx, canvas in np.ndenumerate(self._canvas_array):
            _id = id(canvas)
            if canvas is None or _id in yielded:
                continue
            yield idx, canvas
            yielded.add(_id)

    def show(self, block=False) -> None:
        """Show the grid."""
        from whitecanvas.backend._app import get_app

        # TODO: implement other event loops
        app = get_app(self._backend._app)
        _backend_app = app.get_app()
        out = self._backend_object._plt_show()

        if out is NotImplemented:
            from whitecanvas.backend._window import view

            view(self, self._backend.app)

        if block:
            # TODO: automatically block the event loop or enable ipython
            # GUI mode if needed.
            app.run_app()

    @property
    def background_color(self) -> NDArray[np.floating]:
        """Background color of the canvas."""
        return arr_color(self._backend_object._plt_get_background_color())

    @background_color.setter
    def background_color(self, color):
        self._backend_object._plt_set_background_color(arr_color(color))

    def screenshot(self) -> NDArray[np.uint8]:
        """Return a screenshot of the grid."""
        return self._backend_object._plt_screenshot()

    @property
    def size(self) -> tuple[int, int]:
        """Size in width x height."""
        return self._size

    @size.setter
    def size(self, size: tuple[int, int]):
        w, h = size
        if w <= 0 or h <= 0:
            raise ValueError("Size must be positive")
        self._size = (int(w), int(h))
        self._backend_object._plt_set_figsize(*self._size)

    def _repr_png_(self):
        """Return PNG representation of the widget for QtConsole."""
        from io import BytesIO

        try:
            from imageio import imwrite
        except ImportError:
            return None

        rendered = self.screenshot()
        if rendered is not None:
            with BytesIO() as file_obj:
                imwrite(file_obj, rendered, format="png")
                file_obj.seek(0)
                return file_obj.read()
        return None

    def _ipython_display_(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self._backend_object, "_ipython_display_"):
            return self._backend_object._ipython_display_(*args, **kwargs)
        raise NotImplementedError()

    def _repr_mimebundle_(self, *args: Any, **kwargs: Any) -> dict:
        if hasattr(self._backend_object, "_repr_mimebundle_"):
            return self._backend_object._repr_mimebundle_(*args, **kwargs)
        raise NotImplementedError()

    def _repr_html_(self, *args: Any, **kwargs: Any) -> str:
        if hasattr(self._backend_object, "_repr_html_"):
            return self._backend_object._repr_html_(*args, **kwargs)
        raise NotImplementedError()

    def to_html(self, file: str | None = None) -> str:
        """Return HTML representation of the grid."""
        html = self._backend.get("to_html")(self._backend_object)
        if file is not None:
            Path(file).write_text(html, encoding="utf-8")
        return html


class CanvasVGrid(CanvasGrid):
    @override
    def __init__(
        self,
        heights: list[int],
        *,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(heights, [1], backend=backend)

    @override
    def __getitem__(self, key: int) -> Canvas:
        canvas = self._canvas_array[key, 0]
        if canvas is None:
            raise ValueError(f"Canvas at {key} is not set")
        return canvas

    @override
    @classmethod
    def uniform(
        cls,
        nrows: int = 1,
        *,
        backend: Backend | str | None = None,
    ) -> CanvasVGrid:
        return CanvasVGrid([1] * nrows, backend=backend)

    @override
    def add_canvas(self, row: int, **kwargs) -> Canvas:
        return super().add_canvas(row, 0, **kwargs)

    @override
    def iter_add_canvas(self, **kwargs) -> Iterator[Canvas]:
        for row in range(len(self._heights)):
            yield self.add_canvas(row, **kwargs)


class CanvasHGrid(CanvasGrid):
    @override
    def __init__(
        self,
        widths: list[int],
        *,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__([1], widths, backend=backend)

    @override
    def __getitem__(self, key: int) -> Canvas:
        canvas = self._canvas_array[0, key]
        if canvas is None:
            raise ValueError(f"Canvas at {key} is not set")
        return canvas

    @override
    @classmethod
    def uniform(
        cls,
        ncols: int = 1,
        *,
        backend: Backend | str | None = None,
    ) -> CanvasHGrid:
        return CanvasHGrid([1] * ncols, backend=backend)

    @override
    def add_canvas(self, col: int, **kwargs) -> Canvas:
        return super().add_canvas(0, col, **kwargs)

    @override
    def iter_add_canvas(self, **kwargs) -> Iterator[Canvas]:
        for col in range(len(self._widths)):
            yield self.add_canvas(col, **kwargs)


class _CanvasWithGrid(CanvasBase):
    def __init__(self, canvas: Canvas, grid: CanvasGrid):
        self._main_canvas = canvas
        self._grid = grid
        super().__init__(palette=canvas._color_palette)

    def _get_backend(self) -> Backend:
        """Return the backend."""
        return self._main_canvas._backend

    def _canvas(self):
        return self._main_canvas._backend_object

    @property
    def native(self) -> Any:
        """The native backend object."""
        return self._main_canvas.native

    def show(self, block: bool = False) -> None:
        """Show the canvas using the method defined in the backend."""
        self._grid.show(block=block)

    @property
    def background_color(self) -> NDArray[np.floating]:
        """Background color of the canvas."""
        return self._grid.background_color

    @background_color.setter
    def background_color(self, color):
        self._grid.background_color = color

    @property
    def size(self) -> tuple[float, float]:
        """Size of the canvas"""
        return self._grid.size

    @size.setter
    def size(self, size: tuple[float, float]):
        self._grid.size = size

    def screenshot(self) -> NDArray[np.uint8]:
        """Return a screenshot of the grid."""
        return self._grid.screenshot()

    def _repr_png_(self):
        """Return PNG representation of the widget for QtConsole."""
        return self._grid._repr_png_()

    def _repr_mimebundle_(self, *args: Any, **kwargs: Any) -> dict:
        return self._grid._repr_mimebundle_(*args, **kwargs)

    def _ipython_display_(self, *args: Any, **kwargs: Any) -> Any:
        return self._grid._ipython_display_(*args, **kwargs)

    def _repr_html_(self, *args: Any, **kwargs: Any) -> str:
        return self._grid._repr_html_(*args, **kwargs)

    def to_html(self, file: str | None = None) -> str:
        """Return HTML representation of the canvas."""
        return self._grid.to_html(file=file)


class SingleCanvas(_CanvasWithGrid):
    """
    A canvas without other subplots.

    This class is the simplest form of canvas. In `matplotlib` terms, it is a figure
    with a single axes.
    """

    def __init__(self, grid: CanvasGrid):
        if grid.shape != (1, 1):
            raise ValueError(f"Grid shape must be (1, 1), got {grid.shape}")
        self._grid = grid
        _it = grid.iter_canvas()
        _, canvas = next(_it)
        if next(_it, None) is not None:
            raise ValueError("Grid must have only one canvas")
        self._main_canvas = canvas
        super().__init__(canvas, grid)

        # NOTE: events, dims etc are not shared between the main canvas and the
        # SingleCanvas instance. To avoid confusion, the first and the only canvas
        # should be replaces with the SingleCanvas instance.
        grid._canvas_array[0, 0] = self
        self.events.drawn.connect(self._main_canvas.events.drawn.emit, unique=True)


_0or1 = Literal[0, 1]


class JointCanvas(_CanvasWithGrid):
    """
    Grid with a main (joint) canvas and two marginal canvases.

    The marginal canvases shares the x-axis and y-axis with the main canvas.
    """

    def __init__(
        self,
        loc: tuple[_0or1, _0or1] = (1, 0),
        palette: str | ColormapType | None = None,
        backend: Backend | str | None = None,
    ):
        widths = [1, 1]
        heights = [1, 1]
        rloc, cloc = loc
        if rloc not in (0, 1) or cloc not in (0, 1):
            raise ValueError(f"Invalid location {loc!r}.")
        widths[rloc] = heights[cloc] = 3
        grid = CanvasGrid(widths, heights, backend=Backend(backend))
        canvas = grid.add_canvas(rloc, cloc, palette=palette)
        self._x_canvas = grid.add_canvas(1 - rloc, cloc)
        self._y_canvas = grid.add_canvas(rloc, 1 - cloc)

        super().__init__(canvas, grid)

        # NOTE: events, dims etc are not shared between the main canvas and the
        # JointCanvas instance. To avoid confusion, the main canvas should be replaces
        # with the JointCanvas instance.
        grid._canvas_array[rloc, cloc] = self
        self.events.drawn.connect(canvas.events.drawn.emit, unique=True)

        # link axes
        self._x_linker = link_axes([self._main_canvas.x, self._x_canvas.x])
        self._y_linker = link_axes([self._main_canvas.y, self._y_canvas.y])

    @property
    def x_canvas(self) -> Canvas:
        """The canvas at the x-axis."""
        return self._x_canvas

    @property
    def y_canvas(self) -> Canvas:
        """The canvas at the y-axis."""
        return self._y_canvas
