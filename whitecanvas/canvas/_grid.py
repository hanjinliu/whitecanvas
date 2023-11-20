from __future__ import annotations
from typing import Iterator

from typing_extensions import override
import numpy as np
from numpy.typing import NDArray
from whitecanvas import protocols
from whitecanvas.backend import Backend
from whitecanvas.canvas import Canvas, CanvasBase
from whitecanvas.utils.normalize import norm_color
from whitecanvas.theme import get_theme


class CanvasGrid:
    def __init__(
        self,
        heights: list[int],
        widths: list[int],
        *,
        link_x: bool = False,
        link_y: bool = False,
        backend: Backend | str | None = None,
    ) -> None:
        self._heights = heights
        self._widths = widths
        self._backend_installer = Backend(backend)
        self._backend_object = self._create_backend()
        self._canvas_array = np.empty((len(heights), len(widths)), dtype=object)
        self._canvas_array.fill(None)

        # link axes
        self._x_linked = link_x
        self._y_linked = link_y

        # update settings
        theme = get_theme()
        self.background_color = theme.background_color

    @classmethod
    def uniform(
        cls,
        nrows: int = 1,
        ncols: int = 1,
        *,
        link_x: bool = False,
        link_y: bool = False,
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
        return CanvasGrid(
            [1] * nrows, [1] * ncols, link_x=link_x, link_y=link_y,
            backend=backend,
        )  # fmt: skip

    @property
    def shape(self) -> tuple[int, int]:
        """The (row, col) shape of the grid"""
        return self._canvas_array.shape

    @property
    def x_linked(self) -> bool:
        """Whether the x-axis of all canvases are linked."""
        return self._x_linked

    @x_linked.setter
    def x_linked(self, value: bool):
        value = bool(value)
        if value == self._x_linked:
            return
        if value:
            for _, canvas in self.iter_canvas():
                canvas.x.lim_changed.connect(self._align_xlims, unique=True)
        else:
            for _, canvas in self.iter_canvas():
                canvas.x.lim_changed.disconnect(self._align_xlims)
        self._x_linked = value

    @property
    def y_linked(self) -> bool:
        """Whether the y-axis of all canvases are linked."""
        return self._y_linked

    @y_linked.setter
    def y_linked(self, value: bool):
        value = bool(value)
        if value == self._y_linked:
            return
        if value:
            for _, canvas in self.iter_canvas():
                canvas.y.lim_changed.connect(self._align_ylims, unique=True)
        else:
            for _, canvas in self.iter_canvas():
                canvas.y.lim_changed.disconnect(self._align_ylims)
        self._y_linked = value

    def __repr__(self) -> str:
        cname = type(self).__name__
        return f"<{cname} {self._heights} x {self._widths}>"

    def __getitem__(self, key: tuple[int, int]) -> Canvas:
        canvas = self._canvas_array[key]
        if canvas is None:
            raise IndexError(f"Canvas at {key} is not set")
        return canvas

    def _create_backend(self) -> protocols.CanvasGridProtocol:
        return self._backend_installer.get("CanvasGrid")(self._heights, self._widths)

    def _align_xlims(self, lim: tuple[float, float]):
        for _, canvas in self.iter_canvas():
            with canvas.x.lim_changed.blocked():
                canvas.x.lim = lim

    def _align_ylims(self, lim: tuple[float, float]):
        for _, canvas in self.iter_canvas():
            with canvas.y.lim_changed.blocked():
                canvas.y.lim = lim

    def add_canvas(
        self,
        row: int,
        col: int,
        rowspan: int = 1,
        colspan: int = 1,
        **kwargs,
    ) -> Canvas:
        """Add a canvas to the grid at the given position"""
        if rowspan < 1 or colspan < 1:
            raise ValueError(
                f"rowspan and colspan must be positive, got {rowspan} and {colspan}"
            )
        r1 = row + rowspan
        c1 = col + colspan
        for idx, item in np.ndenumerate(self._canvas_array[row:r1, col:c1]):
            if item is not None:
                raise ValueError(f"Canvas already exists at {idx}")
        backend_canvas = self._backend_object._plt_add_canvas(
            row, col, rowspan, colspan
        )
        canvas = self._canvas_array[row:r1, col:c1] = Canvas.from_backend(
            backend_canvas, backend=self._backend_installer, **kwargs
        )
        # Now backend axes/viewbox are created, we can install mouse events
        canvas._install_mouse_events()

        # link axes if needed
        if self.x_linked:
            canvas.x.lim_changed.connect(self._align_xlims, unique=True)
        if self.y_linked:
            canvas.y.lim_changed.connect(self._align_ylims, unique=True)
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

    def show(self) -> None:
        """Show the grid."""
        self._backend_object._plt_set_visible(True)

    def hide(self) -> None:
        """Hide the grid."""
        self._backend_object._plt_set_visible(False)

    @property
    def background_color(self) -> NDArray[np.floating]:
        """Background color of the canvas."""
        return norm_color(self._backend_object._plt_get_background_color())

    @background_color.setter
    def background_color(self, color):
        self._backend_object._plt_set_background_color(norm_color(color))

    def screenshot(self) -> NDArray[np.uint8]:
        """Return a screenshot of the grid."""
        return self._backend_object._plt_screenshot()

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


class CanvasVGrid(CanvasGrid):
    @override
    def __init__(
        self,
        heights: list[int],
        *,
        link_x: bool = False,
        link_y: bool = False,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__(heights, [1], link_x=link_x, link_y=link_y, backend=backend)

    @override
    def __getitem__(self, key: int) -> Canvas:
        canvas = self._canvas_array[key, 0]
        if canvas is None:
            raise IndexError(f"Canvas at {key} is not set")
        return canvas

    @override
    @classmethod
    def uniform(
        cls, nrows: int = 1, *, backend: Backend | str | None = None
    ) -> CanvasVGrid:
        return CanvasVGrid([1] * nrows, backend=backend)

    @override
    def add_canvas(self, row: int, span: int = 1, **kwargs) -> Canvas:
        return super().add_canvas(row, 0, span, 1, **kwargs)


class CanvasHGrid(CanvasGrid):
    @override
    def __init__(
        self,
        widths: list[int],
        *,
        link_x: bool = False,
        link_y: bool = False,
        backend: Backend | str | None = None,
    ) -> None:
        super().__init__([1], widths, link_x=link_x, link_y=link_y, backend=backend)

    @override
    def __getitem__(self, key: int) -> Canvas:
        canvas = self._canvas_array[0, key]
        if canvas is None:
            raise IndexError(f"Canvas at {key} is not set")
        return canvas

    @override
    @classmethod
    def uniform(
        cls, ncols: int = 1, *, backend: Backend | str | None = None
    ) -> CanvasHGrid:
        return CanvasHGrid([1] * ncols, backend=backend)

    @override
    def add_canvas(self, col: int, span: int = 1, **kwargs) -> Canvas:
        return super().add_canvas(0, col, 1, span, **kwargs)


class SingleCanvas(CanvasBase):
    def __init__(self, grid: CanvasGrid):
        if grid.shape != (1, 1):
            raise ValueError(f"Grid shape must be (1, 1), got {grid.shape}")
        self._grid = grid
        _it = grid.iter_canvas()
        _, canvas = next(_it)
        if next(_it, None) is not None:
            raise ValueError("Grid must have only one canvas")
        self._main_canvas = canvas
        super().__init__(palette=self._main_canvas._color_palette)

    def _get_backend(self) -> Backend:
        """Return the backend."""
        return self._main_canvas._backend_installer

    def _canvas(self):
        return self._main_canvas._backend

    def show(self) -> None:
        """Show the grid."""
        self._grid.show()

    def hide(self) -> None:
        """Hide the grid."""
        self._grid.hide()

    @property
    def background_color(self) -> NDArray[np.floating]:
        """Background color of the canvas."""
        return self._grid.background_color

    @background_color.setter
    def background_color(self, color):
        self._grid.background_color = color

    def screenshot(self) -> NDArray[np.uint8]:
        """Return a screenshot of the grid."""
        return self._grid.screenshot()

    def _repr_png_(self):
        """Return PNG representation of the widget for QtConsole."""
        return self._grid._repr_png_()
