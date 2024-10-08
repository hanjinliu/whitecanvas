from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, overload

import numpy as np
from cmap import Color
from numpy.typing import NDArray
from psygnal import Signal, SignalGroup
from typing_extensions import override

from whitecanvas import protocols
from whitecanvas._json_utils import CustomEncoder, color_to_hex
from whitecanvas.backend import Backend
from whitecanvas.canvas import Canvas, CanvasBase
from whitecanvas.canvas._linker import link_axes
from whitecanvas.layers._deserialize import construct_layers
from whitecanvas.theme import get_theme
from whitecanvas.utils.normalize import arr_color

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.types import ColormapType


class GridEvents(SignalGroup):
    drawn = Signal()


class _Serializable(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    @abstractmethod
    def from_dict(
        cls, d: dict[str, Any], backend: Backend | str | None = None
    ) -> Self: ...

    @classmethod
    def read_json(
        cls,
        path_or_str: str | Path,
        *,
        backend: Backend | str | None = None,
    ) -> Self:
        """
        Construct a canvas grid from a JSON file or string.

        Parameters
        ----------
        path_or_str : str or Path
            The path to the JSON file or the JSON string.
        backend : Backend or str, optional
            The backend of the canvas.
        """

        if isinstance(path_or_str, Path):
            txt = path_or_str.read_text()
        elif not isinstance(path_or_str, str):
            raise TypeError(f"Expected a path or a string, got {path_or_str!r}")
        else:
            if "{" in path_or_str:
                txt = path_or_str
            else:
                txt = Path(path_or_str).read_text()
        return cls.from_dict(json.loads(txt), backend=backend)

    @overload
    def write_json(self) -> str: ...
    @overload
    def write_json(self, path: str | Path) -> None: ...

    def write_json(self, path: str | Path | None = None) -> str | None:
        """Return a JSON string or write to a file."""

        txt = json.dumps(color_to_hex(self.to_dict()), indent=2, cls=CustomEncoder)
        if path is None:
            return txt
        if isinstance(path, str):
            path = Path(path)
        path.write_text(txt)
        return None


class CanvasGrid(_Serializable):
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
        for (_r, _), _canvas in self._iter_canvas():
            to_link.append(_canvas.x)
            if hide_ticks and _r != self.shape[0] - 1:
                _canvas.x.ticks.visible = False
        self._x_linker_ref = link_axes(to_link)
        if future:
            self._x_linked = True
            if hide_ticks:
                self._backend_object._plt_set_spacings(6, 6)
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
        for (_, _c), _canvas in self._iter_canvas():
            to_link.append(_canvas.y)
            if hide_ticks and _c != 0:
                _canvas.y.ticks.visible = False
        self._y_linker_ref = link_axes(to_link)
        if future:
            self._y_linked = True
            if hide_ticks:
                self._backend_object._plt_set_spacings(6, 6)
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
        for _ in self._iter_add_canvas(palette=palette):
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
        canvas.events.drawn.connect(self.events.drawn.emit, unique=True, max_args=None)
        return canvas

    def add_canvas_3d(
        self,
        row: int,
        col: int,
        rowspan: int = 1,
        colspan: int = 1,
        *,
        palette: str | None = None,
    ):
        """Add a canvas to the grid at the given position"""
        from whitecanvas.canvas.canvas3d._base import Canvas3D

        for idx, item in np.ndenumerate(self._canvas_array[row, col]):
            if item is not None:
                raise ValueError(f"Canvas already exists at {idx}")
        backend_canvas = self._backend_object._plt_add_canvas_3d(
            row, col, rowspan, colspan
        )
        canvas = self._canvas_array[row, col] = Canvas3D.from_backend(
            backend_canvas,
            backend=self._backend,
            palette=palette,
        )
        # Now backend axes/viewbox are created, we can install mouse events
        # canvas._install_mouse_events()

        # link axes if needed
        if self._x_linked:
            self._x_linker_ref.link(canvas.x)
        if self._y_linked:
            self._y_linker_ref.link(canvas.y)
        canvas.events.drawn.connect(self.events.drawn.emit, unique=True, max_args=None)
        return canvas

    def _iter_add_canvas(self, **kwargs) -> Iterator[Canvas]:
        for row in range(len(self._heights)):
            for col in range(len(self._widths)):
                yield self.add_canvas(row, col, **kwargs)

    def _iter_canvas(self) -> Iterator[tuple[tuple[int, int], Canvas]]:
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

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        _type_expected = f"{cls.__module__}.{cls.__name__}"
        if (_type := d.get("type")) and _type != _type_expected:
            raise ValueError(f"Expected type {_type_expected!r}, got {_type!r}")
        self = cls(d["heights"], d["widths"], backend=backend)
        if "size" in d:
            self.size = d["size"]
        if "background_color" in d:
            self.background_color = d["background_color"]
        for canvas_dict in d["canvas_array"]:
            if canvas_dict["canvas"]["type"].endswith("3D"):
                canvas = self.add_canvas_3d(canvas_dict["row"], canvas_dict["col"])
            else:
                canvas = self.add_canvas(canvas_dict["row"], canvas_dict["col"])
            canvas._update_from_dict(canvas_dict["canvas"])
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": f"{type(self).__module__}.{type(self).__name__}",
            "heights": self._heights,
            "widths": self._widths,
            "size": self.size,
            "background_color": Color(self.background_color).hex,
            "canvas_array": [
                {"row": r, "col": c, "canvas": canvas.to_dict()}
                for (r, c), canvas in self._iter_canvas()
            ],
        }

    def copy(self, backend: Backend | str | None = None) -> Self:
        """Make a copy of the canvas."""
        if backend is None:
            backend = self._backend
        return self.from_dict(self.to_dict(), backend=backend)

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
    @classmethod
    def _from_heights(
        cls,
        heights: list[int],
        *,
        backend: Backend | str | None = None,
    ) -> CanvasVGrid:
        return CanvasVGrid(heights, [1], backend=backend)

    @override
    def __getitem__(self, key: int) -> Canvas:
        canvas = self._canvas_array[key, 0]
        if canvas is None:
            raise ValueError(f"Canvas at {key} is not set")
        return canvas

    @override
    def add_canvas(self, row: int, **kwargs) -> Canvas:
        return super().add_canvas(row, 0, **kwargs)

    @override
    def _iter_add_canvas(self, **kwargs) -> Iterator[Canvas]:
        for row in range(len(self._heights)):
            yield self.add_canvas(row, **kwargs)


class CanvasHGrid(CanvasGrid):
    @classmethod
    def _from_widths(
        cls,
        widths: list[int],
        *,
        backend: Backend | str | None = None,
    ) -> CanvasHGrid:
        return CanvasHGrid([1], widths, backend=backend)

    @override
    def __getitem__(self, key: int) -> Canvas:
        canvas = self._canvas_array[0, key]
        if canvas is None:
            raise ValueError(f"Canvas at {key} is not set")
        return canvas

    @override
    def add_canvas(self, col: int, **kwargs) -> Canvas:
        return super().add_canvas(0, col, **kwargs)

    @override
    def _iter_add_canvas(self, **kwargs) -> Iterator[Canvas]:
        for col in range(len(self._widths)):
            yield self.add_canvas(col, **kwargs)


class _CanvasWithGrid(CanvasBase):
    def __init__(self, canvas: Canvas, grid: CanvasGrid):
        self._main_canvas = canvas
        self._grid = grid
        super().__init__(palette=canvas._color_palette)

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

    def _get_backend(self) -> Backend:
        """Return the backend."""
        return self._main_canvas._backend

    def _canvas(self):
        return self._main_canvas._backend_object

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


class SingleCanvas(_CanvasWithGrid, _Serializable):
    """
    A canvas without other subplots.

    This class is the simplest form of canvas. In `matplotlib` terms, it is a figure
    with a single axes.
    """

    def __init__(self, canvas: Canvas, grid: CanvasGrid):
        super().__init__(canvas, grid)

        # NOTE: events, dims etc are not shared between the main canvas and the
        # SingleCanvas instance. To avoid confusion, the first and the only canvas
        # should be replaces with the SingleCanvas instance.
        self.mouse = grid[0, 0].mouse
        grid._canvas_array[0, 0] = self
        self.events.drawn.connect(
            self._main_canvas.events.drawn.emit, unique=True, max_args=None
        )

    @classmethod
    def _new(cls, palette=None, backend=None) -> SingleCanvas:
        _grid = CanvasGrid([1], [1], backend=backend)
        _grid.add_canvas(0, 0, palette=palette)
        return SingleCanvas._from_grid(_grid)

    @classmethod
    def _from_grid(cls, grid: CanvasGrid) -> SingleCanvas:
        if grid.shape != (1, 1):
            raise ValueError(f"Grid shape must be (1, 1), got {grid.shape}")
        _it = grid._iter_canvas()
        _, canvas = next(_it)
        if next(_it, None) is not None:
            raise ValueError("Grid must have only one canvas")
        return cls(canvas, grid)

    @classmethod
    def from_dict(cls, d: dict[str, Any], backend: Backend | str | None = None) -> Self:
        """Create a SingleCanvas instance from a dictionary."""
        _type_expected = f"{cls.__module__}.{cls.__name__}"
        if (_type := d.get("type")) and _type != _type_expected:
            raise ValueError(f"Expected type {_type_expected!r}, got {_type!r}")
        self = cls._new(backend=backend, palette=d.get("palette"))
        self.layers.clear()
        self.layers.extend(construct_layers(d["layers"], backend=backend))
        self.x.update(d.get("x", {}))
        self.y.update(d.get("y", {}))
        self.title.update(d.get("title", {}))
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the canvas."""
        return {
            "type": f"{self.__module__}.{self.__class__.__name__}",
            "palette": self._color_palette,
            "layers": [layer.to_dict() for layer in self.layers],
            "title": self.title.to_dict(),
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
        }

    def copy(self, backend: Backend | str | None = None) -> Self:
        """Make a copy of the canvas."""
        if backend is None:
            backend = self._main_canvas._backend
        return self.from_dict(self.to_dict(), backend=backend)
