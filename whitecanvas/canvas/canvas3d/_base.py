from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from whitecanvas import protocols, theme
from whitecanvas.backend import Backend, patch_dummy_backend
from whitecanvas.canvas import _namespaces as _ns
from whitecanvas.canvas._base import CanvasNDBase
from whitecanvas.layers import layer3d
from whitecanvas.layers._base import Layer
from whitecanvas.types import (
    ArrayLike1D,
    ColormapType,
    ColorType,
    Hatch,
    LineStyle,
    Orientation,
)
from whitecanvas.utils.normalize import normalize_xyz

if TYPE_CHECKING:
    from whitecanvas.canvas import CanvasGrid

_L3D = TypeVar("_L3D", bound=layer3d.Layer3D)


class Canvas3DBase(CanvasNDBase):
    _CURRENT_INSTANCE: Canvas3DBase | None = None

    z = _ns.ZAxisNamespace()

    def __init__(self, *, palette: ColormapType | None = None):
        super().__init__(palette=palette)
        self._autoscale_enabled = True
        if not self._get_backend().is_dummy():
            self._init_canvas()

    def _init_canvas(self):
        # default colors and font
        if self._get_backend().name in ("matplotlib", "plotly"):
            _t = theme.get_theme()
            _ft = _t.font
            self.x.color = _t.foreground_color
            self.y.color = _t.foreground_color
            self.x.label.update(family=_ft.family, color=_ft.color, size=_ft.size)
            self.y.label.update(family=_ft.family, color=_ft.color, size=_ft.size)
            self.title.update(family=_ft.family, color=_ft.color, size=_ft.size)
            self.x.ticks.update(family=_ft.family, color=_ft.color, size=_ft.size)
            self.y.ticks.update(family=_ft.family, color=_ft.color, size=_ft.size)

            # lim changed events
            canvas = self._canvas()
            canvas._plt_connect_xlim_changed(self._emit_xlim_changed)
            canvas._plt_connect_ylim_changed(self._emit_ylim_changed)
            canvas._plt_connect_zlim_changed(self._emit_zlim_changed)

            if hasattr(canvas, "_plt_canvas_hook"):
                canvas._plt_canvas_hook(self)

        # connect layer events
        self.layers.events.inserted.connect(
            self._cb_inserted, unique=True, max_args=None
        )
        self.layers.events.removed.connect(self._cb_removed, unique=True, max_args=None)
        self.layers.events.reordered.connect(
            self._cb_reordered, unique=True, max_args=None
        )
        self.layers.events.connect(self._draw_canvas, unique=True, max_args=None)

    def _autoscale_for_layer(
        self,
        layer: Layer,
        pad_rel: float | None = None,
        maybe_empty: bool = True,
    ):
        """This function will be called when a layer is inserted to the canvas."""
        if not self.autoscale_enabled:
            return
        if pad_rel is None:
            pad_rel = 0 if layer._NO_PADDING_NEEDED else 0.025
        xmin, xmax, ymin, ymax, zmin, zmax = layer.bbox_hint()
        if len(self.layers) > 1 or not maybe_empty:
            # NOTE: if there was no layer, so backend may not have xlim/ylim,
            # or they may be set to a default value.
            _xmin, _xmax = self.x.lim
            _ymin, _ymax = self.y.lim
            _dx = (_xmax - _xmin) * pad_rel
            _dy = (_ymax - _ymin) * pad_rel
            xmin = np.min([xmin, _xmin + _dx])
            xmax = np.max([xmax, _xmax - _dx])
            ymin = np.min([ymin, _ymin + _dy])
            ymax = np.max([ymax, _ymax - _dy])

        # this happens when there is <= 1 data
        xmin, xmax = _prep_lims(
            (xmin, xmax),
            self.x.lim,
            pad_rel,
            layer,
            additional=not layer._ATTACH_TO_AXIS
            or getattr(layer, "orient", None) is not Orientation.HORIZONTAL,
        )
        ymin, ymax = _prep_lims(
            (ymin, ymax),
            self.y.lim,
            pad_rel,
            layer,
            additional=not layer._ATTACH_TO_AXIS
            or getattr(layer, "orient", None) is not Orientation.VERTICAL,
        )
        zmin, zmax = _prep_lims((zmin, zmax), self.z.lim, pad_rel, layer)
        self.x.lim = (xmin, xmax)
        self.y.lim = (ymin, ymax)
        self.z.lim = (zmin, zmax)

    def _emit_xlim_changed(self, lim):
        self.x.events.lim.emit(lim)

    def _emit_ylim_changed(self, lim):
        self.y.events.lim.emit(lim)

    def _emit_zlim_changed(self, lim):
        self.z.events.lim.emit(lim)

    @property
    def native(self) -> Any:
        """The native backend object."""
        return self._canvas()._plt_get_native()

    @property
    def autoscale_enabled(self) -> bool:
        """Whether autoscale is enabled."""
        return self._autoscale_enabled

    @overload
    def add_line(
        self, x: ArrayLike1D, y: ArrayLike1D, z: ArrayLike1D, *,
        name: str | None = None, color: ColorType | None = None,
        width: float | None = None, style: str | LineStyle | None = None,
        alpha: float = 1.0, antialias: bool = True,
    ) -> layer3d.Line3D:  # fmt: skip
        ...

    @overload
    def add_line(
        self, data: Any, *, name: str | None = None, color: ColorType | None = None,
        width: float | None = None, style: str | LineStyle | None = None,
        alpha: float = 1.0, antialias: bool = True,
    ) -> layer3d.Line3D:  # fmt: skip
        ...

    def add_line(
        self,
        *args,
        name=None,
        color=None,
        width=None,
        style=None,
        alpha=1.0,
        antialias=True,
    ) -> layer3d.Line3D:
        xdata, ydata, zdata = normalize_xyz(*args)
        name = self._coerce_name(name)
        color = self._generate_colors(color)
        width = theme._default("line.width", width)
        style = theme._default("line.style", style)
        layer = layer3d.Line3D(
            xdata, ydata, zdata, name=name, color=color, width=width, style=style,
            alpha=alpha, antialias=antialias, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    @overload
    def add_markers(
        self, data: Any, *, name: str | None = None, symbol: str | None = None,
        size: float | None = None, color: ColorType | None = None, alpha: float = 1.0,
        hatch: str | Hatch | None = None,
    ) -> layer3d.Markers3D:  # fmt: skip
        ...

    @overload
    def add_markers(
        self, x: ArrayLike1D, y: ArrayLike1D, z: ArrayLike1D, *,
        name: str | None = None, symbol: str | None = None, size: float | None = None,
        color: ColorType | None = None, alpha: float = 1.0,
        hatch: str | Hatch | None = None,
    ) -> layer3d.Markers3D:  # fmt: skip
        ...

    def add_markers(
        self,
        *args,
        name=None,
        symbol=None,
        size=None,
        color=None,
        alpha=1.0,
        hatch=None,
    ) -> layer3d.Markers3D:
        """
        Add markers (scatter plot).

        >>> canvas.add_markers(x, y)  # standard usage
        >>> canvas.add_markers(y)  # use 0, 1, ... for the x values

        Parameters
        ----------
        name : str, optional
            Name of the layer.
        symbol : str or Symbol, optional
            Marker symbols. Use the theme default if not specified.
        size : float, optional
            Marker size. Use the theme default if not specified.
        color : color-like, optional
            Color of the marker faces.
        alpha : float, default 1.0
            Alpha channel of the marker faces.
        hatch : str or Hatch, optional
            Pattern of the marker faces. Use the theme default if not specified.

        Returns
        -------
        Markers
            The markers layer.
        """
        xdata, ydata, zdata = normalize_xyz(*args)
        name = self._coerce_name(name)
        color = self._generate_colors(color)
        symbol = theme._default("markers.symbol", symbol)
        size = theme._default("markers.size", size)
        hatch = theme._default("markers.hatch", hatch)
        layer = layer3d.Markers3D(
            xdata, ydata, zdata, name=name, symbol=symbol, size=size, color=color,
            alpha=alpha, hatch=hatch, backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    def add_mesh(
        self,
        verts: NDArray[np.floating],
        faces: NDArray[np.intp],
        *,
        name: str | None = None,
        color: ColorType | None = None,
        hatch: str | Hatch | None = None,
        alpha: float = 1.0,
    ) -> layer3d.Mesh3D:
        """
        Add a mesh layer.

        Parameters
        ----------
        verts : 2D array-like
            Vertices of the mesh.
        faces : 2D array-like
            Faces of the mesh.
        name : str, optional
            Name of the layer.
        color : color-like, optional
            Color of the surface.
        hatch : str or Hatch, optional
            Hatch pattern of the surface faces.
        alpha : float, default 1.0
            Alpha channel of the surface faces.
        """
        name = self._coerce_name(name)
        color = self._generate_colors(color)
        hatch = theme._default("markers.hatch", hatch)
        layer = layer3d.Mesh3D(
            verts, faces, name=name, color=color, hatch=hatch, alpha=alpha,
            backend=self._get_backend(),
        )  # fmt: skip
        return self.add_layer(layer)

    @overload
    def add_surface(
        self, z: NDArray[np.floating], *, name: str | None =None,
        color: ColorType | None = None, hatch: str | Hatch | None = None,
        alpha: float = 1.0,
    ) -> layer3d.Mesh3D:  # fmt: skip
        ...

    @overload
    def add_surface(
        self, x: NDArray[np.floating], y: NDArray[np.floating], z: NDArray[np.floating],
        *, name: str | None =None, color: ColorType | None = None,
        hatch: str | Hatch | None = None, alpha: float = 1.0,
    ) -> layer3d.Mesh3D:  # fmt: skip
        ...

    def add_surface(
        self,
        *args,
        name: str | None = None,
        color: ColorType | None = None,
        hatch: str | Hatch | None = None,
        alpha: float = 1.0,
    ) -> layer3d.Mesh3D:
        """
        Add a surface plot defined by z = f(x, y).

        The x, y are usually calculated by `np.meshgrid` or `np.indices`.

        Parameters
        ----------
        name : str, optional
            Name of the layer.
        color : color-like, optional
            Color of the surface.
        hatch : str or Hatch, optional
            Hatch pattern of the surface faces.
        alpha : float, default 1.0
            Alpha channel of the surface faces.
        """
        if len(args) == 1:
            z = np.asarray(args[0])
            if z.ndim != 2:
                raise ValueError(f"Expected 2D array for z, got shape {z.shape}")
            x, y = np.indices(z.shape)
        elif len(args) == 3:
            x = np.asarray(args[0])
            y = np.asarray(args[1])
            z = np.asarray(args[2])
        else:
            raise TypeError(f"Expected 1 or 3 positional arguments, got {len(args)}")
        nx, ny = x.shape
        verts = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        # (i, j), (i + 1, j), (i, j + 1)
        # (i + 1, j + 1), (i + 1, j), (i, j + 1)
        corners = (np.arange(nx - 1)[..., None] + np.arange(ny - 1) * nx).ravel()
        face0 = np.stack((corners, corners + 1, corners + nx), axis=1)
        face1 = np.stack((corners + 1, corners + nx + 1, corners + nx), axis=1)
        faces = np.concatenate((face0, face1), axis=0)
        return self.add_mesh(
            verts, faces, name=name, color=color, hatch=hatch, alpha=alpha
        )

    def add_layer(self, layer: _L3D) -> _L3D:
        """Add a layer to the canvas."""
        self.layers.append(layer)
        return layer


def _prep_lims(new_lim, current_lim, pad_rel, layer: Layer, additional=False):
    small_diff = 1e-6
    xmin, xmax = new_lim
    if np.isnan(xmax) or np.isnan(xmin):
        xmin, xmax = current_lim
    elif xmax - xmin < small_diff:
        xmin -= 0.05
        xmax += 0.05
    else:
        dx = (xmax - xmin) * pad_rel
        if xmin != 0 or additional:
            xmin -= dx
        xmax += dx
    return xmin, xmax


class Canvas3D(Canvas3DBase):
    _CURRENT_INSTANCE: Canvas3D | None = None

    def __init__(
        self,
        backend: str | None = None,
        *,
        palette: ColormapType | None = None,
    ):
        self._backend = Backend(backend)
        self._backend_object = self._create_backend_object()
        super().__init__(palette=palette)
        self.__class__._CURRENT_INSTANCE = self

    @classmethod
    def from_backend(
        cls,
        obj: protocols.CanvasProtocol,
        *,
        palette: ColormapType | None = None,
        backend: str | None = None,
    ) -> Self:
        """Create a canvas object from a backend object."""
        with patch_dummy_backend() as name:
            # this patch will delay initialization by "_init_canvas" until the backend
            # objects are created.
            self = cls(backend=name, palette=palette)
        self._backend = Backend(backend)
        self._backend_object = obj
        # self._init_canvas()
        return self

    def _create_backend_object(self) -> protocols.CanvasProtocol:
        return self._backend.get_submodule("canvas3d").Canvas3D()

    def _get_backend(self):
        return self._backend

    def _canvas(self) -> protocols.CanvasProtocol:
        return self._backend_object


class _Canvas3DWithGrid(Canvas3DBase):
    def __init__(self, canvas: Canvas3D, grid: CanvasGrid):
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


class SingleCanvas3D(_Canvas3DWithGrid):
    """
    A canvas without other subplots.

    This class is the simplest form of canvas. In `matplotlib` terms, it is a figure
    with a single axes.
    """

    def __init__(self, grid: CanvasGrid):
        if grid.shape != (1, 1):
            raise ValueError(f"Grid shape must be (1, 1), got {grid.shape}")
        self._grid = grid
        _it = grid._iter_canvas()
        _, canvas = next(_it)
        if next(_it, None) is not None:
            raise ValueError("Grid must have only one canvas")
        self._main_canvas = canvas
        super().__init__(canvas, grid)

        # NOTE: events, dims etc are not shared between the main canvas and the
        # SingleCanvas instance. To avoid confusion, the first and the only canvas
        # should be replaces with the SingleCanvas instance.
        # self.mouse = grid[0, 0].mouse
        grid._canvas_array[0, 0] = self
        # self.events.drawn.connect(
        #     self._main_canvas.events.drawn.emit, unique=True, max_args=None
        # )
