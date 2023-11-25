from typing import Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray
from cmap import Colormap
from whitecanvas.types import LineStyle, Symbol, FacePattern, Orientation

Array1D = NDArray[np.number]


@runtime_checkable
class BaseProtocol(Protocol):
    def _plt_get_visible(self) -> bool:
        """Return the visibility."""

    def _plt_set_visible(self, visible: bool):
        """Set the visibility."""


@runtime_checkable
class XYDataProtocol(BaseProtocol, Protocol):
    def _plt_get_data(self) -> tuple[Array1D, Array1D]:
        """Return the x and y array."""

    def _plt_set_data(self, xdata: Array1D, ydata: Array1D):
        """Set the x and y array."""


@runtime_checkable
class XYYDataProtocol(BaseProtocol, Protocol):
    def _plt_get_data(self) -> tuple[Array1D, Array1D, Array1D]:
        """Return the x and y array."""

    def _plt_set_data(self, xdata: Array1D, ydata0: Array1D, ydata1: Array1D):
        """Set the x and y array."""


@runtime_checkable
class OrientedXYYDataProtocol(BaseProtocol, Protocol):
    def _plt_get_vertical_data(self) -> tuple[Array1D, Array1D, Array1D]:
        """Return the vertical representative data."""

    def _plt_get_horizontal_data(self) -> tuple[Array1D, Array1D, Array1D]:
        """Return the horizontal representative data."""

    def _plt_set_vertical_data(self, xdata: Array1D, ydata0: Array1D, ydata1: Array1D):
        """Set the vertical representative data."""

    def _plt_set_horizontal_data(
        self, xdata: Array1D, ydata0: Array1D, ydata1: Array1D
    ):
        """Set the horizontal representative data."""


@runtime_checkable
class XXYYDataProtocol(BaseProtocol, Protocol):
    def _plt_get_data(self) -> tuple[Array1D, Array1D, Array1D, Array1D]:
        """Return the x and y array."""

    def _plt_set_data(
        self, xdata0: Array1D, xdata1: Array1D, ydata0: Array1D, ydata1: Array1D
    ):
        """Set the x and y array."""


@runtime_checkable
class HasFaces(Protocol):
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        """Return the face color."""

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        """Set the face color."""

    def _plt_get_face_pattern(self) -> FacePattern:
        """Return the face pattern."""

    def _plt_set_face_pattern(self, pattern: FacePattern):
        """Set the face pattern."""


@runtime_checkable
class HasMultiFaces(Protocol):
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        """Return the face color."""

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        """Set the face color."""

    def _plt_get_face_pattern(self) -> list[FacePattern]:
        """Return the face pattern."""

    def _plt_set_face_pattern(self, pattern: FacePattern | list[FacePattern]):
        """Set the face pattern."""


@runtime_checkable
class HasEdges(Protocol):
    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        """Return the edge color."""

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        """Set the edge color."""

    def _plt_get_edge_width(self) -> float:
        """Return the edge width."""

    def _plt_set_edge_width(self, width: float):
        """Set the edge width."""

    def _plt_get_edge_style(self) -> LineStyle:
        """Return the edge style."""

    def _plt_set_edge_style(self, style: LineStyle):
        """Set the edge style."""


@runtime_checkable
class HasMultiEdges(Protocol):
    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        """Return the edge color."""

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        """Set the edge color."""

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        """Return the edge width."""

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        """Set the edge width."""

    def _plt_get_edge_style(self) -> list[LineStyle]:
        """Return the edge style."""

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        """Set the edge style."""


@runtime_checkable
class HasText(BaseProtocol, Protocol):
    def _plt_get_text(self) -> str:
        """Return the text."""

    def _plt_set_text(self, text: list[str]):
        """Set the text."""

    def _plt_get_text_color(self) -> NDArray[np.float32]:
        """Return the text color."""

    def _plt_set_text_color(self, color: NDArray[np.float32]):
        """Set the text color."""

    def _plt_get_text_size(self) -> float:
        """Return the text size."""

    def _plt_set_text_size(self, size: float):
        """Set the text size."""

    def _plt_get_text_position(self) -> tuple[float, float]:
        """Return the text position."""

    def _plt_set_text_position(self, position: tuple[float, float]):
        """Set the text position."""

    def _plt_get_text_anchor(self) -> tuple[float, float]:
        """Return the text position."""

    def _plt_set_text_anchor(self, position: tuple[float, float]):
        """Set the text position."""

    def _plt_get_text_rotation(self) -> float:
        """Return the text rotation in degree."""

    def _plt_set_text_rotation(self, rotation: float):
        """Set the text rotation in degree."""

    def _plt_get_text_fontfamily(self) -> str:
        """Return the text font family."""

    def _plt_set_text_fontfamily(self, family: str):
        """Set the text font family."""


@runtime_checkable
class LineProtocol(XYDataProtocol, HasEdges, Protocol):
    def _plt_get_antialias(self) -> bool:
        """Return the anti alias."""

    def _plt_set_antialias(self, antialias: bool):
        """Set the anti alias."""


@runtime_checkable
class MultiLineProtocol(XYDataProtocol, HasEdges, Protocol):
    def _plt_get_data(self) -> list[NDArray[np.number]]:
        """Return the x and y array."""

    def _plt_set_data(self, data: list[NDArray[np.number]]):
        """Set the x and y array."""

    def _plt_get_edge_color(self) -> NDArray[np.float32]:
        """Return the edge color."""

    def _plt_set_edge_color(self, color: NDArray[np.float32]):
        """Set the edge color."""

    def _plt_get_edge_width(self) -> NDArray[np.floating]:
        """Return the edge width."""

    def _plt_set_edge_width(self, width: float | NDArray[np.floating]):
        """Set the edge width."""

    def _plt_get_edge_style(self) -> list[LineStyle]:
        """Return the edge style."""

    def _plt_set_edge_style(self, style: LineStyle | list[LineStyle]):
        """Set the edge style."""

    def _plt_get_antialias(self) -> bool:
        """Return the anti alias."""

    def _plt_set_antialias(self, antialias: bool):
        """Set the anti alias."""


@runtime_checkable
class BarProtocol(XXYYDataProtocol, HasMultiFaces, HasMultiEdges, Protocol):
    """Protocols for plt.bar, plt.errorbar"""


@runtime_checkable
class MarkersProtocol(XYDataProtocol, HasFaces, HasEdges, Protocol):
    def _plt_get_symbol(self) -> Symbol:
        """Return the symbol."""

    def _plt_set_symbol(self, symbol: Symbol):
        """Set the symbol."""

    def _plt_get_symbol_size(self) -> float:
        """Return the symbol size."""

    def _plt_set_symbol_size(self, size: float):
        """Set the symbol size."""


@runtime_checkable
class HeteroMarkersProtocol(XYDataProtocol, HasMultiFaces, HasMultiEdges, Protocol):
    def _plt_get_symbol(self) -> Symbol:
        """Return the symbol."""

    def _plt_set_symbol(self, symbol: Symbol):
        """Set the symbol."""

    def _plt_get_symbol_size(self) -> NDArray[np.floating]:
        """Return the symbol size."""

    def _plt_set_symbol_size(self, size: float | NDArray[np.floating]):
        """Set the symbol size."""


@runtime_checkable
class BandProtocol(OrientedXYYDataProtocol, HasFaces, HasEdges, Protocol):
    pass


@runtime_checkable
class ErrorbarProtocol(OrientedXYYDataProtocol, HasEdges, Protocol):
    def _plt_get_capsize(self) -> float:
        """Return the capsize of the error bar edges."""

    def _plt_set_capsize(self, capsize: float, orient: Orientation):
        """Set the capsize of the error bar edges."""


@runtime_checkable
class TextProtocol(HasText, HasEdges, HasFaces, Protocol):
    pass


@runtime_checkable
class ImageProtocol(BaseProtocol, Protocol):
    def _plt_get_data(self) -> NDArray[np.number]:
        """Return the image data."""

    def _plt_set_data(self, data: NDArray[np.number]):
        """Set the image data."""

    def _plt_get_colormap(self) -> Colormap:
        """Return the colormap."""

    def _plt_set_colormap(self, colormap: Colormap):
        """Set the colormap."""

    def _plt_get_clim(self) -> tuple[float, float]:
        """Return the clim."""

    def _plt_set_clim(self, clim: tuple[float, float]):
        """Set the clim."""


@runtime_checkable
class RangeDataProtocol(HasFaces, HasEdges, Protocol):
    def _plt_get_ranges(self) -> NDArray[np.number]:  # (N, 2)
        """Return the ranges."""

    def _plt_set_ranges(self, range: NDArray[np.number]):
        """Set the ranges."""
