from typing import Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray
from neoplot.types import LineStyle, Symbol


@runtime_checkable
class BaseProtocol(Protocol):
    def _plt_get_visible(self) -> bool:
        """Return the visibility."""

    def _plt_set_visible(self, visible: bool):
        """Set the visibility."""

    def _plt_get_zorder(self) -> int:
        """Return the zorder."""

    def _plt_set_zorder(self, zorder: int):
        """Set the zorder."""


@runtime_checkable
class XYDataProtocol(BaseProtocol, Protocol):
    def _plt_get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the x and y array."""

    def _plt_set_data(self, xdata: np.ndarray, ydata: np.ndarray):
        """Set the x and y array."""


@runtime_checkable
class XYYDataProtocol(BaseProtocol, Protocol):
    def _plt_get_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the x and y array."""

    def _plt_set_data(self, xdata: np.ndarray, ydata0: np.ndarray, ydata1: np.ndarray):
        """Set the x and y array."""


@runtime_checkable
class HasFaces(Protocol):
    def _plt_get_face_color(self) -> NDArray[np.float32]:
        """Return the face color."""

    def _plt_set_face_color(self, color: NDArray[np.float32]):
        """Set the face color."""


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

    def _plt_get_text_anchor(self) -> tuple[float, float]:
        """Return the text position."""

    def _plt_set_text_anchor(self, position: tuple[float, float]):
        """Set the text position."""

    def _plt_get_text_rotation(self) -> float:
        """Return the text rotation."""

    def _plt_set_text_rotation(self, rotation: float):
        """Set the text rotation."""


@runtime_checkable
class LineProtocol(XYDataProtocol, HasEdges, Protocol):
    def _plt_get_antialias(self) -> bool:
        """Return the anti alias."""

    def _plt_set_antialias(self, antialias: bool):
        """Set the anti alias."""


@runtime_checkable
class BarProtocol(XYYDataProtocol, HasFaces, HasEdges, Protocol):
    """Protocols for plt.bar, plt.errorbar"""

    def _plt_get_bar_width(self) -> float:
        """Return the bar width."""

    def _plt_set_bar_width(self, width: float):
        """Set the bar width."""


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
class FillBetweenProtocol(XYYDataProtocol, HasFaces, HasEdges, Protocol):
    pass


@runtime_checkable
class RangeDataProtocol(HasFaces, HasEdges, Protocol):
    def _plt_get_ranges(self) -> np.ndarray:  # (N, 2)
        """Return the ranges."""

    def _plt_set_ranges(self, range: np.ndarray):
        """Set the ranges."""
