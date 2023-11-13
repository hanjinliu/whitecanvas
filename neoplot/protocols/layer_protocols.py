from typing import Callable, Protocol, runtime_checkable
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
    def _plt_get_data(self) -> np.ndarray:  # (N, 2)
        """Return the x and y array."""

    def _plt_set_data(self, data: np.ndarray):
        """Set the x and y array."""

@runtime_checkable
class HasLine(BaseProtocol, Protocol):
    def _plt_get_line_color(self) -> NDArray[np.float32]:
        """Return the line color."""
    
    def _plt_set_line_color(self, color: NDArray[np.float32]):
        """Set the line color."""
    
    def _plt_get_line_width(self) -> float:
        """Return the line width."""
        
    def _plt_set_line_width(self, width: float):
        """Set the line width."""
    
    def _plt_get_line_style(self) -> LineStyle:
        """Return the line style."""
    
    def _plt_set_line_style(self, style: LineStyle):
        """Set the line style."""
        
    def _plt_get_antialias(self) -> bool:
        """Return the anti alias."""
    
    def _plt_set_antialias(self, antialias: bool):
        """Set the anti alias."""

@runtime_checkable
class HasSymbol(BaseProtocol, Protocol):

    def _plt_get_symbol(self) -> Symbol:
        """Return the symbol."""
    
    def _plt_set_symbol(self, symbol: Symbol):
        """Set the symbol."""
    
    def _plt_get_symbol_size(self) -> float:
        """Return the symbol size."""
    
    def _plt_set_symbol_size(self, size: float):
        """Set the symbol size."""
    
    def _plt_get_symbol_face_color(self) -> NDArray[np.float32]:  # (N, 4)
        """Return the symbol face color."""
    
    def _plt_set_symbol_face_color(self, color: NDArray[np.float32]):
        """Set the symbol face color."""
    
    def _plt_get_symbol_edge_color(self) -> NDArray[np.float32]:  # (N, 4)
        """Return the symbol edge color."""
    
    def _plt_set_symbol_edge_color(self, color: NDArray[np.float32]):
        """Set the symbol edge color."""

@runtime_checkable
class ModelDataProtocol(HasLine, Protocol):
    def _plt_get_model(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the x and y array."""

    def _plt_set_model(self, model: Callable[[np.ndarray], np.ndarray]):
        """Set the x and y array."""


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
class BarProtocol(XYDataProtocol, Protocol):
    def _plt_get_bar_width(self) -> float:
        """Return the bar width."""
        
    def _plt_set_bar_width(self, width: float):
        """Set the bar width."""
    
    def _plt_get_bar_color(self) -> NDArray[np.float32]:
        """Return the bar color."""
    
    def _plt_set_bar_color(self, color: NDArray[np.float32]):
        """Set the bar color."""

@runtime_checkable
class HasRanges(BaseProtocol, Protocol):
    def _plt_get_ranges(self) -> np.ndarray:  # (N, 2)
        """Return the ranges."""
    
    def _plt_set_ranges(self, range: np.ndarray):
        """Set the ranges."""

    def _plt_get_range_color(self) -> NDArray[np.float32]:
        """Return the range color."""
    
    def _plt_set_range_color(self, color: NDArray[np.float32]):
        """Set the range color."""


@runtime_checkable
class LineProtocol(XYDataProtocol, HasLine, HasSymbol, Protocol):
    pass

@runtime_checkable
class ScatterProtocol(XYDataProtocol, HasSymbol, Protocol):
    pass

@runtime_checkable
class RangesProtocol(HasRanges, HasText, Protocol):
    pass
