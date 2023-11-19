from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray
from .layer_protocols import BaseProtocol
from whitecanvas.types import MouseEvent


@runtime_checkable
class HasVisibility(Protocol):
    def _plt_get_visible(self) -> bool:
        """Get visibility of canvas"""

    def _plt_set_visible(self, visible: bool):
        """Set visibility of canvas"""


@runtime_checkable
class HasLayers(Protocol):
    def _plt_insert_layer(self, idx: int, layer: BaseProtocol):
        """Add layer to the canvas (no need for reordering)"""

    def _plt_remove_layer(self, layer: BaseProtocol):
        """Remove layer from the canvas"""


@runtime_checkable
class CanvasProtocol(HasVisibility, HasLayers, Protocol):
    def _plt_get_title(self) -> TextLabelProtocol:
        """Get title handler"""

    def _plt_get_xaxis(self) -> AxisProtocol:
        """Get x axis handler"""

    def _plt_get_yaxis(self) -> AxisProtocol:
        """Get y axis handler"""

    def _plt_get_xlabel(self) -> TextLabelProtocol:
        """Get x label handler"""

    def _plt_get_ylabel(self) -> TextLabelProtocol:
        """Get y label handler"""

    def _plt_get_aspect_ratio(self) -> float | None:
        """Get aspect ratio of canvas"""

    def _plt_set_aspect_ratio(self, ratio: float | None):
        """Set aspect ratio of canvas"""

    def _plt_connect_mouse_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

    def _plt_connect_mouse_drag(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

    def _plt_connect_mouse_double_click(self, callback: Callable[[MouseEvent], None]):
        """Connect callback to clicked event"""

    def _plt_connect_xlim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        """Connect callback to x-limits changed event"""

    def _plt_connect_ylim_changed(
        self, callback: Callable[[tuple[float, float]], None]
    ):
        """Connect callback to y-limits changed event"""


@runtime_checkable
class TextLabelProtocol(HasVisibility, Protocol):
    def _plt_get_text(self) -> str:
        """Get text of label"""

    def _plt_set_text(self, title: str):
        """Set text of label"""

    def _plt_get_size(self) -> float:
        """Get size of text"""

    def _plt_set_size(self, size: float):
        """Set size of text"""

    def _plt_get_color(self) -> str:
        """Get color of text"""

    def _plt_set_color(self, color):
        """Set color of text"""

    def _plt_get_fontfamily(self) -> str:
        """Get font family of text"""

    def _plt_set_fontfamily(self, font):
        """Set font family of text"""


@runtime_checkable
class AxisProtocol(Protocol):
    def _plt_get_limits(self) -> tuple[float, float]:
        """Get limits of axis"""

    def _plt_set_limits(self, limits: tuple[float, float]):
        """Set limits of axis"""

    def _plt_get_color(self) -> np.ndarray:
        """Get color of axis"""

    def _plt_set_color(self, color):
        """Set color of axis"""

    def _plt_flip(self) -> None:
        """Flip axis"""


@runtime_checkable
class AxisGridProtocol(Protocol):
    def _plt_get_visible(self) -> bool:
        """Get visibility of axis grid"""

    def _plt_set_visible(self, visible: bool):
        """Set visibility of axis grid"""

    def _plt_get_color(self) -> np.ndarray:
        """Get color of axis grid"""

    def _plt_set_color(self, color):
        """Set color of axis grid"""

    def _plt_get_width(self) -> float:
        """Get width of axis grid"""

    def _plt_set_width(self, width: float):
        """Set width of axis grid"""


@runtime_checkable
class CanvasGridProtocol(HasVisibility, Protocol):
    def _plt_add_canvas(
        self, row: int, col: int, rowspan: int, colspan: int
    ) -> CanvasProtocol:
        """Add canvas to the grid"""

    def _plt_get_background_color(self):
        """Get background color of canvas"""

    def _plt_set_background_color(self, color):
        """Set background color of canvas"""

    def _plt_screenshot(self) -> NDArray[np.uint8]:
        """Get screenshot of canvas"""
