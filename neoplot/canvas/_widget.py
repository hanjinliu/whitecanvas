from ._base import CanvasBase
from neoplot.protocols import CanvasProtocol


class Canvas(CanvasBase[CanvasProtocol]):
    def _create_backend(self) -> CanvasProtocol:
        return self._backend_installer.get("Canvas")()

    def _canvas(self) -> CanvasProtocol:
        return self._backend
