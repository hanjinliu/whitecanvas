from ._base import CanvasBase
from neoplot.protocols import MainWindowProtocol


class MainCanvas(CanvasBase[MainWindowProtocol]):
    def _create_backend(self):
        return self._backend_installer.get("MainCanvas")()

    def _canvas(self):
        return self._backend._plt_get_canvas()

    def show(self):
        return self._backend._plt_set_visible(True)

    def hide(self):
        return self._backend._plt_set_visible(False)
