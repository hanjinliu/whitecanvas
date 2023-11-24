class PyQtLayer:
    def _plt_get_visible(self) -> bool:
        return self.isVisible()

    def _plt_set_visible(self, visible: bool):
        self.setVisible(visible)

    def _plt_set_zorder(self, zorder: int):
        self.setZValue(zorder)
