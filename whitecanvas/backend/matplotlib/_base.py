class MplLayer:
    def _plt_get_visible(self):
        return self.get_visible()

    def _plt_set_visible(self, visible):
        self.set_visible(visible)

    def _plt_set_zorder(self, zorder: int):
        self.set_zorder(zorder)
