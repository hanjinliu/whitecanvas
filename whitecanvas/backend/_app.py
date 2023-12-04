from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from IPython import InteractiveShell
    from qtpy.QtWidgets import QApplication
    from vispy.app import use_app
    from wx import App as wxApp


class Application(ABC):
    @abstractmethod
    def get_app(self):
        """Get Application."""

    @abstractmethod
    def run_app(self):
        """Start the event loop."""


class QtApplication(Application):
    _APP: QApplication | None = None

    def get_app(self):
        """Get QApplication."""
        self.gui_qt()
        app = self.instance()
        if app is None:
            app = self.create_application()
        self._APP = app
        return app

    def create_application(self):
        from qtpy.QtWidgets import QApplication
        from qtpy.QtCore import Qt

        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
        return QApplication([])

    def run_app(self):
        """Start the event loop."""
        return self.get_app().exec_()

    def instance(self):
        from qtpy.QtWidgets import QApplication

        return QApplication.instance()

    def gui_qt(self):
        """Call "%gui qt" magic."""
        shell = get_shell()

        if shell and shell.active_eventloop != "qt":
            shell.enable_gui("qt")
        return None


class WxApplication(Application):
    _APP: wxApp | None = None

    def get_app(self):
        """Get WxApplication."""
        self.gui_wx()
        app = self.instance()
        if app is None:
            app = self.create_application()
        self._APP = app
        return app

    def create_application(self):
        try:
            from wx import App as wxApp
        except ImportError:
            from wx import PySimpleApp as wxApp

        app = wxApp()
        app.SetExitOnFrameDelete(True)
        return app

    def run_app(self):
        """Start the event loop."""
        return self.get_app().MainLoop()

    def instance(self):
        import wx

        return wx.GetApp()

    def gui_wx(self):
        """Call "%gui wx" magic."""
        shell = get_shell()

        if shell and shell.active_eventloop != "wx":
            shell.enable_gui("wx")
        return None


class TkApplication(Application):
    _APP: QApplication | None = None

    def get_app(self):
        """Get QApplication."""
        self.gui_tk()
        app = self.instance()
        if app is None:
            app = self.create_application()
        self._APP = app
        return app

    def create_application(self):
        from qtpy.QtWidgets import QApplication
        from qtpy.QtCore import Qt

        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
        return QApplication([])

    def run_app(self):
        """Start the event loop."""
        if self.instance() is None:
            return self.get_app().exec_()

    def instance(self):
        from qtpy.QtWidgets import QApplication

        return QApplication.instance()

    def gui_tk(self):
        """Call "%gui tk" magic."""
        shell = get_shell()

        if shell and shell.active_eventloop != "tk":
            shell.enable_gui("tk")
        return None


class EmptyApplication(Application):
    def get_app(self):
        return None

    def run_app(self):
        return None


def get_app(name: str) -> Application:
    if name == "qt":
        return QtApplication()
    elif name == "wx":
        return WxApplication()
    else:
        return EmptyApplication()


def get_shell() -> InteractiveShell | None:
    """Get ipython shell if available."""
    if "IPython" in sys.modules:
        from IPython import get_ipython

        return get_ipython()
    else:
        return None
