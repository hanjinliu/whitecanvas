import sys
from qtpy.QtWidgets import QApplication

APPLICATION: "QApplication | None" = None


def get_shell():
    """Get ipython shell if available."""
    if "IPython" in sys.modules:
        from IPython import get_ipython

        return get_ipython()
    else:
        return None


def gui_qt():
    """Call "%gui qt" magic."""
    shell = get_shell()

    if shell and shell.active_eventloop != "qt":
        shell.enable_gui("qt")
    return None


def gui_qt_is_active() -> bool:
    """Check if Qt app is active."""
    return QApplication.instance() is not None


def get_app():
    """Get QApplication."""
    gui_qt()
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    global APPLICATION
    APPLICATION = app
    return app


def run_app():
    """Start the event loop."""
    if not gui_qt_is_active():
        return get_app().exec_()
