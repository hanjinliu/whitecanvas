from whitecanvas.canvas import CanvasGrid


def view(canvas: CanvasGrid, app: str = "qt"):
    if app == "qt":
        _view_qt(canvas)


_instance = None


def _view_qt(canvas: CanvasGrid):
    global _instance
    from ._qt import QtMainWindow

    main = QtMainWindow(canvas)
    main.show()
    _instance = main
    return main
