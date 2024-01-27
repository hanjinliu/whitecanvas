from whitecanvas.canvas import CanvasBase, CanvasGrid


def view(grid: CanvasGrid, app: str = "qt"):
    if app == "qt":
        return _view_qt(grid)
    elif app == "tk":
        return _view_tk(grid)
    raise ValueError(f"Appication {app} not supported")


def make_dim_slider(canvas: CanvasBase, app: str = "qt"):
    if app == "qt":
        sl = _slider_qt(canvas)
    elif app == "tk":
        sl = _slider_tk(canvas)
    else:
        raise ValueError(f"Appication {app} not supported")
    return sl


def _view_qt(grid: CanvasGrid):
    from whitecanvas.backend._window._qt import QtMainWindow

    main = QtMainWindow(grid)
    main.show()
    return main


def _view_tk(grid: CanvasGrid):
    from whitecanvas.backend._window._tk import TkMainWindow

    main = TkMainWindow(grid)
    main.mainloop()
    return main


def _slider_qt(canvas: CanvasBase):
    from whitecanvas.backend._window._qt import QtDimSliders

    return QtDimSliders.from_canvas(canvas)


def _slider_tk(canvas: CanvasGrid):
    from whitecanvas.backend._window._tk import TkDimSliders

    return TkDimSliders.from_canvas(canvas)
