from __future__ import annotations

from typing import TYPE_CHECKING

from whitecanvas.canvas import Canvas
from whitecanvas.plot._canvases import current_canvas


def _make_method(name: str, pref: str = "add_"):
    fname = f"{pref}{name}"

    def _inner(*args, **kwargs):
        meth = getattr(current_canvas(), fname)
        return meth(*args, **kwargs)

    canvas_func = getattr(Canvas, fname)
    doc = canvas_func.__doc__
    assert isinstance(doc, str)
    _inner.__doc__ = doc.replace(f">>> canvas.{pref}", ">>> plt.")
    _inner.__name__ = name
    _inner.__qualname__ = name
    _inner.__module__ = "whitecanvas.plot._methods"
    _inner.__annotations__ = canvas_func.__annotations__
    return _inner


if TYPE_CHECKING:
    _CANVAS = Canvas(backend="mock")
    # add-layer methods
    line = _CANVAS.add_line
    hline = _CANVAS.add_hline
    vline = _CANVAS.add_vline
    markers = _CANVAS.add_markers
    hist = _CANVAS.add_hist
    bars = _CANVAS.add_bars
    band = _CANVAS.add_band
    errorbars = _CANVAS.add_errorbars
    infline = _CANVAS.add_infline
    infcurve = _CANVAS.add_infcurve
    spans = _CANVAS.add_spans
    kde = _CANVAS.add_kde
    rug = _CANVAS.add_rug
    text = _CANVAS.add_text
    # categorical methods
    cat = _CANVAS.cat
    cat_x = _CANVAS.cat_x
    cat_y = _CANVAS.cat_y
    cat_xy = _CANVAS.cat_xy
    # update methods
    update_axes = _CANVAS.update_axes
    update_labels = _CANVAS.update_labels
    update_font = _CANVAS.update_font
    # others
    legend = _CANVAS.add_legend
else:
    line = _make_method("line")
    hline = _make_method("hline")
    vline = _make_method("vline")
    markers = _make_method("markers")
    hist = _make_method("hist")
    bars = _make_method("bars")
    band = _make_method("band")
    errorbars = _make_method("errorbars")
    infline = _make_method("infline")
    infcurve = _make_method("infcurve")
    spans = _make_method("spans")
    kde = _make_method("kde")
    rug = _make_method("rug")
    text = _make_method("text")
    cat = _make_method("cat", pref="")
    cat_x = _make_method("cat_x", pref="")
    cat_y = _make_method("cat_y", pref="")
    cat_xy = _make_method("cat_xy", pref="")
    update_axes = _make_method("update_axes", pref="")
    update_labels = _make_method("update_labels", pref="")
    update_font = _make_method("update_font", pref="")
    legend = _make_method("legend", pref="add_")
