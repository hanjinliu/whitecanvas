import numpy as np

from whitecanvas import new_canvas
from whitecanvas.layers import Layer

from ._utils import assert_color_equal, assert_color_array_equal


def _test_visibility(layer: Layer):
    layer.visible
    layer.visible = False
    layer.visible = True

def test_line(backend: str):
    canvas = new_canvas(backend=backend)
    canvas.add_line(np.arange(10), np.zeros(10))
    layer = canvas.add_line(np.zeros(10))

    repr(layer)
    layer.color
    layer.color = [1.0, 0.0, 0.0, 1.0]
    assert_color_equal(layer.color, "red")
    layer.style
    layer.style = ":"
    assert layer.style == ":"
    layer.width
    layer.width = 2
    assert layer.width == 2
    _test_visibility(layer)

def test_markers(backend: str):
    canvas = new_canvas(backend=backend)
    canvas.add_markers(np.arange(10), np.zeros(10))
    layer = canvas.add_markers(np.zeros(10))

    repr(layer)
    repr(layer.face)
    repr(layer.edge)
    layer.face.color
    layer.face.alpha
    layer.face.color = [1.0, 0.0, 0.0, 1.0]
    assert_color_equal(layer.face.color, "red")
    layer.face.hatch
    layer.face.hatch = "/"
    assert layer.face.hatch == "/"

    layer.edge.color
    layer.edge.alpha
    layer.edge.color = [0.0, 0.0, 1.0, 1.0]
    assert_color_equal(layer.edge.color, "blue")
    layer.edge.style
    layer.edge.style = ":"
    assert layer.edge.style == ":"
    layer.edge.width
    layer.edge.width = 2
    assert layer.edge.width == 2

    assert layer.symbol == "o"
    layer.symbol = "+"
    assert layer.symbol == "+"

    layer.size
    layer.size = 20
    assert layer.size == 20

    for sym in ["o", "s", "^", "v", "<", ">", "D", "x", "+", "*", "|", "_"]:
        layer.symbol = sym
        assert layer.symbol == sym
    _test_visibility(layer)

def test_bars(backend: str):
    canvas = new_canvas(backend=backend)
    canvas.add_bars(np.arange(10), np.zeros(10), bottom=np.ones(10))
    layer = canvas.add_bars(np.arange(10), np.zeros(10))

    repr(layer)
    repr(layer.face)
    repr(layer.edge)
    layer.face.color
    layer.face.alpha
    layer.face.color = [1.0, 0.0, 0.0, 1.0]
    assert_color_equal(layer.face.color, "red")
    layer.face.hatch
    layer.face.hatch = "/"
    assert layer.face.hatch == "/"

    layer.edge.color
    layer.edge.alpha
    layer.edge.color = [0.0, 0.0, 1.0, 1.0]
    assert_color_equal(layer.edge.color, "blue")
    layer.edge.style
    layer.edge.style = ":"
    assert layer.edge.style == ":"
    layer.edge.width
    layer.edge.width = 2
    assert layer.edge.width == 2

    layer.bar_width = 0.5
    assert layer.bar_width == 0.5
    _test_visibility(layer)

def test_infcurve(backend: str):
    canvas = new_canvas(backend=backend)

    layer = canvas.add_infcurve(lambda arr: np.sin(arr / 5))

    layer.color = [1.0, 0.0, 0.0, 1.0]
    assert_color_equal(layer.color, "red")
    layer.style = ":"
    assert layer.style == ":"
    layer.width = 2
    assert layer.width == 2
    _test_visibility(layer)

def test_band(backend: str):
    canvas = new_canvas(backend=backend)

    x = np.arange(5)
    layer = canvas.add_band(x, x - 1, x ** 2 / 2)

    repr(layer.face)
    repr(layer.edge)
    layer.face.color
    layer.face.color = [1.0, 0.0, 0.0, 1.0]
    assert_color_equal(layer.face.color, "red")
    layer.face.hatch
    layer.face.hatch = "/"
    assert layer.face.hatch == "/"

    layer.edge.color
    layer.edge.color = [0.0, 0.0, 1.0, 1.0]
    assert_color_equal(layer.edge.color, "blue")
    layer.edge.style
    layer.edge.style = ":"
    assert layer.edge.style == ":"
    layer.edge.width
    layer.edge.width = 2
    assert layer.edge.width == 2
    _test_visibility(layer)

def test_image(backend: str):
    canvas = new_canvas(backend=backend)

    layer = canvas.add_image(np.random.random((10, 10)) * 2)

    layer.cmap = "viridis"
    assert layer.cmap == "viridis"
    layer.clim = (0.5, 1.5)
    assert layer.clim == (0.5, 1.5)
    _test_visibility(layer)

def test_errorbars(backend: str):
    canvas = new_canvas(backend=backend)

    layer = canvas.add_errorbars(np.arange(10), np.zeros(10), np.ones(10))

    layer.color = [1.0, 0.0, 0.0, 1.0]
    assert_color_array_equal(layer.color, "red")
    layer.style = ":"
    assert all(s == ":" for s in layer.style)
    layer.width = 2
    assert all(w == 2 for w in layer.width)
    _test_visibility(layer)

    layer = canvas.add_errorbars(np.arange(10), np.zeros(10), np.ones(10), capsize=0.2)

    layer.color = [1.0, 0.0, 0.0, 1.0]
    assert_color_array_equal(layer.color, "red")
    layer.style = "-."
    assert all(s == "-." for s in layer.style)
    layer.width = 3
    assert all(w == 3 for w in layer.width)
    _test_visibility(layer)

def test_texts(backend: str):
    canvas = new_canvas(backend=backend)

    layer = canvas.add_text(np.arange(10), np.zeros(10), list("abcdefghij"))

    repr(layer)
    repr(layer.face)
    repr(layer.edge)
    repr(layer.font)
    layer.anchor
    for anc in [
        "top", "bottom", "left", "right", "center", "top_left", "top_right",
        "bottom_left", "bottom_right"
    ]:
        layer.anchor = anc
        assert layer.anchor == anc

    layer.size
    layer.size = 28
    assert layer.size == 28

    assert layer.ndata == 10
    assert layer.string == list("abcdefghij")
    layer.string = list("ABCDEFGHIJ")
    assert layer.string == list("ABCDEFGHIJ")
    layer.face.color
    layer.face.color = "green"
    layer.face.hatch
    layer.face.hatch = "/"
    layer.face.alpha
    layer.face.alpha = 0.6

    layer.edge.color
    layer.edge.color = "blue"
    layer.edge.style
    layer.edge.style = ":"
    layer.edge.width
    layer.edge.width = 2
    layer.edge.alpha
    layer.edge.alpha = 0.6
    _test_visibility(layer)

def test_with_text(backend: str):
    canvas = new_canvas(backend=backend)
    x = np.arange(10)
    y = np.sqrt(x)
    canvas.add_line(x, y).with_text([f"{i}" for i in range(10)]).add_text_offset(0.1 ,0.1)
    canvas.add_markers(x, y).with_text([f"{i}" for i in range(10)]).add_text_offset(0.1 ,0.1)
    canvas.add_bars(x, y).with_text([f"{i}" for i in range(10)]).add_text_offset(0.1 ,0.1)
    canvas.add_line(x, y).with_text("x={x:.2f}, y={y:.2f}")
    canvas.add_markers(x, y).with_text("x={x:.2f}, y={y:.2f}")
    canvas.add_bars(x, y).with_text("x={x:.2f}, y={y:.2f}")
    canvas.add_line(x, y).with_markers().with_text([f"{i}" for i in range(10)])
    canvas.add_line(x, y).with_xerr(y/4).with_text([f"{i}" for i in range(10)])
    canvas.add_line(x, y).with_yerr(y/4).with_text([f"{i}" for i in range(10)])
    canvas.add_line(x, y).with_markers().with_text("x={x:2f}, i={i}")
    canvas.add_line(x, y).with_xerr(y/4).with_text("x={x:2f}, i={i}")
    canvas.add_line(x, y).with_yerr(y/4).with_text("x={x:2f}, i={i}")
    canvas.add_markers(x, y).with_network([[0, 1], [4, 3]]).with_text([f"{i}" for i in range(10)])
    canvas.add_markers(x, y).with_network([[0, 1], [4, 3]]).with_text("state-{i}")
    canvas.add_bars(x, y).with_xerr(y/4).with_text([f"{i}" for i in range(10)])
    canvas.add_bars(x, y).with_yerr(y/4).with_text([f"{i}" for i in range(10)])
    canvas.add_bars(x, y).with_xerr(y/4).with_text("{x:1f}, {y:1f},")
    canvas.add_bars(x, y).with_yerr(y/4).with_text("{x:1f}, {y:1f}")
