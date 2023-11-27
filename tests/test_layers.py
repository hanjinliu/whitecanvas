import pytest
from whitecanvas import new_canvas
from whitecanvas.layers import Line, Markers, Layer
import numpy as np
from ._utils import assert_color_equal

BACKENDS = ["matplotlib", "pyqtgraph", "plotly", "bokeh", "vispy"]

def _test_visibility(layer: Layer):
    layer.visible
    layer.visible = False
    layer.visible = True

@pytest.mark.parametrize("backend", BACKENDS)
def test_color(backend: str):
    def _is_color_array(arr):
        return isinstance(arr, np.ndarray) and arr.dtype.kind == "f" and arr.shape == (4,)

    layer = Line(np.arange(10), np.zeros(10), backend=backend, color="red")
    assert _is_color_array(layer.color)

    assert_color_equal(layer.color, "red")

    layer = Markers(np.arange(10), np.zeros(10), backend=backend, color="cyan").with_edge(color="white")

    assert _is_color_array(layer.face.color)
    assert _is_color_array(layer.edge.color)

    assert_color_equal(layer.face.color, "cyan")
    assert_color_equal(layer.edge.color, "white")

@pytest.mark.parametrize("backend", BACKENDS)
def test_line(backend: str):
    canvas = new_canvas(backend=backend)
    canvas.add_line(np.arange(10), np.zeros(10))
    layer = canvas.add_line(np.zeros(10))

    layer.color = [1.0, 0.0, 0.0, 1.0]
    assert_color_equal(layer.color, "red")
    layer.style = ":"
    assert layer.style == ":"
    layer.width = 2
    assert layer.width == 2
    _test_visibility(layer)

@pytest.mark.parametrize("backend", BACKENDS)
def test_markers(backend: str):
    canvas = new_canvas(backend=backend)
    canvas.add_markers(np.arange(10), np.zeros(10))
    layer = canvas.add_markers(np.zeros(10))

    layer.face.color = [1.0, 0.0, 0.0, 1.0]
    assert_color_equal(layer.face.color, "red")
    layer.face.pattern = "/"
    assert layer.face.pattern == "/"

    layer.edge.color = [0.0, 0.0, 1.0, 1.0]
    assert_color_equal(layer.edge.color, "blue")
    layer.edge.style = ":"
    assert layer.edge.style == ":"
    layer.edge.width = 2
    assert layer.edge.width == 2

    assert layer.symbol == "o"
    layer.symbol = "+"
    assert layer.symbol == "+"

    layer.size = 20
    assert layer.size == 20
    _test_visibility(layer)

@pytest.mark.parametrize("backend", BACKENDS)
def test_bars(backend: str):
    canvas = new_canvas(backend=backend)
    canvas.add_bars(np.arange(10), np.zeros(10), bottom=np.ones(10))
    layer = canvas.add_bars(np.arange(10), np.zeros(10))

    layer.face.color = [1.0, 0.0, 0.0, 1.0]
    assert_color_equal(layer.face.color, "red")
    layer.face.pattern = "/"
    assert layer.face.pattern == "/"

    layer.edge.color = [0.0, 0.0, 1.0, 1.0]
    assert_color_equal(layer.edge.color, "blue")
    layer.edge.style = ":"
    assert layer.edge.style == ":"
    layer.edge.width = 2
    assert layer.edge.width == 2

    layer.bar_width = 0.5
    assert layer.bar_width == 0.5
    _test_visibility(layer)

@pytest.mark.parametrize("backend", BACKENDS)
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

@pytest.mark.parametrize("backend", BACKENDS)
def test_band(backend: str):
    canvas = new_canvas(backend=backend)

    x = np.arange(5)
    layer = canvas.add_band(x, x - 1, x ** 2 / 2)

    layer.face.color = [1.0, 0.0, 0.0, 1.0]
    assert_color_equal(layer.face.color, "red")
    layer.face.pattern = "/"
    assert layer.face.pattern == "/"

    layer.edge.color = [0.0, 0.0, 1.0, 1.0]
    assert_color_equal(layer.edge.color, "blue")
    layer.edge.style = ":"
    assert layer.edge.style == ":"
    layer.edge.width = 2
    assert layer.edge.width == 2
    _test_visibility(layer)

@pytest.mark.parametrize("backend", BACKENDS)
def test_image(backend: str):
    canvas = new_canvas(backend=backend)

    layer = canvas.add_image(np.random.random((10, 10)) * 2)

    layer.cmap = "viridis"
    assert layer.cmap == "viridis"
    layer.clim = (0.5, 1.5)
    assert layer.clim == (0.5, 1.5)
    _test_visibility(layer)

@pytest.mark.parametrize("backend", BACKENDS)
def test_errorbars(backend: str):
    canvas = new_canvas(backend=backend)

    layer = canvas.add_errorbars(np.arange(10), np.zeros(10), np.ones(10))

    layer.color = [1.0, 0.0, 0.0, 1.0]
    assert_color_equal(layer.color, "red")
    layer.style = ":"
    assert layer.style == ":"
    layer.width = 2
    assert layer.width == 2
    _test_visibility(layer)
