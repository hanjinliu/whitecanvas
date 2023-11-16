import pytest
import whitecanvas
from whitecanvas.layers import Line, Markers
import numpy as np
from numpy.testing import assert_allclose
from cmap import Color

BACKENDS = ["matplotlib", "pyqtgraph"]

@pytest.mark.parametrize("backend", BACKENDS)
def test_color(backend: str):
    def _is_color_array(arr):
        return isinstance(arr, np.ndarray) and arr.dtype.kind == "f" and arr.shape == (4,)

    layer = Line(np.arange(10), np.zeros(10), backend=backend, color="red")
    assert _is_color_array(layer.color)

    assert Color(layer.color) == Color("red")

    layer = Markers(np.arange(10), np.zeros(10), backend=backend, face_color="cyan", edge_color="white")

    assert _is_color_array(layer.face_color)
    assert _is_color_array(layer.edge_color)

    assert Color(layer.face_color) == Color("cyan")
    assert Color(layer.edge_color) == Color("white")

@pytest.mark.parametrize("backend", BACKENDS)
def test_line(backend: str):
    canvas = whitecanvas.Canvas(backend=backend)
    canvas.add_line(np.arange(10), np.zeros(10))
    layer = canvas.add_line(np.zeros(10))
    layer.color = [1.0, 0.0, 0.0, 1.0]
    assert_allclose(layer.color, [1.0, 0.0, 0.0, 1.0])
    layer.style = ":"
    assert layer.style == ":"
    layer.width = 2
    assert layer.width == 2
