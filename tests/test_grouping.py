import pytest
from whitecanvas import new_canvas
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

BACKENDS = ["matplotlib", "pyqtgraph", "plotly", "bokeh", "vispy"]

@pytest.mark.parametrize("backend", BACKENDS)
def test_line_marker_text(backend: str):
    canvas = new_canvas(backend=backend)
    layer = canvas.add_line(
        np.arange(10), np.arange(10) * 2
    ).with_markers(
        symbol="s", size=10, color="red"
    ).with_text(
        [f"Data-{i}" for i in range(10)], color="blue"
    ).with_yerr(
        np.ones(10), color="blue"
    )
    assert_array_equal(layer.data.x, np.arange(10))
    assert_array_equal(layer.data.y, np.arange(10) * 2)
    assert len(canvas.layers) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_bar_err_text(backend: str):
    canvas = new_canvas(backend=backend)
    layer = canvas.add_bars(
        np.arange(10), np.arange(10) * 2
    ).with_err(
        np.ones(10), color="blue"
    ).with_text(
        [f"Data-{i}" for i in range(10)], color="blue"
    )
    assert_array_equal(layer.data.x, np.arange(10))
    assert_allclose(layer.data.y, np.arange(10) * 2, atol=1e-6)
    assert len(canvas.layers) == 1

@pytest.mark.parametrize("backend", BACKENDS)
def test_stem(backend: str):
    canvas = new_canvas(backend=backend)
    layer = canvas.add_markers(
        np.arange(10), np.arange(10) * 2
    ).with_stem()
    assert_array_equal(layer.data.x, np.arange(10))
    assert_allclose(layer.data.y, np.arange(10) * 2, atol=1e-6)
    assert len(canvas.layers) == 1

@pytest.mark.parametrize("backend", BACKENDS)
def test_network(backend: str):
    canvas = new_canvas(backend=backend)
    layer = canvas.add_markers(
        np.arange(10), np.arange(10) * 2
    ).with_network([[0, 1], [0, 2], [1, 3]])
    assert_array_equal(layer.nodes.data.x, np.arange(10))
    assert_array_equal(layer.nodes.data.y, np.arange(10) * 2)
    assert len(canvas.layers) == 1
