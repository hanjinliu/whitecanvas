import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from whitecanvas import new_canvas


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


def test_marker_err(backend: str):
    canvas = new_canvas(backend=backend)
    rng = np.random.default_rng(0)
    data = rng.random((10, 2))
    layer = canvas.add_markers(data).with_xerr(np.ones(10), color="blue")
    layer = canvas.add_markers(data).with_face(color="red").with_yerr(np.ones(10), color="blue")

    assert_allclose(layer.data.x, data[:, 0], atol=1e-6)
    assert_allclose(layer.data.y, data[:, 1], atol=1e-6)

    canvas.add_markers(rng.normal(size=(50, 2))).color_by_density()

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

def test_stem(backend: str):
    canvas = new_canvas(backend=backend)
    layer = canvas.add_markers(
        np.arange(10), np.arange(10) * 2
    ).with_stem()
    assert_array_equal(layer.data.x, np.arange(10))
    assert_allclose(layer.data.y, np.arange(10) * 2, atol=1e-6)
    assert len(canvas.layers) == 1

def test_network(backend: str):
    canvas = new_canvas(backend=backend)
    layer = canvas.add_markers(
        np.arange(10), np.arange(10) * 2
    ).with_network([[0, 1], [0, 2], [1, 3]])
    assert_array_equal(layer.nodes.data.x, np.arange(10))
    assert_array_equal(layer.nodes.data.y, np.arange(10) * 2)
    assert len(canvas.layers) == 1

def test_line_fill(backend: str):
    canvas = new_canvas(backend=backend)
    layer = canvas.add_line([-1.5, -1, 0, 1, 1.5]).with_yfill()
    layer = canvas.add_line([-1.5, -1, 0, 1, 1.5], [0, 0, 0, 0, 0]).with_yfill()
