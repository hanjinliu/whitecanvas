import numpy as np
from numpy.testing import assert_allclose

import pytest
import whitecanvas as wc
from whitecanvas import new_canvas

from ._utils import assert_color_equal


def test_namespaces(backend: str):
    canvas = new_canvas(backend=backend)
    canvas.title.text = "Title-0"
    assert canvas.title.text == "Title-0"
    canvas.title.color = "red"
    assert_color_equal(canvas.title.color, "red")
    canvas.title.size = 20
    assert canvas.title.size == 20
    canvas.title.family = "Arial"
    assert canvas.title.family == "Arial"

    canvas.x.label.text = "X-Label-0"
    assert canvas.x.label.text == "X-Label-0"
    canvas.x.label.color = "red"
    assert_color_equal(canvas.x.label.color, "red")
    canvas.x.label.size = 20
    assert canvas.x.label.size == 20
    canvas.x.label.family = "Arial"
    assert canvas.x.label.family == "Arial"

    canvas.y.label.text = "Y-Label-0"
    assert canvas.y.label.text == "Y-Label-0"
    canvas.y.label.color = "red"
    assert_color_equal(canvas.y.label.color, "red")
    canvas.y.label.size = 20
    assert canvas.y.label.size == 20
    canvas.y.label.family = "Arial"
    assert canvas.y.label.family == "Arial"

def test_namespace_pointing_at_different_objects():
    c0 = new_canvas(backend="matplotlib")
    c1 = new_canvas(backend="matplotlib")
    assert c0.title is not c1.title
    assert c0.x is not c1.x
    c0.title.text = "Title-0"
    c1.title.text = "Title-1"
    assert c0.title.text == "Title-0"
    assert c1.title.text == "Title-1"
    c0.x.color = "red"
    c1.x.color = "blue"
    assert_color_equal(c0.x.color, "red")
    assert_color_equal(c1.x.color, "blue")

def test_grid(backend: str):
    cgrid = wc.new_grid(2, 2, backend=backend).link_x().link_y()
    c00 = cgrid.add_canvas(0, 0)
    c01 = cgrid.add_canvas(0, 1)
    c10 = cgrid.add_canvas(1, 0)
    c11 = cgrid.add_canvas(1, 1)

    c00.add_line([0, 1, 2], [0, 1, 2])
    c01.add_hist([0, 1, 2, 3, 4, 3, 2, 1])
    c10.add_markers([3, 1, 2])
    c11.add_rug([1, 3, 2, 1.2])

    assert_allclose(c00.x.lim, c01.x.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c00.x.lim, c10.x.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c00.x.lim, c11.x.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c00.y.lim, c01.y.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c00.y.lim, c10.y.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c00.y.lim, c11.y.lim, rtol=1e-6, atol=1e-6)

    assert len(c00.layers) == 1
    assert len(c01.layers) == 1
    assert len(c10.layers) == 1
    assert len(c11.layers) == 1


def test_grid_nonuniform(backend: str):
    cgrid = wc.new_grid(
        [2, 1], [2, 1], backend=backend
    ).link_x().link_y()
    c00 = cgrid.add_canvas(0, 0)
    c01 = cgrid.add_canvas(0, 1)
    c10 = cgrid.add_canvas(1, 0)
    c11 = cgrid.add_canvas(1, 1)

    c00.add_line([0, 1, 2], [0, 1, 2])
    c01.add_hist([0, 1, 2, 3, 4, 3, 2, 1])
    c10.add_markers([3, 1, 2])
    c11.add_rug([1, 3, 2, 1.2])

    assert_allclose(c00.x.lim, c01.x.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c00.x.lim, c10.x.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c00.x.lim, c11.x.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c00.y.lim, c01.y.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c00.y.lim, c10.y.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c00.y.lim, c11.y.lim, rtol=1e-6, atol=1e-6)

    assert len(c00.layers) == 1
    assert len(c01.layers) == 1
    assert len(c10.layers) == 1
    assert len(c11.layers) == 1

def test_vgrid_hgrid(backend: str):
    cgrid = wc.new_col(2, backend=backend).link_x().link_y()
    c0 = cgrid.add_canvas(0)
    c1 = cgrid.add_canvas(1)

    c0.add_line([0, 1, 2], [0, 1, 2])
    c1.add_hist([0, 1, 2, 3, 4, 3, 2, 1])

    assert_allclose(c0.x.lim, c1.x.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c0.y.lim, c1.y.lim, rtol=1e-6, atol=1e-6)

    assert len(c0.layers) == 1
    assert len(c1.layers) == 1

    cgrid = wc.new_row(2, backend=backend).link_x().link_y()
    c0 = cgrid.add_canvas(0)
    c1 = cgrid.add_canvas(1)

    c0.add_line([0, 1, 2], [0, 1, 2])
    c1.add_hist([0, 1, 2, 3, 4, 3, 2, 1])

    assert_allclose(c0.x.lim, c1.x.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c0.y.lim, c1.y.lim, rtol=1e-6, atol=1e-6)

    assert len(c0.layers) == 1
    assert len(c1.layers) == 1

def test_unlink(backend: str):
    grid = wc.new_row(2, backend=backend).fill()
    linker = wc.link_axes(grid[0].x, grid[1].x)
    grid[0].x.lim = (10, 11)
    assert grid[0].x.lim == pytest.approx((10, 11))
    assert grid[1].x.lim == pytest.approx((10, 11))
    linker.unlink_all()
    grid[0].x.lim = (20, 21)
    assert grid[0].x.lim == pytest.approx((20, 21))
    assert grid[1].x.lim == pytest.approx((10, 11))

def test_jointgrid(backend: str):
    rng = np.random.default_rng(0)
    joint = wc.new_jointgrid(backend=backend).with_hist().with_kde().with_rug()
    joint.add_markers(rng.random(100), rng.random(100), color="red")
