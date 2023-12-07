from whitecanvas import new_canvas
import whitecanvas as wc
import pytest
from numpy.testing import assert_allclose
from ._utils import assert_color_equal

BACKENDS = ["matplotlib", "pyqtgraph", "plotly", "bokeh", "vispy"]

@pytest.mark.parametrize("backend", BACKENDS)
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

@pytest.mark.parametrize("backend", BACKENDS)
def test_grid(backend: str):
    cgrid = wc.grid(2, 2, link_x=True, link_y=True, backend=backend)
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_grid_nonuniform(backend: str):
    cgrid = wc.grid_nonuniform([2, 1], [2, 1], link_x=True, link_y=True, backend=backend)
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

@pytest.mark.parametrize("backend", BACKENDS)
def test_vgrid_hgrid(backend: str):
    cgrid = wc.vgrid(2, backend=backend, link_x=True, link_y=True)
    c0 = cgrid.add_canvas(0)
    c1 = cgrid.add_canvas(1)

    c0.add_line([0, 1, 2], [0, 1, 2])
    c1.add_hist([0, 1, 2, 3, 4, 3, 2, 1])

    assert_allclose(c0.x.lim, c1.x.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c0.y.lim, c1.y.lim, rtol=1e-6, atol=1e-6)

    assert len(c0.layers) == 1
    assert len(c1.layers) == 1

    cgrid = wc.hgrid(2, backend=backend, link_x=True, link_y=True)
    c0 = cgrid.add_canvas(0)
    c1 = cgrid.add_canvas(1)

    c0.add_line([0, 1, 2], [0, 1, 2])
    c1.add_hist([0, 1, 2, 3, 4, 3, 2, 1])

    assert_allclose(c0.x.lim, c1.x.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c0.y.lim, c1.y.lim, rtol=1e-6, atol=1e-6)

    assert len(c0.layers) == 1
    assert len(c1.layers) == 1
