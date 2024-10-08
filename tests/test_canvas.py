import sys
from pathlib import Path
import tempfile
import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt

import pytest
import whitecanvas as wc
from whitecanvas import new_canvas, wrap_canvas, new_canvas_3d
from whitecanvas.types import Point, MouseEvent

from ._utils import assert_color_equal, filter_warning


def test_namespaces(backend: str):
    canvas = new_canvas(backend=backend)
    canvas.aspect_ratio
    canvas.aspect_ratio = 1
    assert canvas.aspect_ratio == 1

    canvas.background_color

    canvas.title.text = "Title-0"
    assert canvas.title.text == "Title-0"
    canvas.title.color = "red"
    assert_color_equal(canvas.title.color, "red")
    canvas.title.size = 20
    assert canvas.title.size == 20
    canvas.title.family = "Arial"
    assert canvas.title.family == "Arial"
    canvas.title.visible = False
    assert not canvas.title.visible
    canvas.title.visible = True
    assert canvas.title.visible

    canvas.x.label.text = "X-Label-0"
    assert canvas.x.label.text == "X-Label-0"
    canvas.x.label.color = "red"
    assert_color_equal(canvas.x.label.color, "red")
    canvas.x.label.size = 20
    assert canvas.x.label.size == 20
    canvas.x.label.family = "Arial"
    assert canvas.x.label.family == "Arial"
    canvas.x.label.visible = False
    assert not canvas.x.label.visible
    canvas.x.label.visible = True
    assert canvas.x.label.visible

    canvas.y.label.text = "Y-Label-0"
    assert canvas.y.label.text == "Y-Label-0"
    canvas.y.label.color = "red"
    assert_color_equal(canvas.y.label.color, "red")
    canvas.y.label.size = 20
    assert canvas.y.label.size == 20
    canvas.y.label.family = "Arial"
    assert canvas.y.label.family == "Arial"

    # test lim
    canvas.x.lim = (-1, 1)
    assert canvas.x.lim == pytest.approx((-1, 1))
    canvas.x.lim = (-0.4, None)
    assert canvas.x.lim == pytest.approx((-0.4, 1))
    canvas.x.lim = (None, 0.6)
    assert canvas.x.lim == pytest.approx((-0.4, 0.6))

    # test errors
    with pytest.raises(ValueError):
        canvas.x.lim = (1, 0)
    with pytest.raises(TypeError):
        canvas.x.label = 0

    repr(canvas.x)
    repr(canvas.title)
    repr(canvas.x.ticks)

    if backend != "pyqtgraph":  # not implemented in pyqtgraph
        canvas.x.ticks.rotation = 45
        assert canvas.x.ticks.rotation == pytest.approx(45)

    canvas.x.ticks.visible = False
    assert not canvas.x.ticks.visible
    canvas.x.ticks.visible = True
    assert canvas.x.ticks.visible
    canvas.x.ticks.set_labels([0, 1, 2], ["a", "b", "c"])

    # get tick positions and labels are still hard to implement
    if backend in ("mock", "pyqtgraph", "plotly"):
        assert_allclose(canvas.x.ticks.pos, [0, 1, 2])
        assert canvas.x.ticks.labels == ["a", "b", "c"]
    canvas.x.ticks.reset_labels()

    canvas.x.set_gridlines()
    canvas.copy()
    canvas.read_json(canvas.write_json(), backend=backend)

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

def test_update_methods():
    canvas = new_canvas(backend="mock")
    canvas.update_axes(visible=False)
    canvas.update_labels(title="Title", x="X", y="Y")
    canvas.update_font(size=24, color="red", family="Arial")

def test_native_and_wrapping(backend: str):
    canvas = new_canvas(backend=backend)
    assert canvas.native is not None
    if backend == "mock":
        return
    new = wrap_canvas(canvas.native)
    repr(new._get_backend())
    assert new._get_backend() == canvas._get_backend()

def test_grid(backend: str):
    cgrid = wc.new_grid(2, 2, backend=backend).link_x().link_y()
    c00 = cgrid.add_canvas(0, 0)
    c01 = cgrid.add_canvas(0, 1)
    c10 = cgrid.add_canvas(1, 0)
    c11 = cgrid.add_canvas(1, 1)
    assert cgrid[0, 0] is not cgrid[1, 0]
    assert cgrid[0, 1] is not cgrid[1, 1]
    assert cgrid[0, 1].visible
    with filter_warning(backend, "plotly"):
        cgrid[0, 1].visible = False

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

    new = cgrid.copy()
    assert len(new[0, 0].layers) == 1
    assert len(new[0, 1].layers) == 1
    assert len(new[1, 0].layers) == 1
    assert len(new[1, 1].layers) == 1

    cgrid.read_json(cgrid.write_json(), backend=backend)

def test_grid_nonuniform(backend: str):
    cgrid = wc.new_grid([2, 1], [2, 1], backend=backend, size=(100, 100)).link_x().link_y()
    repr(cgrid)
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

    new = cgrid.copy()
    assert len(new[0, 0].layers) == 1
    assert len(new[0, 1].layers) == 1
    assert len(new[1, 0].layers) == 1
    assert len(new[1, 1].layers) == 1
    cgrid.read_json(cgrid.write_json(), backend=backend)


def test_vgrid_hgrid(backend: str):
    cgrid = wc.new_col(2, backend=backend, size=(100, 100)).link_x().link_y()
    repr(cgrid)
    c0 = cgrid.add_canvas(0)
    c1 = cgrid.add_canvas(1)
    assert cgrid[0] is not cgrid[1]

    c0.add_line([0, 1, 2], [0, 1, 2])
    c1.add_hist([0, 1, 2, 3, 4, 3, 2, 1])

    assert_allclose(c0.x.lim, c1.x.lim, rtol=1e-6, atol=1e-6)
    assert_allclose(c0.y.lim, c1.y.lim, rtol=1e-6, atol=1e-6)

    assert len(c0.layers) == 1
    assert len(c1.layers) == 1

    cgrid = wc.new_row(2, backend=backend, size=(100, 100)).link_x().link_y()
    c0 = cgrid.add_canvas(0)
    c1 = cgrid.add_canvas(1)
    assert cgrid[0] is not cgrid[1]

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

    grid = wc.new_row(2, backend=backend).fill().link_x().link_y()

def test_jointgrid(backend: str):
    rng = np.random.default_rng(0)
    joint = wc.new_jointgrid(backend=backend, size=(100, 100)).with_hist().with_kde().with_rug()
    joint.add_markers(rng.random(100), rng.random(100), color="red")
    if backend != "vispy":
        joint.add_legend()
    joint.copy()
    joint.read_json(joint.write_json(), backend=backend)

def test_legend(backend: str):
    if backend == "vispy":
        pytest.skip("vispy does not support legend")
    canvas = new_canvas(backend=backend)
    canvas.add_line([0, 1, 2], [0, 1, 2])
    canvas.add_markers([0, 1, 2], [0, 1, 2], name="markers")
    canvas.add_bars([0, 1, 2], [0, 1, 2], name="bars")
    canvas.add_line([3, 4, 5], [1, 2, 1], name="plot").with_markers()
    canvas.add_line([3, 4, 5], [1, 2, 1], name="plot").with_xband([1, 1, 1])
    canvas.add_line([3, 4, 5], [1, 2, 1], name="plot").with_yband([1, 1, 1])
    canvas.add_line([3, 4, 5], [2, 3, 2], name="line+err").with_yerr([1, 1, 1])
    canvas.add_markers([3, 4, 5], [3, 4, 3], name="markers+err").with_xerr([1, 1, 1])
    canvas.add_line([3, 4, 5], [4, 5, 4], name="plot+err").with_markers().with_xerr([1, 1, 1]).with_yerr([1, 1, 1])
    canvas.add_markers([3, 4, 5], [5, 6, 5], name="markers+err+err").with_stem()
    canvas.add_legend(location="bottom_right")

def test_animation():
    from whitecanvas.animation import Animation

    canvas = new_canvas(backend="matplotlib")
    anim = Animation(canvas)
    x = np.linspace(0, 2 * np.pi, 100)
    line = canvas.add_line(x, np.sin(x + 0), name="line")
    for i in anim.iter_range(3):
        line.set_data(x, np.sin(x + i * np.pi / 3))
    with tempfile.TemporaryDirectory() as tmpdir:
        anim.save(Path(tmpdir) / "test.gif")
    assert anim.asarray().ndim == 4

def test_layer_handling(backend: str):
    canvas = new_canvas(backend=backend)
    canvas.add_line([0, 1, 2], name="l_0")
    canvas.add_markers([0, 1, 2], name="l_1").with_xerr([0.1, 0.1, 0.1])
    repr(canvas.layers)
    assert canvas.layers[0].name == "l_0"
    assert canvas.layers[0] is canvas.layers["l_0"]
    assert canvas.layers.get("l_0") is canvas.layers[0]
    canvas.layers.move(1, 0)
    canvas.layers.pop()
    canvas.layers.pop()
    assert len(canvas.layers) == 0

def test_multidim():
    canvas = new_canvas(backend="matplotlib")
    x = np.arange(5)
    ys = [x, x ** 2, x ** 3]
    canvas.dims.in_axes("T").add_line(x, ys).copy().write_json()
    canvas.dims.add_markers(x, ys, color="red", symbol="+").copy().write_json()
    canvas.dims.add_rug(ys).copy().write_json()
    canvas.dims.add_band(x, ys, [y0 + 1 for y0 in ys]).copy().write_json()
    canvas.dims.add_errorbars(x, ys, [y - 1 for y in ys]).copy().write_json()
    canvas.dims.add_text(x, ys, ["a", "b", "c", "d", "e"]).copy().write_json()
    img = np.zeros((2, 5, 5))
    canvas.dims.add_image(img).copy().write_json()
    canvas.dims.set_indices(1)
    canvas.dims.set_indices({"T": 0})
    canvas.dims.set_values({"T": 1})
    canvas._repr_png_()

def test_multidim_slider():
    # TODO: how to test other app backends?
    plt.close("all")
    canvas = new_canvas(backend="matplotlib:qt")
    x = np.arange(5)
    ys = [x, x ** 2, x ** 3]
    canvas.dims.add_line(x, ys)
    slider = canvas.dims.create_widget()
    slider._emit_changed()

def test_second_x(backend: str):
    if backend in ("vispy", "plotly"):
        pytest.skip(f"{backend} does not support second_x")
    canvas = new_canvas(backend=backend)
    other = canvas.install_second_x()
    canvas.add_line([0, 1, 2], [0, 1, 2], name="line")
    other.add_line([0, 1, 2], [0, 1, 2], name="line")
    canvas.y.lim = (0, 2)
    assert canvas.y.lim == pytest.approx((0, 2))
    assert other.y.lim == pytest.approx((0, 2))

    canvas.copy()
    canvas.read_json(canvas.write_json(), backend=backend)

def test_second_y(backend: str):
    if backend in ("vispy",):
        pytest.skip(f"{backend} does not support second_y")
    canvas = new_canvas(backend=backend)
    other = canvas.install_second_y()
    canvas.add_line([0, 1, 2], [0, 1, 2], name="line")
    other.add_line([0, 1, 2], [0, 1, 2], name="line")
    canvas.x.lim = (0, 2)
    assert canvas.x.lim == pytest.approx((0, 2))
    assert other.x.lim == pytest.approx((0, 2))

    canvas.copy()
    canvas.read_json(canvas.write_json(), backend=backend)

def test_inset(backend: str):
    if backend in ("vispy", "plotly", "bokeh"):
        pytest.skip(f"{backend} does not support inset")
    canvas = new_canvas(backend=backend)
    inset = canvas.install_inset((0.5, 0.96, 0.6, 0.96))
    canvas.add_line([0, 1, 2], [0, 1, 2], name="line")
    inset.add_line([0, 1, 2], [0, 1, 2], name="line")
    canvas.copy()
    canvas.read_json(canvas.write_json(), backend=backend)

def test_to_html(backend: str):
    if backend not in ("plotly", "bokeh"):
        pytest.skip(f"{backend} does not support to_html")
    canvas = new_canvas(backend=backend)
    canvas.add_line([0, 1, 2], [0, 1, 2], name="line")
    canvas.to_html()

def test_screenshot(backend: str):
    if backend in ("plotly", "bokeh", "mock"):
        pytest.skip(f"{backend} does not support screenshot")
    if backend == "vispy":
        backend += ":qt"
    canvas = new_canvas(backend=backend)
    canvas.add_line([0, 1, 2], [0, 1, 2], name="line")
    if sys.platform != "win32":  # NOTE: testing in GitHub Actions fails for some reason
        canvas._repr_png_()

def test_emulating_mouse_click():
    canvas = new_canvas("mock")

    clicked_history: list[Point] = []
    double_clicked_history: list[Point] = []

    @canvas.mouse.clicked.connect
    def _clicked(ev: MouseEvent):
        clicked_history.append(ev.pos)

    @canvas.mouse.double_clicked.connect
    def _double_clicked(ev: MouseEvent):
        double_clicked_history.append(ev.pos)

    canvas.mouse.emulate_click((1, 1))
    assert clicked_history == [(1, 1)]
    assert double_clicked_history == []

    canvas.mouse.emulate_click((1, 2))
    assert clicked_history == [(1, 1), (1, 2)]
    assert double_clicked_history == []

    canvas.mouse.emulate_double_click((-1, -1))
    assert clicked_history == [(1, 1), (1, 2)]
    assert double_clicked_history == [(-1, -1)]

def test_emulating_mouse_move():
    canvas = new_canvas("mock")
    history: list[str] = []

    @canvas.mouse.moved.connect
    def _move(ev: MouseEvent):
        if ev.button != "right":
            return
        history.append("press")
        yield
        while ev.type != "release":
            history.append("move")
            yield
        history.append("release")

    canvas.mouse.emulate_drag([(1, 1), (1, 2), (1, 3)], button="left")
    assert history == []
    canvas.mouse.emulate_drag([(1, 1), (1, 2), (1, 3)], button="right")
    assert history == ["press", "move", "move", "release"]

def test_canvas_3d(backend: str):
    if backend not in ("matplotlib", "vispy", "plotly"):
        pytest.skip(f"{backend} does not support 3d")
    canvas = new_canvas_3d(backend=backend)
    if backend != "vispy":
        canvas.update_axes(visible=True, color="gray")
        canvas.update_font(size=13, color="pink", family="Arial")
    canvas.update_labels(title="Title", x="X", y="Y", z="Z")
    canvas.add_line([0, 1, 2], [0, 1, 2], [0, 1, 2])
    canvas.add_markers([0, 1, 2], [0, 1, 2], [0, 1, 2])
    if backend == "matplotlib":
        canvas.add_vectors([0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 1, 2], [0, 1, 2], [1, 1, 1])
    canvas.add_surface(np.array([[2,3,2], [3,2,3]]))
    canvas.read_json(canvas.write_json(), backend=backend)
