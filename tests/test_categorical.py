import numpy as np

from whitecanvas import new_canvas
from ._utils import assert_color_array_equal
import pytest

def test_cat(backend: str):
    canvas = new_canvas(backend=backend)
    rng = np.random.default_rng(1642)
    df = {
        "x": rng.normal(size=30),
        "y": rng.normal(size=30),
        "label": np.repeat(["A", "B", "C"], 10),
    }
    canvas.cat(df, "x", "y").add_line()
    canvas.cat(df, "x", "y").add_line(color="label")
    canvas.cat(df, "x", "y").add_markers()
    canvas.cat(df, "x", "y").add_markers(color="label")
    canvas.cat(df, "x", "y").add_markers(hatch="label")
    canvas.cat(df, "x", "y").add_hist2d(bins=(5, 4))
    canvas.cat(df, "x", "y").along_x().add_hist(bins=5)
    canvas.cat(df, "x", "y").along_x().add_hist(bins=5, color="label")
    canvas.cat(df, "x", "y").along_y().add_hist(bins=6)
    canvas.cat(df, "x", "y").along_y().add_hist(bins=6, color="label")

@pytest.mark.parametrize("orient", ["v", "h"])
def test_cat_plots(backend: str, orient: str):
    canvas = new_canvas(backend=backend)
    df = {
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
        "c": ["P", "Q"] * 15,
    }
    if orient == "v":
        cat_plt = canvas.cat_x(df, "label", "y")
    else:
        cat_plt = canvas.cat_y(df, "y", "label")
    cat_plt.add_stripplot(color="c")
    cat_plt.add_swarmplot(color="c")
    cat_plt.add_boxplot(color="c")
    cat_plt.add_violinplot(color="c")

def test_colored_plots(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
    }

    canvas.cat(df, "x", "y").add_markers(color="label")
    canvas.cat(df, "x", "y").add_line(color="label")

def test_markers(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "size": np.arange(30) / 2 + 8,
        "label0": np.repeat(["A", "B", "C"], 10),  # [A, A, ..., B, B, ..., C, C, ...]
        "label1": ["One"] * 10 + ["Two"] * 20,
    }

    _c = canvas.cat(df, "x", "y")
    out = _c.add_markers(color="label0", size="size", symbol="label1")
    assert len(set(out._base_layer.symbol[:10])) == 1
    assert len(set(out._base_layer.symbol[10:])) == 1

    out = _c.add_markers(color="label1", size="size", hatch="label0")
    assert len(set(out._base_layer.face.hatch[:10])) == 1
    assert len(set(out._base_layer.face.hatch[10:20])) == 1
    assert len(set(out._base_layer.face.hatch[20:])) == 1

    out = _c.add_markers(color="label1").with_edge(color="label0")
    assert len(np.unique(out._base_layer.edge.color[:10], axis=0)) == 1
    assert len(np.unique(out._base_layer.edge.color[10:20], axis=0)) == 1
    assert len(np.unique(out._base_layer.edge.color[20:], axis=0)) == 1

    # test scalar color
    out = _c.add_markers(color="black")
    assert_color_array_equal(out._base_layer.face.color, "black")

    out = _c.add_markers(color="transparent").with_edge_colormap("size")

def test_heatmap(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": ["A", "B", "A", "B", "A", "B"],
        "y": ["P", "P", "Q", "Q", "R", "R"],
        "z": [1, 2, 3, 4, 5, 6],
    }
    im = canvas.cat_xy(df, "x", "y").first().add_heatmap(value="z")
    canvas.imref(im).add_text()

    df = {
        "x": ["A", "A", "A", "B", "A", "B"],
        "y": ["P", "Q", "Q", "Q", "P", "Q"],
        "z": [1.1, 2.1, 3.4, 6.4, 1.1, 6.8],
    }
    im = canvas.cat_xy(df, "x", "y").mean().add_heatmap(value="z", fill=-1)
    canvas.imref(im).add_text(fmt=".1f")
    assert im.clim == (1.1, 6.6)
