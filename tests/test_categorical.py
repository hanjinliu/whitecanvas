import numpy as np

from whitecanvas import new_canvas
from ._utils import assert_color_array_equal
import pytest

@pytest.mark.parametrize("orient", ["v", "h"])
def test_cat_plots(backend: str, orient: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
    }

    canvas.cat(df).add_stripplot("label", "y", orient=orient)
    canvas.cat(df).add_swarmplot("label", "y", orient=orient)
    canvas.cat(df).add_boxplot("label", "y", orient=orient)
    canvas.cat(df).add_violinplot("label", "y", orient=orient)
    canvas.cat(df).add_countplot("label", orient=orient)

def test_colored_plots(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
    }

    canvas.cat(df).add_markers("x", "y", color="label")
    canvas.cat(df).add_line("x", "y", color="label")

def test_markers(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "size": np.arange(30) / 2 + 8,
        "label0": np.repeat(["A", "B", "C"], 10),  # [A, A, ..., B, B, ..., C, C, ...]
        "label1": ["One"] * 10 + ["Two"] * 20,
    }

    _c = canvas.cat(df)
    out = _c.add_markers("x", "y", color="label0", size="size", symbol="label1")
    assert len(set(out._base_layer.symbol[:10])) == 1
    assert len(set(out._base_layer.symbol[10:])) == 1

    out = _c.add_markers("x", "y", color="label1", size="size", hatch="label0")
    assert len(set(out._base_layer.face.hatch[:10])) == 1
    assert len(set(out._base_layer.face.hatch[10:20])) == 1
    assert len(set(out._base_layer.face.hatch[20:])) == 1

    out = _c.add_markers("x", "y", color="label1").with_edge(color="label0")
    assert len(np.unique(out._base_layer.edge.color[:10], axis=0)) == 1
    assert len(np.unique(out._base_layer.edge.color[10:20], axis=0)) == 1
    assert len(np.unique(out._base_layer.edge.color[20:], axis=0)) == 1

    # test scalar color
    out = _c.add_markers("x", "y", color="black")
    assert_color_array_equal(out._base_layer.face.color, "black")

    out = _c.add_markers("x", "y", color="transparent").with_edge_colormap("size")

def test_heatmap(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": ["A", "B", "A", "B", "A", "B"],
        "y": ["P", "P", "Q", "Q", "R", "R"],
        "z": [1, 2, 3, 4, 5, 6],
    }
    im = canvas.cat(df).add_heatmap("x", "y", value="z")
    canvas.imref(im).add_text()

    df = {
        "x": ["A", "A", "A", "B", "A", "B"],
        "y": ["P", "Q", "Q", "Q", "P", "Q"],
        "z": [1.1, 2.1, 3.4, 6.4, 1.1, 6.8],
    }
    with pytest.raises(ValueError):
        # has duplication
        canvas.cat(df).add_heatmap("x", "y", value="z")
    im = canvas.cat(df).mean().add_heatmap("x", "y", value="z", fill=-1)
    canvas.imref(im).add_text(fmt=".1f")
    assert im.clim == (1.1, 6.6)
