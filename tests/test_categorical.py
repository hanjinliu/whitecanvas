import numpy as np

from whitecanvas import new_canvas


def test_cat_plots(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
    }

    canvas.cat(df).add_stripplot("label", "y")
    canvas.cat(df).add_swarmplot("label", "y")
    canvas.cat(df).add_boxplot("label", "y")
    canvas.cat(df).add_violinplot("label", "y")
    canvas.cat(df).add_countplot("label")

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
        "label0": np.repeat(["A", "B", "C"], 10),
        "label1": ["One"] * 10 + ["Two"] * 20,
    }

    out = canvas.cat(df).add_markers(
        "x", "y", color="label0", size="size", symbol="label1"
    )
    assert len(set(out._base_layer.symbol[:10])) == 1
    assert len(set(out._base_layer.symbol[10:])) == 1
    canvas.cat(df).add_markers("x", "y", color="label1", size="size", hatch="label0")
    assert len(set(out._base_layer.face.hatch[:10])) == 1
    assert len(set(out._base_layer.face.hatch[10:])) == 1
