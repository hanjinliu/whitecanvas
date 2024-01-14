import numpy as np
import pytest

from whitecanvas import new_canvas

BACKENDS = ["matplotlib", "pyqtgraph", "plotly", "bokeh", "vispy"]

@pytest.mark.parametrize("backend", BACKENDS)
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

@pytest.mark.parametrize("backend", BACKENDS)
def test_colored_plots(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
    }

    canvas.cat(df).add_markers("x", "y", color="label")
    canvas.cat(df).add_line("x", "y", color="label")
