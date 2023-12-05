import pytest
from whitecanvas import new_canvas
import numpy as np

BACKENDS = ["matplotlib", "pyqtgraph", "plotly", "bokeh", "vispy"]

@pytest.mark.parametrize("backend", BACKENDS)
def test_cat_plots(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
    }

    canvas.cat(df, by="label").add_stripplot(y="y")
    canvas.cat(df, by="label").add_swarmplot(y="y")
    canvas.cat(df, by="label").add_boxplot(y="y")
    canvas.cat(df, by="label").add_violinplot(y="y")
    canvas.cat(df, by="label").add_countplot()

@pytest.mark.parametrize("backend", BACKENDS)
def test_colored_plots(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
    }

    canvas.colorize(df, by="label").add_markers("x", "y")
    canvas.colorize(df, by="label").add_line("x", "y")
    canvas.colorize(df, by="label").add_hist("y")
    canvas.colorize(df, by="label").add_cdf(value_column="y")
