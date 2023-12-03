import pytest
from whitecanvas import new_canvas
from whitecanvas.layers import Line, Markers, Layer
import numpy as np
from ._utils import assert_color_equal

BACKENDS = ["matplotlib", "pyqtgraph", "plotly", "bokeh", "vispy"]

@pytest.mark.parametrize("backend", BACKENDS)
def test_cat_plots(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
    }

    canvas.cat(df, by="label").to_stripplot(y="y")
    canvas.cat(df, by="label").to_swarmplot(y="y")
    canvas.cat(df, by="label").to_boxplot(y="y")
    canvas.cat(df, by="label").to_violinplot(y="y")
    canvas.cat(df, by="label").to_countplot()

@pytest.mark.parametrize("backend", BACKENDS)
def test_colored_plots(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
    }

    canvas.colorize(df, by="label").to_markers("x", "y")
    canvas.colorize(df, by="label").to_lines("x", "y")
    canvas.colorize(df, by="label").to_hist("y")
    canvas.colorize(df, by="label").to_cdf(value_column="y")
