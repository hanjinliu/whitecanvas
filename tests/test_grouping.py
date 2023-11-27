import pytest
from whitecanvas import new_canvas
import numpy as np
from numpy.testing import assert_array_equal
from cmap import Color

BACKENDS = ["matplotlib", "pyqtgraph", "plotly", "bokeh", "vispy"]

@pytest.mark.parametrize("backend", BACKENDS)
def test_line_marker_text(backend: str):
    canvas = new_canvas(backend=backend)
    layer = canvas.add_line(
        np.arange(10), np.arange(10) * 2
    ).with_markers(
        symbol="s", size=10, color="red"
    ).with_text(
        [f"Data-{i}" for i in range(10)], color="blue"
    )
    assert_array_equal(layer.data.x, np.arange(10))
    assert_array_equal(layer.data.y, np.arange(10) * 2)
    assert len(canvas.layers) == 1
