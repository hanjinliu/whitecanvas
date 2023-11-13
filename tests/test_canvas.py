import pytest
import neoplot
import numpy as np
from numpy.testing import assert_allclose

@pytest.mark.parametrize("backend", ["matplotlib", "pyqtgraph"])
def test_line(backend: str):
    canvas = neoplot.Canvas(backend=backend)
    canvas.add_line(np.arange(10), np.zeros(10))
    layer = canvas.add_line(np.zeros(10))
    layer.line.color = [1, 0, 0, 1]
    assert_allclose(layer.line.color, [1, 0, 0, 1])
    layer.line.style = ":"
    assert layer.line.style == ":"
    layer.line.width = 2
    assert layer.line.width == 2
