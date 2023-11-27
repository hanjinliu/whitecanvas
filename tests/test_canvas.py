from whitecanvas import new_canvas
import pytest
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
    canvas.title.fontfamily = "Arial"
    assert canvas.title.fontfamily == "Arial"

    canvas.x.label.text = "X-Label-0"
    assert canvas.x.label.text == "X-Label-0"
    canvas.x.label.color = "red"
    assert_color_equal(canvas.x.label.color, "red")
    canvas.x.label.size = 20
    assert canvas.x.label.size == 20
    canvas.x.label.fontfamily = "Arial"
    assert canvas.x.label.fontfamily == "Arial"

    canvas.y.label.text = "Y-Label-0"
    assert canvas.y.label.text == "Y-Label-0"
    canvas.y.label.color = "red"
    assert_color_equal(canvas.y.label.color, "red")
    canvas.y.label.size = 20
    assert canvas.y.label.size == 20
    canvas.y.label.fontfamily = "Arial"
    assert canvas.y.label.fontfamily == "Arial"
