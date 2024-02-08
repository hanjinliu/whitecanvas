import pytest
from whitecanvas import new_canvas
import numpy as np


def test_hist():
    canvas = new_canvas(backend="mock")
    rng = np.random.default_rng(1642)
    data = np.concatenate([rng.normal(size=50), rng.normal(1.2, size=40)])
    canvas.add_hist(data, bins="auto")
    canvas.add_hist(data, bins=10)
    canvas.add_hist(data, bins=10, limits=(-2, 2))
    canvas.add_hist(data, bins=[-2, 0, 1, 2], shape="step")
    canvas.add_hist(data, bins=[-2, 0, 1, 2], shape="polygon")
    canvas.add_hist(data, bins=[-2, 0, 1, 2], shape="bars")
    with pytest.raises(ValueError):
        canvas.add_hist(data, bins=[-2, 0, 1, 2], shape="not-a-shape")
    canvas.add_hist(data, bins=[-2, -1, 0, 1.4, 2.5], kind="density")
    canvas.add_hist(data, bins=[-2, -1, 0, 1.4, 2.5], kind="probability")
    canvas.add_hist(data, bins=[-2, -1, 0, 1.4, 2.5], kind="frequency")
    canvas.add_hist(data, bins=[-2, -1, 0, 1.4, 2.5], kind="percent")

def test_hover_template():
    canvas = new_canvas(backend="mock")
    layer = canvas.add_markers([1, 2, 3], [4, 5, 6])
    layer.with_hover_template("const-text")
    layer.with_hover_template("{x}")
    layer.with_hover_template("{x}, {y}")
    layer.with_hover_template("{i}")
    layer.with_hover_template("a: {a}", extra={"a": [0, 1, 4]})
    layer.with_hover_template("x: {x}", extra={"x": [0, 1, 4]})
    layer.with_hover_template("a: {a:.2f}", extra={"a": [0, 1, 4.0], "b": [1, 2, 3]})
    with pytest.raises(KeyError):
        layer.with_hover_template("c: {c}", extra={"a": [0, 1, 4], "b": [1, 2, 3]})
    with pytest.raises(ValueError):
        layer.with_hover_template("c: {c}", extra={"c": [0, 1, 4, 5]})
