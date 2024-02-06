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
