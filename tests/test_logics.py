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
    layer = canvas.add_hist(data, bins=[-2, -1, 0, 1.4, 2.5], kind="percent")
    layer.data
    layer.data = data + 1
    layer.shape
    layer.shape = "step"
    layer.shape = "polygon"
    layer.shape = "bars"
    layer.zorders
    layer.zorders = [0, 1]
    layer.kind
    layer.kind = "density"
    layer.limits
    layer.limits = (-2, 2)
    layer.edges
    layer.update_edges([-2, 0, 1, 2])

def test_kde():
    canvas = new_canvas(backend="mock")
    rng = np.random.default_rng(1642)
    data = np.concatenate([rng.normal(size=50), rng.normal(1.2, size=40)])
    canvas.add_kde(data)
    layer = canvas.add_kde(data, band_width=0.6)
    layer.bottom
    layer.bottom = 1
    layer.band_width
    layer.band_width = 0.5

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

def test_fit():
    canvas = new_canvas(backend="mock")
    rng = np.random.default_rng(1642)
    x = rng.normal(size=50)
    y = rng.normal(size=50) + 2 * x
    layer = canvas.add_markers(x, y)
    canvas.fit(layer).linear(color="red")
    canvas.fit(layer).polynomial(2)

def test_between():
    canvas = new_canvas(backend="mock")
    layer0 = canvas.add_markers([0] * 5, [0, 1, 2, 3, 4])
    layer1 = canvas.add_markers([1] * 5, [5, 1, 4, 2, 4])
    layer2 = canvas.add_markers([2] * 4, [0, 1, 2, 3])
    canvas.between(layer0, layer1).connect_points()
    with pytest.raises(ValueError):
        canvas.between(layer0, layer2).connect_points()

def test_melt():
    canvas = new_canvas(backend="mock")
    rng = np.random.default_rng(1642)
    df = {"x": rng.normal(size=15), "y": rng.normal(size=15)}
    canvas.cat_x(df).melt().add_boxplot(color="variable")

def test_sort():
    rng = np.random.default_rng(1642)
    df = {
        "x": ["a"] * 14 + ["b"] * 10 + ["c"] * 16,
        "y": ["p"] * 20 + ["q"] * 20,
        "val": rng.normal(size=40),
    }

    # simple
    canvas = new_canvas(backend="mock")
    canvas.cat_x(df, "x", "val").sort_in_order(["b", "a", "c"]).add_boxplot()
    assert canvas.x.ticks.labels == ["b", "a", "c"]

    # with color
    canvas = new_canvas(backend="mock")
    canvas.cat_x(df, "x", "val").sort_in_order(["b", "a", "c"]).add_boxplot(color="y")
    assert canvas.x.ticks.labels == ["b", "a", "c"]

    # ascending sort
    canvas = new_canvas(backend="mock")
    canvas.cat_x(df, "x", "val").sort(ascending=True).add_stripplot()
    assert canvas.x.ticks.labels == ["a", "b", "c"]

    # descending sort
    canvas = new_canvas(backend="mock")
    canvas.cat_x(df, "x", "val").sort(ascending=False).add_stripplot()
    assert canvas.x.ticks.labels == ["c", "b", "a"]

    # two keys
    canvas = new_canvas(backend="mock")
    (
        canvas
        .cat_y(df, "val", ["x", "y"])
        .sort_in_order([("b", "q"), ("a", "p"), ("c", "q"), ("c", "p"), ("a", "q"), ("b", "p")])
        .add_swarmplot()
    )
    assert canvas.y.ticks.labels == ["b\nq", "a\np", "c\nq", "c\np", "a\nq", "b\np"]

    # xy categorical plotter
    canvas = new_canvas(backend="mock")
    (
        canvas
        .cat_xy(df, "x", "y")
        .sort_in_order(x=["b", "a", "c"], y=["q", "p"])
        .mean()
        .add_heatmap("val")
    )
    assert canvas.x.ticks.labels == ["b", "a", "c"]
    assert canvas.y.ticks.labels == ["q", "p"]

def test_matplotlib_tooltip():
    canvas = new_canvas(backend="matplotlib")
    canvas.add_markers([1, 2, 3], [4, 5, 6])
    canvas._canvas()._set_tooltip((2, 5), "tooltip")
    canvas._canvas()._hide_tooltip()
