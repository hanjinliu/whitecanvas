import numpy as np

from whitecanvas import new_canvas, read_canvas
from whitecanvas.core import new_jointgrid
from ._utils import assert_color_array_equal, filter_warning
import pytest

def test_cat(backend: str):
    canvas = new_canvas(backend=backend)
    rng = np.random.default_rng(1642)
    df = {
        "x": rng.normal(size=30),
        "y": rng.normal(size=30),
        "label": np.repeat(["A", "B", "C"], 10),
    }
    cplt = canvas.cat(df, "x", "y")
    cplt.add_line().update_style("--").copy()
    cplt.add_line(color="label").update_style("label").with_markers().copy()
    cplt.add_markers().copy()
    cplt.add_markers(color="label").copy()
    cplt.add_markers(hatch="label").copy()
    cplt.add_pointplot(color="label").copy()
    cplt.add_hist2d(bins=5).copy()
    cplt.add_hist2d(bins=(5, 4)).copy()
    cplt.add_hist2d(bins="auto").copy()
    cplt.add_hist2d(bins=(5, 4), color="label").copy()
    cplt.add_hist2d(bins=("auto", 5)).copy()
    cplt.add_kde2d().copy()
    cplt.add_kde2d(color="label").copy()
    cplt.along_x().add_hist(bins=5).copy()
    cplt.along_x().add_hist(bins=5, color="label").copy()
    cplt.along_y().add_hist(bins=6).copy()
    hist = cplt.along_y().add_hist(bins=6, color="label").copy()
    cplt.along_x().add_kde().copy()
    kde = cplt.along_x().add_kde(color="label").copy()
    cplt.along_x().add_rug().copy()
    with filter_warning(backend, "plotly"):
        cplt.along_x().add_rug(color="label").copy()
    hist.update_color("black").copy()
    kde.update_color("label").copy()
    hist.update_width(1.5).copy()
    kde.update_width(1.5).copy()
    hist.update_style(":").copy()
    kde.update_style("label").copy()
    hist.update_hatch("label").copy()
    kde.update_hatch("/").copy()
    with filter_warning(backend, "vispy"):
        canvas.add_legend()
    read_canvas(canvas.write_json())

@pytest.mark.parametrize("orient", ["v", "h"])
def test_cat_plots(backend: str, orient: str):
    canvas = new_canvas(backend=backend)
    df = {
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
        "c": ["P", "Q"] * 15,
        "val": np.cos(np.arange(30) / 10),
    }
    if orient == "v":
        cat_plt = canvas.cat_x(df, "label", "y")
    else:
        cat_plt = canvas.cat_y(df, "y", "label")
    cat_plt.add_stripplot(color="c", alpha=0.8).move(0.1).copy()
    cat_plt.add_swarmplot(color="c", alpha="val").move(0.1).copy()
    cat_plt.add_boxplot(color="c").with_edge().with_outliers(ratio=0.5).copy()
    with filter_warning(backend, "plotly"):
        box = cat_plt.add_boxplot(color="c").as_edge_only().move(0.1).copy()
        box.update_color_palette(["blue", "red"], alpha=0.9, cycle_by="c")
        box = cat_plt.add_boxplot(hatch="c").as_edge_only().move(0.1).copy()
        box.update_hatch_palette(["/", "x"])
        box.update_const(color="black", hatch="+")
    cat_plt.add_violinplot(color="c").with_rug().copy()
    cat_plt.add_violinplot(color="c").with_outliers(ratio=0.5).copy()
    cat_plt.add_violinplot(color="c").with_box().copy()
    cat_plt.add_violinplot(color="c").as_edge_only().move(0.1).with_strip().copy()
    cat_plt.add_violinplot(color="c").with_swarm().copy()
    cat_plt.add_pointplot(color="c").err_by_se().err_by_sd().err_by_quantile().est_by_mean().est_by_median().move(0.1).copy()
    cat_plt.add_barplot(color="c").err_by_se().err_by_sd().err_by_quantile().est_by_mean().est_by_median().move(0.1).copy()
    with filter_warning(backend, "plotly"):
        cat_plt.add_rugplot(color="c").scale_by_density().move(0.1).copy()
    cat_plt.add_heatmap_hist(bins=4).copy()
    cat_plt.add_heatmap_hist(bins=4, color="c").copy()
    read_canvas(canvas.write_json())

def test_single_point():
    import warnings

    warnings.filterwarnings("error")  # to make sure std is not called with single point
    canvas = new_canvas(backend="mock")
    df = {"cat": ["a", "b", "b", "c", "c", "c"], "val": np.arange(6)}
    cat_plt = canvas.cat_x(df, "cat", "val")
    cat_plt.add_barplot()
    cat_plt.add_boxplot()
    cat_plt.add_heatmap_hist()
    cat_plt.add_pointplot()
    cat_plt.add_violinplot()

def test_cat_plots_with_sequential_color():
    df = {
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
        "c": np.random.default_rng(1642).normal(size=30),
        "val": np.cos(np.arange(30) / 10),
    }
    canvas = new_canvas(backend="mock")
    cat_plt = canvas.cat_y(df, "y", "label")
    cat_plt.add_stripplot(color="c", alpha=0.8).copy()
    cat_plt.add_swarmplot(color="c", alpha="val").copy()
    cat_plt.add_rugplot(color="c").copy()

    canvas = new_canvas(backend="mock")
    cat_plt = canvas.cat(df, "y", "val")
    cat_plt.add_markers(color="c").copy()

def test_markers(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "size": np.arange(30) / 2 + 8,
        "label0": np.repeat(["A", "B", "C"], 10),  # [A, A, ..., B, B, ..., C, C, ...]
        "label1": ["One"] * 10 + ["Two"] * 20,
    }

    _c = canvas.cat(df, "x", "y")
    out = _c.add_markers(color="label0", size="size", symbol="label1")
    out_copy = out.copy()
    assert all(out.base.symbol == out_copy.base.symbol)
    assert len(set(out.base.symbol[:10])) == 1
    assert len(set(out.base.symbol[10:])) == 1

    out = _c.add_markers(color="label1", size="size", hatch="label0").copy()
    assert len(set(out.base.face.hatch[:10])) == 1
    assert len(set(out.base.face.hatch[10:20])) == 1
    assert len(set(out.base.face.hatch[20:])) == 1

    out = _c.add_markers(color="label1").with_edge(color="label0").copy()
    assert len(np.unique(out.base.edge.color[:10], axis=0)) == 1
    assert len(np.unique(out.base.edge.color[10:20], axis=0)) == 1
    assert len(np.unique(out.base.edge.color[20:], axis=0)) == 1

    # test scalar color
    out = _c.add_markers(color="black").copy()
    assert_color_array_equal(out.base.face.color, "black")

    out = _c.add_markers(color="transparent").update_edge_colormap("size").copy()
    _c.mean_for_each("label0").add_markers(symbol="D").copy()

def test_cat_xy(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "x": ["A", "B", "A", "B", "A", "B"],
        "y": ["P", "P", "Q", "Q", "R", "R"],
        "z": [1, 2, 3, 4, 5, 6],
    }
    im = canvas.cat_xy(df, "x", "y").first().add_heatmap(value="z")
    im.cmap
    im.cmap = "jet"
    im.clim
    im.clim = (0, 1)
    im.with_text()
    im.with_colorbar(orient="horizontal")

    df = {
        "x": ["A", "A", "A", "B", "A", "B"],
        "y": ["P", "Q", "Q", "Q", "P", "Q"],
        "z": [1.1, 2.1, 3.4, 6.4, 1.1, 6.8],
    }
    im = (
        canvas.cat_xy(df, "x", "y")
        .mean()
        .add_heatmap(value="z", fill=-1)
    )
    im.with_text(fmt=".1f")
    assert im.clim == (1.1, 6.6)

    canvas.cat_xy(df, "x", "y").first().add_markers(value="z")

    if backend != "vispy":
        canvas.add_legend()
    read_canvas(canvas.write_json())

@pytest.mark.parametrize("orient", ["v", "h"])
def test_agg(backend: str, orient: str):
    canvas = new_canvas(backend=backend)
    df = {
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
        "c": ["P", "Q"] * 15,
    }
    if orient == "v":
        cat_plt = canvas.cat_x(df, "label", "y")
    else:
        cat_plt = canvas.cat_y(df, "y", "label")
    cat_plt.mean().add_line(color="c", alpha=0.8)
    cat_plt.mean().add_markers(color="c", alpha=0.8)
    cat_plt.mean().add_bars(color="c", alpha=0.8).as_edge_only()
    cat_plt.std().add_line(color="c", width=2.5)
    cat_plt.sum().add_line()
    cat_plt.median().add_bars(color="c").update_width(1.5)
    cat_plt.max().add_markers()
    cat_plt.min().add_bars(color="c").update_hatch("/")
    cat_plt.first().add_line(color="c")
    if orient == "v":
        canvas.cat_x(df, x="label").count().add_line(color="c")
    else:
        canvas.cat_y(df, y="label").count().add_line(color="c")

    cat_plt.mean_for_each("c").add_stripplot()
    cat_plt.median_for_each("c").add_stripplot()
    cat_plt.std_for_each("c").add_stripplot()
    cat_plt.sum_for_each("c").add_stripplot()
    cat_plt.min_for_each("c").add_stripplot()
    cat_plt.max_for_each("c").add_stripplot()

def test_cat_legend(backend: str):
    if backend == "vispy":
        pytest.skip("vispy does not support legend")
    canvas = new_canvas(backend=backend)
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
    }

    _c = canvas.cat(df, "x", "y")
    _c.add_line(color="label").move(0.1, 0.1)
    _c.add_markers(color="label").move(0.1, 0.1)
    canvas.add_legend()

def test_catx_legend(backend: str):
    if backend == "vispy":
        pytest.skip("vispy does not support legend")
    canvas = new_canvas(backend=backend)
    df = {
        "x": ["P", "Q"] * 15,
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
    }
    _c = canvas.cat_x(df, "x", "y")
    _c.add_boxplot(color="label", alpha=0.8)
    _c.add_violinplot(color="label", alpha=0.8).with_rug()
    _c.add_pointplot(color="label", alpha=0.8).err_by_se()
    _c.add_barplot(color="label", alpha=0.8)
    canvas.add_legend()

def test_marker_legend():
    canvas = new_canvas("mock")
    df = {
        "x": ["P", "Q"] * 15,
        "y": np.arange(30),
        "z": np.sin(np.arange(30) / 10),
        "label": np.repeat(["A", "B", "C"], 10),
    }
    canvas.cat_x(df, "x", "y").add_stripplot(color="label")
    canvas.cat_x(df, "x", "y").add_stripplot().update_size("z")
    canvas.cat_x(df, "x", "y").add_stripplot().update_colormap("z")
    canvas.cat_x(df, "x", "y").add_stripplot(symbol="label")
    canvas.cat_x(df, "x", "y").add_stripplot(hatch="label")
    canvas.add_legend()

def test_single_value():
    canvas = new_canvas("mock")
    df = {
        "x": ["P", "Q"] * 15,
        "y": np.arange(30),
        "z": np.ones(30),
        "label": np.repeat(["A", "B", "C"], 10),
    }
    canvas.cat_x(df, "x", "y").add_stripplot(color="label")
    canvas.cat_x(df, "x", "y").add_stripplot().update_size("z")
    canvas.cat_x(df, "x", "y").add_stripplot().update_colormap("z")
    canvas.add_legend()

@pytest.mark.parametrize("orient", ["v", "h"])
def test_numeric_axis(backend: str, orient: str):
    canvas = new_canvas(backend=backend)
    df = {
        "y": np.arange(30),
        "label": np.repeat([2, 5, 6], 10),
        "c": ["P", "Q"] * 15,
    }
    if orient == "v":
        cat_plt = canvas.cat_x(df, "label", "y", numeric_axis=True)
    else:
        cat_plt = canvas.cat_y(df, "y", "label", numeric_axis=True)
    cat_plt.add_stripplot(color="c", dodge=True).move(0.1)
    cat_plt.add_swarmplot(color="c").move(0.1)
    cat_plt.add_boxplot(color="c").move(0.1).with_outliers(ratio=0.5)
    with filter_warning(backend, "plotly"):
        cat_plt.add_boxplot(color="c").as_edge_only()
    cat_plt.add_violinplot(color="c").move(0.1).with_rug()
    cat_plt.add_violinplot(color="c").with_outliers(ratio=0.5)
    cat_plt.add_violinplot(color="c").with_box()
    cat_plt.add_violinplot(color="c").as_edge_only().with_strip()
    cat_plt.add_violinplot(color="c").with_swarm()
    cat_plt.add_pointplot(color="c").move(0.1).err_by_se().err_by_sd().err_by_quantile().est_by_mean().est_by_median()
    cat_plt.add_barplot(color="c").move(0.1).err_by_se().err_by_sd().err_by_quantile().est_by_mean().est_by_median()
    with filter_warning(backend, "plotly"):
        cat_plt.add_rugplot(color="c").move(0.1).scale_by_density()

def test_stack(backend: str):
    canvas = new_canvas(backend=backend)
    df = {
        "y": np.arange(30),
        "label": np.repeat([2, 5, 6], 10),
        "c": ["P", "Q"] * 15,
    }
    cat_plt = canvas.cat_x(df, "label", "y", numeric_axis=True)
    cat_plt.stack("c").add_bars(color="c")
    area = cat_plt.stack("c").add_area(hatch="c")
    area.update_color("c")
    area.update_color("black")
    area.update_hatch("/")
    area.update_style("--")
    area.update_style("c")
    area.move(0.1, 0.1)
    with filter_warning(backend, "vispy"):
        canvas.add_legend()

def test_joint_cat(backend: str):
    joint = new_jointgrid(backend=backend, loc=(0, 0), size=(180, 180))
    df = {
        "x": np.arange(30),
        "y": np.arange(30),
        "c": np.repeat(["A", "B", "C"], 10),
    }
    joint.cat(df, "x", "y").add_hist2d()
    joint.cat(df, "x", "y").add_markers(color="c")

def test_pandas_and_polars():
    import pandas as pd
    import polars as pl

    canvas = new_canvas(backend="mock")
    _dict = {
        "y": np.arange(30),
        "label": np.repeat(["A", "B", "C"], 10),
        "c": ["P", "Q"] * 15,
    }
    df_pd = pd.DataFrame(_dict)
    df_pl = pl.DataFrame(_dict)

    cat_pd = canvas.cat_x(df_pd, "label", "y")
    cat_pl = canvas.cat_x(df_pl, "label", "y")
    cat_pd.add_swarmplot(color="c")
    cat_pd.mean().add_markers(color="c")
    cat_pd.first().add_markers(color="c")

    cat_pl.add_swarmplot(color="c")
    cat_pl.sort().add_stripplot(color="c")
    cat_pl.mean().add_markers(color="c")
    cat_pl.first().add_markers(color="c")
