# Categorical &times; Numerical Data

In this section, following data will be used as an example:

``` python
import numpy as np
from whitecanvas import new_canvas

rng = np.random.default_rng(12345)
df = {
    "category": ["A"] * 40 + ["B"] * 50,
    "observation": np.concatenate([rng.random(40), rng.random(50) + 1.3]),
    "replicate": [0] * 23 + [1] * 17 + [0] * 22 + [1] * 28,
    "temperature": rng.normal(scale=2.8, size=90) + 22.0,
}
```

How can we visualize the distributions for each category? There are several plots that
use categorical axis as either the x- or y-axis, and numerical axis as the other.
Examples are:

- Strip plot
- Swarm plot
- Violin plot
- Box plot

Aside from the categorical axis, data points may further be grouped by other features,
such as the marker symbol and the marker size. Things are even more complicated when
the markers represent numerical values, such as their size being proportional to the
value, or colored by a colormap.

`whitecanvas` provides a consistent and simple interface to handle all these cases.
Methods used for this purpose are `cat_x` and `cat_y`, where `cat_x` will deem the
x-axis as categorical, and `cat_y` will do the same for the y-axis.

``` python
#!skip
canvas = new_canvas("matplotlib")

# create the categorical plotter.
cat_plt_x = canvas.cat_x(df, x="category", y="observation")
cat_plt_y = canvas.cat_y(df, x="observation", y="category")
```

`cat_x` and `cat_y` use the argument `x=` and `y=` to specify the columns that are used
for the plot, where `x=` is the categorical axis for `cat_x` and `y=` for `cat_y`.

``` note
This is one of the important difference between `seaborn`. In `seaborn`, `orient` are
used to specify the orientation of the plots. This design forces the user to add the
argument `orient=` to every plot even though the orientation rarely changes during the
use of the same figure. In `whitecanvas`, you don't have to specify the orientation
once a categorical plotter is created by either `cat_x` or `cat_y`.
```

Multiplt columns can be used for the categorical axis, but only one column can be used
for the numerical axis.

``` python
#!skip
# OK
canvas.cat_x(df, x=["category", "replicate"], y="observation")
# OK
canvas.cat_y(df, x="observation", y=["category", "replicate"])
# NG
canvas.cat_x(df, x="category", y=["observation", "temperature"])
```

## Non-marker-type Plots

Since plots without data point markers are more straightforward, we will start with
them. It includes `add_violinplot`, `add_boxplot`, `add_pointplot` and `add_barplot`.

``` python
#!name: categorical_axis_violin_0
canvas = new_canvas("matplotlib")
canvas.cat_x(df, x="category", y="observation").add_violinplot()
canvas.show()
```

Violins can also be shown in different color. Specify the `color=` argument to do that.

``` python
#!name: categorical_axis_violin_1
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_violinplot(color="replicate")
)
canvas.show()
```

By default, groups with different colors do not overlap. This is controlled by the
`dodge=` argument. Set `dodge=False` to make them overlap (although it is not the way
we usually do).

``` python
#!name: categorical_axis_violin_2
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_violinplot(color="replicate", dodge=False)
)
canvas.show()
```

`hatch=` can also be specified in a similar way. It will change the hatch pattern of the
violins.

``` python
#!name: categorical_axis_violin_4
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_violinplot(hatch="replicate")
)
canvas.show()
```

`color` and `hatch` can overlap with each other or the `x=` or `y=` argument.

``` python
#!name: categorical_axis_violin_5
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_violinplot(color="category")
)
canvas.show()
```

The color palette of the canvas is used to paint categories. If you want to change it
after the layer is added, use `update_color_palette` method.

``` python
#!name: categorical_axis_violin_6
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_violinplot(color="replicate")
    .update_color_palette(["pink", "teal"])
)
canvas.show()
```

`add_boxplot`, `add_pointplot` and `add_barplot` is very similar to `add_violinplot`.

``` python
#!name: categorical_axis_many_plots
#!width: 700
from whitecanvas import new_row

canvas = new_row(3, size=(1600, 600), backend="matplotlib")

c0 = canvas.add_canvas(0)
c0.cat_x(df, x="category", y="observation").add_boxplot()
c0.title = "boxplot"

c1 = canvas.add_canvas(1)
c1.cat_x(df, x="category", y="observation").add_pointplot()
c1.title = "pointplot"

c2 = canvas.add_canvas(2)
c2.cat_x(df, x="category", y="observation").add_barplot()
c2.title = "barplot"

canvas.show()
```

## Marker-type Plots

Marker-type plots use a marker to represent each data point.

### Strip plot

``` python
#!name: categorical_axis_stripplot
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_stripplot(color="replicate")
)
canvas.show()
```

``` python
#!name: categorical_axis_stripplot_dodge
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_stripplot(color="replicate", dodge=True)
)
canvas.show()
```

As for the `Markers` layer, `as_edge_only` will convert the face features to the edge features.

``` python
#!name: categorical_axis_stripplot_dodge_edge_only
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_stripplot(color="replicate", dodge=True)
    .as_edge_only(width=2)
)
canvas.show()
```

`with_hover_template` is also defined. All the column names can be used in the template
format string.

``` python
#!skip
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_stripplot(color="replicate", dodge=True)
    .with_hover_template("{category} (rep={replicate})")
)
canvas.show()
```

Each marker size can represent a numerical value. `with_size` will map the numerical
values of a column to the size of the markers.

``` python
#!name: categorical_axis_stripplot_by_size
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_stripplot()
    .with_size("temperature")
)
canvas.show()
```

Similarly, each marker color can represent a numerical value. `update_colormap` will map
the value with an arbitrary colormap.

``` python
#!name: categorical_axis_stripplot_by_color
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_stripplot()
    .update_colormap("temperature", cmap="coolwarm")
)
canvas.show()
```

### Swarm plot

Swarm plot (or beeswarm plot) is another way to visualize all the data points with
markers. In swarm plot, the outline of the markers represents the distribution of the
data.

``` python
#!name: categorical_axis_swarmplot
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_swarmplot(sort=True)
    .update_colormap("temperature", cmap="coolwarm")
)
canvas.show()
```

### Rug plot

Although rug plot does not directly use markers, it also use a line to represent each
data point.

``` python
#!name: categorical_axis_rugplot
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_rugplot(color="replicate", dodge=True)
)
canvas.show()
```

Some methods defined for marker-type plots can also be used for rug plot.

``` python
#!name: categorical_axis_rugplot_colormap
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_rugplot()
    .update_colormap("temperature", cmap="coolwarm")
)
canvas.show()
```

`scale_by_density` will change the length of the rugs to represent the density of the
data points.

``` python
#!name: categorical_axis_rugplot_density
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_rugplot(color="replicate", dodge=True)
    .scale_by_density()
)
canvas.show()
```

## Overlaying Plots

Some types of plots are implemented with methods to efficiently overlay them with other
plots. All of them use method chaining so that the arguments can be auto-completed.

### Rug plot over violin plot

Violin plot can be overlaid with rug plot using `with_rug` method.

``` python
#!name: categorical_axis_violin_with_rug
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_violinplot(color="replicate")
    .with_rug(color="#purple")
)
canvas.show()
```

### Box plot over violin plot

Violin plot can be overlaid with box plot using `with_box` method. Color of the box plot
follows the convention of other plotting softwares by default.

``` python
#!name: categorical_axis_violin_with_box
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_violinplot(color="replicate")
    .with_box(width=2.0, extent=0.05)
)
canvas.show()
```

If the violins are edge only, the box plot will be filled with the same color.

``` python
#!name: categorical_axis_violin_with_box_edge_only
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_violinplot(color="replicate")
    .as_edge_only()
    .with_box(width=2.0, extent=0.05)
)
canvas.show()
```

### Markers over violin plot

Violin plot has `with_strip` and `with_swarm` methods to overlay markers.

``` python
#!name: categorical_axis_violin_with_strip
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_violinplot(color="replicate")
    .with_strip(symbol="D", size=8, color="black")
)
```

``` python
#!name: categorical_axis_violin_with_swarm
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_violinplot(color="replicate")
    .with_swarm(size=8, color="black")
)
```

### Add outliers to box plot

Box plot is usually combined with outlier markers. `with_outliers` method will add
outliers and optionally change the whisker lengths.

``` python
#!name: categorical_axis_box_with_outliers
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_boxplot(color="replicate")
    .with_outliers()
)
```

If the box plot is edge only, the outliers will be the same.

``` python
#!name: categorical_axis_box_with_outliers
canvas = new_canvas("matplotlib")
(
    canvas
    .cat_x(df, x="category", y="observation")
    .add_boxplot(color="replicate")
    .as_edge_only()
    .with_outliers()
)
```
