# Categorical &times; Numerical Data

In this section, following data will be used as an example:

``` python
import numpy as np
from whitecanvas import new_canvas

rng = np.random.default_rng(12345)
df = {
    "category": ["A"] * 40 + ["B"] * 50,
    "observation": np.concatenate([rng.random(40), rng.random(50) + 1.3]),
    "replicate": [0] * 20 + [1] * 20 + [0] * 25 + [1] * 25,
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

`add_boxplot`, `add_pointplot` and `add_barplot` is very similar to `add_violinplot`.

``` python
#!name: categorical_axis_many_plots
#!width: 700
from whitecanvas import hgrid

canvas = hgrid(ncols=3, size=(1600, 600), backend="matplotlib")

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

TODO

## Aggregation
