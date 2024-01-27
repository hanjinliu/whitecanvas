# Categorical Axis

There are several plots that use categorical axis. Examples are:

- Strip plot
- Swarm plot
- Violin plot
- Box plot

Aside from the categorical axis, data points may further be grouped by other features,
such as the marker symbol and the marker size. Things are even more complicated when
the markers represent numerical values, such as their size being proportional to the
value, or colored by a colormap.

`whitecanvas` provides a consistent and simple interface to handle all these cases. In
this section, following data will be used as an example:

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

## Non-marker plots

Since plots without data point markers are more straightforward, we will start with
them. It includes `add_violinplot`, `add_boxplot`, `add_pointplot` and `add_barplot`.

``` python
#!name: categorical_axis_violin_0
canvas = new_canvas("matplotlib")
canvas.cat(df).add_violinplot("category", "observation")
canvas.show()
```

The first argument of `add_violinplot` is the column that defines the offset (shift
from 0 in the categorical axis). The second one is the column that is used for the
values.

Offset can be defined by multiple columns. You can pass a sequence of column names to
do that.

``` python
#!name: categorical_axis_violin_1
canvas = new_canvas("matplotlib")
canvas.cat(df).add_violinplot(["category", "replicate"], "observation")
canvas.show()
```

Violons can also be shown in different color. Specify the `color=` argument to do that.

``` python
#!name: categorical_axis_violin_2
canvas = new_canvas("matplotlib")
canvas.cat(df).add_violinplot("category", "observation", color="replicate")
canvas.show()
```

You can see that the violins overlaps. It is because only "category" is used for the
offsets. Offsets, colors and other properties are calculated **independently**.

To separate them, we need to add "replicate" to the offset.

``` python
#!name: categorical_axis_violin_3
canvas = new_canvas("matplotlib")
canvas.cat(df).add_violinplot(
    offset=["category", "replicate"],
    value="observation",
    color="replicate"
)
canvas.show()
```

`hatch=` can also be specified in a similar way. Again, All the properties are
independent.

``` python
#!name: categorical_axis_violin_4
canvas = new_canvas("matplotlib")
canvas.cat(df).add_violinplot(
    offset=["category", "replicate"],
    value="observation",
    color="replicate",
    hatch="category",
)
canvas
```

!!! note
    This is different from the `seaborn` interface, where `hue=` and `dodge=` are used
    to separate groups. As you can see in these examples, this is how `whitecanvas`
    can easily handle more complicated cases without confusion.

`add_boxplot` is very similar to `add_violinplot`.

``` python
#!name: categorical_axis_boxplot_0
canvas = new_canvas("matplotlib")
canvas.cat(df).add_boxplot(
    offset=["category", "replicate"],
    value="observation",
    color="replicate",
    hatch="category",
)
canvas
```
