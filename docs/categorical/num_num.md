# Numerical &times; Numerical Data

## Categorical Lines and Markers

Line plot and scatter plot use numerical values for both x and y axes. In this case,
the plot is categorized by such as color, marker symbol, etc.

``` python
from whitecanvas import new_canvas

# sample data
df = {
    "label": ["A"] * 5 + ["B"] * 5,
    "x": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
    "y": [3, 1, 2, 4, 3, 5, 3, 3, 1, 2],
}
```

By setting `color=` to one of the column name, lines are split by the column and
different colors are used for each group.

``` python
#!name: categorical_add_line_color
canvas = new_canvas("matplotlib")
canvas.cat(df, "x", "y").add_line(color="label")
canvas.show()
```

By setting `style=`, different line styles are used instead. In the following example,
`color="black"` means that all the lines should be the same color (black).

``` python
#!name: categorical_add_line_style
canvas = new_canvas("matplotlib")
canvas.cat(df, "x", "y").add_line(color="black", style="label")
canvas.show()
```

In the case of markers, you can use symbols to distinguish groups.

``` python
#!name: categorical_add_markers_symbol
canvas = new_canvas("matplotlib")
canvas.cat(df, "x", "y").add_markers(symbol="label")
canvas.show()
```

## Distribution of Numerical Data

There are several ways to visualize the distribution of numerical data.

- Histogram
- Kernel Density Estimation (KDE)

These representations only use one array of numerical data. Therefore, either `x` or `y` should be empty in the `cat` method.

``` python
import numpy as np

rng = np.random.default_rng(12345)

# sample data
df = {
    "label": ["A"] * 60 + ["B"] * 30 + ["C"] * 40,
    "value": rng.normal(size=130),
}
```

`x="value"` means that the x-axis being "value" and the y-axis being the count.
Arguments forwards to the `histogram` method of `numpy`.

``` python
#!name: cat_hist_x
canvas = new_canvas("matplotlib")
canvas.cat(df, x="value").add_hist(bins=10)
canvas.show()
```

To transpose the histogram, use `y="value"`.

``` python
#!name: cat_hist_y
canvas = new_canvas("matplotlib")
canvas.cat(df, y="value").add_hist(bins=10)
canvas.show()
```

If both `x` and `y` are set, the plotter cannot determine which axis to use. To tell
the plotter which axis to use, call `along_x()` or `along_y()` to restrict the
dimension.

``` python
canvas = new_canvas("matplotlib")
# canvas.cat(df, x="label", y="value").add_hist(bins=10)  # This will raise an error
canvas.cat(df, x="label", y="value").along_x().add_hist(bins=10)
canvas.show()
```
