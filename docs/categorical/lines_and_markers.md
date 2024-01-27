# Categorical Lines and Markers

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
canvas.cat(df).add_line("x", "y", color="label")
canvas.show()
```

By setting `style=`, different line styles are used instead. In the following example,
`color="black"` means that all the lines should be the same color (black).

``` python
#!name: categorical_add_line_style
canvas = new_canvas("matplotlib")
canvas.cat(df).add_line("x", "y", color="black", style="label")
canvas.show()
```

In the case of markers, you can use symbols to distinguish groups.

``` python
#!name: categorical_add_markers_symbol
canvas = new_canvas("matplotlib")
canvas.cat(df).add_markers("x", "y", symbol="label")
canvas.show()
```
