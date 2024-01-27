# Layer Groups

To plot complex layers, `whitecanvas` uses the layer-grouping strategy. There are
several built-in layer groups.

- `Plot` = `Line` + `Markers`
- `LineBand` = `Line` + `Band`
- `LabeledLine` = `Line` + `Errorbar` &times;2 + `Texts`
- `LabeledMarkers` = `Markers` + `Errorbar` &times;2 + `Texts`
- `LabeledBars` = `Bars` + `Errorbar` &times;2 + `Texts`
- `LabeledPlot` = `Plot` + `Errorbar` &times;2 + `Texts`
- `Stem` = `Markers` + `MultiLine`
- `Graph` = `Markers` + `MultiLine` + `Texts`

These layer groups can be derived from primitive layers. For example, in the following
code, markers are added to the line at the node positions, resulting in a `Plot` layer.

``` python
#!name: layer_groups_line_markers
from whitecanvas import new_canvas

canvas = new_canvas("matplotlib")

canvas.add_line(
    [0, 1, 2], [3, 2, 1], color="black",
).with_markers(
    symbol="o", color="red"
)
canvas.show()
```

The `Plot` layer can be further converted into a `LabeledPlot` layer by adding error
bars using `with_yerr` method.

``` python
#!name: layer_groups_line_markers_yerr
from whitecanvas import new_canvas

canvas = new_canvas("matplotlib")

canvas.add_line(
    [0, 1, 2], [3, 2, 1], color="black",
).with_markers(
    symbol="o", color="red"
).with_yerr(
    [0.1, 0.2, 0.3]
)
canvas.show()
```
