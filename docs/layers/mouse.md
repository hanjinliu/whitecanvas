# Mouse Interactivity

Layers in `whitecanvas` support mouse interactivity. Following table shows what feature
is supported in which backend.

| Feature  | `matplotlib` | `plotly` | `bokeh` | `pyqtgraph` | `vispy` |
|:--------:|:------------:|:--------:|:-------:|:-----------:|:-------:|
| Hovering | &check;      | &check;  | &check; | &check;     | &cross; |
| Picking  | &check;      | &check;  | &cross; | &check;     | &cross; |

## Hover Text

Hover text is very useful for interactive data exploration.

!!! note
    To demonstrate the hover texts in this document, we use `plotly` backend. This does
    not mean that hover text is not supported in `matplotlib`.

### Give a sequence of hover texts

`with_hover_text` method sets the hover text to the layer. Following example shows how
to set custom hover texts to markers.

``` python
#!html: add_markers_with_hover_text
import numpy as np
from whitecanvas import new_canvas

canvas = new_canvas("plotly", size=(400, 300))
x = np.arange(10)
y = np.sin(x)

layer = (
    canvas
    .add_markers(x, y)
    .with_hover_text([f"point {i}" for i in range(10)])
)
canvas.show()
```

### Give a hover template

If the hover texts are to be determined by the internal data, it's better to define a
template for the hover text. `with_hover_template` method sets the hover template to the
layer.

``` python
#!html: add_markers_with_hover_template
import numpy as np
from whitecanvas import new_canvas

canvas = new_canvas("plotly", size=(400, 300))
x = np.arange(10)
y = np.sin(x)

layer = (
    canvas
    .add_markers(x, y)
    .with_hover_template("x={x:.2f}, y={y:.2f}, i={i}")
)
canvas.show()
```

## Picking Data Points

Layers also have a `clicked` event. You can connect a callback to the event to handle
the picking. Use `layer.events.clicked.connect` syntax to do it.

!!! warning
    The callback function is defined on the Python side. This means that if the backend
    uses JavaScript like `plotly`, the callback cannot be executed. If you constructed
    a `plotly` canvas with Jupyter Notebook backend by `canvas("plotly:nb")`, however,
    callbacks are not disabled owing to the [FigureWidget](https://plotly.com/python/figurewidget/) of `plotly`.

``` python
#!skip
import numpy as np
from whitecanvas import new_canvas

canvas = new_canvas("matplotlib:qt")
layer = canvas.add_markers([0, 1, 2], [0, 0, 0])

@layer.events.clicked.connect
def _on_pick(picked):
    print(f"picked indices: {picked}")
```
