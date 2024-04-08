# Selection Tools

Interactive selection in a canvas is an useful feature for many applications. Although
[mouse events](../events/mouse_events.md) provide a complete functionality for manual
selection, implementing a selection tool can be a tedious task.

`whitecanvas` has a built-in, ready-to-use selection tool that can be easily added to
your canvas. Currently, the selection tool is only available for following backends.

| `matplotlib` | `plotly` | `bokeh` | `pyqtgraph` | `vispy` |
|:------------:|:--------:|:-------:|:-----------:|:-------:|
| &check;      | &cross;  | &cross; | &check;     | &check; |

## Line Selection Tool

Line selection tools can be created by [`line_selector`][whitecanvas.tools.line_selector]
function. Because selection temporarily adds a new layer to the canvas, the selection
tool requires a canvas object as an argument. Once selection is done, a selector emits a
`changed` signal.

``` python hl_lines="6"
#!name: selection_tools_line.py
from whitecanvas import new_canvas
from whitecanvas.tools import line_selector

canvas = new_canvas("matplotlib")

selector = line_selector(canvas)  # make a selector

# connect a callback function
@selector.changed.connect
def _on_selection(sel):
    print(sel)

canvas.mouse.emulate_drag([(0.2, 0.2), (0.8, 0.6)], button="left")
canvas.show()
```

``` title="Output"
LineSelection(start=Point(x=0.2, y=0.2), end=Point(x=0.8, y=0.6))
```

The emitted object `LineSelection` is a named tuple of start and end `Point`s, which is
again a named tuple of x and y coordinates.

The `changed` signals are emitted when the selection is done. If you want to make the
signal emitted during the mouse drag,

``` python
#!skip
selector = line_selector(canvas, tracking=True)
```

### Styling Selection

As all the selection is a layer, you can style the selection.

``` python hl_lines="7-9"
#!name: selection_tools_line_styled.py
from whitecanvas import new_canvas
from whitecanvas.tools import line_selector

canvas = new_canvas("matplotlib")
selector = line_selector(canvas)  # make a selector

selector.color = "red"
selector.width = 3
selector.style = "--"

canvas.mouse.emulate_drag([(0.2, 0.2), (0.8, 0.6)], button="left")
canvas.show()
```

## Rectangle Selection Tool

Rectangle selector([`rect_selector`][whitecanvas.tools.rect_selector]) is similar to
the line selector, but it emits a `Rect` object.

``` python hl_lines="6"
#!name: selection_tools_rect.py
from whitecanvas import new_canvas
from whitecanvas.tools import rect_selector

canvas = new_canvas("matplotlib")

selector = rect_selector(canvas)  # make a selector

# connect a callback function
@selector.changed.connect
def _on_selection(sel):
    print(sel)

canvas.mouse.emulate_drag([(0.2, 0.2), (0.8, 0.6)], button="left")
canvas.show()
```

``` title="Output"
Rect(left=0.2, right=0.8, bottom=0.2, top=0.6)
```

## Other Selection Tools

- X-span selection ([xspan_selector][whitecanvas.tools.xspan_selector])
- Y-span selection ([yspan_selector][whitecanvas.tools.yspan_selector])
- Lasso (free-hand) selection ([lasso_selector][whitecanvas.tools.lasso_selector])
- Polygon selection ([polygon_selector][whitecanvas.tools.polygon_selector])

## Check If Selection Contains Points

Mouse selection is usually used to select data points. To do this, you have to connect a
callback function that checks if points are inside the selection area.

Following example demonstrates how to highlight selected points by a rectangle
selection.

``` python hl_lines="14"
#!name: selection_contains.py
import numpy as np
from whitecanvas import new_canvas
from whitecanvas.tools import rect_selector

rng = np.random.default_rng(1234)
canvas = new_canvas("matplotlib")
layer = canvas.add_markers(rng.random(100), rng.random(100))

selector = rect_selector(canvas)

@selector.changed.connect
def _on_selection():
    # get the indices of points inside the selection area
    indices = selector.contains_points(layer.data)
    # highlight selected points
    layer.with_face_multi(alpha=np.where(indices, 1, 0.2))

canvas.mouse.emulate_drag([(0.2, 0.2), (0.8, 0.6)])
canvas.show()
```

Same methods are defined for other selection tools with area.
