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

The line selection tool can be created by `line_selector` function. Because selection
temporarily adds a new layer to the canvas, the selection tool requires a canvas object
as an argument. Once selection is done, a selector emits a `changed` signal.

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

## Other Selection Tools

- Rectangle selection ([rect_selector][whitecanvas.tools.rect_selector])
- X-span selection ([xspan_selector][whitecanvas.tools.xspan_selector])
- Y-span selection ([yspan_selector][whitecanvas.tools.yspan_selector])
- Lasso (free-hand) selection ([lasso_selector][whitecanvas.tools.lasso_selector])
