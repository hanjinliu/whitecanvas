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
from whitecanvas import new_canvas
from whitecanvas.tools import line_selector

canvas = new_canvas("matplotlib:qt")

selector = line_selector(canvas)
@selector.changed.connect
def _on_selection(sel):
    print(sel)
```
