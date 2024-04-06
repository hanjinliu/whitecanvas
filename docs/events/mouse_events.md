# Mouse Events

Interactive visualization is one of the most powerful aspect of Python.

## Click Events

The click events can be captured by `clicked`. Use `connect()` method to connect a
callback function to the event. The callback function must accept a single argument of
type `MouseEvent`.

``` python
from whitecanvas import new_canvas
from whitecanvas.types import MouseEvent

canvas = new_canvas("matplotlib:qt")

@canvas.mouse.clicked.connect
def _on_click(ev: MouseEvent):
    print("pos:", ev.pos)  # the (x, y) coordinate
    print("button", ev.button)  # mouse button enum
    print("modifiers", ev.modifiers)  # mouse modifier enums
```

Double-click events are very similar. They can be captured by `double_clicked`.

``` python
from whitecanvas import new_canvas
from whitecanvas.types import MouseEvent

canvas = new_canvas("matplotlib:qt")

@canvas.mouse.double_clicked.connect
def _on_click(ev: MouseEvent):
    print("pos:", ev.pos)  # the (x, y) coordinate
    print("button", ev.button)  # mouse button enum
    print("modifiers", ev.modifiers)  # mouse modifier enums
```

It is useful to filter the mouse button and modifier inside the callback function.

``` python
@canvas.mouse.clicked.connect
def _on_click(ev: MouseEvent):
    if ev.button == "left":
        print("left button clicked")
    if ev.modifiers == "ctrl":
        print("ctrl key is pressed")
```

## Move Events

Mouse move events need a different architecture. Unlike other events, the callback
function must be a generator function. every time the mouse moves, the generator
proceeds.

``` python
from whitecanvas import new_canvas
from whitecanvas.types import MouseEvent

canvas = new_canvas("matplotlib:qt")

@canvas.mouse.moved.connect
def _on_move(ev: MouseEvent):
    if ev.type != "press":
        return
    print("pressed")
    yield
    while ev.type == "move":
        print("moved to:", ev.pos)
        yield
    print("released")
```

## Emulating Mouse Events

Mouse events can be emulated by the `emulate_*` methods.

- [emulate_click()][whitecanvas.canvas._namespaces.MouseNamespace.emulate_click]
- [emulate_double_click()][whitecanvas.canvas._namespaces.MouseNamespace.emulate_double_click]
- [emulate_hover()][whitecanvas.canvas._namespaces.MouseNamespace.emulate_hover]
- [emulate_drag()][whitecanvas.canvas._namespaces.MouseNamespace.emulate_drag]

``` python
#!skip
from whitecanvas import new_canvas
from whitecanvas.types import MouseEvent

canvas = new_canvas("matplotlib:qt")
canvas.mouse.emulate_click((100, 100), button="left")
canvas.mouse.emulate_double_click((100, 100), button="left", modifiers="ctrl")
canvas.mouse.emulate_hover([(100, 100), (100, 101), (100, 102)])
canvas.mouse.emulate_drag([(100, 100), (100, 101), (100, 102)], button="left")
```
