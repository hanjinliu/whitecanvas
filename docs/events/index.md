# Event Handling

Listening to the changes in the canvas or layer states is a common task in interactive
plotting. Although different backend plotting libraries implement their own event
handling systems, `whitecanvas` provides a unified system using the
[psygnal](https://psygnal.readthedocs.io) library.

The common syntax is to use `connect` function to connect callback function to the
event handler.

``` python
from whitecanvas import new_canvas

canvas = new_canvas()

# connect callback function
@canvas.x.events.lim.connect
def _xlim_changed(lim):
    print(f"canvas.x.lim changed to {lim}")
```

- [Canvas Events](canvas_events.md)
- [Layer Events](layer_events.md)
- [Mouse Events](mouse_events.md)
