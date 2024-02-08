# Face&Edge-type Layers

There are several layers that is composed of faces and edges.

- `Bars` ... a layer composed of bars.
- `Band` ... a layer composed of a band region (fill-between region).
- `Spans` ... a layer composed of infinitely long spans.

These layers have two namespaces: `face` and `edge`. `face` has following properties:

- `color` ... color of the faces. Any color-like object is accepted.
- `hatch` ... hatch pattern of the faces. Should be one of `""`, `"-"`, `"|`, `"+"`,
  `"/"`, `"\\"`, `"x"` or `"."`.

!!! note
    `hatch` is not supported in some backends.

`edge` has following properties:

- `color` ... color of the lines. Any color-like object is accepted.
- `width` ... width of the lines. Should be a non-negative number.
- `style` ... style of the lines. Should be one of `"-"`, `":"`, `"-."`, `"--"`.

!!! note
    `style` is not supported in some backends.

Methods for adding these layers always configure the `face` properties with the
arguments. You can use the `with_edge` method of the output layer to set edge
properties. This separation is very helpful to prevent the confusion of the arguments,
especially the colors.

Following example uses [`add_bars`][whitecanvas.canvas.CanvasBase.add_bars] and
[`add_spans`][whitecanvas.canvas.CanvasBase.add_spans] methods to create `Bars` and
`Spans` layers.

``` python
#!name: face_layers_with_edge
import numpy as np
from whitecanvas import new_canvas

canvas = new_canvas("matplotlib")

bars_layer = (
    canvas
    .add_bars([0, 1, 2, 3], [3, 4, 1, 2], color="yellow")
    .with_edge(color="black")
)

spans_layer = (
    canvas
    .add_spans([[0.2, 0.8], [1.4, 2.1], [1.8, 3.0]], color="blue")
    .with_edge(color="black")
)
canvas.y.lim = (0, 5)
canvas.show()
```

All the properties can be set via properties of `face` and `edge`, or the `update`
method.

``` python
#!skip
bars_layer.face.color = "yellow"
bars_layer.face.hatch = "x"

spans_layer.edge.color = "black"
spans_layer.edge.width = 2
spans_layer.edge.style = "--"

# use `update`
bars_layer.face.update(color="yellow", hatch="x")
spans_layer.edge.update(color="black", width=2, style="--")
```

## Multi-face and Multi-edge

As for [`Markers`](markers.md), `Bars` and `Spans` supports multi-face and multi-edge.

``` python
#!name: face_layers_multifaces
import numpy as np
from whitecanvas import new_canvas

canvas = new_canvas("matplotlib")

layer = (
    canvas
    .add_bars([0, 1, 2, 3], [3, 4, 1, 2])
    .with_face_multi(color=["red", "#00FF00", "rgb(0, 0, 255)", "black"])
)
canvas.show()
```

After calling `with_face_multi`, the layer `face` property will return arrays instead
of scalar values.

``` python
#!skip
layer.face.color  # (N, 4) array of RGBA colors
layer.face.hatch  # (N,) array of hatchs
layer.face.alpha # (N,) array of alpha values
```
