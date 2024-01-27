# Face&Edge-type Layers

There are several layers that is composed of faces and edges.

- `Markers` ... a layers composed of markers for scatter plots.
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

``` python
#!name: face_layers_with_edge
import numpy as np
from whitecanvas import new_canvas

canvas = new_canvas("matplotlib")

layer = canvas.add_markers(np.arange(10), color="yellow").with_edge(color="black")
canvas.show()
```

All the properties can be set via properties of `face` and `edge`, or the `update`
method.

``` python
layer.face.color = "yellow"
layer.face.hatch = "x"

layer.edge.color = "black"
layer.edge.width = 2
layer.edge.style = "--"

# use `update`
layer.face.update(color="yellow", hatch="x")
layer.edge.update(color="black", width=2, style="--")
```

## Multi-faces and Multi-edges

`Markers` and `Bars` supports multi-faces and multi-edges. This means that you can
create a layer with multiple colors, widths, etc.

To do this, you have to call `with_face_multi` or `with_edge_multi` method.
Here's an example of `Markers` with multi-faces.

``` python
#!name: face_layers_multifaces
import numpy as np
from whitecanvas import new_canvas

canvas = new_canvas("matplotlib")

layer = canvas.add_markers(
    np.arange(10),
).with_face_multi(
    color=np.random.random((10, 3)),  # random colors
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
