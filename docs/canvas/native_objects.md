# Working with the Backend Objects

## Convert a backend object to `whitecanvas` canvas

The `wrap_canvas` converts a backend object to a `whitecanvas` canvas.

``` python
#!skip
from whitecanvas import wrap_canvas
import matplotlib.pyplot as plt

fig, axes = plt.subplots()
axes.set_title("Title")  # operations in the backend side

canvas = wrap_canvas(axes)  # axes --> canvas
canvas.add_line([0, 1, 2, 3, 4])

plt.show()  # backend methods still work
```

## Retrieve the backend object from a `whitecanvas` canvas

The backend object is available at the `native` property.

``` python
#!skip
from whitecanvas import new_canvas

canvas = new_canvas("matplotlib")
canvas.native
```

``` title="Output"
<Axes: >
```

## Combine `whitecanvas` with Applications

### 1. Control `matplotlib` Qt application with `whitecanvas`

``` python
#!skip
# use `%gui qt` in IPython
from whitecanvas import new_canvas
import matplotlib.pyplot as plt
from qtpy import QtWidgets as QtW

canvas = new_canvas("matplotlib:qt")
qt_canvas = canvas.native.get_figure().canvas

main = QtW.QMainWindow()
main.setWindowTitle("myapp")
main.setCentralWidget(qt_canvas)
main.show()
```

Since the `canvas` points to the same canvas as in the Qt application, you can control
the application with `whitecanvas` API.

``` python
#!skip
canvas.add_line([0, 1, 2, 3, 4])
```
