# Canvas Grid

## Vertical/Horizontal Grid

``` python
#!name: canvas_grid_vertical
from whitecanvas import vgrid

canvas = vgrid(3, backend="matplotlib")

c0 = canvas.add_canvas(0)
c0.add_text(0, 0, "Canvas 0")
c1 = canvas.add_canvas(1)
c1.add_text(0, 0, "Canvas 1")
c2 = canvas.add_canvas(2)
c2.add_text(0, 0, "Canvas 2")
canvas.show()
```


``` python
#!name: canvas_grid_horizontal
from whitecanvas import hgrid

canvas = hgrid(3, backend="matplotlib")

c0 = canvas.add_canvas(0)
c0.add_text(0, 0, "Canvas 0")
c1 = canvas.add_canvas(1)
c1.add_text(0, 0, "Canvas 1")
c2 = canvas.add_canvas(2)
c2.add_text(0, 0, "Canvas 2")
canvas.show()
```

## 2D Grid

``` python
#!name: canvas_grid_2d
from whitecanvas import grid

canvas = grid(2, 2, backend="matplotlib")

for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    c = canvas.add_canvas(i, j)
    c.add_text(0, 0, f"Canvas ({i}, {j})")
canvas.show()
```

## Non-uniform Grid

The `*_nonuniform` functions allow you to create a grid with non-uniform sizes.
Instead of specifying the number of rows and columns, these functions take a list of size ratios.

``` python
#!name: canvas_grid_2d_nonuniform

from whitecanvas import grid_nonuniform

canvas = grid_nonuniform([1, 2], [2, 1], backend="matplotlib")

for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    c = canvas.add_canvas(i, j)
    c.add_text(0, 0, f"Canvas ({i}, {j})")
canvas.show()
```
