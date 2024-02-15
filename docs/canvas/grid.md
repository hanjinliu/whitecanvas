# Canvas Grid

A "canvas grid" is a grid of canvases (which is called "figure" in `matplotlib`).
A grid is composed of multiple canvas objects, so that grid itself does not have
either layers or the `add_*` methods.

Once a grid is created, you can add chid canvases using the `add_canvas` method.
The signature of the method differs between 1D and 2D grid.

## Vertical/Horizontal Grid

``` python
#!name: canvas_grid_vertical
from whitecanvas import new_col

grid = new_col(3, backend="matplotlib")

c0 = grid.add_canvas(0)
c0.add_text(0, 0, "Canvas 0")
c1 = grid.add_canvas(1)
c1.add_text(0, 0, "Canvas 1")
c2 = grid.add_canvas(2)
c2.add_text(0, 0, "Canvas 2")
grid.show()
```


``` python
#!name: canvas_grid_horizontal
from whitecanvas import new_row

grid = new_row(3, backend="matplotlib")

c0 = grid.add_canvas(0)
c0.add_text(0, 0, "Canvas 0")
c1 = grid.add_canvas(1)
c1.add_text(0, 0, "Canvas 1")
c2 = grid.add_canvas(2)
c2.add_text(0, 0, "Canvas 2")
grid.show()
```

## 2D Grid

``` python
#!name: canvas_grid_2d
from whitecanvas import new_grid

grid = new_grid(2, 2, backend="matplotlib")

for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    c = grid.add_canvas(i, j)
    c.add_text(0, 0, f"Canvas ({i}, {j})")
grid.show()
```

## Non-uniform Grid

The `*_nonuniform` functions allow you to create a grid with non-uniform sizes.
Instead of specifying the number of rows and columns, these functions take a list of size ratios.

!!! note
    This feature is work in progress. Some backends does not support it yet.

``` python
#!name: canvas_grid_2d_nonuniform

from whitecanvas import grid_nonuniform

grid = grid_nonuniform([1, 2], [2, 1], backend="matplotlib")

for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    c = grid.add_canvas(i, j)
    c.add_text(0, 0, f"Canvas ({i}, {j})")
grid.show()
```
