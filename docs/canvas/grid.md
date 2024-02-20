# Canvas Grid

A "canvas grid" is a grid of canvases (which is called "figure" in `matplotlib`).
A grid is composed of multiple canvas objects, so that grid itself does not have
either layers or the `add_*` methods.

Once a grid is created, you can add chid canvases using the `add_canvas` method, or fill the grid with canvases using the `fill` method.
The signature of the method differs between 1D and 2D grid.

## Vertical/Horizontal Grid

This is an example of a grid with a single column, and the canvases are added one by one.

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

This is an example of a grid with a single row, and the canvases are added at once.

``` python
#!name: canvas_grid_horizontal
from whitecanvas import new_row

grid = new_row(3, backend="matplotlib").fill()

grid[0].add_text(0, 0, "Canvas 0")
grid[1].add_text(0, 0, "Canvas 1")
grid[2].add_text(0, 0, "Canvas 2")
grid.show()
```

## 2D Grid

The `new_grid` function creates a 2D grid. Since it is a 2D grid, you need to specify
two integers for the number of rows and columns.

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

If a sequence of integers are given, the grid will be non-uniform.

!!! note
    This feature is work in progress. Some backends does not support it yet.

``` python
#!name: canvas_grid_2d_nonuniform

from whitecanvas import new_grid

grid = new_grid([1, 2], [2, 1], backend="matplotlib")

for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    c = grid.add_canvas(i, j)
    c.add_text(0, 0, f"Canvas ({i}, {j})")
grid.show()
```
