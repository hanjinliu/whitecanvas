# whitecanvas

[![PyPI - Version](https://img.shields.io/pypi/v/whitecanvas.svg)](https://pypi.org/project/whitecanvas)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/whitecanvas.svg)](https://pypi.org/project/whitecanvas)

A type safe and backend independent plotting library for Python, aiming at not the simplest, but the tidiest API.

&rarr; [Documentation](https://hanjinliu.github.io/whitecanvas/)

|**matplotlib**||**pyqtgraph**|
|:--------:|:-:|:-------:|
|![](https://github.com/hanjinliu/whitecanvas/blob/main/images/raincloud_matplotlib.png)|**Rain-cloud plot** in different backends ([Source code](https://github.com/hanjinliu/whitecanvas/blob/main/examples/raincloud_plot.py))|![](https://github.com/hanjinliu/whitecanvas/blob/main/images/raincloud_pyqtgraph.png)|
|**vispy**|**plotly**|**bokeh**|
![](https://github.com/hanjinliu/whitecanvas/blob/main/images/raincloud_vispy.png)|![](https://github.com/hanjinliu/whitecanvas/blob/main/images/raincloud_plotly.png)|![](https://github.com/hanjinliu/whitecanvas/blob/main/images/raincloud_bokeh.png)|

-----

## Installation

```console
pip install whitecanvas -U
```

## Type safety

In `whitecanvas`, each component is configured separately by `with_*` methods.
This architecture makes function arguments highly consistent and allows you to
write type-safe codes.

```python
import numpy as np
from whitecanvas import new_canvas

canvas = new_canvas()  # make a new canvas

# sample data
N = 10
xdata = np.linspace(0, np.pi * 2, N)
ydata = np.sin(xdata)
yerr = np.ones(N) / 3

# add layer
layer = (
    canvas
    .add_line(xdata, ydata, color="blue")
    .with_markers(color="violet", symbol="s")
    .with_edge(color="blue")
    .with_yerr(yerr, capsize=0.2, color="black")
)

canvas.show()  # show canvas
```

![](https://github.com/hanjinliu/whitecanvas/blob/main/images/sin_with_err_matplotlib.png)

## Backend independency

One of the ultimate goal of `whitecanvas` is "visualize data everywhere".
Currently supported backends are:

- `matplotlib`
- `pyqtgraph`
- `vispy`
- `plotly`
- `bokeh`

If you want other backends, please feel free to [open an issue](https://github.com/hanjinliu/whitecanvas/issues).
