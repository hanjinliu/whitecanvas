# whitecanvas

[![PyPI - Version](https://img.shields.io/pypi/v/whitecanvas.svg)](https://pypi.org/project/whitecanvas)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/whitecanvas.svg)](https://pypi.org/project/whitecanvas)

A type safe and backend independent plotting library for Python.

-----

```python
import numpy as np
import whitecanvas as cnv

canvas = cnv.new_canvas()  # make a new canvas

# sample data
N = 10
xdata = np.linspace(0, np.pi * 2, N)
ydata = np.sin(xdata)

layer = canvas.add_line(  # add a line
    xdata,
    ydata,
    color="blue",
).with_markers(     # group with markers
    color="violet",
    symbol="s",
).with_edge(        # setup edges
    color="blue",
).with_yerr(        # group with errorbars
    np.ones(N) / 3,
    capsize=0.2,
    color="black",
)

canvas.show()  # show canvas
```

## Installation

```console
pip install whitecanvas -U
```

### Backend independency

Currently supported backends are:

- `matplotlib`
- `pyqtgraph`
