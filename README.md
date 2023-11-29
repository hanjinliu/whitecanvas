# whitecanvas

[![PyPI - Version](https://img.shields.io/pypi/v/whitecanvas.svg)](https://pypi.org/project/whitecanvas)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/whitecanvas.svg)](https://pypi.org/project/whitecanvas)

A type safe and backend independent plotting library for Python.

|matplotlib|pyqtgraph|vispy|plotly|bokeh|
|:--------:|:-------:|:---:|:----:|:---:|
|<img src="images/raincloud_matplotlib.png" alt="drawing" width="200"/>|<img src="images/raincloud_pyqtgraph.png" alt="drawing" width="200"/>|<img src="images/raincloud_vispy.png" alt="drawing" width="200"/>|<img src="images/raincloud_plotly.png" alt="drawing" width="200"/>|<img src="images/raincloud_bokeh.png" alt="drawing" width="200"/>|

-----

## Installation

```console
pip install whitecanvas -U
```

```python
import numpy as np
from whitecanvas as new_canvas

canvas = new_canvas()  # make a new canvas

# sample data
N = 10
xdata = np.linspace(0, np.pi * 2, N)
ydata = np.sin(xdata)

layer = (
    canvas
    .add_line(xdata, ydata, color="blue")
    .with_markers(color="violet", symbol="s")
    .with_edge(color="blue")
    .with_yerr(np.ones(N) / 3, capsize=0.2, color="black")
)

canvas.show()  # show canvas
```

### Backend independency

Currently supported backends are:

- `matplotlib`
- `pyqtgraph`
- `vispy`
- `plotly`
- `bokeh`
