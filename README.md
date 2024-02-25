# whitecanvas

[![PyPI - Version](https://img.shields.io/pypi/v/whitecanvas.svg)](https://pypi.org/project/whitecanvas)
[![Python package index download statistics](https://img.shields.io/pypi/dm/whitecanvas.svg)](https://pypistats.org/packages/whitecanvas)
[![codecov](https://codecov.io/gh/hanjinliu/whitecanvas/graph/badge.svg?token=MYLNFOpEnA)](https://codecov.io/gh/hanjinliu/whitecanvas)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/whitecanvas.svg)](https://pypi.org/project/whitecanvas)

A type safe and backend independent plotting library for Python, aiming at not the simplest, but the tidiest API.

## Installation

```console
pip install whitecanvas -U
```

## Project Philosophy

#### Type safety

All the methods should be designed to have nice signature, and should return the same
type of object, so that your program can be statically checked by the IDE.

#### Backend independency

Every plotting library has their own strength and weakness. Same code should work on
different backends, so that you can choose the best one for different purposes.

Currently supported backends are `matplotlib`, `pyqtgraph`, `vispy`, `plotly` and
`bokeh`. If you want other backends, please feel free to
[open an issue](https://github.com/hanjinliu/whitecanvas/issues).

#### API tidiness

Most of (probably all of) the plotting libraries rely on the large
number of arguments to configure the plot elements. They are usually hard to remember,
forcing you to look up the documentation every time you want to make a plot.

`whitecanvas` tries to organize the methods, namespaces and arguments carefully so that you can make any kind of plot only with the help of the IDE's auto-completion and
suggestions.

## Documentation

Documentation is available [here](https://hanjinliu.github.io/whitecanvas/).

## Examples

[&rarr; Find more examples](https://github.com/hanjinliu/whitecanvas/blob/main/examples)

#### Rain-cloud plot in matplotlib

[&rarr; source](https://github.com/hanjinliu/whitecanvas/blob/main/examples/raincloud_plot.py)

![](https://github.com/hanjinliu/whitecanvas/blob/main/images/raincloud.png)

#### Super plot in matplotlib

[&rarr; source](https://github.com/hanjinliu/whitecanvas/blob/main/examples/superplot.py)

![](https://github.com/hanjinliu/whitecanvas/blob/main/images/superplot.png)

#### Joint plot in matplotlib

[&rarr; source](https://github.com/hanjinliu/whitecanvas/blob/main/examples/joint_grid.py)

![](https://github.com/hanjinliu/whitecanvas/blob/main/images/jointgrid.png)

#### Heatmap with text in pyqtgraph

[&rarr; source](https://github.com/hanjinliu/whitecanvas/blob/main/examples/heatmap_with_text.py)

![](https://github.com/hanjinliu/whitecanvas/blob/main/images/heatmap.png)

#### Curve fitting in bokeh

[&rarr; source](https://github.com/hanjinliu/whitecanvas/blob/main/examples/curve_fit.py)

![](https://github.com/hanjinliu/whitecanvas/blob/main/images/curve_fit.png)

-----
