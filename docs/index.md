# whitecanvas

`whitecanvas` is a type safe and backend independent plotting library for Python.

## Source

[Jump to GitHub repository](https://github.com/hanjinliu/whitecanvas).

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

Most of (probably all of) the plotting libraries rely on the large number of arguments
to configure the plot elements. They are usually hard to remember, forcing you to look
up the documentation every time you want to make a plot.

`whitecanvas` tries to organize the methods, namespaces and arguments carefully so that you can make any kind of plot only with the help of the IDE's auto-completion and
suggestions.

## Installation

`whitecanvas` is available on PyPI.

``` bash
pip install whitecanvas -U
```

You can also install backend optional dependencies using one of the
following commands.

``` bash
pip install whitecanvas[matplotlib] -U
pip install whitecanvas[pyqtgraph] -U
pip install whitecanvas[vispy] -U
pip install whitecanvas[plotly] -U
pip install whitecanvas[bokeh] -U
```
