================
Line-type Layers
================

There are several layers that is composed of only lines.

- :class:`Line` ... a simple line
- :class:`InfLine` ... a straight line that extends to infinity
- :class:`InfCurve` ... a curve that extends to infinity
- :class:`Errorbar` ... lines representing error bars
- :class:`Rug` ... lines representing rug plots

These layers have following properties in common.

- ``color`` ... color of the lines. Any color-like object is accepted.
- ``width`` ... width of the lines. Should be a non-negative number.
- ``style`` ... style of the lines. Should be one of ``"-"``, ``":"``, ``"-."``, ``"--"``.

.. note::

    ``style`` is not supported in some backends.

These properties can be configured in function calls, via properties or the :meth:`update`
method.

.. code-block:: python

    import numpy as np
    from whitecanvas import new_canvas

    canvas = new_canvas("matplotlib")

    # function call
    layer = canvas.add_line(np.arange(10), color="black", width=2, style=":")

    # properties
    layer.color = "#FF36D9"
    layer.width = 2.5
    layer.style = "-"

    # update method
    layer.update(color=[0.0, 1.0, 0.0, 1.0], width=1, style="--")
