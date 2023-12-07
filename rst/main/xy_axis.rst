==================
Customize X/Y axis
==================

Properties related to the X/Y axis can be customized using the :attr:`x`` and :attr:`y`
namespaces.

.. toctree::
    :maxdepth: 1

Limits
======

.. code-block:: python

    from whitecanvas import new_canvas

    canvas = new_canvas(backend="matplotlib")

    canvas.x.lim = (0, 10)
    canvas.x.color = "red"
    canvas.x.flipped = True
    canvas.x.set_gridlines(color="gray", width=1, style=":")

Labels
======

You can set x/y labels using the :attr:`label` property.

.. code-block:: python

    from whitecanvas import new_canvas

    canvas = new_canvas(backend="matplotlib")

    canvas.x.label = "X axis"
    canvas.y.label = "Y axis"

The :attr:`label` property is actually another namespace. You can specify the text,
font size, etc. separately.

.. code-block:: python

    canvas.x.label.text = "X axis"
    canvas.x.label.size = 20
    canvas.x.label.family = "Arial"
    canvas.x.label.color = "red"

Ticks
=====

# TODO
