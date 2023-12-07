==================
Customize X/Y axis
==================


.. code-block:: python

    from whitecanvas import new_canvas

    canvas = new_canvas(backend="matplotlib")

    canvas.x.lim = (0, 10)
