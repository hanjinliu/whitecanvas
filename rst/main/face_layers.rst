=====================
Face&edge-type Layers
=====================

There are several layers that is composed of faces and edges.

- :class:`Markers` ... a layers composed of markers for scatter plots.
- :class:`Bars` ... a layer composed of bars.
- :class:`Band` ... a layer composed of a band region (fill-between region).
- :class:`Spans` ... a layer composed of infinitely long spans.

These layers have two namespaces: :attr:`face` and :attr:`edge`.
:attr:`face` has following properties:

- ``color`` ... color of the faces. Any color-like object is accepted.
- ``pattern`` ... pattern of the faces. Should be one of ``""``, ``"-"``, ``"|``, ``"+"``,
    ``"/"``, ``"\\"``, ``"x"`` or ``"."``.

.. note::

    ``pattern`` is not supported in some backends.

:attr:`edge` has following properties:

- ``color`` ... color of the lines. Any color-like object is accepted.
- ``width`` ... width of the lines. Should be a non-negative number.
- ``style`` ... style of the lines. Should be one of ``"-"``, ``":"``, ``"-."``, ``"--"``.

.. note::

    ``style`` is not supported in some backends.

Methods for adding these layers always configure the :attr:`face` properties with the
arguments. You can use the :meth:`with_edges` method of the output layer to set edge
properties. This separation is very helpful to prevent the confusion of the arguments,
especially the colors.

.. code-block:: python

    import numpy as np
    from whitecanvas import new_canvas

    canvas = new_canvas("matplotlib")

    layer = canvas.add_markers(np.arange(10), color="yellow").with_edges(color="black")

All the properties can be set via properties of :attr:`face` and :attr:`edge`, or the
:meth:`update` method.

.. code-block:: python

    layer.face.color = "yellow"
    layer.face.pattern = "x"

    layer.edge.color = "black"
    layer.edge.width = 2
    layer.edge.style = "--"

    # use `update`
    layer.face.update(color="yellow", pattern="x")
    layer.edge.update(color="black", width=2, style="--")
