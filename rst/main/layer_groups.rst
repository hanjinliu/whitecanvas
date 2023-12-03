==============
Layer Grouping
==============

To plot complex layers, :mod:`whitecanvas` uses the layer-grouping architecture.
There are several layer groups.

- :class:`Plot` ... :class:`Line` + :class:`Markers`
- :class:`LineBand` ... :class:`Line` + :class:`Band`
- :class:`LabeledLine` ... :class:`Line` + :class:`Errorbar` x2 + :class:`TextGroup`
- :class:`LabeledMarkers` ... :class:`Markers` + :class:`Errorbar` x2 + :class:`TextGroup`
- :class:`LabeledBars` ... :class:`Bars` + :class:`Errorbar` x2 + :class:`TextGroup`
- :class:`LabeledPlot` ... :class:`Plot` + :class:`Errorbar` x2 + :class:`TextGroup`
- :class:`Stem` ... :class:`Markers` + :class:`MultiLine`
- :class:`Graph` ... :class:`Markers` + :class:`MultiLine` + :class:`TextGroup`

Although

.. code-block:: python

    from whitecanvas import new_canvas

    canvas = new_canvas("matplotlib")

    canvas.add_line(
        [0, 1, 2], [3, 2, 1], color="black",
    ).with_markers(
        symbol="o", color="red"
    )

    canvas.add_markers(
        [0, 1, 2], [3, 2, 1], symbol="o", color="red"
    ).with_lines(
        color="black"
    )
