=================
Categorical Plots
=================

Existing Python plotting libraries such as :mod:`seaborn` nad :mod:`plotly`
have excellent support for high-level categorical plotting methods that use
DataFrame objects as input.

In :mod:`whitecanvas`, we provide similar functionality, but these methods
does not depend on any external plotting libraries or DataFrames.

The ``cat`` namespace
=====================

The ``cat`` namespace converts a tabular data into a categorical plotter.
Currently, following objects are allowed as input:

- :class:`dict` of array-like objects
- :class:`pandas.DataFrame`
- :class:`polars.DataFrame`

.. code-block:: python

    from whitecanvas import new_canvas

    canvas = new_canvas("matplotlib")

    # sample data
    df = {
        "label": ["A"] * 60 + ["B"] * 30 + ["C"] * 40,
        "value": np.random.normal(size=130),
    }

    canvas.cat(df, by="label").add_stripplot("value")
    canvas.show()

You can directly pass a categorized ``dict`` object. In this case, you should
not specify the column name parameters.

.. code-block:: python

    from whitecanvas import new_canvas

    canvas = new_canvas("matplotlib")

    # sample data
    df = {
        "A": np.random.normal(size=60),
        "B": np.random.normal(size=30),
        "C": np.random.normal(size=40),
    }

    canvas.cat(df).add_stripplot()
    canvas.show()

Several plotting methods are available in the ``cat`` namespace:

- :meth:`add_stripplot`
- :meth:`add_boxplot`
- :meth:`add_violinplot`
- :meth:`add_swarmplot`
- :meth:`add_countplot`
