# Categorical Plot

Existing Python plotting libraries such as `seaborn` and `plotly` have an excellent
support for high-level categorical plotting methods that use DataFrame objects as input.

In `whitecanvas`, similar functions are provided, but these methods do not depend on
any external plotting libraries or DataFrames, and are more flexible in some cases.

## The `cat` Method

The `cat` method converts a tabular data into a categorical plotter. Currently,
following objects are allowed as input:

- `dict` of array-like objects
- `pandas.DataFrame`
- `polars.DataFrame`
- `pyarrow.Table`

Following example shows how to make a strip plot.

``` python
#!name: categorical_add_stripplot
import numpy as np
from whitecanvas import new_canvas

canvas = new_canvas("matplotlib")
rng = np.random.default_rng(12345)

# sample data
df = {
    "label": ["A"] * 60 + ["B"] * 30 + ["C"] * 40,
    "value": rng.normal(size=130),
}

canvas.cat(df).add_stripplot("label", "value").with_edge(color="black")
canvas.show()
```
