import matplotlib.pyplot as plt
import pytest


@pytest.fixture(
    scope="function",
    params=["matplotlib", "pyqtgraph", "plotly", "bokeh", "vispy"]
)
def backend(request):
    yield request.param
    if request.param == "matplotlib":
        plt.close("all")
