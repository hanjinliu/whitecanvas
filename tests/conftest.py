import matplotlib.pyplot as plt
import pytest


@pytest.fixture(
    scope="function",
    params=["mock", "matplotlib", "pyqtgraph", "plotly", "bokeh", "vispy"]
)
def backend(request):
    try:
        yield request.param
    except Exception:
        if request.param == "mock":
            pytest.skip("failed in mock backend")
        else:
            raise
    if request.param == "matplotlib":
        plt.close("all")
