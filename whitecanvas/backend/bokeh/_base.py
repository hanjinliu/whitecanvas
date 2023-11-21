import bokeh.models as bk_models
from whitecanvas.protocols import BaseProtocol
from whitecanvas.types import LineStyle


class BokehLayer(BaseProtocol):
    _model: bk_models.Model
    _data: bk_models.ColumnDataSource


def to_bokeh_line_style(style: LineStyle) -> str:
    if style is LineStyle.SOLID:
        return "solid"
    elif style is LineStyle.DASH:
        return "dashed"
    elif style is LineStyle.DOT:
        return "dotted"
    elif style is LineStyle.DASH_DOT:
        return "dashdot"


def from_bokeh_line_style(style: str) -> LineStyle:
    if style == "solid":
        return LineStyle.SOLID
    elif style == "dashed":
        return LineStyle.DASH
    elif style == "dotted":
        return LineStyle.DOT
    elif style in ("dashdot", "dotdash"):
        return LineStyle.DASH_DOT
    else:
        return LineStyle.SOLID
