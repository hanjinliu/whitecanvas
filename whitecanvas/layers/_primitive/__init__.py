from whitecanvas.layers._primitive.band import Band
from whitecanvas.layers._primitive.bars import Bars
from whitecanvas.layers._primitive.errorbars import Errorbars
from whitecanvas.layers._primitive.image import Image
from whitecanvas.layers._primitive.inf_curve import InfCurve, InfLine
from whitecanvas.layers._primitive.line import Line, LineStep, MultiLine
from whitecanvas.layers._primitive.markers import Markers
from whitecanvas.layers._primitive.rects import Rects
from whitecanvas.layers._primitive.rug import Rug
from whitecanvas.layers._primitive.spans import Spans
from whitecanvas.layers._primitive.text import Texts
from whitecanvas.layers._primitive.vectors import Vectors

__all__ = [
    "Line",
    "LineStep",
    "MultiLine",
    "Bars",
    "Markers",
    "Band",
    "Spans",
    "Errorbars",
    "Rug",
    "InfLine",
    "InfCurve",
    "Texts",
    "Image",
    "Rects",
    "Vectors",
]

# register layer type strings
from whitecanvas.layers._deserialize import register_layer_type

register_layer_type(Line)
register_layer_type(LineStep)
register_layer_type(MultiLine)
register_layer_type(Bars)
register_layer_type(Markers)
register_layer_type(Band)
register_layer_type(Spans)
register_layer_type(Errorbars)
register_layer_type(Rug)
register_layer_type(InfLine)
register_layer_type(InfCurve)
register_layer_type(Texts)
register_layer_type(Image)
register_layer_type(Rects)
register_layer_type(Vectors)

del register_layer_type
