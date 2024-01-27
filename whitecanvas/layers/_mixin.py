from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Iterator,
    Sequence,
    SupportsIndex,
    TypeVar,
    overload,
)
from weakref import WeakValueDictionary

import numpy as np
from numpy.typing import NDArray
from psygnal import Signal, SignalGroup

from whitecanvas.layers._base import DataBoundLayer, LayerEvents, PrimitiveLayer
from whitecanvas.protocols import layer_protocols as _lp
from whitecanvas.theme import get_theme
from whitecanvas.types import (
    ColorType,
    Hatch,
    LineStyle,
    XYTextData,
    _Void,
)
from whitecanvas.utils.normalize import arr_color, as_any_1d_array, as_color_array

if TYPE_CHECKING:
    from typing_extensions import Self

    from whitecanvas.layers.group._collections import LayerCollectionBase

_L = TypeVar("_L", bound=PrimitiveLayer)
_void = _Void()


class FaceEvents(SignalGroup):
    """Events of the face changes."""

    color = Signal(object)
    hatch = Signal(object)


class EdgeEvents(SignalGroup):
    """Events of the edge changes."""

    color = Signal(object)
    style = Signal(object)
    width = Signal(object)


class LayerNamespace(ABC, Generic[_L]):
    _properties = ()
    events: SignalGroup
    _events_class: type[SignalGroup]

    def __init__(self, layer: _L | None = None) -> None:
        self._layer = layer
        self._instances: WeakValueDictionary[int, _L] = WeakValueDictionary()
        self.events = self.__class__._events_class()
        self.__setattr__ = self._setattr

    def __repr__(self) -> str:
        attrs = ", ".join(f"{p}={getattr(self, p)!r}" for p in self._properties)
        return f"{type(self).__name__}(layer={self._layer!r}, {attrs})"

    def __repr_pretty__(self, p, cycle):
        attrs = ",\n    ".join(f"{p}={getattr(self, p)!r}" for p in self._properties)
        p.text(f"{type(self).__name__}(\n    layer={self._layer!r}, {attrs}\n)")

    def __get__(self, obj, owner) -> Self:
        if obj is None:
            return self
        _id = id(obj)
        if (ns := self._instances.get(_id, None)) is None:
            ns = self._instances[_id] = self.__class__(obj)
        return ns

    def _setattr(self, name: str, value: Any) -> None:
        raise AttributeError(f"Cannot set attribute {name!r} on {self!r}")


class FaceNamespace(LayerNamespace[PrimitiveLayer[_lp.HasFaces]]):
    _properties = ("color", "hatch")
    events: FaceEvents
    _events_class = FaceEvents

    @abstractproperty
    def color(self) -> NDArray[np.floating]:
        raise NotImplementedError

    @abstractproperty
    def hatch(self) -> Hatch | EnumArray[Hatch]:
        raise NotImplementedError

    @abstractproperty
    def alpha(self) -> float | NDArray[np.floating]:
        raise NotImplementedError

    @abstractmethod
    def update(self, color=None, hatch=None, alpha=1.0):
        ...


class EdgeNamespace(LayerNamespace[PrimitiveLayer[_lp.HasEdges]]):
    _properties = ("color", "style", "width")
    events: EdgeEvents
    _events_class = EdgeEvents

    @abstractproperty
    def color(self) -> NDArray[np.floating]:
        raise NotImplementedError

    @abstractproperty
    def width(self) -> float | NDArray[np.floating]:
        raise NotImplementedError

    @abstractproperty
    def style(self) -> LineStyle | EnumArray[LineStyle]:
        raise NotImplementedError

    @abstractproperty
    def alpha(self) -> float | NDArray[np.floating]:
        raise NotImplementedError

    @abstractmethod
    def update(self, color=None, width=None, style=None, alpha=1.0):
        ...


class MonoFace(FaceNamespace):
    @property
    def color(self) -> NDArray[np.floating]:
        return self._layer._backend._plt_get_face_color()

    @color.setter
    def color(self, value: ColorType):
        col = arr_color(value)
        self._layer._backend._plt_set_face_color(col)
        self.events.color.emit(col)

    @property
    def hatch(self) -> Hatch:
        return self._layer._backend._plt_get_face_hatch()

    @hatch.setter
    def hatch(self, value: str | Hatch):
        hatch = Hatch(value)
        self._layer._backend._plt_set_face_hatch(hatch)
        self.events.hatch.emit(hatch)

    @property
    def alpha(self) -> float:
        return float(self.color[3])

    @alpha.setter
    def alpha(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {value!r}")
        self.color = (*self.color[:3], value)

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        hatch: Hatch | str | _Void = _void,
        alpha: float | _Void = _void,
    ) -> _L:
        """
        Update the face properties.

        Parameters
        ----------
        color : ColorType, optional
            Color of the face.
        hatch : FacePattern, optional
            Fill hatch of the face.
        alpha : float, optional
            Alpha value of the face.
        """
        if color is not _void:
            self.color = color
        if hatch is not _void:
            self.hatch = hatch
        if alpha is not _void:
            self.alpha = alpha
        return self._layer


class ConstFace(FaceNamespace):
    @property
    def color(self) -> NDArray[np.floating]:
        return self._layer._backend._plt_get_face_color()[0]

    @color.setter
    def color(self, value: ColorType):
        col = arr_color(value)
        self._layer._backend._plt_set_face_color(col)
        self.events.color.emit(col)

    @property
    def hatch(self) -> Hatch:
        return self._layer._backend._plt_get_face_hatch()[0]

    @hatch.setter
    def hatch(self, value: str | Hatch):
        hatch = Hatch(value)
        self._layer._backend._plt_set_face_hatch(hatch)
        self.events.hatch.emit(hatch)

    @property
    def alpha(self) -> float:
        return float(self.color[3])

    @alpha.setter
    def alpha(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {value!r}")
        try:
            self.color = (*self.color[:3], value)
        except IndexError:
            pass  # when the layer is empty

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        hatch: Hatch | str | _Void = _void,
        alpha: float | _Void = _void,
    ) -> _L:
        """
        Update the face properties.

        Parameters
        ----------
        color : ColorType, optional
            Color of the face.
        hatch : FacePattern, optional
            Fill hatch of the face.
        alpha : float, optional
            Alpha value of the face.
        """
        if color is not _void:
            self.color = color
        if hatch is not _void:
            self.hatch = hatch
        if alpha is not _void:
            self.alpha = alpha
        return self._layer


class MonoEdge(EdgeNamespace):
    @property
    def color(self) -> NDArray[np.floating]:
        return self._layer._backend._plt_get_edge_color()

    @color.setter
    def color(self, value: ColorType):
        col = arr_color(value)
        self._layer._backend._plt_set_edge_color(col)
        self.events.color.emit(col)

    @property
    def style(self) -> LineStyle:
        return self._layer._backend._plt_get_edge_style()

    @style.setter
    def style(self, value: str | LineStyle):
        style = LineStyle(value)
        self._layer._backend._plt_set_edge_style(style)
        self.events.style.emit(style)

    @property
    def width(self) -> float:
        return self._layer._backend._plt_get_edge_width()

    @width.setter
    def width(self, value: float):
        if value < 0:
            raise ValueError(f"Edge width must be non-negative, got {value!r}")
        value = float(value)
        self._layer._backend._plt_set_edge_width(value)
        self.events.width.emit(value)

    @property
    def alpha(self) -> float:
        """The alpha value of the edge."""
        return float(self.color[3])

    @alpha.setter
    def alpha(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {value!r}")
        self.color = (*self.color[:3], value)

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        style: LineStyle | str | _Void = _void,
        width: float | _Void = _void,
        alpha: float | _Void = _void,
    ) -> _L:
        if color is not _void:
            self.color = color
        if style is not _void:
            self.style = style
        if width is not _void:
            self.width = width
        if alpha is not _void:
            self.alpha = alpha
        return self._layer


class ConstEdge(EdgeNamespace):
    _properties = ("color", "style", "width")

    @property
    def color(self) -> NDArray[np.floating]:
        return self._layer._backend._plt_get_edge_color()[0]

    @color.setter
    def color(self, value: ColorType):
        col = arr_color(value)
        self._layer._backend._plt_set_edge_color(col)
        self.events.color.emit(col)

    @property
    def style(self) -> LineStyle:
        """Edge style."""
        return self._layer._backend._plt_get_edge_style()[0]

    @style.setter
    def style(self, value: str | LineStyle):
        style = LineStyle(value)
        self._layer._backend._plt_set_edge_style(style)
        self.events.style.emit(style)

    @property
    def width(self) -> float:
        """Edge width."""
        return self._layer._backend._plt_get_edge_width()[0]

    @width.setter
    def width(self, value: float):
        if value < 0:
            raise ValueError(f"Edge width must be non-negative, got {value!r}")
        value = float(value)
        self._layer._backend._plt_set_edge_width(value)
        self.events.width.emit(value)

    @property
    def alpha(self) -> float:
        """The alpha value of the edge."""
        return float(self.color[3])

    @alpha.setter
    def alpha(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {value!r}")
        self.color = (*self.color[:3], value)

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        style: LineStyle | str | _Void = _void,
        width: float | _Void = _void,
        alpha: float | _Void = _void,
    ) -> _L:
        if color is not _void:
            self.color = color
        if style is not _void:
            self.style = style
        if width is not _void:
            self.width = width
        if alpha is not _void:
            self.alpha = alpha
        return self._layer


class MultiFace(FaceNamespace):
    @property
    def color(self) -> NDArray[np.floating]:
        """Face color of the bar."""
        return self._layer._backend._plt_get_face_color()

    @color.setter
    def color(self, color):
        col = as_color_array(color, self._layer.ndata)
        self._layer._backend._plt_set_face_color(col)
        self.events.color.emit(col)

    @property
    def hatch(self) -> EnumArray[Hatch]:
        """Face fill hatches."""
        return np.asarray(self._layer._backend._plt_get_face_hatch(), dtype=object)

    @hatch.setter
    def hatch(self, hatch: str | Hatch | Iterable[str | Hatch]):
        if isinstance(hatch, str):
            hatch = Hatch(hatch)
        elif isinstance(hatch, Hatch):
            pass
        else:
            hatch = [Hatch(p) for p in hatch]
        self._layer._backend._plt_set_face_hatch(hatch)
        self.events.hatch.emit(hatch)

    @property
    def alpha(self) -> float:
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, value):
        color = self.color.copy()
        color[:, 3] = value
        self.color = color

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        hatch: Hatch | str | _Void = _void,
        alpha: float | _Void = _void,
    ) -> _L:
        """
        Update the face properties.

        Parameters
        ----------
        color : ColorType, optional
            Color of the face.
        hatch : FacePattern, optional
            Fill hatch of the face.
        alpha : float, optional
            Alpha value of the face.
        """
        if color is not _void:
            self.color = color
        if hatch is not _void:
            self.hatch = hatch
        if alpha is not _void:
            self.alpha = alpha
        return self._layer


class MultiEdge(EdgeNamespace):
    @property
    def color(self) -> NDArray[np.floating]:
        """Edge color of the bar."""
        return self._layer._backend._plt_get_edge_color()

    @color.setter
    def color(self, color):
        col = as_color_array(color, self._layer.ndata)
        self._layer._backend._plt_set_edge_color(col)
        self.events.color.emit(col)

    @property
    def width(self) -> NDArray[np.floating]:
        """Edge widths."""
        return self._layer._backend._plt_get_edge_width()

    @width.setter
    def width(self, width: float | Iterable[float]):
        if not isinstance(width, (int, float, np.number)):
            width = np.asarray(width, dtype=np.float32)
            if width.shape != (self._layer.ndata,):
                raise ValueError(
                    "Width must be a scalar or an array of length "
                    f"{self._layer.ndata}, got {width.shape!r}"
                )
        else:
            width = float(width)
        self._layer._backend._plt_set_edge_width(width)
        self.events.width.emit(width)

    @property
    def style(self) -> EnumArray[LineStyle]:
        """Edge styles."""
        return np.asarray(self._layer._backend._plt_get_edge_style(), dtype=object)

    @style.setter
    def style(self, style: str | LineStyle | Iterable[str | LineStyle]):
        if isinstance(style, str):
            style = LineStyle(style)
        elif isinstance(style, LineStyle):
            pass
        else:
            style = [LineStyle(s) for s in style]
        self._layer._backend._plt_set_edge_style(style)
        self.events.style.emit(style)

    @property
    def alpha(self) -> float:
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, value):
        color = self.color.copy()
        color[:, 3] = value
        self.color = color

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        style: LineStyle | str | _Void = _void,
        width: float | _Void = _void,
        alpha: float | _Void = _void,
    ):
        if color is not _void:
            self.color = color
        if style is not _void:
            self.style = style
        if width is not _void:
            self.width = width
        if alpha is not _void:
            self.alpha = alpha
        return self._layer


_NFace = TypeVar("_NFace", bound=FaceNamespace)
_NEdge = TypeVar("_NEdge", bound=EdgeNamespace)


class FaceEdgeMixinEvents(LayerEvents):
    face = Signal(object)
    edge = Signal(object)


class AbstractFaceEdgeMixin(Generic[_NFace, _NEdge]):
    events: FaceEdgeMixinEvents
    _events_class = FaceEdgeMixinEvents

    def __init__(self, face: _NFace, edge: _NEdge):
        self._face_namespace = face
        self._edge_namespace = edge

    def _init_events(self):
        self._face_namespace.events.connect(self.events.face.emit)
        self._edge_namespace.events.connect(self.events.edge.emit)
        # _face_namespace may change! _make_sure_hatch_visible should connected to
        # the layer event.
        self.events.face.connect(self._make_sure_hatch_visible)

    @property
    def face(self) -> _NFace:
        """The face namespace."""
        return self._face_namespace

    @property
    def edge(self) -> _NEdge:
        """The edge namespace."""
        return self._edge_namespace

    def with_face(
        self,
        *,
        color: ColorType | _Void = _void,
        hatch: Hatch | str = Hatch.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the face properties."""
        self.face.update(color=color, hatch=hatch, alpha=alpha)
        return self

    def with_edge(
        self,
        *,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the edge properties."""
        if color is None:
            color = get_theme().foreground_color
        self.edge.update(color=color, style=style, width=width, alpha=alpha)
        return self

    def _make_sure_hatch_visible(self):
        pass


class FaceEdgeMixin(AbstractFaceEdgeMixin[MonoFace, MonoEdge]):
    def __init__(self):
        super().__init__(MonoFace(self), MonoEdge(self))

    def _make_sure_hatch_visible(self):
        if self.edge.width == 0:
            self.edge.width = 1
            self.edge.color = get_theme().foreground_color


class MultiFaceEdgeMixin(AbstractFaceEdgeMixin[_NFace, _NEdge]):
    """Mixin for layers with multiple faces and edges."""

    def __init__(self):
        super().__init__(ConstFace(self), ConstEdge(self))

    def with_face(
        self,
        color: ColorType | _Void = _void,
        hatch: Hatch | str = Hatch.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the face properties."""
        if not isinstance(self._face_namespace, ConstFace):
            self._face_namespace.events.disconnect()
            self._face_namespace = ConstFace(self)  # type: ignore
            self._face_namespace.events.connect(self.events.face.emit)
        self.face.update(color=color, hatch=hatch, alpha=alpha)
        return self

    def with_edge(
        self,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the edge properties."""
        if color is None:
            color = get_theme().foreground_color
        if not isinstance(self._edge_namespace, ConstEdge):
            self._edge_namespace.events.disconnect()
            self._edge_namespace = ConstEdge(self)  # type: ignore
            self._edge_namespace.events.connect(self.events.edge.emit)
        self.edge.update(color=color, style=style, width=width, alpha=alpha)
        return self

    def with_face_multi(
        self,
        color: ColorType | Sequence[ColorType] | _Void = _void,
        hatch: str | Hatch | Sequence[str | Hatch] = Hatch.SOLID,
        alpha: float = 1,
    ) -> Self:
        if not isinstance(self._face_namespace, MultiFace):
            self._face_namespace.events.disconnect()
            self._face_namespace = MultiFace(self)  # type: ignore
            self._face_namespace.events.connect(self.events.face.emit)
        self.face.update(color=color, hatch=hatch, alpha=alpha)
        return self

    def with_edge_multi(
        self,
        color: ColorType | Sequence[ColorType] | None = None,
        width: float | Sequence[float] = 1,
        style: str | LineStyle | list[str | LineStyle] = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the edge properties."""
        if color is None:
            color = get_theme().foreground_color
        if not isinstance(self._edge_namespace, MultiEdge):
            self._edge_namespace.events.disconnect()
            self._edge_namespace = MultiEdge(self)  # type: ignore
            self._edge_namespace.events.connect(self.events.edge.emit)
        self.edge.update(color=color, style=style, width=width, alpha=alpha)
        return self

    def _make_sure_hatch_visible(self):
        _is_no_width = self.edge.width == 0
        if isinstance(self._edge_namespace, MultiEdge):
            if np.any(_is_no_width):
                ec = np.array(get_theme().foreground_color, dtype=np.float32)
                self.edge.width = np.where(_is_no_width, 1, self.edge.width)
                ec_old = self.edge.color
                ec_old[_is_no_width] = ec[np.newaxis]
                self.edge.color = ec_old
        else:
            if _is_no_width:
                self.edge.width = 1
                self.edge.color = get_theme().foreground_color


class CollectionFace(FaceNamespace):
    _layer: LayerCollectionBase

    def _iter_children(self) -> Iterator[FaceEdgeMixin]:
        yield from iter(self._layer)

    @property
    def color(self) -> NDArray[np.floating]:
        """Face color of the bar."""
        cols = [layer.face.color for layer in self._iter_children()]
        return np.stack(cols, axis=0)

    @color.setter
    def color(self, color):
        layers = list(self._iter_children())
        ndata = len(layers)
        col = as_color_array(color, ndata)
        for layer, c in zip(layers, col):
            layer.face.color = c
        self.events.color.emit(col)

    @property
    def hatch(self) -> EnumArray[Hatch]:
        """Face fill hatch."""
        hatches = [layer.face.hatch for layer in self._iter_children()]
        return np.array(hatches, dtype=object)

    @hatch.setter
    def hatch(self, hatch: str | Hatch | Iterable[str | Hatch]):
        layers = list(self._iter_children())
        ndata = len(layers)
        hatches = as_any_1d_array(hatch, ndata, dtype=object)
        for layer, ptn in zip(layers, hatches):
            layer.face.hatch = ptn
        self.events.hatch.emit(hatch)

    @property
    def alpha(self) -> float:
        """Alpha value of the face."""
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, value):
        color = self.color.copy()
        color[:, 3] = value
        self.color = color

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        hatch: Hatch | str | _Void = _void,
        alpha: float | _Void = _void,
    ) -> _L:
        """
        Update the face properties.

        Parameters
        ----------
        color : ColorType, optional
            Color of the face.
        hatch : FacePattern, optional
            Fill hatch of the face.
        alpha : float, optional
            Alpha value of the face.
        """
        if color is not _void:
            self.color = color
        if hatch is not _void:
            self.hatch = hatch
        if alpha is not _void:
            self.alpha = alpha
        return self._layer


class CollectionEdge(EdgeNamespace):
    _layer: LayerCollectionBase

    def _iter_children(self) -> Iterator[FaceEdgeMixin]:
        yield from iter(self._layer)

    @property
    def color(self) -> NDArray[np.floating]:
        """Face colors of the collection."""
        cols = [layer.edge.color for layer in self._iter_children()]
        return np.stack(cols, axis=0)

    @color.setter
    def color(self, color):
        layers = list(self._iter_children())
        ndata = len(layers)
        col = as_color_array(color, ndata)
        for layer, c in zip(layers, col):
            layer.edge.color = c
        self.events.color.emit(col)

    @property
    def width(self) -> NDArray[np.float32]:
        """Edge widths."""
        widths = [layer.edge.width for layer in self._iter_children()]
        return np.array(widths, dtype=np.float32)

    @width.setter
    def width(self, width: float | Iterable[float]):
        layers = list(self._iter_children())
        widths = as_any_1d_array(width, len(layers), dtype=np.float32)
        for layer, w in zip(layers, widths):
            layer.edge.width = w
        self.events.width.emit(width)

    @property
    def style(self) -> EnumArray[LineStyle]:
        """Edge styles."""
        styles = [layer.edge.style for layer in self._iter_children()]
        return np.array(styles, dtype=object)

    @style.setter
    def style(self, style: str | LineStyle | Iterable[str | LineStyle]):
        layers = list(self._iter_children())
        styles = as_any_1d_array(style, len(layers), dtype=object)
        for layer, ls in zip(layers, styles):
            layer.edge.style = ls
        self.events.style.emit(style)

    @property
    def alpha(self) -> float:
        return self.color[:, 3]

    @alpha.setter
    def alpha(self, value):
        color = self.color.copy()
        color[:, 3] = value
        self.color = color

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        style: LineStyle | str | _Void = _void,
        width: float | _Void = _void,
        alpha: float | _Void = _void,
    ):
        if color is not _void:
            self.color = color
        if style is not _void:
            self.style = style
        if width is not _void:
            self.width = width
        if alpha is not _void:
            self.alpha = alpha
        return self._layer


class CollectionFaceEdgeMixin(AbstractFaceEdgeMixin[CollectionFace, CollectionEdge]):
    def __init__(self):
        super().__init__(CollectionFace(self), CollectionEdge(self))

    def _make_sure_hatch_visible(self):
        _is_no_width = self.edge.width == 0
        if np.any(_is_no_width):
            ec = np.array(get_theme().foreground_color, dtype=np.float32)
            self.edge.width = np.where(_is_no_width, 1, self.edge.width)
            ec_old = self.edge.color
            ec_old[_is_no_width] = ec[np.newaxis]
            self.edge.color = ec_old


# just for typing
_E = TypeVar("_E", bound=Enum)


class EnumArray(Generic[_E]):
    @overload
    def __getitem__(self, key: SupportsIndex) -> _E:
        ...

    @overload
    def __getitem__(
        self, key: slice | NDArray[np.integer] | list[SupportsIndex]
    ) -> EnumArray[_E]:
        ...

    def __iter__(self) -> Iterator[_E]:
        ...


class FontEvents(SignalGroup):
    color = Signal(object)
    size = Signal(object)
    family = Signal(object)


class FontNamespace(LayerNamespace[PrimitiveLayer[_lp.HasText]]):
    _properties = ("color", "size", "family")
    events: FontEvents
    _events_class = FontEvents

    @abstractproperty
    def color(self):
        raise NotImplementedError

    @abstractproperty
    def size(self):
        raise NotImplementedError

    @abstractproperty
    def family(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, *, color=_void, size=_void, family=_void):
        raise NotImplementedError


class ConstFont(FontNamespace):
    @property
    def color(self):
        return self._layer._backend._plt_get_text_color()[0]

    @color.setter
    def color(self, value):
        if value is None:
            value = get_theme().foreground_color
        col = arr_color(value)
        self._layer._backend._plt_set_text_color(col)
        self.events.color.emit(col)

    @property
    def size(self):
        return self._layer._backend._plt_get_text_size()[0]

    @size.setter
    def size(self, value):
        if value is None:
            value = get_theme().font.size
        self._layer._backend._plt_set_text_size(value)
        self.events.size.emit(value)

    @property
    def family(self):
        return self._layer._backend._plt_get_text_fontfamily()[0]

    @family.setter
    def family(self, value):
        if value is None:
            value = get_theme().font.family
        if not isinstance(value, str):
            raise TypeError(f"fontfamily must be a string, got {type(value)}.")
        self._layer._backend._plt_set_text_fontfamily(value)
        self.events.family.emit(value)

    def update(
        self,
        *,
        color: ColorType | _Void = _void,
        size: float | _Void = _void,
        family: str | _Void = _void,
    ):
        if color is not _void:
            self.color = color
        if size is not _void:
            self.size = size
        if family is not _void:
            self.family = family
        return self._layer


class MultiFont(FontNamespace):
    @property
    def color(self):
        return self._layer._backend._plt_get_text_color()

    @color.setter
    def color(self, value):
        if value is None:
            value = get_theme().foreground_color
        col = as_color_array(value, self._layer.ntexts)
        self._layer._backend._plt_set_text_color(col)
        self.events.color.emit(col)

    @property
    def size(self):
        return self._layer._backend._plt_get_text_size()

    @size.setter
    def size(self, value):
        if value is None:
            value = get_theme().font.size
        sizes = as_any_1d_array(value, self._layer.ntexts, dtype=np.float32)
        self._layer._backend._plt_set_text_size(sizes)
        self.events.size.emit(sizes)

    @property
    def family(self):
        return self._layer._backend._plt_get_text_fontfamily()

    @family.setter
    def family(self, value):
        if value is None:
            value = get_theme().font.family
        family = as_any_1d_array(value, self._layer.ntexts, dtype=object)
        self._layer._backend._plt_set_text_fontfamily(family)
        self.events.family.emit(family)

    def update(
        self,
        *,
        color: ColorType | Sequence[ColorType] | _Void = _void,
        size: float | Sequence[float] | _Void = _void,
        family: str | Sequence[str] | _Void = _void,
    ):
        if color is not _void:
            self.color = color
        if size is not _void:
            self.size = size
        if family is not _void:
            self.family = family
        return self._layer


_NFont = TypeVar("_NFont", bound=FontNamespace)


class TextMixinEvents(LayerEvents):
    face = Signal(object)
    edge = Signal(object)
    font = Signal(object)


class TextMixin(
    DataBoundLayer[_lp.TextProtocol, XYTextData],
    Generic[_NFace, _NEdge, _NFont],
):
    face: _NFace
    edge: _NEdge
    _face_namespace: _NFace
    _edge_namespace: _NEdge
    _font_namespace: _NFont

    events: TextMixinEvents
    _events_class = TextMixinEvents

    def __init__(self, name: str | None = None):
        self._face_namespace = ConstFace(self)
        self._edge_namespace = ConstEdge(self)
        self._font_namespace = ConstFont(self)
        super().__init__(name=name)
        self._face_namespace.events.connect(self.events.face.emit)
        self._edge_namespace.events.connect(self.events.edge.emit)
        self._font_namespace.events.connect(self.events.font.emit)

    @property
    def face(self) -> _NFace:
        """Namespace of the text background face."""
        return self._face_namespace

    @property
    def edge(self) -> _NEdge:
        """Namespace of the text background edge."""
        return self._edge_namespace

    @property
    def font(self) -> _NFont:
        """Namespace of the text font."""
        return self._font_namespace

    def with_face(
        self,
        color: ColorType | _Void = _void,
        hatch: Hatch | str = Hatch.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the face properties."""
        if not isinstance(self._face_namespace, ConstFace):
            self._face_namespace.events.disconnect()
            self._face_namespace = ConstFace(self)  # type: ignore
            self._face_namespace.events.connect(self.events.face.emit)
        self.face.update(color=color, hatch=hatch, alpha=alpha)
        return self

    def with_edge(
        self,
        color: ColorType | None = None,
        width: float = 1.0,
        style: LineStyle | str = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the edge properties."""
        if color is None:
            color = get_theme().foreground_color
        if not isinstance(self._edge_namespace, ConstEdge):
            self._edge_namespace.events.disconnect()
            self._edge_namespace = ConstEdge(self)  # type: ignore
            self._edge_namespace.events.connect(self.events.edge.emit)
        self.edge.update(color=color, style=style, width=width, alpha=alpha)
        return self

    def with_face_multi(
        self,
        color: ColorType | Sequence[ColorType] | _Void = _void,
        hatch: str | Hatch | Sequence[str | Hatch] = Hatch.SOLID,
        alpha: float = 1,
    ) -> Self:
        if not isinstance(self._face_namespace, MultiFace):
            self._face_namespace.events.disconnect()
            self._face_namespace = MultiFace(self)  # type: ignore
            self._face_namespace.events.connect(self.events.face.emit)
        self.face.update(color=color, hatch=hatch, alpha=alpha)
        return self

    def with_edge_multi(
        self,
        color: ColorType | Sequence[ColorType] | None = None,
        width: float | Sequence[float] = 1,
        style: str | LineStyle | list[str | LineStyle] = LineStyle.SOLID,
        alpha: float = 1,
    ) -> Self:
        """Update the edge properties."""
        if color is None:
            color = get_theme().foreground_color
        if not isinstance(self._edge_namespace, MultiEdge):
            self._edge_namespace.events.disconnect()
            self._edge_namespace = MultiEdge(self)  # type: ignore
            self._edge_namespace.events.connect(self.events.edge.emit)
        self.edge.update(color=color, style=style, width=width, alpha=alpha)
        return self

    def with_font(
        self,
        color: ColorType | None = None,
        size: float | None = None,
        family: str | None = None,
    ) -> Self:
        """Update the face properties."""
        if color is None:
            color = self.font.color
        if not isinstance(self._font_namespace, ConstFace):
            self._font_namespace.events.disconnect()
            self._font_namespace = ConstFace(self)  # type: ignore
            self._font_namespace.events.connect(self.events.font.emit)
        self.font.update(color=color, size=size, family=family)
        return self

    def with_font_multi(
        self,
        color: ColorType | Sequence[ColorType] | None = None,
        size: float | Sequence[float] | None = None,
        family: str | Sequence[str] | None = None,
    ) -> Self:
        """Update the face properties."""
        if color is None:
            color = self.font.color
        if not isinstance(self._font_namespace, MultiFace):
            self._font_namespace.events.disconnect()
            self._font_namespace = MultiFace(self)  # type: ignore
            self._font_namespace.events.connect(self.events.font.emit)
        self.font.update(color=color, size=size, family=family)
        return self
