from __future__ import annotations

from typing import Generic, TypeVar

from neoplot.protocols import BaseProtocol
from neoplot.backend import Backend

_P = TypeVar("_P", bound=BaseProtocol)

class Layer(Generic[_P]):
    _backend: _P
    _name: str

    @property
    def visible(self) -> bool:
        """Return true if the layer is visible"""
        return self._backend._plt_get_visible()
    
    @visible.setter
    def visible(self, visible: bool):
        """Set the visibility of the layer"""
        self._backend._plt_set_visible(visible)
    
    @property
    def name(self) -> str:
        """Name of this layer."""
        return self._name
    
    @name.setter
    def name(self, name: str):
        """Set the name of this layer."""
        self._name = str(name)
    
    @property
    def zorder(self) -> int:
        return self._backend._plt_get_zorder()
    
    @zorder.setter
    def zorder(self, zorder: int):
        self._backend._plt_set_zorder(zorder)
    
    
    def _create_backend(self, backend: Backend, *args) -> _P:
        return backend.get(type(self).__name__)(*args)

    def __repr__(self):
        return f"{self.__class__.__name__}<{self.name!r}>"
