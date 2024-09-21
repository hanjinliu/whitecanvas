from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Generic,
    TypeVar,
    overload,
)

from psygnal import throttled

from whitecanvas.types import MouseEvent

if TYPE_CHECKING:
    from typing_extensions import Self

_R = TypeVar("_R")
_S = TypeVar("_S")
_Y = TypeVar("_Y")

GeneratorFunction = Callable[..., Generator[_R, _S, _Y]]
_G = TypeVar("_G", bound=GeneratorFunction)
_F = TypeVar("_F", bound=Callable)


class _Slot(Generic[_G]):
    def __init__(self, slot: _G, msec: int = 0, leading: bool = True):
        self._slot_orig = slot
        if not inspect.isgeneratorfunction(slot):
            fn = _to_generator_function(slot)
        else:
            fn = slot
        if msec > 0:
            fn = throttled(fn, timeout=msec, leading=leading)
        self._slot = fn
        self._generator: Generator | None = None
        self._current_event: MouseEvent | None = None

    def next(self, ev: MouseEvent):
        """Advance the slot."""
        if self._generator is None:
            self._generator = self._slot(ev)
            self._current_event = ev
        try:
            self._current_event.update(ev)
            next(self._generator)
        except StopIteration:
            self._generator = None
            return


def _to_generator_function(f: Callable[[MouseEvent], Any | None]):
    def _f(ev: MouseEvent):
        yield f(ev)

    return _f


class _MouseSignalMixin:
    def __init__(self):
        self._slots: list[_Slot] = []
        self._name: str | None = None

    def __set_name__(self, owner: type[Any], name: str) -> None:
        if self._name is None:
            self._name = name

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> Self:
        if instance is None:
            return self
        out = self._copy()
        setattr(instance, self._name, out)
        return out

    def _copy(self):
        new = self.__class__()
        new._name = self._name
        return new


class MouseMoveSignal(_MouseSignalMixin):
    @overload
    def connect(self, slot: _G, *, msec: int = 0, leading: bool = True) -> _G: ...

    @overload
    def connect(
        self, slot: None = None, *, msec: int = 0, leading: bool = True
    ) -> Callable[[_G], _G]: ...

    def connect(self, slot=None, *, msec: int = 0, leading: bool = True):
        """
        Connect a mouse move callback.

        >>> @canvas.mouse.moved.connect
        >>> def on_mouse_move(ev: MouseEvent):
        ...     print(started)
        ...     yield
        ...     while ev.button == "left":
        ...         print("dragging")
        ...         yield
        ...     print("ended")
        """

        def _inner(slot: _F) -> _F:
            if not callable(slot):
                raise TypeError(f"Can only connect callable object, got {slot!r}")
            self._slots.append(_Slot(slot, msec=msec, leading=leading))
            return slot

        return _inner(slot) if slot is not None else _inner

    def disconnect(
        self, slot: GeneratorFunction | None = None, missing_ok: bool = True
    ) -> None:
        """Disconnect the slot (all by default)."""
        if slot is None:
            return self._slots.clear()
        i = -1
        for _i, _slot in enumerate(self._slots):
            if _slot._slot_orig is slot:
                i = _i
                break
        if i > 0:
            self._slots.pop(i)
        elif not missing_ok:
            raise ValueError(f"Slot {slot!r} not found")
        return

    def emit(self, ev: MouseEvent) -> None:
        """Emit the mouse event"""
        for slot in self._slots:
            slot.next(ev)


class MouseSignal(_MouseSignalMixin, Generic[_R]):
    def __init__(self, typ: type[_R]):
        super().__init__()
        self._typ = typ

    def connect(self, slot: _F | None = None) -> _F:
        """
        Connect a mouse clicked callback.

        >>> @canvas.mouse.clicked.connect
        >>> def on_mouse_clicked(ev: MouseEvent):
        ...     print(ev.pos)
        ...     print(ev.button)
        ...     print(ev.modifiers)
        ...     print(ev.type)
        """
        if not callable(slot):
            raise TypeError(f"Can only connect callable object, got {slot!r}")
        self._slots.append(slot)
        return slot

    def disconnect(self, slot: Callable | None = None, missing_ok: bool = True) -> None:
        if slot is None:
            return self._slots.clear()
        i = -1
        for _i, _slot in enumerate(self._slots):
            if _slot is slot:
                i = _i
                break
        if i > 0:
            self._slots.pop(i)
        elif not missing_ok:
            raise ValueError(f"Slot {slot!r} not found")
        return

    def emit(self, *args) -> None:
        for slot in self._slots:
            slot(*args)

    def _copy(self):
        new = self.__class__(self._typ)
        new._name = self._name
        return new
