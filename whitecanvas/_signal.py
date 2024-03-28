from __future__ import annotations

import inspect
from typing import Any, Callable, Generator, Generic, Sequence, TypeVar, overload

from psygnal import throttled

from whitecanvas.types import Modifier, MouseButton, MouseEvent, MouseEventType, Point

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


class MouseMoveSignal:
    def __init__(self):
        self._slots: list[_Slot] = []

    @overload
    def connect(self, slot: _G, *, msec: int = 0, leading: bool = True) -> _G: ...

    @overload
    def connect(
        self, slot: None = None, *, msec: int = 0, leading: bool = True
    ) -> Callable[[_G], _G]: ...

    def connect(self, slot=None, *, msec: int = 0, leading: bool = True):
        """Connect a mouse move callback."""

        def _inner(slot: _F) -> _F:
            if not callable(slot):
                raise TypeError(f"Can only connect callable object, got {slot!r}")
            self._slots.append(_Slot(slot, msec=msec, leading=leading))
            return slot

        return _inner(slot) if slot is not None else _inner

    def disconnect(
        self, slot: GeneratorFunction | None = None, missing_ok: bool = True
    ) -> None:
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

    def emulate_drag(
        self,
        positions: Sequence[tuple[float, float]],
        *,
        button: str | MouseButton = MouseButton.LEFT,
        modifiers: str | Modifier | Sequence[str | Modifier] = (),
    ):
        """Emulate a mouse move event."""
        if isinstance(modifiers, str):
            _modifiers = (Modifier(modifiers),)
        elif isinstance(modifiers, Modifier):
            _modifiers = MouseButton(button)
        else:
            _modifiers = tuple(Modifier(m) for m in modifiers)

        ev = MouseEvent(
            MouseButton(button),
            _modifiers,
            Point(*positions[0]),
            MouseEventType.CLICK,
        )
        self.emit(ev)

        for pos in positions[1:]:
            ev = MouseEvent(
                MouseButton(button),
                _modifiers,
                Point(*pos),
                MouseEventType.MOVE,
            )
            self.emit(ev)

        ev = MouseEvent(
            MouseButton(button),
            _modifiers,
            Point(*positions[-1]),
            MouseEventType.RELEASE,
        )
        self.emit(ev)


class MouseSignal(Generic[_R]):
    def __init__(self, typ: type[_R]):
        self._slots: list[Callable[[_R], Any | None]] = []
        self._typ = typ

    def connect(self, slot: _F | None = None) -> _F:
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
