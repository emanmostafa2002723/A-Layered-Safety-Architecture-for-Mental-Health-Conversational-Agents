"""Bounded in-memory storage for content-free recent safety state."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
import time
from typing import Callable, TypeVar

from .response_policy.session_state import RecentSafetyState


T = TypeVar("T")


@dataclass
class _StateEntry:
    state: RecentSafetyState
    updated_monotonic: float


class InMemorySafetyStateStore:
    """Thread-safe development store that never contains message text.

    Production deployments with more than one API process must replace this
    with an encrypted, access-controlled shared store implementing the same
    transactional interface.
    """

    production_ready = False
    version = "in-memory-safety-state/1.0"

    def __init__(self, *, max_sessions: int = 10_000, ttl_seconds: float = 3600.0) -> None:
        if max_sessions < 1 or ttl_seconds <= 0:
            raise ValueError("max_sessions and ttl_seconds must be positive")
        self.max_sessions = int(max_sessions)
        self.ttl_seconds = float(ttl_seconds)
        self._entries: OrderedDict[str, _StateEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, session_id: str) -> RecentSafetyState:
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            entry = self._entries.get(session_id)
            if entry is None:
                return RecentSafetyState()
            self._entries.move_to_end(session_id)
            return entry.state

    def put(self, session_id: str, state: RecentSafetyState) -> None:
        if not isinstance(state, RecentSafetyState):
            raise TypeError("state must be a RecentSafetyState")
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            self._entries[session_id] = _StateEntry(state, now)
            self._entries.move_to_end(session_id)
            while len(self._entries) > self.max_sessions:
                self._entries.popitem(last=False)

    def transaction(
        self,
        session_id: str,
        operation: Callable[[RecentSafetyState], tuple[RecentSafetyState, T]],
    ) -> T:
        """Read/update one session atomically and return operation output."""
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            current = self._entries.get(session_id)
            state = current.state if current else RecentSafetyState()
            updated, result = operation(state)
            if not isinstance(updated, RecentSafetyState):
                raise TypeError("transaction must return a RecentSafetyState")
            self._entries[session_id] = _StateEntry(updated, now)
            self._entries.move_to_end(session_id)
            while len(self._entries) > self.max_sessions:
                self._entries.popitem(last=False)
            return result

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._entries.pop(session_id, None) is not None

    def _expire(self, now: float) -> None:
        stale = [
            key for key, value in self._entries.items()
            if now - value.updated_monotonic >= self.ttl_seconds
        ]
        for key in stale:
            self._entries.pop(key, None)

