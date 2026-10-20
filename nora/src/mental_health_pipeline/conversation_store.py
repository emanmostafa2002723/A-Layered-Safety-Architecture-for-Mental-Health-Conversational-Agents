"""Bounded, ephemeral chat context, separate from content-free safety state."""
from __future__ import annotations

from collections import OrderedDict
import threading
import time


class InMemoryConversationStore:
    """Keep only recent user turns and the responses actually shown to them.

    No disk persistence or logging. Production instances need an authenticated
    session boundary and a reviewed shared-store retention policy.
    """

    production_ready = False

    def __init__(self, *, max_sessions=1000, max_turns=12,
                 max_session_chars=16000, ttl_seconds=3600.0):
        if min(max_sessions, max_turns, max_session_chars, ttl_seconds) <= 0:
            raise ValueError("conversation limits must be positive")
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self.max_session_chars = max_session_chars
        self.ttl_seconds = ttl_seconds
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    def get(self, session_id):
        with self._lock:
            self._expire()
            entry = self._entries.get(session_id)
            if entry is None:
                return ()
            self._entries.move_to_end(session_id)
            return tuple(dict(message) for message in entry[1])

    def append(self, session_id, user_text, displayed_response):
        with self._lock:
            self._expire()
            messages = list(self.get(session_id))
            messages.extend(({"role": "user", "content": user_text},
                             {"role": "assistant", "content": displayed_response}))
            while len(messages) > self.max_turns * 2 or sum(len(m["content"]) for m in messages) > self.max_session_chars:
                del messages[:2]
            self._entries[session_id] = (time.monotonic(), tuple(messages))
            self._entries.move_to_end(session_id)
            while len(self._entries) > self.max_sessions:
                self._entries.popitem(last=False)

    def delete(self, session_id):
        with self._lock:
            return self._entries.pop(session_id, None) is not None

    def _expire(self):
        now = time.monotonic()
        for key in [key for key, (updated, _) in self._entries.items()
                    if now - updated >= self.ttl_seconds]:
            del self._entries[key]
