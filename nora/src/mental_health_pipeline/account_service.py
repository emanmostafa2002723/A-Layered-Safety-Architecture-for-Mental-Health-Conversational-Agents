"""Owner-scoped account chats over the existing guarded pipeline."""
from __future__ import annotations

import threading
import uuid
import time

from dataclasses import replace
from .account_store import AccountError, AccountStore
from .personal_memory import select_notes
from .schemas import PipelineRequest, RequestScope


class AccountChatService:
    production_ready = False

    def __init__(self, pipeline, store: AccountStore):
        self.pipeline, self.store = pipeline, store
        # One GPU pipeline, one process: serializes consent/delete and inference.
        # Routes acquire this before authenticating to avoid logout/delete races.
        self.lock = threading.RLock()

    @staticmethod
    def session_key(owner, chat):
        return 'account:'+owner+':'+chat

    def _clear_chat(self, owner, chat, *, clear_safety=True):
        key = self.session_key(owner, chat)
        self.pipeline.conversation_store.delete(key)
        if clear_safety:
            self.pipeline.components.state_store.delete(key)

    def chat(self, owner, chat, message, *, country=None, location_confirmed=False):
        with self.lock:
            response_started = time.perf_counter()
            # Ownership is checked before reading history, notes or invoking a model.
            turn = self.store.next_turn(owner, chat)
            key = self.session_key(owner, chat)
            profile = self.store.profile(owner)
            if profile['history_enabled'] and not self.pipeline.conversation_store.get(key):
                for entry in self.store.history(owner, chat)[-12:]:
                    shown = entry['response']['response']
                    if entry['response'].get('resources') and '\n\n' in shown:
                        shown = shown.split('\n\n',1)[1]
                    if shown.strip():
                        self.pipeline.conversation_store.append(key, entry['user'], shown)
            memory_started = time.perf_counter()
            notes = select_notes([row['note'] for row in self.store.memories(owner)], message,
                                 now=self.store.clock())
            memory_lookup_ms = (time.perf_counter()-memory_started)*1000
            request = PipelineRequest(message, key, RequestScope(age_eligibility_18_plus=True,
                current_location_country=country, location_confirmed=location_confirmed),
                turn_number=turn, request_id=str(uuid.uuid4()), personal_memory=notes)
            request.validate()
            previous_context = self.pipeline.conversation_store.get(key)
            previous_state = self.pipeline.components.state_store.get(key)
            try:
                response = self.pipeline.process(request)
                public = response.public_dict()
                self.store.commit_turn(owner,chat,message,public,
                    memory_allowed=response.route == 'normal_chat',expected_turn=turn)
            except Exception:
                # The browser will receive an error, not this generated reply.
                # Restore context and delivery observations, retaining any new
                # conservative risk observation made by the input guardrails.
                self.pipeline.conversation_store.delete(key)
                for i in range(0,len(previous_context),2):
                    self.pipeline.conversation_store.append(key,previous_context[i]['content'],
                                                            previous_context[i+1]['content'])
                current = self.pipeline.components.state_store.get(key)
                self.pipeline.components.state_store.put(key,replace(current,
                    resources_last_shown_turn=previous_state.resources_last_shown_turn,
                    direct_safety_question_pending=previous_state.direct_safety_question_pending))
                raise
            return {**public, 'memory_notes_supplied': len(notes) if response.route == 'normal_chat' else 0,
                    'memory_lookup_ms': round(memory_lookup_ms, 3),
                    'response_seconds': round(time.perf_counter()-response_started, 3),
                    'text_released': bool(response.response.strip())}

    def preferences(self, owner, *, memory_enabled, history_enabled):
        with self.lock:
            previous = self.store.profile(owner)
            profile = self.store.preferences(owner, memory_enabled=memory_enabled, history_enabled=history_enabled)
            if previous['history_enabled'] and not history_enabled:
                for chat in self.store.chats(owner):
                    self._clear_chat(owner, chat['id'],clear_safety=False)
            return profile

    def delete_chat(self, owner, chat):
        with self.lock:
            self.store.delete_chat(owner, chat)
            self._clear_chat(owner, chat)

    def delete_account(self, owner):
        with self.lock:
            chats = self.store.chats(owner)
            self.store.delete_account(owner)
            for chat in chats:
                self._clear_chat(owner, chat['id'])
