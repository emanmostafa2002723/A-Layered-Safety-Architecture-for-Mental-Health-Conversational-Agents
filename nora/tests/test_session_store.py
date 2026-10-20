from mental_health_pipeline.response_policy.session_state import RecentSafetyState
from mental_health_pipeline.session_store import InMemorySafetyStateStore


def test_store_contains_only_typed_content_free_state():
    store = InMemorySafetyStateStore(max_sessions=2)
    state = RecentSafetyState(last_updated_turn=0)
    store.put("session-1", state)
    assert store.get("session-1") == state


def test_transaction_updates_atomically():
    store = InMemorySafetyStateStore()

    def update(state):
        updated = RecentSafetyState(last_updated_turn=state.last_updated_turn + 1)
        return updated, updated.last_updated_turn

    assert store.transaction("s1", update) == 0
    assert store.transaction("s1", update) == 1


def test_lru_bound_evicts_oldest_session():
    store = InMemorySafetyStateStore(max_sessions=1)
    store.put("s1", RecentSafetyState(last_updated_turn=0))
    store.put("s2", RecentSafetyState(last_updated_turn=1))
    assert store.get("s1").last_updated_turn == -1
    assert store.get("s2").last_updated_turn == 1
