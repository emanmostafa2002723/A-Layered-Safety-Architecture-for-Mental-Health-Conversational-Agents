"""Additional synthetic adversarial/transaction tests; no model accuracy claims."""
from dataclasses import replace
import json
import math
import sqlite3

import pytest
from fastapi.testclient import TestClient

from test_account_memory import store, user, FakePipeline
from mental_health_pipeline.account_service import AccountChatService
from mental_health_pipeline.account_api import create_account_app
from mental_health_pipeline.personal_memory import explicit_reports, select_notes, MemoryNote


@pytest.mark.parametrize('message,expected', [
    ("I can't sleep?", ()),
    ("I can't sleep? No, I sleep well.", ()),
    ("I can't sleep. Actually, that's not true.", ()),
    ("I can't sleep. This is a fictional example.", ()),
    ("I can't sleep. What should I do?", (('sleep','reported'),)),
    ("I can't sleep. Actually I am sleeping better now.", (('sleep','improved'),)),
])
def test_questions_corrections_and_fiction_are_not_stored_as_facts(message, expected):
    assert explicit_reports(message) == expected


@pytest.mark.parametrize('message,expected_topics', [
    ('Do you remember what I said about sleep?', {'sleep'}),
    ('Can we check in on my stress?', {'stress'}),
    ('What do you know about me?', {'sleep','stress'}),
    ('Remember my sleep and stress?', {'sleep','stress'}),
])
def test_targeted_recall_does_not_include_unrelated_notes(message, expected_topics):
    notes = (MemoryNote('sleep','reported',100.,200.), MemoryNote('stress','reported',100.,200.))
    assert {n.topic for n in select_notes(notes,message,now=150.)} == expected_topics


def test_save_failure_is_atomic_and_unshown_reply_not_in_context(store, monkeypatch):
    owner = user(store)
    store.preferences(owner, memory_enabled=True, history_enabled=True)
    chat = store.new_chat(owner)['id']
    pipe = FakePipeline()
    service = AccountChatService(pipe, store)
    service.chat(owner, chat, 'A first normal message.')
    key = service.session_key(owner, chat)
    before = pipe.conversation_store.get(key)
    encrypt = store._encrypt
    def fail_on_note(owner, kind, value):
        if kind.startswith('memory:'):
            raise sqlite3.OperationalError('synthetic disk failure')
        return encrypt(owner,kind,value)
    monkeypatch.setattr(store,'_encrypt',fail_on_note)
    with pytest.raises(sqlite3.OperationalError):
        service.chat(owner, chat, "I can't sleep.")
    assert store.next_turn(owner,chat) == 2
    assert len(store.history(owner,chat)) == 1
    assert pipe.conversation_store.get(key) == before
    assert not store.memories(owner)


def test_history_opt_out_clears_content_without_resetting_same_chat_safety(store):
    owner = user(store)
    store.preferences(owner, memory_enabled=True, history_enabled=True)
    chat = store.new_chat(owner)['id']
    pipe = FakePipeline()
    service = AccountChatService(pipe, store)
    service.chat(owner,chat,'I feel stressed.')
    key = service.session_key(owner,chat)
    state = replace(pipe.components.state_store.get(key),last_updated_turn=1,
                    direct_safety_question_pending=True)
    pipe.components.state_store.put(key,state)
    service.preferences(owner,memory_enabled=True,history_enabled=False)
    assert not pipe.conversation_store.get(key)
    assert pipe.components.state_store.get(key).direct_safety_question_pending
    service.delete_chat(owner,chat)
    assert not pipe.components.state_store.get(key).direct_safety_question_pending


@pytest.mark.parametrize('message', ['   ', 'x\ud800', '\u00e9'*9000],
                         ids=['blank','invalid-unicode','over-utf8-byte-limit'])
def test_invalid_message_is_422_before_inference_or_storage(store, message):
    from test_account_memory import PASSWORD
    pipe = FakePipeline()
    app = create_account_app(AccountChatService(pipe,store))
    headers = {'X-Memory-Client':'1','Origin':'http://127.0.0.1:7861','Content-Type':'application/json'}
    with TestClient(app) as client:
        assert client.post('/accounts/login',headers=headers,json={'alias':'alice','password':PASSWORD}).status_code == 200
        chat = client.post('/accounts/chats',headers=headers,json={}).json()['id']
        # ASCII JSON lets invalid Unicode reach validation rather than the HTTP client encoder.
        result = client.post('/accounts/chats/'+chat+'/messages',headers=headers,
                             content=json.dumps({'message':message},
                                 ensure_ascii=any(0xD800 <= ord(c) <= 0xDFFF for c in message)))
        assert result.status_code == 422
        assert message not in result.text
        assert not pipe.requests


@pytest.mark.parametrize('observed,expires', [(100.,math.inf),(math.nan,200.),(-1.,200.)])
def test_invalid_timestamp_cannot_create_a_permanent_note(observed,expires):
    with pytest.raises(ValueError):
        MemoryNote('sleep','reported',observed,expires)


def test_oversized_transport_returns_413_without_breaking_security_headers(store):
    from test_account_memory import PASSWORD
    app = create_account_app(AccountChatService(FakePipeline(),store))
    headers = {'X-Memory-Client':'1','Origin':'http://127.0.0.1:7861','Content-Type':'application/json'}
    with TestClient(app) as client:
        result = client.post('/accounts/login',headers=headers,content='x'*21000)
        assert result.status_code == 413
        assert result.headers['cache-control'] == 'no-store'
        assert "frame-ancestors 'none'" in result.headers['content-security-policy']


@pytest.mark.parametrize('current,candidate,route,expected', [
    ('I feel overwhelmed.', 'A few things that might help: first, try a break. Second, write a list.', 'normal_chat',True),
    ('I feel overwhelmed. Please give me tips.', 'A few things that might help: first, try a break.', 'normal_chat',False),
    ('I feel overwhelmed.', 'What has felt hardest about today?', 'normal_chat',False),
    ('I feel unsafe.', 'A few things that might help: first, seek human help.', 'crisis_response',False),
])
def test_listening_preference_is_proportionate_and_cannot_override_safety(current,candidate,route,expected):
    from types import SimpleNamespace
    from mental_health_pipeline.conversation_quality import ignored_listening_preference
    packet=SimpleNamespace(system_route=SimpleNamespace(value=route),crisis_labels=())
    notes=(MemoryNote('support_preference','listening',100.,200.),)
    assert ignored_listening_preference(current,candidate,notes,packet) is expected


def test_note_date_is_not_a_previous_meeting():
    from types import SimpleNamespace
    from mental_health_pipeline.conversation_quality import unsupported_personal_memory
    packet=SimpleNamespace(system_route=SimpleNamespace(value='normal_chat'),crisis_labels=())
    note=MemoryNote('sleep','improved',100.,200.)
    assert unsupported_personal_memory('Can we check in about sleep?',
        'How has it been since our last check-in on October 18?',(),(note,),packet)


def test_restored_transcript_does_not_feed_application_contact_block_to_generator(store):
    owner=user(store)
    store.preferences(owner,memory_enabled=False,history_enabled=True)
    chat=store.new_chat(owner)['id']
    store.save_turn(owner,chat,'An earlier synthetic disclosure.',
        {'response':'APPLICATION CONTACT BLOCK\n\nA previously displayed supportive reply.',
         'resources':{'country_code':'CA'}})
    pipe=FakePipeline()
    service=AccountChatService(pipe,store)
    service.chat(owner,chat,'Another message.')
    messages=pipe.conversation_store.get(service.session_key(owner,chat))
    assert messages[1]['content']=='A previously displayed supportive reply.'
    assert all('APPLICATION CONTACT BLOCK' not in item['content'] for item in messages)


def test_rollback_preserves_new_risk_but_not_claimed_resource_delivery(store,monkeypatch):
    owner=user(store)
    chat=store.new_chat(owner)['id']
    class RiskPipeline(FakePipeline):
        def process(self,request):
            key=request.session_id
            state=self.components.state_store.get(key)
            self.components.state_store.put(key,replace(state,last_updated_turn=1,
                current_or_imminent_seen=True,resources_last_shown_turn=1,
                direct_safety_question_pending=True))
            return super().process(request)
    pipe=RiskPipeline()
    service=AccountChatService(pipe,store)
    def fail(*args,**kwargs): raise sqlite3.OperationalError('synthetic failure')
    monkeypatch.setattr(store,'commit_turn',fail)
    with pytest.raises(sqlite3.OperationalError):service.chat(owner,chat,'A fictional message.')
    state=pipe.components.state_store.get(service.session_key(owner,chat))
    assert state.current_or_imminent_seen
    assert state.resources_last_shown_turn is None
    assert not state.direct_safety_question_pending
    assert not pipe.conversation_store.get(service.session_key(owner,chat))


def test_invalid_password_unicode_and_coerced_adult_confirmation_are_rejected(store):
    pipe=FakePipeline()
    app=create_account_app(AccountChatService(pipe,store))
    headers={'X-Memory-Client':'1','Content-Type':'application/json'}
    with TestClient(app) as client:
        assert client.post('/accounts/login',headers=headers,
            content=json.dumps({'alias':'alice','password':'bad\ud800'},ensure_ascii=True)).status_code==422
        assert client.post('/accounts/register',headers=headers,
            json={'alias':'fictional','password':'fictional-password-only','adult_confirmed':'yes'}).status_code==422


def test_general_prompt_matches_depth_to_content_not_a_fixed_short_length():
    from mental_health_pipeline.generation import CONDITIONAL_SYSTEM_PROMPT
    assert 'Match depth to the concern' in CONDITIONAL_SYSTEM_PROMPT
    assert 'not just a question' in CONDITIONAL_SYSTEM_PROMPT
    assert '2–4' not in CONDITIONAL_SYSTEM_PROMPT
    assert len(CONDITIONAL_SYSTEM_PROMPT.splitlines())==2


def test_optional_notes_are_rendered_as_direct_reports_not_case_records():
    from mental_health_pipeline.generation import GenerationRequest,PromptBuilder,CONDITIONAL_SYSTEM_PROMPT
    note=MemoryNote('sleep','reported',100.,200.)
    envelope=PromptBuilder(CONDITIONAL_SYSTEM_PROMPT).build(GenerationRequest('How about my sleep?', '', personal_memory=(note,)))
    assert ': You reported difficulty sleeping.' in envelope.messages[-1]['content']
    assert 'The user reported' not in envelope.messages[-1]['content']
