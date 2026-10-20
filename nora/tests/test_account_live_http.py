"""Real loopback HTTP with synthetic pipeline doubles; no learned-model claims."""
from contextlib import contextmanager
import socket
import threading
import time

import httpx
import uvicorn
import pytest

from test_account_memory import store, user, FakePipeline, PASSWORD
from mental_health_pipeline.account_api import create_account_app,AccountWebSettings
from mental_health_pipeline.account_service import AccountChatService


@contextmanager
def running(service):
    sock=socket.socket()
    sock.bind(('127.0.0.1',0))
    port=sock.getsockname()[1]
    origin=f'http://127.0.0.1:{port}'
    app=create_account_app(service,AccountWebSettings(origin))
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,
        access_log=False,log_level='critical'))
    thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True)
    thread.start()
    try:
        deadline=time.monotonic()+10
        while not server.started and thread.is_alive() and time.monotonic()<deadline:
            time.sleep(.02)
        assert server.started,'owned loopback server did not start'
        yield origin
    finally:
        server.should_exit=True
        thread.join(10)
        sock.close()
        assert not thread.is_alive(),'owned loopback server failed to stop'


def test_actual_http_account_lifecycle_and_cross_user_isolation(store):
    pipe=FakePipeline()
    with running(AccountChatService(pipe,store)) as origin:
        headers={'X-Memory-Client':'1','Origin':origin,'Content-Type':'application/json'}
        with httpx.Client(base_url=origin,headers=headers,trust_env=False) as alice, \
             httpx.Client(base_url=origin,headers=headers,trust_env=False) as bob:
            assert alice.get('/').status_code==200
            assert alice.get('/accounts/export').status_code==401
            assert alice.post('/accounts/login',json={'alias':'alice','password':PASSWORD}).status_code==200
            assert bob.post('/accounts/login',json={'alias':'bob','password':PASSWORD}).status_code==200
            assert alice.put('/accounts/preferences',json={'memory_enabled':True,'history_enabled':True}).status_code==200
            chat=alice.post('/accounts/chats',json={}).json()['id']
            assert alice.post('/accounts/chats/'+chat+'/messages',json={'message':"I can't sleep."}).status_code==200
            rows=alice.get('/accounts/memories').json()
            assert len(rows)==1
            assert rows[0]['user_report']=='reported difficulty sleeping'
            assert bob.get('/accounts/memories').json()==[]
            assert bob.get('/accounts/chats/'+chat).status_code==404
            assert bob.delete('/accounts/chats/'+chat).status_code==404
            assert bob.delete('/accounts/memories/'+rows[0]['id']).status_code==200
            assert len(alice.get('/accounts/memories').json())==1
            exported=alice.get('/accounts/export').json()
            assert len(exported['chats'])==1
            assert len(exported['chats'][0]['turns'])==1
            assert alice.post('/accounts/login',content='x'*21000).status_code==413
            assert alice.delete('/accounts/memories/'+rows[0]['id']).status_code==200
            assert alice.get('/accounts/memories').json()==[]
            assert alice.delete('/accounts/chats/'+chat).status_code==200
            assert alice.get('/accounts/chats').json()==[]
            assert alice.delete('/accounts/me').status_code==200
            assert alice.get('/accounts/export').status_code==401
            assert bob.get('/accounts/me').status_code==200


def test_corrupt_ciphertext_fails_closed_without_exposing_contents(store):
    owner=user(store)
    store.preferences(owner,memory_enabled=True,history_enabled=True)
    store.capture(owner,"I can't sleep.")
    with store._transaction() as db:
        db.execute('UPDATE memories SET body=? WHERE user_id=?',(b'corrupt-synthetic-token',owner))
    with running(AccountChatService(FakePipeline(),store)) as origin:
        headers={'X-Memory-Client':'1','Origin':origin,'Content-Type':'application/json'}
        with httpx.Client(base_url=origin,headers=headers,trust_env=False) as client:
            client.post('/accounts/login',json={'alias':'alice','password':PASSWORD})
            result=client.get('/accounts/memories')
            assert result.status_code==503
            assert result.json()=={'detail':'stored_content_unavailable'}
            assert 'corrupt-synthetic' not in result.text
