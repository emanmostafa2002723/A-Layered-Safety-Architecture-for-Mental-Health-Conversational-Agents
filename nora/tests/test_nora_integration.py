"""Connected NORA software contracts, using synthetic accounts and a pipeline double."""
from pathlib import Path
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from test_account_memory import store, user, PASSWORD, FakePipeline
from mental_health_pipeline.account_api import create_account_app
from mental_health_pipeline.account_service import AccountChatService
from mental_health_pipeline.account_store import AccountStore, AccountError
from mental_health_pipeline.account_preferences import InterfacePreferences


def test_preferences_encrypted_durable_and_isolated(store):
    alice, bob = user(store), user(store, 'bob')
    values = InterfacePreferences(display_name='Synthetic nickname',
        avatar={'character': 4, 'colour': 3, 'accessory': 'star'},
        country='CA', location_confirmed=True, bookmarks=['sleep', 'sleep']).model_dump()
    saved = store.update_interface_preferences(alice, values)
    assert saved['bookmarks'] == ['sleep']
    assert store.interface_preferences(bob)['display_name'] == ''
    assert b'Synthetic nickname' not in store.test_path.read_bytes()
    reopened = AccountStore(store.test_path, encryption_key=store.test_key, clock=store.clock)
    try:
        assert reopened.interface_preferences(alice) == saved
    finally:
        reopened.close()
    store.delete_account(alice)
    with store._transaction() as db:
        assert not db.execute('SELECT * FROM interface_preferences WHERE user_id=?', (alice,)).fetchall()


@pytest.mark.parametrize('invalid', [
    {'display_name': 'x' * 33}, {'display_name': 'bad\nname'},
    {'avatar': {'character': 6}}, {'avatar': {'colour': -1}},
    {'avatar': {'character': True}}, {'avatar': {'accessory': 'script'}},
    {'theme': 'dark'}, {'quiet': 'yes'}, {'country': 'ca'},
    {'location_confirmed': True}, {'bookmarks': ['<script>']},
    {'bookmarks': ['x'] * 101}, {'owner': 'bob'},
])
def test_preferences_strict_and_bounded(invalid):
    with pytest.raises(ValidationError):
        InterfacePreferences.model_validate(invalid)


def test_corrupt_profile_does_not_leak_or_reset_settings(store):
    alice = user(store)
    store.update_interface_preferences(alice, InterfacePreferences().model_dump())
    with store._transaction() as db:
        db.execute('UPDATE interface_preferences SET body=? WHERE user_id=?', (b'corrupt', alice))
    with pytest.raises(AccountError, match='stored_content_unavailable'):
        store.interface_preferences(alice)


def test_connected_static_assets_account_api_and_checked_reply(store):
    pipeline = FakePipeline()
    with TestClient(create_account_app(AccountChatService(pipeline, store))) as client:
        home = client.get('/')
        assert home.status_code == 200
        assert '/nora/connection.js' in home.text
        assert "script-src 'self';" in home.headers['content-security-policy']
        assert '<script>' not in home.text
        assert 'rel="icon" type="image/png" sizes="64x64" href="/nora/assets/nora-favicon.png"' in home.text
        assert 'rel="apple-touch-icon" href="/nora/assets/nora-logo.png"' in home.text
        assert home.headers['cache-control'] == 'no-store'
        assert client.get('/nora/assets/nora-logo.png').headers['cache-control'] == 'public, max-age=3600'
        assert client.get('/nora/connection.js').headers['cache-control'] == 'no-store'
        for asset in ('app.js', 'connection.js', 'styles.css', 'assets/nora-logo.png',
                      'assets/nora-avatars.png', 'assets/video-sleep.jpg', 'assets/nora-favicon.png'):
            assert client.get('/nora/' + asset).status_code == 200
        favicon = client.get('/nora/assets/nora-favicon.png')
        assert favicon.headers['content-type'] == 'image/png'
        assert favicon.headers['cache-control'] == 'public, max-age=3600'
        # PNG signature/IHDR: a genuine RGBA PNG, not a JPEG renamed .png.
        assert favicon.content[:8] == b'\x89PNG\r\n\x1a\n'
        assert int.from_bytes(favicon.content[16:20], 'big') == 64
        assert int.from_bytes(favicon.content[20:24], 'big') == 64
        assert favicon.content[25] == 6
        assert client.get('/nora/../../.env.qwen-server').status_code == 404
        assert client.get('/accounts/interface-preferences').status_code == 401
        client.headers.update({'X-Memory-Client': '1'})
        assert client.post('/accounts/login', json={'alias': 'alice', 'password': PASSWORD}).status_code == 200
        assert client.put('/accounts/interface-preferences', json={'display_name': '<b>Alice</b>'}).status_code == 200
        assert client.get('/accounts/interface-preferences').json()['display_name'] == '<b>Alice</b>'
        assert client.put('/accounts/interface-preferences', json={'owner': 'bob'}).status_code == 422
        chat = client.post('/accounts/chats', json={}).json()['id']
        message = 'Original user text <img src=x onerror=bad()>'
        response = client.post('/accounts/chats/' + chat + '/messages', json={'message': message}).json()
        assert pipeline.requests[-1].latest_user_message == message
        assert response['text_released'] is True and response['response_seconds'] >= 0
        pipeline.withhold = True
        withheld = client.post('/accounts/chats/' + chat + '/messages', json={'message': 'Another turn'}).json()
        assert withheld['text_released'] is False and withheld['response'] == ''
        assert client.get('/accounts/export').json()['interface_preferences']['display_name'] == '<b>Alice</b>'
        assert client.post('/accounts/logout', json={}).status_code == 200
        assert client.get('/accounts/interface-preferences').status_code == 401


def test_help_resources_require_explicit_country_confirmation(store):
    with TestClient(create_account_app(AccountChatService(FakePipeline(), store))) as client:
        unspecified = client.get('/support-resources?country=CA').json()
        confirmed = client.get('/support-resources?country=CA&location_confirmed=true').json()
        assert unspecified['country_code'] is None
        assert confirmed['country_code'] == 'CA'
        assert client.get('/health/live').json()['chat_backend'] == 'full_guarded_pipeline'


def test_real_connected_javascript_contracts():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node unavailable for shipped JavaScript contract tests')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root/'tests/nora_ui_contracts.cjs')],
                            capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
