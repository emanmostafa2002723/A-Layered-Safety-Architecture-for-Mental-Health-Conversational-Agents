"""Single-process, encrypted-content SQLite account prototype (not deployment approval)."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
import uuid

from .personal_memory import MemoryNote, explicit_reports

RETENTION_SECONDS = 30 * 86400
SESSION_SECONDS = 12 * 3600
SESSION_IDLE_SECONDS = 30 * 60


class AccountError(ValueError):
    def __init__(self, code, status=400):
        self.code, self.status = code, status
        super().__init__(code)


def password_hash(password, salt):
    # OWASP's scrypt fallback parameters; no plaintext/reversible passwords.
    return hashlib.scrypt(password.encode('utf-8'), salt=salt, n=2**17, r=8, p=1,
                          maxmem=256 * 1024**2, dklen=32)


class AccountStore:
    """Every chat/note operation requires an authenticated owner's ID.

    Notes and transcript bodies are Fernet encrypted; account IDs, timestamps,
    aliases and relationship metadata are not. Keep the key out of backups and
    source control. One application process only; no claim of clinical readiness.
    """
    production_ready = False

    def __init__(self, path, *, encryption_key: bytes, clock=time.time):
        from cryptography.fernet import Fernet
        self._cipher = Fernet(encryption_key)
        self.clock = clock
        self._lock = threading.RLock()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, timeout=10, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript('''
          PRAGMA foreign_keys=ON;
          PRAGMA secure_delete=ON;
          CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, alias TEXT NOT NULL UNIQUE, salt BLOB NOT NULL,
            password_hash BLOB NOT NULL, memory_enabled INTEGER NOT NULL DEFAULT 0,
            history_enabled INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires REAL NOT NULL, last_seen REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created REAL NOT NULL, expires REAL NOT NULL, turn INTEGER NOT NULL DEFAULT 0);
          CREATE INDEX IF NOT EXISTS chat_owner ON chats(user_id);
          CREATE TABLE IF NOT EXISTS turns (
            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            turn INTEGER NOT NULL, body BLOB NOT NULL, expires REAL NOT NULL,
            PRIMARY KEY(chat_id, turn));
          CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            topic TEXT NOT NULL, body BLOB NOT NULL, observed REAL NOT NULL, expires REAL NOT NULL,
            UNIQUE(user_id, topic));
          CREATE TABLE IF NOT EXISTS interface_preferences (
            user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            body BLOB NOT NULL);
        ''')
        self._db.commit()

    @contextmanager
    def _transaction(self):
        with self._lock, self._db:
            self._purge()
            yield self._db

    def _purge(self):
        now = self.clock()
        self._db.execute('DELETE FROM memories WHERE expires<=?', (now,))
        self._db.execute('DELETE FROM turns WHERE expires<=?', (now,))
        self._db.execute('DELETE FROM chats WHERE expires<=?', (now,))
        self._db.execute('DELETE FROM sessions WHERE expires<=? OR last_seen<=?',
                         (now, now-SESSION_IDLE_SECONDS))

    def purge(self):
        with self._transaction():
            pass

    def _encrypt(self, owner, kind, value):
        return self._cipher.encrypt(json.dumps({'owner': owner, 'kind': kind, 'value': value},
                                               ensure_ascii=True).encode())

    def _decrypt(self, owner, kind, body):
        from cryptography.fernet import InvalidToken
        try:
            value = json.loads(self._cipher.decrypt(body))
        except (InvalidToken,ValueError,UnicodeError,TypeError) as error:
            raise AccountError('stored_content_unavailable',503) from error
        if not isinstance(value,dict) or set(value) != {'owner','kind','value'}:
            raise AccountError('stored_content_unavailable',503)
        if value['owner'] != owner or value['kind'] != kind:
            raise AccountError('stored_content_binding_failed', 503)
        return value['value']

    def register(self, alias, password, *, adult_confirmed=False):
        alias = alias.strip().casefold()
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,31}', alias):
            raise AccountError('use_a_3_to_32_character_alias')
        if not adult_confirmed:
            raise AccountError('adult_confirmation_required')
        if not 12 <= len(password) <= 128:
            raise AccountError('password_must_be_12_to_128_characters')
        salt = secrets.token_bytes(16)
        hashed = password_hash(password, salt)
        user_id = str(uuid.uuid4())
        try:
            with self._transaction() as db:
                db.execute('INSERT INTO users(id,alias,salt,password_hash,created) VALUES(?,?,?,?,?)',
                           (user_id, alias, salt, hashed, self.clock()))
        except sqlite3.IntegrityError as error:
            raise AccountError('registration_unavailable', 409) from error
        return self.profile(user_id)

    def login(self, alias, password):
        if not 1 <= len(password) <= 128:
            raise AccountError('invalid_credentials', 401)
        with self._transaction() as db:
            row = db.execute('SELECT * FROM users WHERE alias=?', (alias.strip().casefold(),)).fetchone()
            salt = bytes(row['salt']) if row else b'unknown-alias-slt'
            actual = password_hash(password, salt)
            expected = bytes(row['password_hash']) if row else bytes(32)
            if not hmac.compare_digest(actual, expected) or row is None:
                raise AccountError('invalid_credentials', 401)
            token = secrets.token_urlsafe(32)
            now = self.clock()
            # Bounded simultaneous sessions; signing in invalidates the oldest above five.
            db.execute('DELETE FROM sessions WHERE user_id=? AND token_hash NOT IN '
                       '(SELECT token_hash FROM sessions WHERE user_id=? ORDER BY last_seen DESC LIMIT 4)',
                       (row['id'], row['id']))
            db.execute('INSERT INTO sessions VALUES(?,?,?,?)',
                       (hashlib.sha256(token.encode()).hexdigest(), row['id'], now+SESSION_SECONDS, now))
            return token

    def authenticate(self, token):
        if not isinstance(token, str) or not 20 <= len(token) <= 128:
            raise AccountError('sign_in_required', 401)
        hashed = hashlib.sha256(token.encode()).hexdigest()
        with self._transaction() as db:
            row = db.execute('SELECT user_id FROM sessions WHERE token_hash=?', (hashed,)).fetchone()
            if row is None:
                raise AccountError('sign_in_required', 401)
            db.execute('UPDATE sessions SET last_seen=? WHERE token_hash=?', (self.clock(), hashed))
            return row['user_id']

    def logout(self, token):
        with self._transaction() as db:
            db.execute('DELETE FROM sessions WHERE token_hash=?', (hashlib.sha256(token.encode()).hexdigest(),))

    def profile(self, owner):
        with self._transaction() as db:
            row = db.execute('SELECT id,alias,memory_enabled,history_enabled FROM users WHERE id=?', (owner,)).fetchone()
            if row is None:
                raise AccountError('sign_in_required', 401)
            return {'id': row['id'], 'alias': row['alias'], 'memory_enabled': bool(row['memory_enabled']),
                    'history_enabled': bool(row['history_enabled']), 'retention_days': 30}

    def preferences(self, owner, *, memory_enabled, history_enabled):
        with self._transaction() as db:
            db.execute('UPDATE users SET memory_enabled=?,history_enabled=? WHERE id=?',
                       (int(memory_enabled), int(history_enabled), owner))
            if not memory_enabled:
                db.execute('DELETE FROM memories WHERE user_id=?', (owner,))
            if not history_enabled:
                db.execute('DELETE FROM turns WHERE chat_id IN (SELECT id FROM chats WHERE user_id=?)', (owner,))
        return self.profile(owner)

    def interface_preferences(self, owner):
        from pydantic import ValidationError
        from .account_preferences import InterfacePreferences
        self.profile(owner)
        with self._transaction() as db:
            row = db.execute('SELECT body FROM interface_preferences WHERE user_id=?', (owner,)).fetchone()
            if row is None:
                return InterfacePreferences().model_dump()
            value = self._decrypt(owner, 'interface_preferences', row['body'])
            try:
                return InterfacePreferences.model_validate(value).model_dump()
            except ValidationError as error:
                raise AccountError('stored_content_unavailable', 503) from error

    def update_interface_preferences(self, owner, value):
        from .account_preferences import InterfacePreferences
        cleaned = InterfacePreferences.model_validate(value).model_dump()
        with self._transaction() as db:
            if db.execute('SELECT 1 FROM users WHERE id=?', (owner,)).fetchone() is None:
                raise AccountError('sign_in_required', 401)
            db.execute('INSERT INTO interface_preferences(user_id,body) VALUES(?,?) '
                       'ON CONFLICT(user_id) DO UPDATE SET body=excluded.body',
                       (owner, self._encrypt(owner, 'interface_preferences', cleaned)))
        return cleaned

    def new_chat(self, owner):
        self.profile(owner)
        with self._transaction() as db:
            if db.execute('SELECT COUNT(*) FROM chats WHERE user_id=?', (owner,)).fetchone()[0] >= 100:
                raise AccountError('chat_limit_reached', 409)
            chat_id, now = str(uuid.uuid4()), self.clock()
            db.execute('INSERT INTO chats(id,user_id,created,expires) VALUES(?,?,?,?)',
                       (chat_id, owner, now, now+RETENTION_SECONDS))
        return {'id': chat_id, 'created_at': now, 'expires_at': now+RETENTION_SECONDS}

    def _owned(self, db, owner, chat):
        row = db.execute('SELECT * FROM chats WHERE id=? AND user_id=?', (chat, owner)).fetchone()
        if row is None:
            raise AccountError('chat_not_found', 404)
        return row

    def chats(self, owner):
        with self._transaction() as db:
            return [dict(row) for row in db.execute('SELECT id,created,expires,turn FROM chats '
                    'WHERE user_id=? ORDER BY created DESC', (owner,))]

    def history(self, owner, chat):
        with self._transaction() as db:
            self._owned(db, owner, chat)
            return [self._decrypt(owner, 'turn:'+chat, row['body']) for row in
                    db.execute('SELECT body FROM turns WHERE chat_id=? ORDER BY turn', (chat,))]

    def next_turn(self, owner, chat):
        with self._transaction() as db:
            return self._owned(db, owner, chat)['turn']+1

    def save_turn(self, owner, chat, user_text, public_response):
        profile = self.profile(owner)
        with self._transaction() as db:
            self._write_turn(db,owner,chat,user_text,public_response,profile['history_enabled'])

    def _write_turn(self, db, owner, chat, user_text, public_response, history_enabled, expected_turn=None):
        row = self._owned(db, owner, chat)
        turn = row['turn']+1
        if expected_turn is not None and turn != expected_turn:
            raise AccountError('chat_turn_changed',409)
        db.execute('UPDATE chats SET turn=? WHERE id=?', (turn, chat))
        if history_enabled:
            body = self._encrypt(owner, 'turn:'+chat,
                                 {'turn': turn, 'user': user_text, 'response': public_response})
            db.execute('INSERT INTO turns VALUES(?,?,?,?)',
                       (chat, turn, body, min(row['expires'], self.clock()+RETENTION_SECONDS)))
            db.execute('DELETE FROM turns WHERE chat_id=? AND turn<=?', (chat, turn-200))

    def commit_turn(self, owner, chat, user_text, public_response, *, memory_allowed, expected_turn):
        """Commit an approved public result and its user-only notes atomically.

        A failed memory write cannot leave an unseen response/advanced turn on disk.
        Consent is reread inside the transaction; no transcript is saved when off.
        """
        with self._transaction() as db:
            profile = db.execute('SELECT history_enabled,memory_enabled FROM users WHERE id=?',(owner,)).fetchone()
            if profile is None:
                raise AccountError('sign_in_required',401)
            self._write_turn(db,owner,chat,user_text,public_response,profile['history_enabled'],expected_turn)
            if profile['memory_enabled'] and memory_allowed:
                self._write_notes(db,owner,user_text)

    def capture(self, owner, user_text):
        if not self.profile(owner)['memory_enabled']:
            return
        with self._transaction() as db:
            self._write_notes(db,owner,user_text)

    def _write_notes(self, db, owner, user_text):
        for topic,state in explicit_reports(user_text):
            now = self.clock()
            db.execute('INSERT INTO memories VALUES(?,?,?,?,?,?) ON CONFLICT(user_id,topic) '
                       'DO UPDATE SET body=excluded.body, observed=excluded.observed,expires=excluded.expires',
                       (str(uuid.uuid4()), owner, topic, self._encrypt(owner, 'memory:'+topic, state),
                        now, now+RETENTION_SECONDS))

    def memories(self, owner):
        if not self.profile(owner)['memory_enabled']:
            return []
        with self._transaction() as db:
            return [{'id': row['id'], 'note': MemoryNote(row['topic'],
                    self._decrypt(owner, 'memory:'+row['topic'], row['body']), row['observed'], row['expires'])}
                    for row in db.execute('SELECT * FROM memories WHERE user_id=? ORDER BY observed DESC', (owner,))]

    def forget(self, owner, memory_id=None):
        with self._transaction() as db:
            if memory_id is None:
                db.execute('DELETE FROM memories WHERE user_id=?', (owner,))
            else:
                db.execute('DELETE FROM memories WHERE user_id=? AND id=?', (owner, memory_id))

    def delete_chat(self, owner, chat):
        with self._transaction() as db:
            self._owned(db, owner, chat)
            db.execute('DELETE FROM chats WHERE id=? AND user_id=?', (chat, owner))

    def delete_account(self, owner):
        with self._transaction() as db:
            db.execute('DELETE FROM users WHERE id=?', (owner,))

    def close(self):
        with self._lock:
            self._db.close()
