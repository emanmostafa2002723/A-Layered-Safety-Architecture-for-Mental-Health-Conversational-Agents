"""Local-only account/memory web prototype; no anonymous model endpoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .account_store import AccountError, SESSION_SECONDS
from .account_preferences import InterfacePreferences
from .api import RequestBodyLimitMiddleware, SecurityHeadersMiddleware

COOKIE = 'mh_account'


@dataclass(frozen=True)
class AccountWebSettings:
    origin: str = 'http://127.0.0.1:7861'

    def __post_init__(self):
        parsed = urlsplit(self.origin)
        if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'}
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError('Account prototype is loopback HTTP only; public deployment is not approved')


class Credentials(BaseModel):
    model_config = ConfigDict(extra='forbid')
    alias: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=128)

    @field_validator('password')
    @classmethod
    def valid_password_encoding(cls,value):
        try:
            value.encode('utf-8',errors='strict')
        except UnicodeEncodeError as error:
            raise ValueError('invalid_credentials') from error
        return value


class Registration(Credentials):
    adult_confirmed: bool = Field(default=False,strict=True)


class Preferences(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    memory_enabled: bool
    history_enabled: bool


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid')
    message: str = Field(min_length=1, max_length=16384)
    country: str | None = Field(default=None, max_length=2, pattern='^[A-Za-z]{2}$')
    location_confirmed: bool = False

    @field_validator('message')
    @classmethod
    def valid_utf8_message(cls, value):
        try:
            size = len(value.encode('utf-8',errors='strict'))
        except UnicodeEncodeError as error:
            raise ValueError('invalid_message') from error
        if not value.strip() or size > 16384:
            raise ValueError('invalid_message')
        return value


class AuthLimiter:
    """Bounded single-process abuse throttle, not a distributed gateway."""
    def __init__(self):
        self.rows, self.lock = {}, threading.Lock()

    def check(self, key):
        with self.lock:
            now = time.monotonic()
            self.rows = {k: v for k, v in self.rows.items() if now-v[0] < 60}
            started, count = self.rows.get(key, (now, 0))
            if count >= 6 or (key not in self.rows and len(self.rows) >= 1000):
                raise AccountError('too_many_auth_attempts', 429)
            self.rows[key] = (started, count+1)


def create_account_app(service, settings=None):
    settings = settings or AccountWebSettings()
    limiter = AuthLimiter()

    @asynccontextmanager
    async def lifespan(app):
        # The launcher owns model startup/close. Expiry is enforced before every
        # DB operation; this task also removes idle expired rows every minute.
        import asyncio
        async def purge_loop():
            while True:
                await asyncio.sleep(60)
                await asyncio.to_thread(service.store.purge)
        task = asyncio.create_task(purge_loop())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title='Private account and memory prototype', docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', 'testserver'])
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=20000)

    @app.middleware('http')
    async def local_boundary(request, call_next):
        client = request.client.host if request.client else ''
        if client not in {'127.0.0.1', '::1', 'testclient'}:
            return JSONResponse({'detail': 'local_prototype_only'}, status_code=403)
        if request.method not in {'GET', 'HEAD'}:
            # Custom header + exact origin + no CORS: browser CSRF cannot issue
            # a valid mutation through a simple cross-site form or fetch.
            if (request.headers.get('x-memory-client') != '1'
                    or request.headers.get('origin') not in {None, settings.origin}
                    or request.headers.get('sec-fetch-site') == 'cross-site'
                    or request.headers.get('content-type', '').split(';')[0] != 'application/json'):
                return JSONResponse({'detail': 'invalid_request_origin'}, status_code=403)
        result = await call_next(request)
        legacy = request.url.path == '/dev/account-console'
        result.headers['Content-Security-Policy'] = ("default-src 'self'; script-src 'self'" +
            (" 'unsafe-inline'" if legacy else '') + "; style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-src 'none'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        # Only public, bundled imagery is cacheable. Account/API responses and
        # scripts remain no-store; no private data enters a shared cache.
        if (request.url.path.startswith('/nora/assets/') and
                result.status_code in {200, 304}):
            result.headers['Cache-Control'] = 'public, max-age=3600'
        return result

    @app.exception_handler(AccountError)
    async def account_error(request, error):
        return JSONResponse({'detail': error.code}, status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse({'detail': 'invalid_request'}, status_code=422)

    def owner(request):
        return service.store.authenticate(request.cookies.get(COOKIE))

    @app.get('/', response_class=HTMLResponse)
    def index():
        return HTMLResponse((web_root/'index.html').read_text(encoding='utf-8'))

    web_root = Path(__file__).with_name('web_nora')
    app.mount('/nora', StaticFiles(directory=web_root), name='nora-interface')

    @app.get('/dev/account-console', response_class=HTMLResponse)
    def diagnostic_console():
        from .account_web_ui import ACCOUNT_HTML
        return HTMLResponse(ACCOUNT_HTML)

    @app.get('/health/live')
    def live():
        return {'status': 'live', 'production_ready': False, 'retention_days': 30,
                'interface': 'nora-v2', 'chat_backend': 'full_guarded_pipeline'}

    @app.get('/support-resources')
    def support_resources(country: str | None = None, location_confirmed: bool = False):
        from .resources import DEFAULT_RESOURCE_DIRECTORY
        directory = getattr(service.pipeline.components, 'resources', DEFAULT_RESOURCE_DIRECTORY)
        return directory.resolve(country if location_confirmed else None).to_dict()

    @app.post('/accounts/register')
    def register(body: Registration, request: Request):
        limiter.check(request.client.host)
        with service.lock:
            return service.store.register(body.alias, body.password, adult_confirmed=body.adult_confirmed)

    @app.post('/accounts/login')
    def login(body: Credentials, request: Request):
        limiter.check(request.client.host)
        with service.lock:
            token = service.store.login(body.alias, body.password)
            response = JSONResponse({'signed_in': True})
            # Secure=False ONLY because this factory refuses non-loopback HTTP.
            response.set_cookie(COOKIE, token, httponly=True, samesite='strict', secure=False,
                                path='/accounts', max_age=SESSION_SECONDS)
            return response

    @app.post('/accounts/logout')
    def logout(request: Request):
        with service.lock:
            owner(request)
            service.store.logout(request.cookies[COOKIE])
            response = JSONResponse({'signed_out': True})
            response.delete_cookie(COOKIE, path='/accounts')
            return response

    @app.get('/accounts/me')
    def me(request: Request):
        with service.lock:
            return service.store.profile(owner(request))

    @app.put('/accounts/preferences')
    def preferences(body: Preferences, request: Request):
        with service.lock:
            return service.preferences(owner(request), **body.model_dump())

    @app.get('/accounts/interface-preferences')
    def interface_preferences(request: Request):
        with service.lock:
            return service.store.interface_preferences(owner(request))

    @app.put('/accounts/interface-preferences')
    def update_interface_preferences(body: InterfacePreferences, request: Request):
        with service.lock:
            return service.store.update_interface_preferences(owner(request), body.model_dump())

    @app.get('/accounts/chats')
    def chats(request: Request):
        with service.lock:
            return service.store.chats(owner(request))

    @app.post('/accounts/chats')
    def new_chat(request: Request):
        with service.lock:
            return service.store.new_chat(owner(request))

    @app.get('/accounts/chats/{chat_id}')
    def chat_history(chat_id: str, request: Request):
        with service.lock:
            return service.store.history(owner(request), chat_id)

    @app.post('/accounts/chats/{chat_id}/messages')
    def send_message(chat_id: str, body: Message, request: Request):
        with service.lock:
            try:
                return service.chat(owner(request), chat_id, **body.model_dump())
            except AccountError:
                raise
            except Exception:
                # Never display a raw draft, exception text or unchecked fallback.
                return JSONResponse({'detail': 'pipeline_no_checked_response'}, status_code=503)

    @app.delete('/accounts/chats/{chat_id}')
    def delete_chat(chat_id: str, request: Request):
        with service.lock:
            service.delete_chat(owner(request), chat_id)
            return {'deleted': True}

    @app.get('/accounts/memories')
    def memories(request: Request):
        with service.lock:
            return [{'id': row['id'], **row['note'].prompt_dict(), 'expires_at': row['note'].expires_at}
                    for row in service.store.memories(owner(request))]

    @app.delete('/accounts/memories')
    def forget_all(request: Request):
        with service.lock:
            service.store.forget(owner(request))
            return {'deleted': True}

    @app.delete('/accounts/memories/{memory_id}')
    def forget_one(memory_id: str, request: Request):
        with service.lock:
            service.store.forget(owner(request), memory_id)
            return {'deleted': True}

    @app.get('/accounts/export')
    def export(request: Request):
        with service.lock:
            user = owner(request)
            return {'profile': service.store.profile(user), 'memories': memories(request),
                    'interface_preferences': service.store.interface_preferences(user),
                    'chats': [{'chat': chat, 'turns': service.store.history(user, chat['id'])}
                              for chat in service.store.chats(user)]}

    @app.delete('/accounts/me')
    def delete_me(request: Request):
        with service.lock:
            service.delete_account(owner(request))
            response = JSONResponse({'deleted': True})
            response.delete_cookie(COOKIE, path='/accounts')
            return response

    return app
