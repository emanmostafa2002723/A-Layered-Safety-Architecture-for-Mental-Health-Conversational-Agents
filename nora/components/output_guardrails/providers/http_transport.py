"""Picklable Chat Completions HTTP adapter. Clients are created inside the worker.

HTTP wire format: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
Works with a configured compatible service; no default endpoint or model.
"""
from dataclasses import dataclass
import http.client
import json
import math
import os
import re
from time import monotonic
from urllib.parse import urlsplit
from baseline.contracts import parse_json_object

MAX_HTTP_BYTES = 131072

@dataclass(frozen=True)
class HTTPConfig:
    endpoint: str
    model_id: str
    api_key_env: object = None
    structured_output: bool = True
    allow_loopback_http: bool = False

    def __post_init__(self):
        if type(self.endpoint) is not str or len(self.endpoint) > 2048:
            raise ValueError('Invalid provider endpoint')
        url = urlsplit(self.endpoint)
        if (not url.hostname or url.username is not None or url.password is not None
                or url.query or url.fragment or not url.path or any(c.isspace() for c in self.endpoint)):
            raise ValueError('Invalid provider endpoint')
        if type(self.allow_loopback_http) is not bool or type(self.structured_output) is not bool:
            raise ValueError('Invalid provider flags')
        if url.scheme != 'https' and not (url.scheme == 'http' and self.allow_loopback_http
                and url.hostname in {'127.0.0.1', '::1', 'localhost'}):
            raise ValueError('HTTPS is required outside explicit loopback testing')
        if type(self.model_id) is not str or not self.model_id.strip() or len(self.model_id) > 128:
            raise ValueError('Invalid provider model')
        if self.api_key_env is not None and (type(self.api_key_env) is not str
                or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}', self.api_key_env)):
            raise ValueError('Use an environment variable name for credentials')
        url.port  # Validate a malformed/out-of-range port before any request.

class ChatCompletionsTransport:
    def __init__(self, config, response_schema, schema_name):
        if type(config) is not HTTPConfig or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', schema_name):
            raise ValueError('Invalid transport configuration')
        self.config, self.response_schema, self.schema_name = config, response_schema, schema_name

    def complete(self, messages, timeout_seconds):
        connection = None
        try:
            if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValueError('Invalid request deadline')
            started = monotonic()
            def remaining():
                budget = timeout_seconds - (monotonic() - started)
                if budget <= 0:
                    raise TimeoutError('Provider deadline exceeded')
                return budget
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
            if self.config.api_key_env is not None:
                key = os.environ.get(self.config.api_key_env)
                if not key or any(c in key for c in '\r\n'):
                    raise ValueError('Provider credential unavailable')
                headers['Authorization'] = 'Bearer ' + key
            formatting = {'type': 'json_schema', 'json_schema': {'name': self.schema_name,
                'strict': True, 'schema': self.response_schema}} if self.config.structured_output else {'type': 'json_object'}
            payload = json.dumps({'model': self.config.model_id, 'messages': messages,
                'stream': False, 'response_format': formatting}, allow_nan=False).encode('utf-8')
            if len(payload) > 1048576:
                raise ValueError('Provider request too large')
            url = urlsplit(self.config.endpoint)
            client = http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection
            connection = client(url.hostname, url.port, timeout=remaining())
            connection.request('POST', url.path, body=payload, headers=headers)
            request_socket = connection.sock
            request_socket.settimeout(remaining())
            response = connection.getresponse()
            if response.status != 200:
                # Redirects are never followed, and provider error bodies are never logged.
                raise ValueError('Provider request failed')
            body = bytearray()
            while True:
                # Keep the socket reference even when a Connection: close response
                # detaches it from HTTPConnection; read1 must use the remaining budget.
                request_socket.settimeout(remaining())
                chunk = response.read1(min(16384, MAX_HTTP_BYTES + 1 - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > MAX_HTTP_BYTES:
                    raise ValueError('Provider response too large')
                if response.isclosed():
                    break
            remaining()
            value = parse_json_object(body.decode('utf-8'), MAX_HTTP_BYTES)
            choices = value.get('choices')
            if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
                raise ValueError('Invalid provider response')
            choice, message = choices[0], choices[0].get('message')
            if (choice.get('finish_reason') != 'stop' or type(message) is not dict
                    or message.get('role') != 'assistant' or message.get('refusal')
                    or message.get('tool_calls') or type(message.get('content')) is not str):
                raise ValueError('Incomplete provider response')
            return message['content']
        except TimeoutError:
            raise TimeoutError('Provider deadline exceeded') from None
        except Exception:
            raise ValueError('Provider execution failed') from None
        finally:
            if connection is not None:
                connection.close()
