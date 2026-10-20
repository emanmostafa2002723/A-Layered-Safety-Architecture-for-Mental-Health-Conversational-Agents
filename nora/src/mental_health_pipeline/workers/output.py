"""Isolated standalone output framework. All missing/unapproved providers abstain."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from mental_health_pipeline.workers.common import announce_ready, announce_startup_error, _send


def build_application(root, provider_config):
    sys.path.insert(0, str(root))
    import integration
    Path(integration.__file__).resolve().relative_to(root)
    from integration.service import GuardrailService, ProviderSpec
    from integration.application import GuardedApplication
    from providers.http_transport import HTTPConfig
    settings = json.loads(provider_config.read_text(encoding='utf-8'))
    def provider(role):
        value = settings.get(role)
        # Explicitly refrain from initializing/calling unapproved providers.
        if value is None or value.get('deployment_approved') is not True:
            return None
        if settings.get('data_transmission_authorized') is not True:
            raise ValueError('Provider transmission has not been authorized')
        return ProviderSpec(HTTPConfig(endpoint=value['endpoint'], model_id=value['model_id'],
            api_key_env=value.get('api_key_env'), structured_output=value.get('structured_output', True),
            allow_loopback_http=value.get('allow_loopback_http', False)),
            value['model_revision'], deployment_approved=True)
    return GuardedApplication(GuardrailService(semantic=provider('semantic'), factuality=provider('factuality')))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--component-root', type=Path, required=True)
    p.add_argument('--provider-config', type=Path, required=True)
    a = p.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            app = build_application(a.component_root.resolve(), a.provider_config.resolve())
    except Exception:
        announce_startup_error('output')
        return 2
    announce_ready('output')
    while True:
        line = sys.stdin.buffer.readline(262145)
        if not line:
            return 0
        request_id = 'unknown'
        try:
            if len(line) > 262144 or not line.endswith(b'\n'):
                return 2
            message = json.loads(line)
            request_id = message.get('request_id')
            if message.get('schema_version') != 'worker-request/1.0' or not isinstance(request_id, str) or not request_id:
                raise ValueError('Invalid envelope')
            if message.get('operation') == 'shutdown':
                return 0
            payload = message['payload']
            if message.get('operation') != 'deliver' or set(payload) != {'frozen_input'}:
                raise ValueError('Invalid request')
            displayed = []
            with redirect_stdout(sys.stderr):
                result = app.deliver(payload['frozen_input'], displayed.append)
            text = displayed[0] if len(displayed) == 1 else None
            if result.action != 'DISPLAY' or text != result.text:
                text = None
            _send({'schema_version': 'worker-response/1.0', 'request_id': request_id, 'status': 'ok',
                'result': {'action': 'DISPLAY' if text is not None else 'ABSTAIN', 'text': text, 'route': result.route}})
        except Exception:
            _send({'schema_version': 'worker-response/1.0', 'request_id': request_id,
                'status': 'error', 'error_code': 'output_gate_request_failed'})


if __name__ == '__main__':
    raise SystemExit(main())
