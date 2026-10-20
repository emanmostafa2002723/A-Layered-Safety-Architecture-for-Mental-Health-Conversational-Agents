"""Trusted component descriptions, never constructed from candidate/evidence data.

Only JSON-safe configuration crosses the process boundary. Live SDK state,
closures and custom serialization hooks require an explicit worker factory.
"""
from dataclasses import dataclass
import importlib
import json
import re
from baseline.contracts import _json_safe, parse_json_object

MAX_SPEC_BYTES = 262144
MAX_ARGS_BYTES = 4194304
COMPONENT_KEY = '__guardrail_component_factory__'
NAME_PATTERN = r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*'

@dataclass(frozen=True)
class WorkerSpec:
    module: str
    qualified_class: str
    arguments_json: str = '{}'

    def __post_init__(self):
        if (type(self.module) is not str or type(self.qualified_class) is not str
                or not re.fullmatch(NAME_PATTERN, self.module)
                or not re.fullmatch(NAME_PATTERN, self.qualified_class)
                or self.module == '__main__'):
            raise ValueError('An importable trusted worker class is required')
        parse_json_object(self.arguments_json, MAX_SPEC_BYTES)

def _encode(value, depth=0):
    if depth > 12:
        raise ValueError('Component configuration is too deep')
    if value is None or type(value) in (str, bool, int, float):
        if not _json_safe(value):
            raise ValueError('Invalid component value')
        return value
    if type(value) is list:
        if len(value) > 256:
            raise ValueError('Too many component values')
        return [_encode(v, depth + 1) for v in value]
    if type(value) is dict:
        if len(value) > 64 or COMPONENT_KEY in value or any(type(k) is not str for k in value):
            raise ValueError('Invalid component mapping')
        return {k: _encode(v, depth + 1) for k, v in value.items()}
    spec = as_worker_spec(value, depth + 1)
    return {COMPONENT_KEY: [spec.module, spec.qualified_class, spec.arguments_json]}

def as_worker_spec(component, depth=0):
    if type(component) is WorkerSpec:
        return WorkerSpec(component.module, component.qualified_class, component.arguments_json)
    cls = type(component)
    # Inspect class dictionaries without invoking user attribute/serialization hooks.
    for ancestor in type.__getattribute__(cls, '__mro__'):
        attributes = type.__getattribute__(ancestor, '__dict__')
        for name in ('__reduce__', '__reduce_ex__', '__getstate__', '__getattribute__', '__getattr__'):
            if name in attributes and ancestor is not object:
                raise ValueError('Custom parent-side component hooks are unsupported')
    module = type.__getattribute__(cls, '__module__')
    qualified = type.__getattribute__(cls, '__qualname__')
    # Reject local classes/functions before touching instance configuration.
    if module == '__main__' or '<locals>' in qualified or depth > 12:
        raise ValueError('An explicit top-level worker factory is required')
    attributes = object.__getattribute__(component, '__dict__')
    if type(attributes) is not dict or len(attributes) > 64:
        raise ValueError('An explicit worker factory is required')
    arguments = {key: _encode(value, depth + 1) for key, value in attributes.items()}
    return WorkerSpec(module, qualified, json.dumps(arguments, ensure_ascii=False, allow_nan=False))

def _restore(value, depth=0):
    if depth > 12:
        raise ValueError('Worker configuration is too deep')
    if type(value) is dict and set(value) == {COMPONENT_KEY}:
        fields = value[COMPONENT_KEY]
        if type(fields) is not list or len(fields) != 3:
            raise ValueError('Invalid nested worker factory')
        return construct_worker(WorkerSpec(*fields), depth + 1)
    if type(value) is dict:
        return {key: _restore(v, depth + 1) for key, v in value.items()}
    if type(value) is list:
        return [_restore(v, depth + 1) for v in value]
    return value

def construct_worker(spec, depth=0):
    """Called inside the isolated process; imports/constructors share its deadline."""
    factory = importlib.import_module(spec.module)
    for name in spec.qualified_class.split('.'):
        factory = getattr(factory, name)
    if not isinstance(factory, type):
        raise ValueError('The worker factory must be a class')
    arguments = parse_json_object(spec.arguments_json, MAX_SPEC_BYTES)
    return factory(**{key: _restore(value, depth + 1) for key, value in arguments.items()})

def encode_call_arguments(args):
    if type(args) not in (tuple, list):
        raise ValueError('Invalid worker arguments')
    value = {'args': list(args)}
    if not _json_safe(value):
        raise ValueError('Only strict JSON call arguments are allowed')
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(raw.encode('utf-8')) > MAX_ARGS_BYTES:
        raise ValueError('Worker call is too large')
    return raw
