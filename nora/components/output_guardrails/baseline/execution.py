"""Process boundary for external checker/rewriter calls. No shell or network calls."""
import json
import math
import multiprocessing
from threading import Event, Thread
from time import monotonic
from baseline.contracts import checker_result, validate_checker_result, valid_individual_findings, parse_json_object
from baseline.worker_spec import WorkerSpec, as_worker_spec, construct_worker, encode_call_arguments, MAX_ARGS_BYTES
from config.fusion_policy import execution_health

MAX_RESULT_BYTES = 1048576

class ExecutionFailure(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__('Bounded operation failed')

def _worker(connection, module, qualified_class, arguments_json, method, call_json, expected_checker):
    try:
        target = construct_worker(WorkerSpec(module, qualified_class, arguments_json))
        args = parse_json_object(call_json, MAX_ARGS_BYTES)['args']
        value = getattr(target, method)(*args)
        if expected_checker is not None and not validate_checker_result(value, expected_checker):
            value = checker_result(expected_checker, 'ERROR', valid_individual_findings(value, expected_checker),
                {'error': 'MALFORMED_ISOLATED_RESULT'})
        packet = json.dumps({'value': value}, allow_nan=False).encode('utf-8')
        if len(packet) > MAX_RESULT_BYTES:
            packet = b'{"error":"ERROR"}'
    except TimeoutError:
        packet = b'{"error":"TIMEOUT"}'
    except BaseException:
        # Never serialize exception details, raw prompts, or tracebacks.
        packet = b'{"error":"ERROR"}'
    try:
        connection.send_bytes(packet)
    except Exception:
        pass
    finally:
        connection.close()

def bounded_invoke(target, method, args, timeout_seconds, *, expected_checker=None):
    started = monotonic()
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise ExecutionFailure('ERROR')
    try:
        if (type(method) is not str or method not in {'check', 'rewrite', 'complete'}
                or expected_checker is not None and (type(expected_checker) is not str or not 0 < len(expected_checker) <= 256)):
            raise ValueError('Unsupported worker method')
        spec = as_worker_spec(target)
        call_json = encode_call_arguments(args)
    except Exception:
        raise ExecutionFailure('ERROR') from None
    if monotonic() - started > timeout_seconds:
        raise ExecutionFailure('TIMEOUT')
    try:
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe(duplex=False)
    except Exception:
        raise ExecutionFailure('ERROR') from None
    try:
        process = context.Process(target=_worker,
            args=(child, spec.module, spec.qualified_class, spec.arguments_json,
                method, call_json, expected_checker), daemon=True)
    except Exception:
        parent.close()
        child.close()
        raise ExecutionFailure('ERROR') from None
    received, completed = {}, Event()
    reader = None
    def receive():
        try:
            received['packet'] = parse_json_object(parent.recv_bytes(MAX_RESULT_BYTES).decode('utf-8'), MAX_RESULT_BYTES)
        except Exception:
            received['failed'] = True
        finally:
            completed.set()
    try:
        process.start()
        child.close()
        # poll() only promises some bytes, not a complete framed message. Keep the
        # full receive/decode under a watchdog, including a stalled message body.
        reader = Thread(target=receive, name='guardrail-result-reader', daemon=True)
        reader.start()
        remaining = max(0.0, timeout_seconds - (monotonic() - started))
        if not completed.wait(remaining):
            raise ExecutionFailure('TIMEOUT')
        if monotonic() - started > timeout_seconds:
            raise ExecutionFailure('TIMEOUT')
        if received.get('failed'):
            raise ExecutionFailure('ERROR')
        packet = received['packet']
        if 'error' in packet:
            status = packet['error']
            raise ExecutionFailure(status if type(status) is str and status in {'ERROR', 'TIMEOUT'} else 'ERROR')
        return packet['value']
    except ExecutionFailure:
        raise
    except Exception:
        raise ExecutionFailure('ERROR') from None
    finally:
        child.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(0.1)
            if process.is_alive():
                process.kill()
                process.join(0.1)
            if not process.is_alive():
                process.close()
        # Killing the writer releases a reader waiting for a partial frame.
        if reader is not None:
            reader.join(0.2)
        parent.close()

class BoundedChecker:
    """Wrap a trusted factory; no arbitrary object is pickled in the caller."""
    def __init__(self, checker, name, timeout_seconds):
        self.target, self.name, self.timeout_seconds = checker, name, timeout_seconds

    def check(self, *args):
        started = monotonic()
        try:
            result = bounded_invoke(self.target, 'check', args, self.timeout_seconds, expected_checker=self.name)
            if not validate_checker_result(result, self.name):
                return checker_result(self.name, 'ERROR', valid_individual_findings(result, self.name),
                    {'error': 'MALFORMED_ISOLATED_RESULT'})
            if monotonic() - started > self.timeout_seconds:
                return checker_result(self.name, execution_health([result['status'], 'TIMEOUT']), result['violations'],
                    {'error': 'RESULT_DEADLINE_EXCEEDED'})
            return result
        except ExecutionFailure as exc:
            return checker_result(self.name, exc.status, details={'error': 'ISOLATED_EXECUTION_FAILED'})


class DeadlineChecker(BoundedChecker):
    """Recompute remaining request time at invocation, after previous checkers."""
    def __init__(self, checker, name, timeout_seconds, deadline):
        super().__init__(checker, name, timeout_seconds)
        self.deadline = deadline

    def check(self, *args):
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            return checker_result(self.name, 'TIMEOUT', details={'error':'REQUEST_DEADLINE_EXCEEDED'})
        return BoundedChecker(self.target, self.name, min(self.timeout_seconds, remaining)).check(*args)
