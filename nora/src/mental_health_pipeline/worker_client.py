"""Persistent local JSON-line workers for dependency-isolated model runtimes."""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import uuid
from typing import Any, Mapping, Sequence


class WorkerError(RuntimeError):
    """A worker could not satisfy the versioned local-process contract."""


class WorkerTimeout(WorkerError):
    pass


@dataclass(frozen=True)
class WorkerConfig:
    component: str
    python_executable: Path
    module: str
    arguments: tuple[str, ...]
    source_root: Path
    startup_timeout_seconds: float = 180.0
    request_timeout_seconds: float = 30.0
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Preserve the final POSIX venv symlink: executing its resolved base
        # interpreter would lose the venv's pyvenv.cfg and installed packages.
        executable = Path(self.python_executable).expanduser()
        object.__setattr__(self, "python_executable", executable.parent.resolve() / executable.name)
        object.__setattr__(self, "source_root", Path(self.source_root).resolve())
        if self.component not in {"crisis", "boundary", "output"}:
            raise ValueError("unsupported worker component")
        if not self.module.startswith("mental_health_pipeline.workers."):
            raise ValueError("worker module must be inside the trusted workers package")
        if self.startup_timeout_seconds <= 0 or self.request_timeout_seconds <= 0:
            raise ValueError("worker timeouts must be positive")
        if not all(
            isinstance(key, str) and key and isinstance(value, str)
            for key, value in self.environment.items()
        ):
            raise ValueError("worker environment must contain string keys and values")


class JsonLineWorker:
    """One-at-a-time client for a persistent model process.

    Standard output is reserved for JSON protocol messages. Standard error is
    drained and discarded to avoid deadlock and accidental sensitive logging.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[object] = queue.Queue()
        self._lock = threading.RLock()
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        atexit.register(self.close)

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def readiness(self) -> dict[str, Any]:
        issues: list[str] = []
        if not self.config.python_executable.is_file():
            issues.append("python_executable_missing")
        if not self.config.source_root.is_dir():
            issues.append("pipeline_source_root_missing")
        return {
            "component": self.config.component,
            "ready": not issues,
            "running": self.running,
            "issues": issues,
        }

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            ready = self.readiness()
            if not ready["ready"]:
                raise WorkerError(
                    f"{self.config.component} worker not ready: {','.join(ready['issues'])}"
                )
            env = os.environ.copy()
            env.update(self.config.environment)
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(self.config.source_root) + (
                os.pathsep + existing if existing else ""
            )
            command = [
                str(self.config.python_executable),
                "-m",
                self.config.module,
                *self.config.arguments,
            ]
            # A previous worker may have left an EOF/error sentinel in the
            # queue.  Each process gets a fresh protocol channel so a clean
            # restart cannot consume stale state from its predecessor.
            self._messages = queue.Queue()
            protocol_queue = self._messages
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
            self._reader = threading.Thread(
                target=self._read_stdout,
                args=(protocol_queue,),
                name=f"{self.config.component}-worker-stdout",
                daemon=True,
            )
            self._stderr_reader = threading.Thread(
                target=self._drain_stderr,
                name=f"{self.config.component}-worker-stderr",
                daemon=True,
            )
            self._reader.start()
            self._stderr_reader.start()
            message = self._next_message(self.config.startup_timeout_seconds)
            if not isinstance(message, Mapping) or message.get("schema_version") != "worker-ready/1.0":
                self._terminate_locked()
                raise WorkerError(f"{self.config.component} worker sent an invalid startup response")
            if message.get("status") != "ready" or message.get("component") != self.config.component:
                self._terminate_locked()
                raise WorkerError(f"{self.config.component} worker failed startup")

    def request(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            self.start()
            process = self._process
            if process is None or process.stdin is None:
                raise WorkerError("worker stdin is unavailable")
            request_id = str(uuid.uuid4())
            message = {
                "schema_version": "worker-request/1.0",
                "request_id": request_id,
                "operation": operation,
                "payload": dict(payload),
            }
            try:
                process.stdin.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError, UnicodeError) as error:
                self._terminate_locked()
                raise WorkerError(f"{self.config.component} worker communication failed") from error
            response = self._next_message(self.config.request_timeout_seconds)
            if not isinstance(response, Mapping):
                self._terminate_locked()
                raise WorkerError("worker returned a non-object response")
            if (
                response.get("schema_version") != "worker-response/1.0"
                or response.get("request_id") != request_id
            ):
                self._terminate_locked()
                raise WorkerError("worker response identity mismatch")
            if response.get("status") != "ok" or not isinstance(response.get("result"), Mapping):
                raise WorkerError(f"{self.config.component} worker could not complete the request")
            return response["result"]

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                return
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write(json.dumps({
                        "schema_version": "worker-request/1.0",
                        "request_id": str(uuid.uuid4()),
                        "operation": "shutdown",
                        "payload": {},
                    }) + "\n")
                    process.stdin.flush()
                    process.wait(timeout=3)
                except Exception:
                    pass
            self._terminate_locked()

    def _next_message(self, timeout: float) -> object:
        try:
            item = self._messages.get(timeout=timeout)
        except queue.Empty as error:
            self._terminate_locked()
            raise WorkerTimeout(f"{self.config.component} worker timed out") from error
        if isinstance(item, BaseException):
            self._terminate_locked()
            raise WorkerError(f"{self.config.component} worker protocol ended") from item
        return item

    def _read_stdout(self, messages: queue.Queue[object]) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as error:
                    messages.put(error)
                    return
                messages.put(parsed)
        except BaseException as error:
            messages.put(error)
        finally:
            messages.put(EOFError("worker stdout closed"))

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            for _line in process.stderr:
                # Deliberately discard. Third-party errors can contain model
                # paths or, during inference, sensitive input fragments.
                pass
        except Exception:
            return

    def _terminate_locked(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


class SubprocessCrisisGuardrail:
    version = "crisis-worker-client/1.0"

    def __init__(self, worker: JsonLineWorker) -> None:
        if worker.config.component != "crisis":
            raise ValueError("crisis client requires a crisis worker")
        self.worker = worker

    def assess(self, latest_user_message: str) -> Mapping[str, Any]:
        return self.worker.request("assess", {"latest_user_message": latest_user_message})

    def readiness(self) -> dict[str, Any]:
        return self.worker.readiness()


class SubprocessBoundaryGuardrail:
    version = "boundary-worker-client/1.0"

    def __init__(self, worker: JsonLineWorker) -> None:
        if worker.config.component != "boundary":
            raise ValueError("boundary client requires a boundary worker")
        self.worker = worker

    def predict(self, latest_user_message: str) -> Mapping[str, Any]:
        return self.worker.request("predict", {"latest_user_message": latest_user_message})

    def readiness(self) -> dict[str, Any]:
        return self.worker.readiness()
