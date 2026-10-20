"""Bounded, best-effort speculative retrieval. Never authorizes generation.

A running HTTP request or tensor kernel cannot be forcibly cancelled safely.
Cancellation means cooperative checkpoints and discarding its result, not a
claim that the underlying computation stopped instantly.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import threading
import time


@dataclass
class RetrievalTicket:
    future: object
    cancel_event: threading.Event
    started: float

    def discard(self, reason):
        self.cancel_event.set()
        cancelled = self.future.cancel()
        return {'status':'discarded', 'reason':reason, 'cancel_requested':True,
                'cancelled_before_start':cancelled,
                'work_still_running':not self.future.done(),
                'elapsed_seconds':time.perf_counter()-self.started}


class SpeculativeRetriever:
    """One in-flight retrieval per pipeline, no unbounded speculative queue."""
    def __init__(self):
        self._slot = threading.BoundedSemaphore(1)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rag-speculation')
        self._active_event = None

    def start(self, rag, text):
        if not self._slot.acquire(blocking=False): return None
        started=time.perf_counter()
        event=threading.Event()
        self._active_event = event
        try:
            future=self._pool.submit(rag.retrieve, text, 'normal_chat', cancel_event=event)
        except BaseException:
            self._slot.release(); raise
        future.add_done_callback(lambda _:self._slot.release())
        return RetrievalTicket(future,event,started)

    @staticmethod
    def resolve(ticket, rag, text, route, *, timeout):
        from .rag import RetrievalMode
        mode=rag.policy.mode_for(route)
        if mode is RetrievalMode.BYPASS:
            return rag.retrieve(text,route),ticket.discard('resolved_route_bypasses_rag')
        if mode is not RetrievalMode.STANDARD:
            # Different top-k/topic limits: never reuse unrestricted evidence.
            info=ticket.discard('resolved_route_requires_restricted_retrieval')
            return rag.retrieve(text,route),info
        started=time.perf_counter()
        try:
            result=ticket.future.result(timeout=timeout)
            if result.query != text or result.safety_route != route:
                raise ValueError('speculative result identity mismatch')
            return result,{'status':'accepted','wait_after_policy_seconds':time.perf_counter()-started,
                           'elapsed_seconds':time.perf_counter()-ticket.started,'cancel_requested':False}
        except TimeoutError:
            return None,ticket.discard('speculative_wait_timeout')
        except Exception as exc:
            return None,{'status':'failed','error_type':type(exc).__name__}

    def close(self):
        # Active operations finish cooperatively within backend deadlines.
        if self._active_event is not None: self._active_event.set()
        self._pool.shutdown(wait=False,cancel_futures=True)
