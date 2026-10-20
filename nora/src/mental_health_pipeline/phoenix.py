"""Portable bridge to the supplied Phoenix v3 retrieval implementation.

Safety routes still belong to the application. The handoff's retrieval-needed
decision, child evidence, topic boost and source diversity are preserved.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib
import io
import logging
import re
from pathlib import Path
import sys
import threading
import time
from types import ModuleType

from .rag import HybridRAGBackend, HybridRAGConfig, RetrievalReadiness, check_retrieval_cancelled
from .qwen_router import QwenRouterError


class PhoenixRAGBackend:
    def __init__(self, project: str | Path, config: HybridRAGConfig, *, router=None):
        self.project = Path(project).resolve()
        self.config = config
        self._retriever = None
        self._lock = threading.Lock()
        self._package = None
        self.router = router

    def _module(self, name):
        if self._package is None:
            identifier = "_mhp_phoenix_" + hashlib.sha256(str(self.project).encode()).hexdigest()[:16]
            package = sys.modules.get(identifier)
            if package is None:
                package = ModuleType(identifier)
                package.__path__ = [str(self.project / "scripts")]
                sys.modules[identifier] = package
            self._package = identifier
        return importlib.import_module(f"{self._package}.{name}")

    def readiness(self) -> RetrievalReadiness:
        base = HybridRAGBackend(self.config).readiness()
        errors = list(base.errors)
        for name in ("retriever_v8.py", "retrieval_router.py"):
            if not (self.project / "scripts" / name).is_file():
                errors.append(f"phoenix_script_missing:{name}")
        if self.router is not None:
            error = self.router.readiness_code()
            if error:
                errors.append(error)
        return RetrievalReadiness(not errors, self._retriever is not None, tuple(errors), base.warnings)

    def retrieve(self, query: str, top_k: int = 3):
        return self.retrieve_cancellable(query, top_k)

    def retrieve_cancellable(self, query: str, top_k: int = 3, *, cancel_event=None):
        check_retrieval_cancelled(cancel_event)
        if not isinstance(query, str) or not query.strip() or top_k < 1:
            raise ValueError("non-empty query and positive top_k required")
        started = time.monotonic()
        trace = {"router_backend": "legacy", "router_status": "disabled", "router_latency_ms": 0.0}
        if self.router is not None:
            try:
                decision = self.router.route(query)
                check_retrieval_cancelled(cancel_event)
                backend_name=getattr(self.router,'backend_name','qwen_v4')
                trace = {"router_backend": backend_name, "router_status": "ok",
                         "router_decision": decision.decision, "reason_code": decision.reason_code,
                         "router_latency_ms": decision.latency_ms}
                if getattr(decision,'application_override',None):
                    trace.update(application_override=decision.application_override,raw_model_decision=decision.raw_decision)
                if not decision.requires_retrieval:
                    return {"parent_chunks": [], "retrieval_required": False,
                            "routing_reason": ("qwen_no_retrieve:" if backend_name=='qwen_v4' else backend_name+'_no_retrieve:') + decision.reason_code,
                            "routing_diagnostics": {**trace, "retrieval_latency_ms": 0.0,
                                "total_latency_ms": (time.monotonic() - started) * 1000}}
                from .query_semantics import preserve_query_meaning
                search_query,reasons=preserve_query_meaning(query,decision.canonical_query)
                if reasons:
                    trace.update(query_integrity='original_query_restored',query_integrity_reasons=list(reasons))
                return self._retrieve_documents(search_query, top_k,
                                                trace=trace, started=started, force_retrieval=True, cancel_event=cancel_event)
            except QwenRouterError as error:
                # A router outage is not a safety-classifier outage. Preserve the
                # existing conservative retrieval fallback and make it observable.
                trace = {"router_backend": getattr(self.router,'backend_name','qwen_v4'), "router_status": "fallback",
                         "error_code": error.code,
                         "router_latency_ms": (time.monotonic() - started) * 1000}
                logging.getLogger(__name__).warning("RAG router fallback: %s", error.code)
        check_retrieval_cancelled(cancel_event)
        result = self._legacy_retrieve(query, top_k, trace=trace, started=started, cancel_event=cancel_event)
        return result

    def _legacy_retrieve(self, query, top_k, *, trace, started, cancel_event=None):
        text = " ".join(query.casefold().split())
        # A corpus cannot answer questions about this chat or the user's own
        # update. Avoid injecting unrelated clinical excerpts into those turns.
        memory = re.search(r"\b(?:what did i|who did i|what have i told|do you remember|please remember)\b", text)
        knowledge = re.search(r"\b(?:what is|what are|how does|how do|why does|why do|explain|define|general information|sleep hygiene)\b", text)
        support = (re.search(r"\b(?:someone to listen|need someone to listen|that helped|helped a little|i moved away|she is on the phone|is here now|i can see|please suggest something else|can we talk)\b", text)
                   or ("?" not in text and re.match(r"(?:i (?:feel|have been feeling|tried)|my manager criticized|it is the exam)\b", text)))
        if memory or (support and not knowledge):
            return {"parent_chunks": [], "retrieval_required": False,
                    "routing_intent": "conversation", "routing_reason": "conversation_context_not_corpus",
                    "routing_diagnostics": {**trace, "retrieval_latency_ms": 0.0,
                        "total_latency_ms": (time.monotonic() - started) * 1000}}
        # Social/identity/live-data requests skip even the lazy retriever load.
        decision = self._module("retrieval_router").route_retrieval(query)
        if not decision.requires_retrieval:
            return {"parent_chunks": [], "retrieval_required": False,
                    "routing_intent": decision.intent, "routing_reason": decision.reason,
                    "routing_diagnostics": {**trace, "retrieval_latency_ms": 0.0,
                        "total_latency_ms": (time.monotonic() - started) * 1000}}
        return self._retrieve_documents(query, top_k, trace=trace, started=started, cancel_event=cancel_event)

    def _retrieve_documents(self, query, top_k, *, trace, started, force_retrieval=False, cancel_event=None):
        retrieval_started = time.monotonic()
        while not self._lock.acquire(timeout=0.1):
            check_retrieval_cancelled(cancel_event)
        try:
            check_retrieval_cancelled(cancel_event)
            if self._retriever is None:
                module = self._module("retriever_v8")
                # The supplied loader prints status; no stdout or user text is
                # emitted by the integrated API retrieval adapter.
                with redirect_stdout(io.StringIO()):
                    self._retriever = module.Retriever(
                        processed_dir=self.config.processed_dir,
                        embedding_model=self.config.embedding_model,
                        reranker_model=self.config.reranker_model,
                        embedding_device=self.config.embedding_device,
                        reranker_device=self.config.reranker_device,
                        enable_dense=self.config.enable_dense,
                        enable_reranker=self.config.enable_reranker,
                        local_files_only=not self.config.allow_network,
                    )
            check_retrieval_cancelled(cancel_event)
            result = self._retriever.retrieve(query, top_k=top_k, force_retrieval=force_retrieval)
            check_retrieval_cancelled(cancel_event)
            children = {item["chunk_id"]: item for item in self._retriever.child_chunks}
            # Phoenix v3 builds evidence from matched children instead of entire
            # multi-page parents. Retain the parent metadata and citation IDs.
            for item in result["parent_chunks"]:
                child = children.get(item.get("child_chunk_id"))
                if child and child.get("text"):
                    item["parent_chunk"] = {**item["parent_chunk"], "text": child["text"]}
            result["retrieval_query"] = query
            result["routing_diagnostics"] = {**trace,
                "retrieval_latency_ms": (time.monotonic() - retrieval_started) * 1000,
                "total_latency_ms": (time.monotonic() - started) * 1000}
            return result
        finally:
            self._lock.release()
