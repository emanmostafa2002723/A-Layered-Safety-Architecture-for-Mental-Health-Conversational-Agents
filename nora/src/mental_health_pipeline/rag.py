"""Retrieval adapters for the deployable mental-health pipeline.

This module intentionally has no import-time dependency on FAISS, NumPy,
``sentence-transformers`` or ``rank_bm25``.  A service can therefore perform
configuration checks, expose a health endpoint, and run unit tests without
loading model weights or native libraries.

Safety routing belongs to the input guardrail.  The old retriever's
``NORMAL``/``MONITORED`` value is deliberately ignored: retrieval receives an
already-resolved safety route and may only operate within the corresponding
policy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import threading
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable


class RAGError(RuntimeError):
    """Base class for retrieval failures."""


class RAGReadinessError(RAGError):
    """Raised when configured retrieval artifacts cannot be used."""


class ArtifactIntegrityError(RAGReadinessError):
    """Raised when an artifact does not match its pinned digest."""


class RetrievalMode(str, Enum):
    """How much knowledge-base access a resolved route is allowed."""

    BYPASS = "bypass"
    STANDARD = "standard"
    RESTRICTED = "restricted"


class RetrievalCancelled(RAGError):
    """Cooperative cancellation; no retrieval content may be adopted."""


def check_retrieval_cancelled(event):
    if event is not None and event.is_set():
        raise RetrievalCancelled('retrieval_cancelled')


def _normalise_route(route: object) -> str:
    value = getattr(route, "value", route)
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


DEFAULT_ROUTE_MODES: Mapping[str, RetrievalMode] = MappingProxyType(
    {
        # Boundary route.
        "allow": RetrievalMode.STANDARD,
        # Crisis tier.
        "no_crisis": RetrievalMode.STANDARD,
        "normal": RetrievalMode.STANDARD,
        # Monitored messages may use a smaller, optionally topic-filtered set.
        "monitored": RetrievalMode.RESTRICTED,
        # These routes must use deterministic response policy/resources rather
        # than allowing corpus text to steer an urgent response.
        "crisis": RetrievalMode.BYPASS,
        "urgent": RetrievalMode.BYPASS,
        "urgent_medical_escalation": RetrievalMode.BYPASS,
        "safe_completion": RetrievalMode.BYPASS,
        "abstain": RetrievalMode.BYPASS,
        "review": RetrievalMode.BYPASS,
        "error": RetrievalMode.BYPASS,
        # Application-level SystemRoute values.
        "normal_chat": RetrievalMode.STANDARD,
        "monitored_support": RetrievalMode.RESTRICTED,
        "crisis_response": RetrievalMode.BYPASS,
        "urgent_medical_response": RetrievalMode.BYPASS,
        "clinical_boundary_safe_completion": RetrievalMode.BYPASS,
        "unsupported_scope": RetrievalMode.BYPASS,
    }
)


@dataclass(frozen=True, slots=True)
class RouteRetrievalPolicy:
    """Fail-closed mapping from guardrail route to retrieval permissions."""

    route_modes: Mapping[str, RetrievalMode | str] = field(
        default_factory=lambda: dict(DEFAULT_ROUTE_MODES)
    )
    unknown_route_mode: RetrievalMode = RetrievalMode.BYPASS
    standard_top_k: int = 3
    restricted_top_k: int = 2
    standard_document_char_limit: int = 8_000
    restricted_document_char_limit: int = 3_000
    # An empty tuple means "do not filter by topic".  Deployments can provide
    # a reviewed allow-list without changing retrieval code.
    restricted_topics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.standard_top_k < 1 or self.restricted_top_k < 1:
            raise ValueError("retrieval top_k values must be positive")
        if self.standard_document_char_limit < 1 or self.restricted_document_char_limit < 1:
            raise ValueError("document character limits must be positive")

    def mode_for(self, route: object) -> RetrievalMode:
        raw = self.route_modes.get(_normalise_route(route), self.unknown_route_mode)
        try:
            return raw if isinstance(raw, RetrievalMode) else RetrievalMode(str(raw))
        except ValueError as exc:
            raise ValueError(f"invalid retrieval mode configured for route {route!r}: {raw!r}") from exc


@dataclass(frozen=True, slots=True)
class Citation:
    """Machine-readable citation retained separately from generated prose."""

    source_id: str
    source: str
    title: str
    url: str | None = None
    page: str | None = None
    parent_id: str | None = None
    child_id: str | None = None

    @property
    def citation_id(self) -> str:
        """Readable alias used inside prompt context records."""

        return self.source_id

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "page": self.page,
        }


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    """A single untrusted knowledge-base document selected for generation."""

    text: str
    citation: Citation
    score: float | None = None
    topic: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def with_text_limit(self, maximum: int) -> "RetrievedDocument":
        if len(self.text) <= maximum:
            return self
        return replace(self, text=self.text[:maximum], metadata={**self.metadata, "truncated": True})


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    safety_route: str
    mode: RetrievalMode
    documents: tuple[RetrievedDocument, ...] = ()
    skipped_reason: str | None = None
    # Request-scoped diagnostics; never automatically serialized into the API.
    retrieval_query: str | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    @property
    def citations(self) -> tuple[Citation, ...]:
        return tuple(document.citation for document in self.documents)

    @property
    def was_retrieved(self) -> bool:
        return self.mode is not RetrievalMode.BYPASS and self.skipped_reason is None


@dataclass(frozen=True, slots=True)
class RetrievalReadiness:
    ready: bool
    loaded: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@runtime_checkable
class RetrievalBackend(Protocol):
    """Minimal backend contract, suitable for mocks and alternative stores."""

    def retrieve(self, query: str, top_k: int = 3) -> object:
        """Return documents or an existing-retriever result mapping."""


class RAGService:
    """Safety-route-aware facade around a retrieval backend.

    The backend never gets to decide the safety route.  In particular, a
    ``route`` key returned by the legacy v8 retriever is ignored.
    """

    def __init__(self, backend: RetrievalBackend, policy: RouteRetrievalPolicy | None = None):
        self._backend = backend
        self.policy = policy or RouteRetrievalPolicy()

    def readiness(self) -> RetrievalReadiness:
        probe = getattr(self._backend, "readiness", None)
        if not callable(probe):
            return RetrievalReadiness(
                False, False, ("retrieval backend has no readiness probe",)
            )
        result = probe()
        if not isinstance(result, RetrievalReadiness):
            return RetrievalReadiness(
                False, False, ("retrieval backend returned an invalid readiness result",)
            )
        return result

    def retrieve(self, query: str, safety_route: object, *, cancel_event=None) -> RetrievalResult:
        check_retrieval_cancelled(cancel_event)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        route = _normalise_route(safety_route)
        mode = self.policy.mode_for(route)
        if mode is RetrievalMode.BYPASS:
            return RetrievalResult(
                query=query,
                safety_route=route,
                mode=mode,
                skipped_reason="retrieval_not_permitted_for_route",
            )

        top_k = (
            self.policy.standard_top_k
            if mode is RetrievalMode.STANDARD
            else self.policy.restricted_top_k
        )
        cancellable = getattr(self._backend, 'retrieve_cancellable', None)
        raw_result = (cancellable(query, top_k=top_k, cancel_event=cancel_event)
                      if cancel_event is not None and callable(cancellable)
                      else self._backend.retrieve(query, top_k=top_k))
        check_retrieval_cancelled(cancel_event)
        diagnostics = dict(raw_result.get("routing_diagnostics", {})) if isinstance(raw_result, Mapping) else {}
        retrieval_query = raw_result.get("retrieval_query") if isinstance(raw_result, Mapping) else None
        if isinstance(raw_result, Mapping) and raw_result.get("retrieval_required") is False:
            return RetrievalResult(query=query, safety_route=route, mode=mode,
                skipped_reason=str(raw_result.get("routing_reason") or "retrieval_not_required"),
                retrieval_query=retrieval_query, diagnostics=diagnostics)
        documents = list(_normalise_backend_result(raw_result))
        from .query_semantics import subject_aligned_documents
        aligned=subject_aligned_documents(query,documents)
        if len(aligned)!=len(documents):
            diagnostics['evidence_alignment']={'status':'explicit_subject_filter',
                'before':len(documents),'retained':len(aligned)}
        documents=aligned

        if mode is RetrievalMode.RESTRICTED and self.policy.restricted_topics:
            allowed = {topic.casefold() for topic in self.policy.restricted_topics}
            documents = [
                document
                for document in documents
                if document.topic and document.topic.casefold() in allowed
            ]

        char_limit = (
            self.policy.standard_document_char_limit
            if mode is RetrievalMode.STANDARD
            else self.policy.restricted_document_char_limit
        )
        documents = [document.with_text_limit(char_limit) for document in documents[:top_k]]
        # Citation markers are request-local display IDs, not publisher identity.
        # Filtering must not leave [source-1],[source-3] holes that encourage a
        # valid answer to invent [source-2]. Preserve all provenance metadata.
        documents=[replace(doc,citation=replace(doc.citation,source_id=f'source-{index}'))
                   for index,doc in enumerate(documents,1)]
        return RetrievalResult(
            query=query,
            safety_route=route,
            mode=mode,
            documents=tuple(documents),
            skipped_reason='no_subject_aligned_evidence' if not documents and 'evidence_alignment' in diagnostics else None,
            retrieval_query=retrieval_query, diagnostics=diagnostics,
        )


def _normalise_backend_result(result: object) -> tuple[RetrievedDocument, ...]:
    if isinstance(result, RetrievalResult):
        return result.documents
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        return tuple(_coerce_document(item, index) for index, item in enumerate(result, 1))
    if not isinstance(result, Mapping):
        raise RAGError(f"retrieval backend returned unsupported type: {type(result).__name__}")

    # Existing v8 shape.  The top-level `route` is intentionally not read.
    parents = result.get("parent_chunks", result.get("documents", ()))
    if not isinstance(parents, Sequence) or isinstance(parents, (str, bytes, bytearray)):
        raise RAGError("retrieval backend result must contain a document sequence")
    return tuple(_coerce_document(item, index) for index, item in enumerate(parents, 1))


def _coerce_document(item: object, index: int) -> RetrievedDocument:
    if isinstance(item, RetrievedDocument):
        return item
    if not isinstance(item, Mapping):
        raise RAGError(f"document {index} is not a mapping")

    # v8 nests the parent in `parent_chunk`; a native backend may return it flat.
    parent_value = item.get("parent_chunk", item)
    if not isinstance(parent_value, Mapping):
        raise RAGError(f"document {index} has an invalid parent_chunk")
    text = parent_value.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise RAGError(f"document {index} has no usable text")

    parent_id = _optional_string(parent_value.get("parent_id"))
    child_id = _optional_string(item.get("child_chunk_id") or parent_value.get("chunk_id"))
    citation = Citation(
        source_id=f"source-{index}",
        source=str(parent_value.get("source") or "Unknown source"),
        title=str(parent_value.get("document_title") or parent_value.get("title") or "Untitled"),
        url=_optional_string(parent_value.get("url")),
        page=_optional_string(parent_value.get("page_num") or parent_value.get("page")),
        parent_id=parent_id,
        child_id=child_id,
    )
    score_value = item.get("reranker_score")
    if score_value is None:
        score_value = item.get("score")
    try:
        score = float(score_value) if score_value is not None else None
    except (TypeError, ValueError):
        score = None
    topic = _optional_string(parent_value.get("primary_topic") or parent_value.get("topic"))
    retrievers = item.get("retrievers", ())
    if not isinstance(retrievers, Sequence) or isinstance(retrievers, (str, bytes, bytearray)):
        retrievers = ()
    metadata = {
        "dense_score": item.get("dense_score"),
        "sparse_score": item.get("sparse_score"),
        "retrievers": tuple(str(value) for value in retrievers),
    }
    return RetrievedDocument(text=text, citation=citation, score=score, topic=topic, metadata=metadata)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class BM25Mode(str, Enum):
    """Sparse-index loading strategy."""

    REBUILD_FROM_JSON = "rebuild_from_json"
    VERIFIED_LEGACY_PICKLE = "verified_legacy_pickle"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class HybridRAGConfig:
    """Configuration for the artifacts in the existing ``rag_project``."""

    processed_dir: Path
    embedding_model: str | Path = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str | Path = "BAAI/bge-reranker-large"
    embedding_device: str = "cpu"
    reranker_device: str = "cpu"
    allow_network: bool = False
    enable_dense: bool = True
    enable_sparse: bool = True
    enable_reranker: bool = True
    bm25_mode: BM25Mode = BM25Mode.REBUILD_FROM_JSON
    legacy_bm25_sha256: str | None = None
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rerank_top_k: int = 10
    min_dense_score: float = 0.30
    rrf_constant: int = 60

    def __post_init__(self) -> None:
        object.__setattr__(self, "processed_dir", Path(self.processed_dir).expanduser())
        if not self.embedding_device or not self.reranker_device:
            raise ValueError("embedding_device and reranker_device must be explicit")
        if self.embedding_device == "auto" or self.reranker_device == "auto":
            raise ValueError("device='auto' is not allowed in deployment configuration")
        for name in ("dense_top_k", "sparse_top_k", "rerank_top_k", "rrf_constant"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.bm25_mode is BM25Mode.VERIFIED_LEGACY_PICKLE:
            digest = (self.legacy_bm25_sha256 or "").lower()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("verified legacy BM25 loading requires a 64-character SHA-256")

    @property
    def child_chunks_path(self) -> Path:
        return self.processed_dir / "child_chunks.json"

    @property
    def parent_chunks_path(self) -> Path:
        return self.processed_dir / "parent_chunks.json"

    @property
    def faiss_path(self) -> Path:
        return self.processed_dir / "faiss_index.bin"

    @property
    def bm25_path(self) -> Path:
        return self.processed_dir / "bm25_index.pkl"


class RebuiltBM25Index:
    """Small dependency-free BM25 implementation rebuilt from child JSON."""

    _TOKEN_RE = re.compile(r"\w+", re.UNICODE)

    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.document_count = len(documents)
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequencies: Counter[str] = Counter()

        for document_index, document in enumerate(documents):
            tokens = self.tokenize(document)
            self.lengths.append(len(tokens))
            frequencies = Counter(tokens)
            document_frequencies.update(frequencies.keys())
            for token, frequency in frequencies.items():
                self.postings[token].append((document_index, frequency))

        self.average_length = (
            sum(self.lengths) / self.document_count if self.document_count else 0.0
        )
        self.idf = {
            token: math.log(1.0 + (self.document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequencies.items()
        }

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return cls._TOKEN_RE.findall(text.casefold())

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self.document_count == 0 or self.average_length == 0:
            return []
        scores: defaultdict[int, float] = defaultdict(float)
        for token in self.tokenize(query):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for document_index, term_frequency in self.postings[token]:
                length = self.lengths[document_index]
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length / self.average_length
                )
                scores[document_index] += idf * (
                    term_frequency * (self.k1 + 1.0) / denominator
                )
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_verified_legacy_pickle(path: Path, expected_sha256: str) -> object:
    """Load an explicitly trusted legacy pickle after exact digest verification.

    Pickle is executable.  Consequently there is no "load anything safely"
    mode: callers must pin the digest of an artifact reviewed and distributed
    with the deployment.  New deployments should use ``REBUILD_FROM_JSON``.
    """

    path = Path(path)
    if path.is_symlink():
        raise ArtifactIntegrityError("refusing to load a symlinked legacy pickle")
    if not path.is_file():
        raise RAGReadinessError(f"legacy BM25 artifact is missing: {path}")
    expected = expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ArtifactIntegrityError("a valid pinned SHA-256 is required before unpickling")
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected):
        raise ArtifactIntegrityError(
            f"legacy BM25 SHA-256 mismatch for {path.name}: expected {expected}, got {actual}"
        )
    import pickle  # noqa: S403 - gated by exact allow-listed digest above

    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - see digest gate above


def augment_query(query: str) -> str:
    """Add retrieval vocabulary only; this function never makes a safety decision."""

    lowered = query.casefold()
    additions: list[str] = []
    expansions = (
        (("panic", "anxious", "anxiety", "heart racing"), "panic anxiety grounding breathing"),
        (("hopeless", "depressed", "empty inside"), "depression low mood emotional support"),
        (("trauma", "flashback", "ptsd"), "trauma post-traumatic stress grounding"),
        (("self-harm", "hurt myself", "cut myself"), "self injury coping support"),
    )
    for patterns, expansion in expansions:
        if any(pattern in lowered for pattern in patterns):
            additions.append(expansion)
    return f"{query} {' '.join(additions)}".strip() if additions else query


class HybridRAGBackend:
    """Lazy hybrid backend reading the existing v8 data artifacts.

    It is a replacement adapter around the artifact format, not a modification
    of the research scripts.  Model/device choices are explicit and downloads
    are disabled unless ``allow_network`` is deliberately enabled.
    """

    def __init__(self, config: HybridRAGConfig):
        self.config = config
        self._load_lock = threading.Lock()
        self._retrieval_lock = threading.Lock()
        self._loaded = False
        self._children: list[dict[str, Any]] = []
        self._parents: dict[str, dict[str, Any]] = {}
        self._faiss: object | None = None
        self._embedder: object | None = None
        self._bm25: object | None = None
        self._reranker: object | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def readiness(self) -> RetrievalReadiness:
        errors: list[str] = []
        warnings: list[str] = []
        required = [self.config.child_chunks_path, self.config.parent_chunks_path]
        if self.config.enable_dense:
            required.append(self.config.faiss_path)
        if (
            self.config.enable_sparse
            and self.config.bm25_mode is BM25Mode.VERIFIED_LEGACY_PICKLE
        ):
            required.append(self.config.bm25_path)
        for path in required:
            if not path.is_file():
                errors.append(f"missing retrieval artifact: {path}")
        if self.config.enable_sparse and self.config.bm25_mode is BM25Mode.DISABLED:
            errors.append("sparse retrieval is enabled but BM25 mode is disabled")
        if not self.config.enable_dense and not self.config.enable_sparse:
            errors.append("at least one retrieval strategy must be enabled")
        if self.config.bm25_mode is BM25Mode.VERIFIED_LEGACY_PICKLE:
            warnings.append("legacy pickle mode is enabled; migrate to rebuild_from_json")
        if not self.config.allow_network:
            warnings.append("offline mode: embedding and reranker models must exist locally")
        return RetrievalReadiness(not errors, self._loaded, tuple(errors), tuple(warnings))

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            status = self.readiness()
            if not status.ready:
                raise RAGReadinessError("; ".join(status.errors))

            self._children = _load_json_list(self.config.child_chunks_path, "child chunks")
            parents = _load_json_list(self.config.parent_chunks_path, "parent chunks")
            self._parents = {
                str(parent["parent_id"]): parent
                for parent in parents
                if isinstance(parent, dict) and "parent_id" in parent
            }
            if not self._parents:
                raise RAGReadinessError("parent chunk artifact contains no parent_id records")

            if self.config.enable_sparse:
                if self.config.bm25_mode is BM25Mode.REBUILD_FROM_JSON:
                    self._bm25 = RebuiltBM25Index(
                        [str(chunk.get("text", "")) for chunk in self._children]
                    )
                elif self.config.bm25_mode is BM25Mode.VERIFIED_LEGACY_PICKLE:
                    self._bm25 = load_verified_legacy_pickle(
                        self.config.bm25_path,
                        self.config.legacy_bm25_sha256 or "",
                    )

            if self.config.enable_dense:
                self._load_dense_components()
            if self.config.enable_reranker:
                self._load_reranker()
            self._loaded = True

    def _load_dense_components(self) -> None:
        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RAGReadinessError(
                "dense retrieval requires faiss and sentence-transformers"
            ) from exc
        self._faiss = faiss.read_index(str(self.config.faiss_path))
        try:
            self._embedder = SentenceTransformer(
                str(self.config.embedding_model),
                device=self.config.embedding_device,
                local_files_only=not self.config.allow_network,
            )
        except Exception as exc:
            raise RAGReadinessError(
                "could not load the embedding model; offline mode requires a populated local cache"
            ) from exc

    def _load_reranker(self) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise RAGReadinessError("reranking requires sentence-transformers") from exc
        try:
            self._reranker = CrossEncoder(
                str(self.config.reranker_model),
                device=self.config.reranker_device,
                local_files_only=not self.config.allow_network,
            )
        except Exception as exc:
            raise RAGReadinessError(
                "could not load the reranker; offline mode requires a populated local cache"
            ) from exc

    def retrieve(self, query: str, top_k: int = 3) -> tuple[RetrievedDocument, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.load()

        # SentenceTransformer/CrossEncoder and a shared FAISS index are treated
        # as a single worker until a serving engine with explicit batching is
        # introduced. This prevents accidental concurrent mutation/use.
        with self._retrieval_lock:
            return self._retrieve_loaded(query, top_k)

    def _retrieve_loaded(self, query: str, top_k: int) -> tuple[RetrievedDocument, ...]:

        ranked: dict[int, dict[str, float]] = {}
        if self.config.enable_dense:
            for rank, (index, score) in enumerate(self._dense_search(augment_query(query)), 1):
                entry = ranked.setdefault(index, {})
                entry["dense_score"] = score
                entry["fusion_score"] = entry.get("fusion_score", 0.0) + 1.0 / (
                    self.config.rrf_constant + rank
                )
        if self.config.enable_sparse:
            for rank, (index, score) in enumerate(self._sparse_search(query), 1):
                entry = ranked.setdefault(index, {})
                entry["sparse_score"] = score
                entry["fusion_score"] = entry.get("fusion_score", 0.0) + 1.0 / (
                    self.config.rrf_constant + rank
                )

        candidates = sorted(
            ranked.items(), key=lambda item: (-item[1].get("fusion_score", 0.0), item[0])
        )
        candidates = candidates[: self.config.rerank_top_k]
        if self._reranker is not None and candidates:
            pairs = [(query, str(self._children[index].get("text", ""))) for index, _ in candidates]
            scores = self._reranker.predict(pairs, show_progress_bar=False)
            for (index, values), score in zip(candidates, scores):
                values["reranker_score"] = float(score)
            candidates.sort(key=lambda item: -item[1].get("reranker_score", float("-inf")))

        documents: list[RetrievedDocument] = []
        seen_parents: set[str] = set()
        for child_index, scores in candidates:
            child = self._children[child_index]
            parent_id = str(child.get("parent_id", ""))
            if not parent_id or parent_id in seen_parents:
                continue
            parent = self._parents.get(parent_id)
            if parent is None:
                continue
            seen_parents.add(parent_id)
            documents.append(
                _coerce_document(
                    {
                        "parent_chunk": parent,
                        "child_chunk_id": child.get("chunk_id"),
                        "score": scores.get("fusion_score"),
                        "reranker_score": scores.get("reranker_score"),
                        "dense_score": scores.get("dense_score"),
                        "sparse_score": scores.get("sparse_score"),
                        "retrievers": tuple(
                            name.removesuffix("_score")
                            for name in ("dense_score", "sparse_score")
                            if name in scores
                        ),
                    },
                    len(documents) + 1,
                )
            )
            if len(documents) >= top_k:
                break
        return tuple(documents)

    def _dense_search(self, query: str) -> list[tuple[int, float]]:
        try:
            import numpy as np  # type: ignore
            import faiss  # type: ignore
        except ImportError as exc:
            raise RAGReadinessError("dense retrieval requires numpy and faiss") from exc
        if self._embedder is None or self._faiss is None:
            raise RAGReadinessError("dense retrieval components are not loaded")
        vector = self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        vector = np.asarray(vector, dtype="float32")
        # Normalising again is harmless and protects older sentence-transformers versions.
        faiss.normalize_L2(vector)
        scores, indices = self._faiss.search(vector, self.config.dense_top_k)
        return [
            (int(index), float(score))
            for score, index in zip(scores[0], indices[0])
            if int(index) >= 0 and float(score) >= self.config.min_dense_score
        ]

    def _sparse_search(self, query: str) -> list[tuple[int, float]]:
        if isinstance(self._bm25, RebuiltBM25Index):
            return self._bm25.search(query, self.config.sparse_top_k)
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(RebuiltBM25Index.tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda pair: (-float(pair[1]), pair[0]))
        return [
            (int(index), float(score))
            for index, score in ranked[: self.config.sparse_top_k]
            if float(score) > 0.0
        ]


def _load_json_list(path: Path, description: str) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RAGReadinessError(f"could not load {description} from {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RAGReadinessError(f"{description} must be a JSON list of objects")
    return value


__all__ = [
    "ArtifactIntegrityError",
    "BM25Mode",
    "Citation",
    "HybridRAGBackend",
    "HybridRAGConfig",
    "RAGError",
    "RAGReadinessError",
    "RAGService",
    "RebuiltBM25Index",
    "RetrievalBackend",
    "RetrievalMode",
    "RetrievalReadiness",
    "RetrievalResult",
    "RetrievedDocument",
    "RouteRetrievalPolicy",
    "augment_query",
    "load_verified_legacy_pickle",
    "sha256_file",
]
