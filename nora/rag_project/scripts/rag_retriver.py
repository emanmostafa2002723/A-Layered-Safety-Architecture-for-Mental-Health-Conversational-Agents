"""
retriever_v8.py
---------------
Hybrid retrieval pipeline for the Mental Health RAG knowledge base.

Pipeline (per query):
    1. Augment query with clinical vocabulary (Phase 1's augment_query)
    2. Dense retrieval (FAISS) → top-10 child chunks
    3. Sparse retrieval (BM25) → top-10 child chunks
    4. Fuse (union + deduplicate by chunk_id)
    5. BGE reranker rescore → top-3
    6. Map to parent chunks (deduplicate)
    7. Build citations

Key design:
    - Dense retrieval uses AUGMENTED query (more context for semantic matching)
    - Sparse retrieval uses ORIGINAL query (keyword precision, no dilution)
    - BGE reranker uses ORIGINAL query (direct query-vs-document comparison)

Run test queries:
    python scripts/retriever_v8.py

Use as a module:
    from retriever_v8 import Retriever
    r = Retriever()
    results = r.retrieve("I feel anxious")
    # results = {"parent_chunks": [...], "citations": [...], "route": "MONITORED"}

Install dependencies:
    pip install sentence-transformers faiss-cpu rank-bm25
    # BGE reranker is loaded via sentence-transformers CrossEncoder — no extra package needed
"""

import re
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = Path("data/processed")
FAISS_PATH = PROCESSED_DIR / "faiss_index.bin"
BM25_PATH = PROCESSED_DIR / "bm25_index.pkl"
CHILD_CHUNKS_PATH = PROCESSED_DIR / "child_chunks.json"
PARENT_CHUNKS_PATH = PROCESSED_DIR / "parent_chunks.json"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-large"

DENSE_TOP_K = 10       # candidates from FAISS
SPARSE_TOP_K = 10      # candidates from BM25
RERANK_TOP_K = 10      # candidates to pass to parent mapping (after reranking)
FINAL_TOP_K = 3        # final results after parent mapping + diversity filter
MIN_DENSE_SCORE = 0.30  # minimum cosine similarity for dense retrieval

# Crisis topics — used to flag MONITORED route
CRISIS_TOPICS = {
    "suicidal_ideation", "self_harm",
    "sexual_violence", "domestic_violence", "child_abuse",
    "anxiety_crisis",
}

# v8.1: Query-topic detection — boosts chunks matching the detected topic
QUERY_TOPIC_PATTERNS = {
    "suicidal_ideation": ["suicide", "suicidal", "kill myself", "end my life", "want to die", "don't want to live", "ending my life"],
    "self_harm": ["self-harm", "self harm", "cutting", "hurt myself", "self injury", "self-injury"],
    "sexual_violence": ["sexual assault", "rape", "sexual abuse", "sexual violence", "sexual harassment", "military sexual trauma"],
    "domestic_violence": ["domestic violence", "intimate partner violence", "abusive partner", "abusive relationship", "abusive spouse"],
    "child_abuse": ["child abuse", "child neglect", "child maltreatment", "child endangerment", "child protective services"],
    "anxiety_crisis": ["anxiety", "anxious", "panic", "panic attack", "racing heart", "can't breathe", "hyperventilation"],
    "depression": ["depression", "depressed", "sad", "hopeless", "worthless", "low mood", "anhedonia", "empty"],
    "therapy_techniques": ["cbt", "cognitive behavioral", "therapy", "therapist", "counselling", "psychotherapy", "cognitive restructuring", "behavioral activation", "exposure therapy"],
    "trauma_ptsd": ["ptsd", "trauma", "traumatic", "flashback", "nightmare", "hypervigilance", "trigger", "combat"],
    "psychosis_bipolar": ["psychosis", "psychotic", "schizophrenia", "bipolar", "mania", "manic", "hallucination", "delusion", "paranoia"],
    "substance_use": ["substance", "alcohol", "drug", "drugs", "addiction", "withdrawal", "overdose", "opioid"],
    "medication": ["ssri", "ssris", "snri", "antidepressant", "medication", "dosage", "side effect", "prescription"],
}

TOPIC_BOOST = 0.10


# ============================================================
# QUERY AUGMENTATION (from Phase 1 — unchanged)
# ============================================================

def augment_query(query: str) -> str:
    """
    Expand a user query with related clinical vocabulary.
    Bridges the lexical gap between user language and clinical documents.
    Only adds clinical terms — never changes the original query.
    """
    q = query.lower()
    additions = []

    # --- Suicidal ideation ---
    suicidal_patterns = [
        "don't want to live", "want to die", "end my life",
        "kill myself", "not worth living", "can't go on",
        "no point", "rather be dead", "wish i was dead",
        "life is meaningless", "disappear forever",
        "ending my life", "ending it all", "thoughts of ending",
        "thinking of ending it", "think about ending my life",
        "don't want to be here", "no reason to live",
        "what's the point of living",
    ]
    if any(p in q for p in suicidal_patterns):
        additions.append(
            "suicidal ideation thoughts of suicide "
            "suicide prevention crisis wanting to end life "
            "hopelessness no reason to live"
        )

    # --- Self-harm ---
    self_harm_patterns = [
        "hurt myself", "cut myself", "harm myself",
        "self-harm", "cutting", "burning myself",
        "injure myself", "self injury",
    ]
    if any(p in q for p in self_harm_patterns):
        additions.append(
            "self-harm self-injury nonsuicidal "
            "self-destructive urge to hurt "
            "coping with self-harm"
        )

    # --- Anxiety / panic ---
    anxiety_patterns = [
        "panic attack", "can't breathe", "heart racing",
        "anxiety", "anxious", "losing control",
        "chest tight", "overwhelming fear", "racing heart",
    ]
    if any(p in q for p in anxiety_patterns):
        additions.append(
            "anxiety crisis panic disorder acute anxiety "
            "grounding breathing techniques calm "
            "panic symptoms overwhelmed"
        )

    # --- Depression / hopelessness ---
    depression_patterns = [
        "hopeless", "sad all the time", "empty inside",
        "worthless", "nothing matters", "depressed",
        "no energy", "can't feel anything", "low mood",
    ]
    if any(p in q for p in depression_patterns):
        additions.append(
            "depression hopelessness persistent sadness "
            "depressive episode loss of interest "
            "emotional pain support"
        )

    if additions:
        return query + " " + " ".join(additions)
    return query


def is_crisis_query(query: str) -> bool:
    """
    Check if the query indicates a crisis requiring MONITORED routing.

    BUG 1 FIX (v8.1): Synced with augment_query() patterns.
    Previously, this function had a shorter pattern list than augment_query,
    causing queries that triggered augmentation to NOT trigger MONITORED routing.

    Now triggers MONITORED for:
        - Suicidal ideation (all augment_query suicidal patterns)
        - Self-harm (all augment_query self-harm patterns)
        - Acute anxiety/panic with physical symptoms
          (panic attack, can't breathe, racing heart, chest tight, etc.)

    Does NOT trigger for:
        - General anxiety ("I feel anxious" without physical symptoms)
        - Depression ("I feel sad/hopeless")
        Those stay NORMAL — mental health concerns but not immediate crises.
    """
    q = query.lower()
    crisis_patterns = [
        # --- Suicidal ideation (synced with augment_query) ---
        "don't want to live", "want to die", "end my life",
        "kill myself", "not worth living", "can't go on",
        "no point", "rather be dead", "wish i was dead",
        "don't want to be here", "no reason to live",
        "what's the point of living",
        # v8.1 fix: missing suicidal patterns from augment_query
        "ending my life", "ending it all", "thoughts of ending",
        "thinking of ending it", "think about ending my life",
        "life is meaningless", "disappear forever",
        # --- Self-harm (synced with augment_query) ---
        "hurt myself", "cut myself", "harm myself",
        "self-harm", "cutting", "burning myself",
        "injure myself", "self injury",
        # --- Acute anxiety/panic (physical symptom patterns) ---
        # v8.1 fix: added "racing heart", "heart is racing", "chest is tight",
        # "tightness in chest", "shortness of breath" to catch more phrasings
        "panic attack",
        "can't breathe", "cannot breathe",
        "heart racing", "racing heart", "heart is racing",
        "losing control", "overwhelming fear",
        "chest tight", "chest is tight", "tightness in chest",
        "shortness of breath",
        # NEW: sexual violence
        "sexual assault", "sexual abuse", "sexual violence",
        "rape", "raped", "i was raped",
        "sexual harassment", "sexual coercion",
        "military sexual trauma", "mst",
        # NEW: domestic violence
        "domestic violence", "intimate partner violence",
        "abusive partner", "abusive relationship", "abusive spouse",
        "my partner hits", "my partner hurts",
        "my husband hits", "my wife hits",
        "physical abuse", "emotional abuse",
        "battered", "battering",
        # NEW: child abuse
        "child abuse", "child neglect", "child maltreatment",
        "child endangerment", "child protective services",
        "mandated reporter", "mandatory reporting",
        "my child is being", "my child was abused",
        "child sexual abuse", "inappropriate touching",
        "hitting a child", "hitting my child",
    ]
    return any(p in q for p in crisis_patterns)

def detect_query_topic(query: str) -> str:
    """Detect which topic the query is about. Returns topic name or empty string."""
    q = query.lower()
    topic_scores = {}
    for topic, patterns in QUERY_TOPIC_PATTERNS.items():
        score = sum(1 for p in patterns if p in q)
        if score > 0:
            topic_scores[topic] = score
    if not topic_scores:
        return ""
    return max(topic_scores, key=topic_scores.get)
# ============================================================
# RETRIEVER CLASS
# ============================================================

class Retriever:
    """
    Hybrid retrieval pipeline: Dense (FAISS) + Sparse (BM25) + BGE reranker.

    Usage:
        r = Retriever()
        results = r.retrieve("I feel anxious")
        # results = {
        #     "parent_chunks": [...],
        #     "citations": [...],
        #     "route": "MONITORED",  # or "NORMAL"
        #     "augmented_query": "I feel anxious anxiety crisis panic disorder...",
        # }
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._load_artifacts()

    def _load_artifacts(self):
        """Load FAISS index, BM25 index, child chunks, parent chunks, embedder, reranker."""
        print("Loading retrieval artifacts...")

        # Load child chunks
        with open(CHILD_CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.child_chunks = json.load(f)
        print(f"  Child chunks: {len(self.child_chunks)}")

        # Load parent chunks + build lookup by parent_id
        with open(PARENT_CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.parent_chunks = json.load(f)
        self.parent_lookup = {p["parent_id"]: p for p in self.parent_chunks}
        print(f"  Parent chunks: {len(self.parent_chunks)}")

        # Load FAISS index
        self.faiss_index = faiss.read_index(str(FAISS_PATH))
        print(f"  FAISS index: {self.faiss_index.ntotal} vectors, dim {self.faiss_index.d}")

        # Load BM25 index
        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        print(f"  BM25 index loaded")

        # Load embedder
        print(f"  Loading embedder: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Load reranker (BGE)
        print(f"  Loading reranker: {RERANKER_MODEL}")
        try:
            self.reranker = CrossEncoder(RERANKER_MODEL)
            self.has_reranker = True
            print(f"  Reranker loaded")
        except Exception as e:
            print(f"  WARNING: Could not load reranker: {e}")
            print(f"  Falling back to fusion-only (no reranking)")
            self.reranker = None
            self.has_reranker = False

        print()

    # --------------------------------------------------------
    # DENSE RETRIEVAL (FAISS)
    # --------------------------------------------------------
    def dense_retrieve(self, augmented_query: str, top_k: int = DENSE_TOP_K) -> list:
        """
        Embed the augmented query and search FAISS for top-K child chunks.
        Returns list of {chunk_id, score, chunk} dicts.
        """
        # Embed query
        q_emb = self.embedder.encode(
            [augmented_query], convert_to_tensor=True
        ).cpu().numpy().astype("float32")
        faiss.normalize_L2(q_emb)  # L2 normalize for cosine similarity

        # Search
        scores, indices = self.faiss_index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < MIN_DENSE_SCORE:
                continue
            chunk = self.child_chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "score": float(score),
                "retriever": "dense",
                "chunk": chunk,
            })

        return results

    # --------------------------------------------------------
    # SPARSE RETRIEVAL (BM25)
    # --------------------------------------------------------
    def sparse_retrieve(self, original_query: str, top_k: int = SPARSE_TOP_K) -> list:
        """
        Tokenize the ORIGINAL query (no augmentation) and search BM25 for top-K child chunks.
        Returns list of {chunk_id, score, chunk} dicts.
        """
        # Tokenize query (same way the index was built)
        tokenized_query = re.findall(r"\w+", original_query.lower())

        if not tokenized_query:
            return []

        # Get scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.child_chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "score": float(scores[idx]),
                "retriever": "sparse",
                "chunk": chunk,
            })

        return results

    # --------------------------------------------------------
    # FUSION (union + deduplicate)
    # --------------------------------------------------------
    def fuse(self, dense_results: list, sparse_results: list) -> list:
        """
        Combine dense + sparse results, deduplicate by chunk_id.
        Keep both scores for transparency.
        """
        seen = {}
        fused = []

        for result in dense_results + sparse_results:
            cid = result["chunk_id"]
            if cid in seen:
                # Already seen — add the other retriever's score
                seen[cid]["retrievers"].add(result["retriever"])
                if result["retriever"] == "dense":
                    seen[cid]["dense_score"] = result["score"]
                else:
                    seen[cid]["sparse_score"] = result["score"]
            else:
                entry = {
                    "chunk_id": cid,
                    "chunk": result["chunk"],
                    "retrievers": {result["retriever"]},
                    "dense_score": result["score"] if result["retriever"] == "dense" else None,
                    "sparse_score": result["score"] if result["retriever"] == "sparse" else None,
                }
                seen[cid] = entry
                fused.append(entry)

        return fused

    # --------------------------------------------------------
    # RERANKING (BGE)
    # --------------------------------------------------------
    def rerank(self, original_query: str, candidates: list, top_k: int = FINAL_TOP_K) -> list:
        """
        Use BGE reranker to rescore candidates.
        If reranker not available, fall back to dense score (or sparse if no dense).
        """
        if not candidates:
            return []

        if not self.has_reranker:
            # Fallback: sort by dense score (or sparse if no dense)
            def fallback_score(c):
                d = c.get("dense_score")
                s = c.get("sparse_score")
                return d if d is not None else (s if s is not None else 0)
            candidates_sorted = sorted(candidates, key=fallback_score, reverse=True)
            return candidates_sorted[:top_k]

        # Prepare query-document pairs
        pairs = [(original_query, c["chunk"]["text"]) for c in candidates]

        # Score with BGE reranker
        reranker_scores = self.reranker.predict(pairs, show_progress_bar=False)

        # Attach reranker scores
        for i, score in enumerate(reranker_scores):
            candidates[i]["reranker_score"] = float(score)

        # Sort by reranker score
        candidates_sorted = sorted(
            candidates,
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return candidates_sorted[:top_k]

    # --------------------------------------------------------
    # PARENT MAPPING + DIVERSITY FILTER
    # --------------------------------------------------------
    def map_to_parents(self, child_results: list, final_top_k: int = FINAL_TOP_K) -> list:
        """
        Map top child chunks to their parent chunks.
        Deduplicate parents (multiple children may map to the same parent).
        Apply source diversity filter (BUG 2 fix).
        """
        seen_parents = set()
        parent_results = []

        for result in child_results:
            parent_id = result["chunk"]["parent_id"]
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            parent = self.parent_lookup.get(parent_id)
            if parent is None:
                continue

            parent_results.append({
                "parent_chunk": parent,
                "child_chunk_id": result["chunk"]["chunk_id"],
                "reranker_score": result.get("reranker_score"),
                "dense_score": result.get("dense_score"),
                "sparse_score": result.get("sparse_score"),
                "retrievers": list(result.get("retrievers", set())),
            })

        # BUG 2 FIX: source diversity filter
        parent_results = self._apply_diversity_filter(parent_results, final_top_k)

        return parent_results[:final_top_k]

    def _apply_diversity_filter(self, results: list, top_k: int) -> list:
        """
        BUG 2 FIX (v8.1): Ensure the top-K results come from at least 2
        different sources.

        If the top-K are all from the same source, replace the lowest-ranked
        with the next-best result from a different source (if available).
        This prevents the chatbot from giving narrow, single-source answers.
        """
        if len(results) <= top_k:
            return results

        top_sources = [r["parent_chunk"]["source"] for r in results[:top_k]]
        unique_sources = set(top_sources)

        if len(unique_sources) >= 2:
            # Already diverse — no change needed
            return results

        # All top-K from same source — find a different source
        same_source = top_sources[0]
        for i in range(top_k, len(results)):
            if results[i]["parent_chunk"]["source"] != same_source:
                # Replace the last position (lowest-ranked) with this different source
                results[top_k - 1] = results[i]
                break

        return results

    # --------------------------------------------------------
    # CITATION BUILDING
    # --------------------------------------------------------
    def build_citations(self, parent_results: list) -> list:
        """Build citation strings from parent chunks."""
        citations = []
        for i, result in enumerate(parent_results, 1):
            parent = result["parent_chunk"]
            citation = f"[{i}] {parent['source']} — {parent['document_title']}"
            if parent.get("url"):
                citation += f" ({parent['url']})"
            if parent.get("page_num"):
                citation += f" [page {parent['page_num']}]"
            citations.append(citation)
        return citations

    # --------------------------------------------------------
    # FULL RETRIEVAL PIPELINE
    # --------------------------------------------------------
    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> dict:
        """
        Full hybrid retrieval pipeline.

        Args:
            query: user message
            top_k: number of final parent chunks to return

        Returns:
            {
                "parent_chunks": list of parent chunk dicts (with metadata + text),
                "citations": list of citation strings,
                "route": "MONITORED" or "NORMAL",
                "augmented_query": str,
                "dense_count": int,
                "sparse_count": int,
                "fused_count": int,
            }
        """
        # 1. Determine route
        route = "MONITORED" if is_crisis_query(query) else "NORMAL"

        # 2. Augment query
        augmented = augment_query(query)

        if self.verbose:
            print(f"  [AUGMENTED] {augmented[:100]}...")
            print(f"  [ROUTE] {route}")

        # 3. Dense retrieval (FAISS) — uses augmented query
        dense_results = self.dense_retrieve(augmented, top_k=DENSE_TOP_K)

        # 4. Sparse retrieval (BM25) — uses original query
        sparse_results = self.sparse_retrieve(query, top_k=SPARSE_TOP_K)

        # 5. Fuse
        fused = self.fuse(dense_results, sparse_results)

        # 6. Rerank — get top RERANK_TOP_K candidates (more than final_top_k
        #    so the diversity filter has room to work)
        reranked = self.rerank(query, fused, top_k=RERANK_TOP_K)

        # 6.5 Topic-based query boosting
        query_topic = detect_query_topic(query)
        if query_topic and reranked:
            print(f"  [TOPIC BOOST] Detected topic: {query_topic} (+{TOPIC_BOOST})")
            for result in reranked:
                if result["chunk"]["primary_topic"] == query_topic:
                    current_score = result.get("reranker_score", 0) or 0
                    result["reranker_score"] = current_score + TOPIC_BOOST
            reranked = sorted(reranked, key=lambda x: x.get("reranker_score", 0), reverse=True)

   
        # 7. Map to parents (dedup by parent_id + source diversity filter)
        parent_results = self.map_to_parents(reranked, final_top_k=top_k)

        # 8. Citations
        citations = self.build_citations(parent_results)

        return {
            "parent_chunks": parent_results,
            "citations": citations,
            "route": route,
            "augmented_query": augmented,
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "fused_count": len(fused),
        }


# ============================================================
# TEST QUERIES
# ============================================================

TEST_QUERIES = [
    "I feel anxious and my heart is racing",
    "I sometimes think about ending my life",
    "I don't want to live anymore",
    "How can I calm down during a panic attack?",
    "I feel sad and hopeless all the time",
    "What is cognitive behavioral therapy?",
    "What are SSRIs?",
    "How does PTSD affect people?",
]


# ============================================================
# MAIN — run test queries
# ============================================================

def main():
    print("=" * 60)
    print("RETRIEVER v8 — Hybrid Retrieval (FAISS + BM25 + BGE)")
    print("=" * 60)
    print()

    retriever = Retriever(verbose=True)

    print("=" * 60)
    print(f"Running {len(TEST_QUERIES)} test queries")
    print("=" * 60)

    for i, query in enumerate(TEST_QUERIES, 1):
        print("\n" + "=" * 70)
        print(f"QUERY {i}/{len(TEST_QUERIES)}: {query}")
        print("=" * 70)

        results = retriever.retrieve(query)

        print(f"\n  Route:           {results['route']}")
        print(f"  Augmented query: {results['augmented_query'][:120]}...")
        print(f"  Dense candidates: {results['dense_count']}")
        print(f"  Sparse candidates: {results['sparse_count']}")
        print(f"  Fused candidates:  {results['fused_count']}")

        if not results["parent_chunks"]:
            print("\n  NO RESULTS — knowledge gap detected")
            continue

        print(f"\n  Top {len(results['parent_chunks'])} parent chunks:")

        for j, pr in enumerate(results["parent_chunks"], 1):
            parent = pr["parent_chunk"]
            child = pr["child_chunk_id"]
            reranker_score = pr.get("reranker_score")
            dense_score = pr.get("dense_score")
            sparse_score = pr.get("sparse_score")
            retrievers = pr.get("retrievers", [])

            print(f"\n  --- Result {j} ---")
            print(f"    Source:          {parent['source']}")
            print(f"    Document:        {parent['document_title']}")
            print(f"    Topic:           {parent['primary_topic']}")
            if parent.get("page_num"):
                print(f"    Page:            {parent['page_num']}")
            print(f"    Reranker score:  {reranker_score:.4f}" if reranker_score else "    Reranker score: N/A")
            print(f"    Dense score:     {dense_score:.4f}" if dense_score else "    Dense score: N/A")
            print(f"    Sparse score:    {sparse_score:.4f}" if sparse_score else "    Sparse score: N/A")
            print(f"    Retrievers:      {', '.join(retrievers)}")
            print(f"    Child chunk ID:  {child}")
            # Show first 200 chars of parent text
            preview = parent["text"][:200].replace("\n", " ")
            print(f"    Text preview:    {preview}...")

        print(f"\n  Citations:")
        for c in results["citations"]:
            print(f"    {c}")

    print("\n" + "=" * 60)
    print("Done — retrieval test complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
