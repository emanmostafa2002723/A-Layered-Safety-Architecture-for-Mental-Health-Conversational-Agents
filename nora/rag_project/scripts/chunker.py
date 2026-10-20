"""
chunker_v8.py
-------------
Parent-child chunking for the Mental Health RAG knowledge base.

Pipeline:
    1. Load all 28 preprocessed files (20 web + 8 PDF)
    2. Parse headers + extract [PAGE N] markers (for PDFs)
    3. Split into parent chunks (~1000 tokens)
    4. Split parent chunks into child chunks (~250 tokens, 40 overlap)
    5. Label each chunk with 10-topic dominance_test
    6. Embed child chunks with all-MiniLM-L6-v2
    7. Build FAISS index (dense) + BM25 index (sparse)
    8. Save artifacts to data/processed/

Output artifacts:
    data/processed/parent_chunks.json   — parent chunk store
    data/processed/child_chunks.json    — child chunk store (metadata only)
    data/processed/embeddings.npy       — child embeddings (numpy array)
    data/processed/faiss_index.bin      — FAISS dense index
    data/processed/bm25_index.pkl       — BM25 sparse index (pickled)
    data/processed/chunk_stats.json     — summary statistics

Install dependencies:
    pip install tiktoken rank-bm25 sentence-transformers faiss-cpu

Run:
    python scripts/chunker_v8.py
"""

import re
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import Counter

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken (accurate for Mistral 7B / GPT-style models)."""
        return len(_ENCODER.encode(text))
    HAS_TIKTOKEN = True
except ImportError:
    # Fallback to Phase 1 estimator
    def count_tokens(text: str) -> int:
        """Fallback: estimate tokens as word count × 1.3."""
        return int(len(text.split()) * 1.3)
    HAS_TIKTOKEN = False
    print("WARNING: tiktoken not installed. Using fallback token estimator.")
    print("Install with: pip install tiktoken")

# Try to import rank_bm25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("WARNING: rank_bm25 not installed. BM25 index will be skipped.")
    print("Install with: pip install rank-bm25")


# ============================================================
# CONFIGURATION
# ============================================================

WEB_DIR = Path("data/knowledge_base")
PDF_DIR = Path("data/knowledge_base_phase2")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARENT_CHUNK_SIZE = 1000   # tokens (max)
CHILD_CHUNK_SIZE = 250     # tokens (max)
CHILD_OVERLAP = 40         # tokens
MIN_CHILD_SIZE = 50        # tokens (filter out tiny chunks)
MIN_PARENT_SIZE = 100      # tokens

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# TOPIC KEYWORDS (10 topics)
# ============================================================
# These are provisional — may be adjusted to align with the
# input crisis classifier's classes once those are confirmed.

TOPIC_KEYWORDS = {
    "suicidal_ideation": [
        "suicide", "suicidal", "kill myself", "kill yourself",
        "end your life", "end my life", "end their life",
        "don't want to live", "not worth living",
        "want to die", "wanting to die", "thoughts of death",
        "no reason to live", "better off dead",
        "crisis line", "suicide prevention", "suicidal thoughts",
        "safety plan", "suicide hotline", "988",
        "suicidal ideation", "suicide attempt", "suicide risk",
        "warning signs of suicide", "at risk for suicide",
        "thinking about suicide", "attempting suicide",
        "means to end", "goodbye messages",
        "giving away possessions", "immediate risk",
        "i want to die", "i don't want to be here",
        "i can't go on", "what's the point of living",
        "i wish i was dead", "life is not worth living",
    ],
    "self_harm": [
        "self-harm", "self harm", "self-injury", "self injury",
        "cutting myself", "cut myself", "hurting myself",
        "hurt yourself", "nonsuicidal self-injury",
        "burn myself", "scratching myself", "hitting myself",
        "self-mutilation",
        "self-harming behavior", "self-harm behavior",
        "recurring thoughts of self-harm",
        "urge to self-harm", "history of self-harm",
        "self-destructive behavior",
        "borderline personality", "emotion dysregulation",
        "impulse control", "self-harm and suicidal",
    ],
    "anxiety_crisis": [
        "anxiety", "anxious", "panic attack", "panic disorder",
        "panic", "racing heart", "shortness of breath", "chest pain",
        "feeling of losing control", "grounding technique",
        "deep breathing", "breathing exercise", "5-4-3-2-1",
        "fight or flight", "hyperventilation", "acute anxiety",
        "generalized anxiety", "gad", "worry", "worrying",
        "phobia", "phobic", "social anxiety",
    ],
    "depression": [
        "depression", "depressed", "low mood", "hopeless",
        "hopelessness", "loss of interest", "anhedonia",
        "feeling empty", "persistent sadness", "worthlessness",
        "feeling worthless", "no energy", "fatigue",
        "sleep changes", "depressive episode",
        "major depressive", "mood disorder",
    ],
    "general_support": [
        "coping", "coping strategies", "self-care", "self care",
        "support", "supportive", "active listening", "empathy",
        "emotional support", "mental health", "stress management",
        "relaxation", "mindfulness", "wellness", "peer support",
        "recovery", "resilience",
    ],    "sexual_violence": [
        "sexual violence", "sexual assault", "sexual abuse",
        "rape", "raped", "rapist",
        "sexual harassment", "sexual coercion", "coerced",
        "nonconsensual", "non-consensual", "without consent",
        "sexual trauma", "sexual victimization",
        "survivor of sexual", "victim of sexual",
        "perpetrator of sexual", "sexual perpetrator",
        "forensic examination", "rape kit", "sexual assault kit",
        "date rape", "acquaintance rape", "drug-facilitated",
        "sexual exploitation", "human trafficking for sex",
        "incest", "child sexual abuse",
        "military sexual trauma", "mst",
        "sexual intimacy", "reclaiming intimacy",
    ],
    "domestic_violence": [
        "domestic violence", "intimate partner violence",
        "ipv", "battering", "battered",
        "abusive relationship", "abusive partner", "abusive spouse",
        "abuser", "batterer",
        "physical abuse", "emotional abuse", "psychological abuse",
        "financial abuse", "economic abuse",
        "controlling behavior", "controlling partner",
        "isolation from family", "isolation from friends",
        "gaslighting", "manipulation",
        "restraining order", "protective order",
        "order of protection", "no contact order",
        "domestic abuse", "family violence",
        "cycle of violence", "power and control wheel",
        "leaving an abuser", "safety planning for dv",
        "teen dating violence", "dating abuse",
    ],
    "child_abuse": [
        "child abuse", "child neglect", "child maltreatment",
        "child endangerment", "child protection",
        "child protective services", "cps",
        "mandated reporter", "mandated reporting",
        "mandated report", "mandatory reporting",
        "physical abuse of child", "physical neglect",
        "emotional abuse of child", "emotional neglect",
        "medical neglect", "educational neglect",
        "child welfare", "child welfare system",
        "foster care", "foster home",
        "child sexual abuse", "inappropriate touching",
        "exposure to violence", "witnessing violence",
        "shaken baby", "shaken baby syndrome",
        "ace", "adverse childhood experiences",
        "child protective investigation",
        "safe environment for child", "child safety",
    ],
    "therapy_techniques": [
        "cognitive behavioral", "cbt", "bcbt",
        "thought record", "cognitive restructuring",
        "behavioral activation", "exposure therapy",
        "agenda setting", "homework", "automatic thoughts",
        "cognitive distortions", "unhelpful thinking",
        "challenging thoughts", "evidence for", "evidence against",
        "dialectical behavior", "dbt",
        "psychodynamic", "psychotherapy", "counselling",
        "therapeutic alliance", "case conceptualization",
        "treatment planning", "orienting the patient",
        "problem solving", "relaxation technique",
    ],
    "trauma_ptsd": [
        "trauma", "traumatic", "ptsd", "post-traumatic",
        "flashback", "flashbacks", "nightmare", "nightmares",
        "hypervigilance", "trigger", "triggered", "triggers",
        "dissociation", "dissociative",
        "abuse", "domestic violence", "sexual assault",
        "combat", "war veteran",
        "acute stress", "stress disorder",
        "intrusive memories", "intrusive thoughts",
    ],
    "psychosis_bipolar": [
        "psychosis", "psychotic", "hallucination", "hallucinations",
        "delusion", "delusions", "paranoia", "paranoid",
        "schizophrenia", "schizoaffective",
        "mania", "manic", "hypomania", "hypomanic",
        "bipolar", "bipolar disorder",
        "mood stabilizer", "antipsychotic",
        "reality testing", "thought disorder",
    ],
    "substance_use": [
        "substance", "substance use", "substance abuse",
        "alcohol", "alcoholic", "alcoholism",
        "drug", "drugs", "drug use", "drug abuse",
        "addiction", "addicted", "dependency", "dependent",
        "withdrawal", "withdraw", "detox",
        "opioid", "heroin", "cocaine", "methamphetamine",
        "intoxication", "overdose",
        "relapse", "recovery program", "12-step",
    ],
    "medication": [
        "ssri", "ssris", "snri", "snris",
        "antidepressant", "antidepressants",
        "benzodiazepine", "benzodiazepines",
        "antipsychotic", "antipsychotics",
        "mood stabilizer", "mood stabilizers",
        "medication", "medications", "pharmacology",
        "dosage", "dose", "side effect", "side effects",
        "prescription", "prescribed",
        "fluoxetine", "sertraline", "escitalopram",
        "venlafaxine", "bupropion", "mirtazapine",
    ],
}

# Crisis topics — used by the input guardrail for MONITORED routing
CRISIS_TOPICS = {
    "suicidal_ideation",
    "self_harm",
    "sexual_violence",
    "domestic_violence",
    "child_abuse",
    "anxiety_crisis",
}

# Safety terms — never remove from chunks
SAFETY_TERMS = [
    "988", "911", "crisis", "suicide", "suicidal",
    "emergency", "hotline", "lifeline", "helpline",
    "safe-t", "safety plan", "self-harm", "self harm", "overdose",
]


# ============================================================
# HEADER PARSING + PAGE EXTRACTION
# ============================================================

def parse_header(raw: str) -> dict:
    """Parse the header block (Title/Source/URL/License/Topic) from raw text."""
    header = {
        "title": "Unknown",
        "source": "Unknown",
        "url": "",
        "license": "Unknown",
        "topic": "general",
    }
    if "---" not in raw:
        return header
    header_part = raw.split("---", 1)[0]
    for line in header_part.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()
            if key in header:
                header[key] = val
    return header


def extract_pages(text: str) -> list:
    """
    Split text into pages using [PAGE N] markers.
    Returns list of (page_num, page_text).
    For web files (no markers), returns [(None, full_text)].
    """
    if "[PAGE " not in text:
        return [(None, text)]

    parts = re.split(r"\[PAGE (\d+)\]", text)
    pages = []
    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        page_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if page_text:
            pages.append((page_num, page_text))
    return pages if pages else [(None, text)]


def get_page_for_offset(pages: list, char_offset: int) -> int:
    """
    Given a list of (page_num, page_text) and a character offset
    within the concatenated text, return the page number.
    """
    cumulative = 0
    for page_num, page_text in pages:
        cumulative += len(page_text)
        if char_offset < cumulative:
            return page_num
    return pages[-1][0] if pages else None


# ============================================================
# TOPIC LABELING (dominance_test with 10 topics)
# ============================================================

def dominance_test(text: str) -> dict:
    """
    Determine the primary and secondary topics for a chunk of text.

    Uses keyword matching across 10 topics. Returns:
        {
            "primary_topic": str,
            "secondary_topics": list[str],
            "all_scores": dict[topic -> int],
        }
    """
    text_lower = text.lower()
    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[topic] = score

    total_matches = sum(scores.values())

    if total_matches == 0:
        return {
            "primary_topic": "general_support",
            "secondary_topics": [],
            "all_scores": scores,
        }

    sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_topic, top_score = sorted_topics[0]

    dominance_ratio = top_score / total_matches if total_matches > 0 else 1.0

    # Primary topic needs strong dominance (>= 0.4) and at least 1 match
    if dominance_ratio >= 0.4 and top_score >= 1:
        secondary = [t for t, s in sorted_topics[1:] if s > 0 and t != top_topic]
        # Cap secondary at 3 topics
        secondary = secondary[:3]
        return {
            "primary_topic": top_topic,
            "secondary_topics": secondary,
            "all_scores": scores,
        }
    else:
        # Mixed topic — take top as primary, others as secondary
        secondary = [t for t, s in sorted_topics[1:] if s > 0 and t != top_topic]
        secondary = secondary[:3]
        return {
            "primary_topic": top_topic,
            "secondary_topics": secondary,
            "all_scores": scores,
        }


# ============================================================
# PARENT CHUNK SPLITTING (~1000 tokens)
# ============================================================

def split_paragraphs(text: str) -> list:
    """Split text into paragraphs by double newlines."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_long_paragraph(paragraph: str, max_tokens: int) -> list:
    """Split a paragraph that exceeds max_tokens by sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def split_into_parent_chunks(text: str, pages: list) -> list:
    """
    Split text into parent chunks of ~1000 tokens.
    Tracks page provenance for each chunk.

    Returns list of dicts:
        {"text": str, "page_num": int|None, "token_count": int}
    """
    # Build a mapping: for each character position in the text, which page?
    # We'll track pages by splitting text at page boundaries.
    if pages and pages[0][0] is not None:
        # PDF with page markers — rebuild text with page tracking
        page_boundaries = []
        full_text = ""
        for page_num, page_text in pages:
            start = len(full_text)
            full_text += page_text + "\n\n"
            end = len(full_text)
            page_boundaries.append((start, end, page_num))

        def get_page_at_offset(offset):
            for start, end, pnum in page_boundaries:
                if start <= offset < end:
                    return pnum
            return page_boundaries[-1][2] if page_boundaries else None
    else:
        # Web file — no page markers
        full_text = text
        page_boundaries = None

        def get_page_at_offset(offset):
            return None

    paragraphs = split_paragraphs(full_text)

    parent_chunks = []
    current_paras = []
    current_tokens = 0
    current_start_offset = 0

    offset = 0
    for para in paragraphs:
        # Find where this paragraph starts in the full text
        para_start = full_text.find(para, offset)
        if para_start == -1:
            para_start = offset
        para_end = para_start + len(para)
        offset = para_end

        para_tokens = count_tokens(para)

        # If paragraph alone exceeds parent size, split it
        if para_tokens > PARENT_CHUNK_SIZE:
            # Save current chunk first
            if current_paras:
                chunk_text = "\n\n".join(current_paras)
                page_num = get_page_at_offset(current_start_offset)
                parent_chunks.append({
                    "text": chunk_text,
                    "page_num": page_num,
                    "token_count": count_tokens(chunk_text),
                })
                current_paras = []
                current_tokens = 0

            # Split the long paragraph
            sub_chunks = split_long_paragraph(para, PARENT_CHUNK_SIZE)
            for sub in sub_chunks:
                sub_start = full_text.find(sub, para_start)
                if sub_start == -1:
                    sub_start = para_start
                page_num = get_page_at_offset(sub_start)
                parent_chunks.append({
                    "text": sub,
                    "page_num": page_num,
                    "token_count": count_tokens(sub),
                })
                para_start = sub_start + len(sub)
            continue

        # Normal case: accumulate paragraphs
        if current_tokens + para_tokens > PARENT_CHUNK_SIZE and current_paras:
            # Save current chunk
            chunk_text = "\n\n".join(current_paras)
            page_num = get_page_at_offset(current_start_offset)
            parent_chunks.append({
                "text": chunk_text,
                "page_num": page_num,
                "token_count": count_tokens(chunk_text),
            })
            current_paras = [para]
            current_tokens = para_tokens
            current_start_offset = para_start
        else:
            if not current_paras:
                current_start_offset = para_start
            current_paras.append(para)
            current_tokens += para_tokens

    # Don't forget the last chunk
    if current_paras:
        chunk_text = "\n\n".join(current_paras)
        page_num = get_page_at_offset(current_start_offset)
        parent_chunks.append({
            "text": chunk_text,
            "page_num": page_num,
            "token_count": count_tokens(chunk_text),
        })

    # Filter out tiny parent chunks
    parent_chunks = [c for c in parent_chunks if c["token_count"] >= MIN_PARENT_SIZE]

    return parent_chunks


# ============================================================
# CHILD CHUNK SPLITTING (~250 tokens, 40 overlap)
# ============================================================

def split_sentences(text: str) -> list:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def split_into_child_chunks(parent_text: str, parent_id: str,
                            page_num=None) -> list:
    """
    Split a parent chunk into child chunks of ~250 tokens with 40-token overlap.

    Overlap is implemented by carrying the last 1-2 sentences from the previous
    child chunk into the next one.
    """
    sentences = split_sentences(parent_text)
    if not sentences:
        return []

    child_chunks = []
    current_sents = []
    current_tokens = 0
    is_continuation = False

    # For overlap: track the last few sentences
    overlap_sents = []

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        if current_tokens + sent_tokens > CHILD_CHUNK_SIZE and current_sents:
            # Save current chunk
            chunk_text = " ".join(current_sents)
            if count_tokens(chunk_text) >= MIN_CHILD_SIZE:
                child_chunks.append({
                    "text": chunk_text,
                    "token_count": count_tokens(chunk_text),
                    "page_num": page_num,
                    "is_continuation": is_continuation,
                    "parent_id": parent_id,
                })

            # Set up overlap: carry last 1-2 sentences (~40 tokens)
            overlap_sents = []
            overlap_tokens = 0
            for s in reversed(current_sents):
                if overlap_tokens + count_tokens(s) > CHILD_OVERLAP:
                    break
                overlap_sents.insert(0, s)
                overlap_tokens += count_tokens(s)

            # Start new chunk with overlap
            current_sents = list(overlap_sents) + [sent]
            current_tokens = sum(count_tokens(s) for s in current_sents)
            is_continuation = True
        else:
            current_sents.append(sent)
            current_tokens += sent_tokens

    # Don't forget the last chunk
    if current_sents:
        chunk_text = " ".join(current_sents)
        if count_tokens(chunk_text) >= MIN_CHILD_SIZE:
            child_chunks.append({
                "text": chunk_text,
                "token_count": count_tokens(chunk_text),
                "page_num": page_num,
                "is_continuation": is_continuation,
                "parent_id": parent_id,
            })

    return child_chunks


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_all_documents() -> list:
    """
    Load all preprocessed documents from knowledge_base/ and knowledge_base_phase2/.
    Returns list of dicts with: path, filename, topic_folder, header, body, pages, source_type
    """
    documents = []

    # Web files
    for txt_file in sorted(WEB_DIR.rglob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8", errors="ignore")
        header = parse_header(raw)
        body = raw.split("---", 1)[1].strip() if "---" in raw else raw
        pages = extract_pages(body)
        documents.append({
            "path": str(txt_file),
            "filename": txt_file.name,
            "topic_folder": txt_file.parent.name,
            "header": header,
            "body": body,
            "pages": pages,
            "source_type": "web",
        })

    # PDF files
    for txt_file in sorted(PDF_DIR.rglob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8", errors="ignore")
        header = parse_header(raw)
        body = raw.split("---", 1)[1].strip() if "---" in raw else raw
        pages = extract_pages(body)
        documents.append({
            "path": str(txt_file),
            "filename": txt_file.name,
            "topic_folder": txt_file.parent.name,
            "header": header,
            "body": body,
            "pages": pages,
            "source_type": "pdf",
        })

    return documents


# ============================================================
# EMBEDDING + INDEX BUILDING
# ============================================================

def embed_texts(texts: list, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed a list of texts using sentence-transformers. Returns L2-normalized float32 array."""
    print(f"  Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    print(f"  Embedding {len(texts)} texts...")
    embeddings = embedder.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=64,
    )
    embeddings = embeddings.cpu().numpy().astype("float32")
    # L2 normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS IndexFlatIP (inner product = cosine on normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def build_bm25_index(texts: list):
    """Build a BM25Okapi index from texts. Returns the BM25 object or None."""
    if not HAS_BM25:
        print("  [SKIP] BM25 not available — install rank-bm25")
        return None

    # Tokenize: simple lowercase + split on non-word characters
    tokenized = [re.findall(r"\w+", t.lower()) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("CHUNKER v8 — Parent-Child Chunking")
    print("=" * 60)

    if not HAS_TIKTOKEN:
        print("NOTE: Using fallback token estimator. Install tiktoken for accuracy.")
    print()

    # --- Step 1: Load documents ---
    print("Step 1: Loading documents...")
    documents = load_all_documents()
    print(f"  Loaded {len(documents)} documents")
    print(f"    Web files: {sum(1 for d in documents if d['source_type'] == 'web')}")
    print(f"    PDF files: {sum(1 for d in documents if d['source_type'] == 'pdf')}")
    print()

    # --- Step 2: Split into parent chunks ---
    print("Step 2: Splitting into parent chunks (~1000 tokens)...")
    all_parents = []
    for doc in documents:
        doc_parents = split_into_parent_chunks(doc["body"], doc["pages"])
        for i, parent in enumerate(doc_parents):
            parent_id = f"{doc['filename']}_parent_{i:03d}"
            # Label with topic
            dominance = dominance_test(parent["text"])

            all_parents.append({
                "parent_id": parent_id,
                "filename": doc["filename"],
                "source_type": doc["source_type"],
                "document_title": doc["header"]["title"],
                "source": doc["header"]["source"],
                "url": doc["header"]["url"],
                "license": doc["header"]["license"],
                "topic_folder": doc["topic_folder"],
                "text": parent["text"],
                "page_num": parent["page_num"],
                "token_count": parent["token_count"],
                "primary_topic": dominance["primary_topic"],
                "secondary_topics": dominance["secondary_topics"],
            })

    print(f"  Created {len(all_parents)} parent chunks")
    print()

    # --- Step 3: Split into child chunks ---
    print("Step 3: Splitting into child chunks (~250 tokens, 40 overlap)...")
    all_children = []
    for parent in all_parents:
        children = split_into_child_chunks(
            parent["text"], parent["parent_id"], parent["page_num"]
        )
        for j, child in enumerate(children):
            child_id = f"{parent['parent_id']}_child_{j:03d}"
            # Label with topic
            dominance = dominance_test(child["text"])

            all_children.append({
                "chunk_id": child_id,
                "parent_id": parent["parent_id"],
                "filename": parent["filename"],
                "source_type": parent["source_type"],
                "document_title": parent["document_title"],
                "source": parent["source"],
                "url": parent["url"],
                "license": parent["license"],
                "topic_folder": parent["topic_folder"],
                "text": child["text"],
                "page_num": child["page_num"],
                "token_count": child["token_count"],
                "is_continuation": child["is_continuation"],
                "primary_topic": dominance["primary_topic"],
                "secondary_topics": dominance["secondary_topics"],
                "is_crisis_topic": dominance["primary_topic"] in CRISIS_TOPICS,
            })

    print(f"  Created {len(all_children)} child chunks")
    print()

    # --- Step 4: Topic distribution stats ---
    print("Step 4: Topic distribution...")
    topic_counts = Counter(c["primary_topic"] for c in all_children)
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        marker = " [CRISIS]" if topic in CRISIS_TOPICS else ""
        print(f"    {topic}: {count}{marker}")
    print()

    # --- Step 5: Embed child chunks ---
    print("Step 5: Embedding child chunks...")
    child_texts = [c["text"] for c in all_children]
    embeddings = embed_texts(child_texts)
    print(f"  Embeddings shape: {embeddings.shape}")
    print()

    # --- Step 6: Build FAISS index ---
    print("Step 6: Building FAISS index (dense)...")
    faiss_index = build_faiss_index(embeddings)
    print(f"  FAISS index: {faiss_index.ntotal} vectors, dim {faiss_index.d}")
    print()

    # --- Step 7: Build BM25 index ---
    print("Step 7: Building BM25 index (sparse)...")
    bm25_index = build_bm25_index(child_texts)
    if bm25_index:
        print("  BM25 index built")
    print()

    # --- Step 8: Save artifacts ---
    print("Step 8: Saving artifacts...")

    # Parent chunks
    parent_path = OUTPUT_DIR / "parent_chunks.json"
    with open(parent_path, "w", encoding="utf-8") as f:
        json.dump(all_parents, f, indent=2, ensure_ascii=False)
    print(f"  {parent_path} ({len(all_parents)} parents)")

    # Child chunks (metadata only — no embeddings)
    child_path = OUTPUT_DIR / "child_chunks.json"
    with open(child_path, "w", encoding="utf-8") as f:
        json.dump(all_children, f, indent=2, ensure_ascii=False)
    print(f"  {child_path} ({len(all_children)} children)")

    # Embeddings
    emb_path = OUTPUT_DIR / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"  {emb_path} ({embeddings.shape})")

    # FAISS index
    faiss_path = OUTPUT_DIR / "faiss_index.bin"
    faiss.write_index(faiss_index, str(faiss_path))
    print(f"  {faiss_path}")

    # BM25 index
    if bm25_index:
        bm25_path = OUTPUT_DIR / "bm25_index.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_index, f)
        print(f"  {bm25_path}")

    # Stats
    stats = {
        "total_documents": len(documents),
        "web_documents": sum(1 for d in documents if d["source_type"] == "web"),
        "pdf_documents": sum(1 for d in documents if d["source_type"] == "pdf"),
        "total_parent_chunks": len(all_parents),
        "total_child_chunks": len(all_children),
        "avg_parent_tokens": sum(p["token_count"] for p in all_parents) / max(len(all_parents), 1),
        "avg_child_tokens": sum(c["token_count"] for c in all_children) / max(len(all_children), 1),
        "topic_distribution": dict(topic_counts),
        "crisis_topic_count": sum(1 for c in all_children if c["is_crisis_topic"]),
        "embedding_dim": embeddings.shape[1],
        "has_tiktoken": HAS_TIKTOKEN,
        "has_bm25": HAS_BM25,
    }
    stats_path = OUTPUT_DIR / "chunk_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  {stats_path}")

    print()
    print("=" * 60)
    print("DONE — chunking complete!")
    print("=" * 60)
    print(f"Parent chunks: {len(all_parents)}")
    print(f"Child chunks:  {len(all_children)}")
    print(f"Crisis chunks:  {stats['crisis_topic_count']}")
    print(f"Output:        {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
