# Retrieval and query routing

Knowledge preparation, synthetic retrieval/Q&A cases, query routing, retrieval and reranking.

**Upload status:** structure only; this scaffold contains no component experiment submission or dataset. The component owner should replace this status, add their name and list the completed runs when uploading.

[Research overview](../README.md) · [Team upload instructions](../../CONTRIBUTING.md) · [Data-sharing checklist](../DATA_SHARING.md) · [Current runtime implementation](../../nora/src/mental_health_pipeline/phoenix.py)

## Current integration reference

The selected application uses Qwen3 4B with the one-field query prompt, hybrid retrieval, MiniLM embeddings and BGE reranking. MiniCPM and older multi-field Qwen trials are research alternatives, not the current output guard.

This note identifies the selected integration; it is not evidence that historical experiments, data or weights have been uploaded here.

## Where to put your work

| Folder | Contents |
| --- | --- |
| [preprocessing/](preprocessing/) | Source-to-ready-data transformations, validation and split logic |
| [retrieval/](retrieval/) | Retrieval/index-building code and scoring settings; no index binaries |
| [routing/](routing/) | Query-router prompts/adapters and controlled router comparisons |
| [configs/](configs/) | Versioned settings, label maps, thresholds and dependency references |
| [evaluation/](evaluation/) | Scoring code, frozen-case definitions, metrics and error analysis |
| [notebooks/](notebooks/) | Ordered runnable notebooks with outputs cleared and portable paths |
| [data/](data/) | Dataset cards, provenance and only reviewed, approved data |
| [results/](results/) | One documented folder per run: experiment card, metrics and sanitized outputs |

## What reviewers need from this component

- Record source documents/URLs, permissions, snapshot dates, chunking and parent/child relationships. Do not assume retrieved web text can be redistributed.
- Keep synthetic queries, paraphrases and relevance judgments distinct from the underlying knowledge corpus and generated answer claims.
- Compare no-router, older Qwen, one-field Qwen and MiniCPM trials using the same messages, corpus and retrieval settings; preserve malformed/vague query cases.
- Report routing accuracy, relevant-document retrieval, skipped factual questions, reranking effects and cold/warm latency separately. Record corpus/index checksums, not index binaries in Git.

## Minimum handoff

1. Add a concise experiment history: question → baseline → change → evidence → selected/rejected outcome.
2. Fill an [experiment card](../templates/EXPERIMENT_CARD.md) for each reported run and link its inputs/configs/results.
3. Document exact commands, dependency versions, hardware, seeds and artifact hashes. Mark unavailable information as unknown; do not invent it.
4. Add dataset cards before data content; complete the sharing review before committing any records.
5. Replace this scaffold's status with an honest uploaded/partial/missing summary and link the actual files.

Uploading research here does not deploy it or replace the selected code under [nora/](../../nora/). Runtime changes require a separate reviewed integration update.
