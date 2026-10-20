# Personal memory and accounts

User-scoped memory, retention, account isolation and cross-chat recall experiments.

**Upload status:** structure only; this scaffold contains no component experiment submission or dataset. The component owner should replace this status, add their name and list the completed runs when uploading.

[Research overview](../README.md) · [Team upload instructions](../../CONTRIBUTING.md) · [Data-sharing checklist](../DATA_SHARING.md) · [Current runtime implementation](../../nora/src/mental_health_pipeline/personal_memory.py)

## Current integration reference

The prototype uses conservative deterministic note extraction, not an LLM memory curator. Memory and saved history are independent opt-ins, OFF by default, with separate maximum 30-day retention for note/history content.

This note identifies the selected integration; it is not evidence that historical experiments, data or weights have been uploaded here.

## Where to put your work

| Folder | Contents |
| --- | --- |
| [experiments/](experiments/) | Controlled experiment runners and ablations |
| [configs/](configs/) | Versioned settings, label maps, thresholds and dependency references |
| [evaluation/](evaluation/) | Scoring code, frozen-case definitions, metrics and error analysis |
| [notebooks/](notebooks/) | Ordered runnable notebooks with outputs cleared and portable paths |
| [data/](data/) | Dataset cards, provenance and only reviewed, approved data |
| [results/](results/) | One documented folder per run: experiment card, metrics and sanitized outputs |

## What reviewers need from this component

- Use invented accounts only; never upload real user databases, hashes, session tokens, encryption keys, exports or transcripts.
- Describe extraction rules, note schema, relevance selection, expiry, corrections/deletion and owner-scoped access.
- Compare matched conversations with/without memory for factual recall, unwanted assumptions, reply quality and latency.
- Keep test fixtures synthetic and mark their provenance. Account software tests are not a privacy compliance certification.

## Minimum handoff

1. Add a concise experiment history: question → baseline → change → evidence → selected/rejected outcome.
2. Fill an [experiment card](../templates/EXPERIMENT_CARD.md) for each reported run and link its inputs/configs/results.
3. Document exact commands, dependency versions, hardware, seeds and artifact hashes. Mark unavailable information as unknown; do not invent it.
4. Add dataset cards before data content; complete the sharing review before committing any records.
5. Replace this scaffold's status with an honest uploaded/partial/missing summary and link the actual files.

Uploading research here does not deploy it or replace the selected code under [nora/](../../nora/). Runtime changes require a separate reviewed integration update.
