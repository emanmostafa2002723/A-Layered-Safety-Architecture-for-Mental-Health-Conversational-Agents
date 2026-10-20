# Phoenix generation

Mistral/Phoenix data preparation, LoRA training, prompts and conversational evaluation.

**Upload status:** structure only; this scaffold contains no component experiment submission or dataset. The component owner should replace this status, add their name and list the completed runs when uploading.

[Research overview](../README.md) · [Team upload instructions](../../CONTRIBUTING.md) · [Data-sharing checklist](../DATA_SHARING.md) · [Current runtime implementation](../../nora/src/mental_health_pipeline/generation.py)

## Current integration reference

The selected runtime uses Mistral-7B-Instruct-v0.3 with the Phoenix V6 adapter. Keep earlier Phoenix versions as clearly identified baselines; do not substitute an adapter just because its directory is called final_model.

This note identifies the selected integration; it is not evidence that historical experiments, data or weights have been uploaded here.

## Where to put your work

| Folder | Contents |
| --- | --- |
| [preprocessing/](preprocessing/) | Source-to-ready-data transformations, validation and split logic |
| [training/](training/) | Training/fine-tuning code and launch instructions; no checkpoints |
| [configs/](configs/) | Versioned settings, label maps, thresholds and dependency references |
| [prompts/](prompts/) | Exact prompt versions, generation/annotation instructions and templates |
| [evaluation/](evaluation/) | Scoring code, frozen-case definitions, metrics and error analysis |
| [notebooks/](notebooks/) | Ordered runnable notebooks with outputs cleared and portable paths |
| [data/](data/) | Dataset cards, provenance and only reviewed, approved data |
| [results/](results/) | One documented folder per run: experiment card, metrics and sanitized outputs |

## What reviewers need from this component

- Document base-model revision/tokenizer, adapter identity, dataset versions, trainable parameters, epochs, seeds and dependencies.
- Keep train/validation/test lineage separate, including paraphrases and synthetic source conversations.
- Compare adapters and prompts with matched generation settings; test greetings, follow-ups, topic changes, RAG use, stopping and instruction adherence.
- Store sanitized synthetic example outputs and failure analysis; record adapter SHA-256 and authorized artifact access separately from Git.

## Minimum handoff

1. Add a concise experiment history: question → baseline → change → evidence → selected/rejected outcome.
2. Fill an [experiment card](../templates/EXPERIMENT_CARD.md) for each reported run and link its inputs/configs/results.
3. Document exact commands, dependency versions, hardware, seeds and artifact hashes. Mark unavailable information as unknown; do not invent it.
4. Add dataset cards before data content; complete the sharing review before committing any records.
5. Replace this scaffold's status with an honest uploaded/partial/missing summary and link the actual files.

Uploading research here does not deploy it or replace the selected code under [nora/](../../nora/). Runtime changes require a separate reviewed integration update.
