# Crisis classifier

Incoming crisis assessment, data generation, training, calibration and decoding.

**Upload status:** structure only; this scaffold contains no component experiment submission or dataset. The component owner should replace this status, add their name and list the completed runs when uploading.

[Research overview](../README.md) · [Team upload instructions](../../CONTRIBUTING.md) · [Data-sharing checklist](../DATA_SHARING.md) · [Current runtime implementation](../../nora/src/mental_health_pipeline/fresh_crisis.py)

## Current integration reference

The runnable pipeline selects the fresh full-encoder epoch-3 checkpoint with its paired calibration and presence-gated decoding. Earlier ST01c and S600 work belongs in dated experiments, not as an undocumented replacement for the selected release.

This note identifies the selected integration; it is not evidence that historical experiments, data or weights have been uploaded here.

## Where to put your work

| Folder | Contents |
| --- | --- |
| [preprocessing/](preprocessing/) | Source-to-ready-data transformations, validation and split logic |
| [training/](training/) | Training/fine-tuning code and launch instructions; no checkpoints |
| [configs/](configs/) | Versioned settings, label maps, thresholds and dependency references |
| [evaluation/](evaluation/) | Scoring code, frozen-case definitions, metrics and error analysis |
| [notebooks/](notebooks/) | Ordered runnable notebooks with outputs cleared and portable paths |
| [data/](data/) | Dataset cards, provenance and only reviewed, approved data |
| [results/](results/) | One documented folder per run: experiment card, metrics and sanitized outputs |

## What reviewers need from this component

- Record the architecture/tokenizer identity and exact category, presence and severity label mappings.
- Separate training, calibration/threshold selection and untouched final evaluation; document the presence gate and any category override.
- Compare older ST01c/S600 and fresh epoch runs on the same frozen cases where possible. Include false positives, false negatives, negation, ambiguity and per-class metrics.
- Identify the selected weights and calibration by SHA-256. Weights stay outside Git; record an authorized access location and restoration instructions.

## Minimum handoff

1. Add a concise experiment history: question → baseline → change → evidence → selected/rejected outcome.
2. Fill an [experiment card](../templates/EXPERIMENT_CARD.md) for each reported run and link its inputs/configs/results.
3. Document exact commands, dependency versions, hardware, seeds and artifact hashes. Mark unavailable information as unknown; do not invent it.
4. Add dataset cards before data content; complete the sharing review before committing any records.
5. Replace this scaffold's status with an honest uploaded/partial/missing summary and link the actual files.

Uploading research here does not deploy it or replace the selected code under [nora/](../../nora/). Runtime changes require a separate reviewed integration update.
