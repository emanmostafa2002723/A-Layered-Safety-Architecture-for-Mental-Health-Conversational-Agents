# Output guardrail

Generated-answer checks, safety examples, prompted judges, trained alternatives and final selection.

**Upload status:** structure only; this scaffold contains no component experiment submission or dataset. The component owner should replace this status, add their name and list the completed runs when uploading.

[Research overview](../README.md) · [Team upload instructions](../../CONTRIBUTING.md) · [Data-sharing checklist](../DATA_SHARING.md) · [Current runtime implementation](../../nora/src/mental_health_pipeline/qwen_only_output.py)

## Current integration reference

The selected pipeline uses the Qwen output backend plus lightweight rules, policy/quality checks and the final delivery gate. Output DeBERTa and MiniCPM work should be documented as alternatives, not described as the active selection.

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

- Label the assistant candidate, not merely the user's request. Preserve the user/policy/evidence context needed to judge it.
- Record diagnosis/dosing, harmful content, negation, quotations, paragraph-level cases, false blocks and missed violations.
- Separate Qwen prompting from any actual fine-tuning. Keep DeBERTa and MiniCPM trials identifiable, including rejected approaches and reasons.
- Report safety detection, false refusals, grounding limitations and latency on matched held-out examples. Never claim a prompted check has perfect coverage.

## Minimum handoff

1. Add a concise experiment history: question → baseline → change → evidence → selected/rejected outcome.
2. Fill an [experiment card](../templates/EXPERIMENT_CARD.md) for each reported run and link its inputs/configs/results.
3. Document exact commands, dependency versions, hardware, seeds and artifact hashes. Mark unavailable information as unknown; do not invent it.
4. Add dataset cards before data content; complete the sharing review before committing any records.
5. Replace this scaffold's status with an honest uploaded/partial/missing summary and link the actual files.

Uploading research here does not deploy it or replace the selected code under [nora/](../../nora/). Runtime changes require a separate reviewed integration update.
