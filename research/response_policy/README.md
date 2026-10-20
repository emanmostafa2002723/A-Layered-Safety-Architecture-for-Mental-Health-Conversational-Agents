# Response policy

Label-to-route mapping, conditional instructions, contact display, turn state and safe completion.

**Upload status:** structure only; this scaffold contains no component experiment submission or dataset. The component owner should replace this status, add their name and list the completed runs when uploading.

[Research overview](../README.md) · [Team upload instructions](../../CONTRIBUTING.md) · [Data-sharing checklist](../DATA_SHARING.md) · [Current runtime implementation](../../nora/src/mental_health_pipeline/response_policy/)

## Current integration reference

The application uses short general guidance plus conditional instructions derived from the input guardrails. An ordinary greeting should not receive a crisis checklist; a crisis-count annotation alone must not reclassify every later turn.

This note identifies the selected integration; it is not evidence that historical experiments, data or weights have been uploaded here.

## Where to put your work

| Folder | Contents |
| --- | --- |
| [policies/](policies/) | Versioned label-to-action rules, instruction templates and policy tables |
| [configs/](configs/) | Versioned settings, label maps, thresholds and dependency references |
| [prompts/](prompts/) | Exact prompt versions, generation/annotation instructions and templates |
| [evaluation/](evaluation/) | Scoring code, frozen-case definitions, metrics and error analysis |
| [notebooks/](notebooks/) | Ordered runnable notebooks with outputs cleared and portable paths |
| [data/](data/) | Dataset cards, provenance and only reviewed, approved data |
| [results/](results/) | One documented folder per run: experiment card, metrics and sanitized outputs |

## What reviewers need from this component

- Document input/output contracts, precedence when labels disagree and which instructions are generated for each label combination.
- Preserve policy versions, resource-display conditions, bounded recent safety state and fallback/retry rules.
- Use synthetic multi-turn cases to show normal support, boundary restriction, crisis escalation and recovery from a mistaken label.
- Link each claimed improvement to the actual policy change and comparable outputs. Do not use clinical approval language for software-policy tests.

## Minimum handoff

1. Add a concise experiment history: question → baseline → change → evidence → selected/rejected outcome.
2. Fill an [experiment card](../templates/EXPERIMENT_CARD.md) for each reported run and link its inputs/configs/results.
3. Document exact commands, dependency versions, hardware, seeds and artifact hashes. Mark unavailable information as unknown; do not invent it.
4. Add dataset cards before data content; complete the sharing review before committing any records.
5. Replace this scaffold's status with an honest uploaded/partial/missing summary and link the actual files.

Uploading research here does not deploy it or replace the selected code under [nora/](../../nora/). Runtime changes require a separate reviewed integration update.
