# Full-pipeline integration

Component wiring, sequential/parallel execution, compatibility, latency and regression experiments.

**Upload status:** structure only; this scaffold contains no component experiment submission or dataset. The component owner should replace this status, add their name and list the completed runs when uploading.

[Research overview](../README.md) · [Team upload instructions](../../CONTRIBUTING.md) · [Data-sharing checklist](../DATA_SHARING.md) · [Current runtime implementation](../../nora/src/mental_health_pipeline/orchestrator.py)

## Current integration reference

The shipped baseline runs crisis and boundary assessment in parallel, then applies route-compatible downstream work and complete-response output approval. Speculative RAG and incremental-output work remain experiments unless explicitly enabled and revalidated.

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

- Record the exact pipeline commit and component/asset identities for each run so old server code is not mistaken for the new integration.
- Keep raw-versus-rewritten input tests, router alternatives, serial/parallel trials and output scheduling ablations separate.
- Measure wall-clock end-to-end time and component times, including cold starts, warm runs, device placement, failures and retries. Do not sum nested/parallel timings.
- Describe safe cancellation, missing components, fail-closed behavior and the final approval gate. Research uploads do not change runtime configuration automatically.

## Minimum handoff

1. Add a concise experiment history: question → baseline → change → evidence → selected/rejected outcome.
2. Fill an [experiment card](../templates/EXPERIMENT_CARD.md) for each reported run and link its inputs/configs/results.
3. Document exact commands, dependency versions, hardware, seeds and artifact hashes. Mark unavailable information as unknown; do not invent it.
4. Add dataset cards before data content; complete the sharing review before committing any records.
5. Replace this scaffold's status with an honest uploaded/partial/missing summary and link the actual files.

Uploading research here does not deploy it or replace the selected code under [nora/](../../nora/). Runtime changes require a separate reviewed integration update.
