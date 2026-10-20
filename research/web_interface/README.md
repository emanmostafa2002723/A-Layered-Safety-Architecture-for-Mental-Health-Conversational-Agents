# Web interface and experience

UI design decisions, implementation notes, accessibility checks and frontend performance.

**Upload status:** structure only; this scaffold contains no component experiment submission or dataset. The component owner should replace this status, add their name and list the completed runs when uploading.

[Research overview](../README.md) · [Team upload instructions](../../CONTRIBUTING.md) · [Data-sharing checklist](../DATA_SHARING.md) · [Current runtime implementation](../../nora/src/mental_health_pipeline/web_nora/)

## Current integration reference

The connected NORA interface is plain HTML/CSS/JavaScript backed by the account API. Avatar artwork is not a UI screenshot; label mockups, browser captures and implemented features accurately.

This note identifies the selected integration; it is not evidence that historical experiments, data or weights have been uploaded here.

## Where to put your work

| Folder | Contents |
| --- | --- |
| [design/](design/) | Rights-cleared design sources and synthetic-account screenshots |
| [experiments/](experiments/) | Controlled experiment runners and ablations |
| [evaluation/](evaluation/) | Scoring code, frozen-case definitions, metrics and error analysis |
| [notebooks/](notebooks/) | Ordered runnable notebooks with outputs cleared and portable paths |
| [data/](data/) | Dataset cards, provenance and only reviewed, approved data |
| [results/](results/) | One documented folder per run: experiment card, metrics and sanitized outputs |

## What reviewers need from this component

- Upload design notes/source and rights-cleared assets, with credits and an explanation of implemented versus proposed screens.
- Use screenshots from synthetic demo accounts only; remove names, identifiers, private messages and browser secrets.
- Document keyboard, mobile, text-size, reduced-motion, session-expiry and memory/history control checks. Do not claim manual checks were performed if they were not.
- Separate browser/network/account overhead from model processing; record measurement method and device/browser versions.

## Minimum handoff

1. Add a concise experiment history: question → baseline → change → evidence → selected/rejected outcome.
2. Fill an [experiment card](../templates/EXPERIMENT_CARD.md) for each reported run and link its inputs/configs/results.
3. Document exact commands, dependency versions, hardware, seeds and artifact hashes. Mark unavailable information as unknown; do not invent it.
4. Add dataset cards before data content; complete the sharing review before committing any records.
5. Replace this scaffold's status with an honest uploaded/partial/missing summary and link the actual files.

Uploading research here does not deploy it or replace the selected code under [nora/](../../nora/). Runtime changes require a separate reviewed integration update.
