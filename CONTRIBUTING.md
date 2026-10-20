# Uploading component research to NORA

This guide is for team members adding preprocessing, training, evaluation, notebooks, prompts or reviewed synthetic data. The [research index](research/README.md) assigns a place for each component; owners still need to fill in their names and upload their actual work.

**Keep the runnable application in [nora/](nora/) separate from the research story in [research/](research/).** A research upload must not silently change active weights, thresholds, prompts or web behaviour.

## What to upload

- Source scripts and ordered notebooks for preparation, training and evaluation.
- Versioned configuration, dependency references, exact prompts, seeds and commands.
- Completed experiment cards with linked metrics, failure analysis and selection decisions.
- Dataset cards and only the synthetic/source records approved by the [sharing checklist](research/DATA_SHARING.md).
- Model/artifact identities, hashes and authorized access instructions rather than the weights themselves.

Do not upload whole working directories, archives, virtual environments, downloaded model caches, optimizer states, private account data, API keys or unreviewed conversation logs. The repository remains private; do not change visibility without a new team publication review.

## Choose the right folder

Open the [component table](research/README.md#choose-your-component). Add your component work there, not to the repository root or another person's component. Keep dated/versioned trials and a clear README explaining baseline → changes → evidence → final choice. Templates are incomplete examples, not claims of completed work.

## Option A: GitHub website

1. Sign in with an account that has repository access.
2. Open `research/<your_component>/`, then **Add file → Upload files** (or create a text file). Keep the supplied subfolder organization.
3. Upload only the reviewed files; do not drag in the entire project or a ZIP.
4. Choose a new branch/pull request for the change rather than committing directly to `main`.
5. Explain what is included, what is missing and whether data sharing has been approved. Ask a teammate to review before merging.

For a folder upload, preserve its relative structure and check the final GitHub file list. Empty placeholder folders contain `.gitkeep`; it is safe to remove those specific placeholders once real files are present.

## Option B: Git locally

Run from this repository's root, not from a separate parent workspace. The example is for crisis research; change the component name for your own upload.

```bash
git switch main
git pull --ff-only
git switch -c codex/research-crisis-upload

# Copy the reviewed files into research/crisis_classifier/ first.
git status --short
git add -- research/crisis_classifier/
git diff --cached --stat
git diff --cached --name-only

# Review staged contents; then commit only when ready.
git commit -m "Document crisis classifier experiments"
git push -u origin codex/research-crisis-upload
```

Open a pull request on GitHub. Do not force-push, stage the whole parent workspace, or use `git add -f` to bypass ignored private/model files. Git ignore rules are guardrails, not a comprehensive privacy scanner. A deleted committed secret can remain in history; notify the team immediately and rotate it rather than merely deleting the visible file.

## Before requesting review

- [ ] Component README lists the owner, uploaded runs and honest incomplete/missing items.
- [ ] Every reported run has a completed [experiment card](research/templates/EXPERIMENT_CARD.md) and reproducible command/configuration.
- [ ] Dataset versions have completed [dataset cards](research/templates/DATASET_CARD.md), actual checksums and split definitions.
- [ ] Any record-level upload has documented rights/privacy review for this repository's visibility.
- [ ] Related examples stay in the same split; held-out data was not used for training or threshold selection.
- [ ] Metrics include sample counts, definitions and failures; no invented or selectively renamed final results.
- [ ] Notebook outputs are cleared; no credentials, real personal examples, absolute personal-machine paths or embedded model files.
- [ ] Source links and attribution work; large/restricted artifacts have authorized access instructions instead of content.
- [ ] Nothing in `nora/` changed unless this is a separately explained runtime update.

Use the [research upload PR template](.github/PULL_REQUEST_TEMPLATE/research_upload.md) when opening the review. Team process/manual review is still required; the existing NORA source CI does not automatically approve new research datasets or verify reported experiment results.
