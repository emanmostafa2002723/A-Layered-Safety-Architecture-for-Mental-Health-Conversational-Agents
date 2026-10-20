# Experiment card — TEMPLATE (not a completed run)

Copy into `research/<component>/results/<run_id>/EXPERIMENT_CARD.md` and replace every placeholder. Missing information should be explicitly unknown, not guessed.

## Identity and status

- Component / run ID / owner / contributors: FILL IN
- Run date(s), timezone and completion status: FILL IN
- Code commit / branch / external code revision: FILL IN
- Stage: exploratory / training / validation / calibration / final evaluation / integration
- Outcome: selected / rejected / inconclusive / unfinished, with reason

## Question and comparison

- Problem and motivating examples: FILL IN
- Hypothesis and baseline: FILL IN
- What changed, and what was held fixed: FILL IN
- Prior/follow-up experiment links: FILL IN

## Inputs and reproducibility

- Dataset card(s), version(s), exact split IDs and SHA-256 values: FILL IN
- Architecture/base model/tokenizer/adapter revisions and checkpoint hashes: FILL IN
- Calibration/threshold identity and how chosen: FILL IN
- Config/prompt files, prompt hashes and settings: FILL IN
- OS, Python/packages, hardware, device allocation and resource limits: FILL IN
- Seeds, determinism controls and remaining nondeterminism: FILL IN
- Working directory, installation steps, exact commands and expected outputs: FILL IN
- Required external artifacts, authorized access and missing dependencies: FILL IN

## Evaluation

- Frozen evaluation population and sample counts/per-class support: FILL IN
- Split leakage/deduplication checks: FILL IN
- Metrics, denominators, thresholding and confidence intervals if actually computed: FILL IN
- Annotator/judge model identity, review procedure and disagreements: FILL IN
- Warm-up, cold/warm runs, repetitions, concurrency, timeout/failure handling: FILL IN
- End-to-end wall time versus nested/parallel component time: FILL IN

## Results and failure analysis

Link real metric files/plots and permitted per-case predictions. Include errors, false positives/negatives, incomplete runs and excluded cases with reasons. Do not fill this section with anticipated improvements.

- Results and artifacts: FILL IN
- Limitations and known failure cases: FILL IN
- Decision and evidence for selecting/rejecting: FILL IN
- Reproduction attempted by another person? Who/when/outcome, or not attempted: FILL IN
- Connection to the selected runtime / not integrated: FILL IN

## Sharing review

- Data/privacy/license review, approver/date/scope: FILL IN
- Contains only approved synthetic or otherwise permitted examples? FILL IN
- Clinical/release status and claims not established by this experiment: FILL IN
