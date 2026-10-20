# NORA research and reproducibility

**The experiment story, separate from the application.**

[Project overview](../README.md) · [Upload instructions](../CONTRIBUTING.md) · [Synthetic-data sharing](DATA_SHARING.md) · [Experiment template](templates/EXPERIMENT_CARD.md) · [Dataset template](templates/DATASET_CARD.md)

> **Current status: scaffold only.** These folders reserve places for each component owner's work. This update does not copy historical experiments, upload datasets, train models or manufacture results. Existing runtime tests documented under `nora/` are a different evidence source.

## Choose your component

| Folder | What belongs here | Owner / upload status |
| --- | --- | --- |
| [Crisis classifier](crisis_classifier/) | Incoming crisis assessment, data generation, training, calibration and decoding. | Owner to be assigned · awaiting upload |
| [Input boundary guardrail](boundary_guardrail/) | Synthetic input-boundary cases, preprocessing, training, calibration and boundary decisions. | Owner to be assigned · awaiting upload |
| [Retrieval and query routing](rag/) | Knowledge preparation, synthetic retrieval/Q&A cases, query routing, retrieval and reranking. | Owner to be assigned · awaiting upload |
| [Phoenix generation](generation/) | Mistral/Phoenix data preparation, LoRA training, prompts and conversational evaluation. | Owner to be assigned · awaiting upload |
| [Output guardrail](output_guardrail/) | Generated-answer checks, safety examples, prompted judges, trained alternatives and final selection. | Owner to be assigned · awaiting upload |
| [Response policy](response_policy/) | Label-to-route mapping, conditional instructions, contact display, turn state and safe completion. | Owner to be assigned · awaiting upload |
| [Full-pipeline integration](system_integration/) | Component wiring, sequential/parallel execution, compatibility, latency and regression experiments. | Owner to be assigned · awaiting upload |
| [Personal memory and accounts](personal_memory/) | User-scoped memory, retention, account isolation and cross-chat recall experiments. | Owner to be assigned · awaiting upload |
| [Web interface and experience](web_interface/) | UI design decisions, implementation notes, accessibility checks and frontend performance. | Owner to be assigned · awaiting upload |

Owners should add their names, actual experiment dates and links to completed uploads in their component README. Do not infer authorship from who uploads a file.

## What a reviewer should be able to reconstruct

1. **Starting point:** the baseline, exact model/data versions and initial failure cases.
2. **Data preparation:** provenance, synthetic generation prompts, cleanup, labels, split logic and leakage checks.
3. **Trial:** the change, exact command/configuration, environment, hardware, random seed and checkpoint identity.
4. **Evidence:** evaluation population, metric definitions, per-case errors where safe, latency and failures.
5. **Decision:** why a candidate was selected, rejected or left unapproved; link the actual evidence.
6. **Integration:** which code/artifacts were selected for the runnable pipeline and what remains experimental.

A run should be understandable without access to someone's Downloads folder. Use repository-relative paths and documented local asset mappings. External files need versioned, authorized access references and SHA-256 values.

## Standard organization

Each component has a short README, a data guide, a results guide and only the relevant code folders. Empty directories contain `.gitkeep` so GitHub displays them; remove that placeholder when adding real files if desired.

```text
research/
├── README.md
├── DATA_SHARING.md
├── templates/
│   ├── EXPERIMENT_CARD.md
│   ├── DATASET_CARD.md
│   ├── dataset_manifest.example.json
│   └── run_manifest.example.json
├── crisis_classifier/
├── boundary_guardrail/
├── rag/
├── generation/
├── output_guardrail/
├── response_policy/
├── system_integration/
├── personal_memory/
└── web_interface/
```

Training components include `preprocessing/`, `training/`, `configs/`, `evaluation/` and `notebooks/`; retrieval, policy, UI and integration folders use task-specific equivalents. Each component's own README is the map.

## Submission conventions

- One dataset version per data subfolder; one actual run per results subfolder.
- Dataset files under component `data/` folders retain their committed bytes rather than automatic line-ending conversion; compute checksums on the exact released files. Documentation has separate text handling.
- Preserve failed trials and negative results; distinguish baselines, calibration and final evaluation.
- Notebook files should have cleared outputs and no personal paths, tokens or embedded private examples.
- Upload scripts, configs, prompts, dependency references and reviewed result summaries. Keep checkpoints/index binaries and optimizer states outside Git.
- Use small, reviewed synthetic datasets when sharing is approved. Restricted inputs get a dataset card and reconstruction/access instructions, not a copied corpus.
- Research code is for review/reproduction. Nothing under this folder is automatically imported by the web application.

## Start here

1. Open your component README.
2. Follow [CONTRIBUTING](../CONTRIBUTING.md) and fill the relevant templates.
3. Open a pull request for your component; do not drop an entire working directory or ZIP into the repository.
4. Ask another team member to review privacy, provenance, split integrity and claims before merging.
