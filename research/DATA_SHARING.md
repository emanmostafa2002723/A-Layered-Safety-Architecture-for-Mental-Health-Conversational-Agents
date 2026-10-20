# Should we upload the synthetic data?

**Yes, when it passes a documented sharing review.** Reviewed synthetic classifier/RAG examples, exact labels and generation/splitting code are valuable for reproducing experiments. This scaffold does not approve or upload any actual dataset.

[Research overview](README.md) · [Dataset card](templates/DATASET_CARD.md) · [Manifest example](templates/dataset_manifest.example.json) · [Team instructions](../CONTRIBUTING.md)

## The decision, by data type

| Material | Recommended treatment |
| --- | --- |
| Team-created synthetic crisis/boundary examples | Share the reviewed, frozen version with exact labels, provenance, split assignments, generation prompts and limitations |
| Synthetic RAG queries, paraphrases or question/answer pairs | Share if the prompts, source passages and generated outputs have been reviewed for rights/privacy; include relevance targets and stable document IDs where applicable |
| Original RAG source documents, copied web/book content or third-party datasets | Do not upload by default. Check the source terms and authorization; otherwise publish citations, metadata and rebuild/access instructions only |
| Real user/participant messages, account exports or clinical/private records | Do not upload to this source repository. A private GitHub repository is not a substitute for an authorized sensitive-data process |
| Model checkpoints, adapters, embedding/index binaries and optimizer states | Keep outside Git; record an authorized artifact location, version, size and SHA-256 |
| Metrics, charts and per-example outputs | Share after removing private content and verifying all included examples are permitted; aggregate metrics can still expose small sensitive groups |

## Synthetic does not automatically mean anonymous or redistributable

A model can reproduce or transform material from real examples used in its prompt or source corpus. Review those inputs, copied spans and generated outputs. Do not treat replacing names, marking rows synthetic or hosting them privately as proof of permission or privacy. NIST explains why synthetic-data methods do not automatically provide formal privacy guarantees in its [synthetic-data overview](https://www.nist.gov/blogs/cybersecurity-insights/differentially-private-synthetic-data).

Do not claim differential privacy unless the actual method, parameters and guarantee have been established. This is a project-sharing checklist, not legal advice or privacy certification.

## Review before adding records

- [ ] **Origin:** identify the creator, dates, generation model/version, source datasets/documents and any real-example seeding. Mixed synthetic/real data must be labelled as mixed.
- [ ] **Permissions:** check source-data rights, generation-provider/model terms and team/university permission. State a real applicable data license or access restriction; do not invent an open license.
- [ ] **Privacy:** inspect for names, contact details, addresses, account identifiers, private narratives, credentials and copied sensitive passages. Document the method and limits of review.
- [ ] **Content:** add an appropriate warning for sensitive synthetic mental-health/crisis examples. Do not expose unnecessary harmful procedural detail just to document a label.
- [ ] **Labels:** include exact schema, integer/string mappings, annotation instructions, human/model review and known noisy/ambiguous cases.
- [ ] **Leakage:** group paraphrases, shared source scenarios/conversations and augmentation descendants before splitting. Freeze test data; keep calibration separate when used. Deduplicate across splits and report checks.
- [ ] **Identity:** retain original IDs/version history, row counts, file SHA-256 values and the exact split membership. If text is sanitized after a reported experiment, version that change and disclose that it is not the identical input.
- [ ] **Approval:** record reviewer, review date, approved scope and visibility. Approval for this private repository is not automatic approval for a later public release.

Automatic scans can help find problems but cannot replace the review. If any item is unresolved, upload the card/code/allowed metadata only and mark record-level access restricted.

## What an approved dataset folder should contain

```text
research/<component>/data/<dataset_name>_v1/
├── DATASET_CARD.md           Completed provenance, limitations and sharing review
├── manifest.json             Actual counts, file sizes, hashes and split identities
├── schema.json               Exact fields and label mapping, when appropriate
├── train.jsonl               Approved records only, if this split exists
├── validation.jsonl          Approved records only, if this split exists
├── calibration.jsonl         Separate calibration records, only when applicable
└── test.jsonl                Frozen evaluation records, if approved for sharing
```

These are naming examples, not files already included. CSV is also fine if its schema and encoding are documented. Keep generation/preprocessing code in the component's corresponding folders and link it from the card. Do not publish private prompts or hidden provider reasoning traces.

## What matters especially for our components

**Crisis and boundary:** document presence/category/severity and boundary-policy labels exactly, including negation, temporal status, ambiguous cases, typos and paraphrases. Synthetic test performance is not evidence of clinical validation or representative real-world prevalence.

**RAG:** separate query/relevance evaluation data from the knowledge corpus. A synthetic question generated from a protected passage does not automatically authorize redistributing that passage or its answer. Include permitted source IDs, chunking/rebuild settings and source snapshots/checksums where available.

**Output guard:** label the generated assistant candidate together with the context needed to evaluate it. Do not relabel a user-only input dataset as output-safety ground truth without a documented conversion and review.

## Size and access

Keep ordinary code and small approved text datasets in Git. For large approved datasets, use a team-approved versioned dataset store and link its immutable revision/checksums; a gated location may be appropriate. Do not move sensitive or unlicensed data to a public dataset host to bypass this review.

GitHub blocks ordinary Git files above 100 MiB and recommends separate large-file handling; see [GitHub's official guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Do not use LFS or ZIP files as a reason to commit model stores or private data. [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards) are a useful reference for provenance, usage and limitations documentation; using that format does not itself approve publication.
