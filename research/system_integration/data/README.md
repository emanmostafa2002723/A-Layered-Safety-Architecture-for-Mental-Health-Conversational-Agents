# Data for full-pipeline integration

**No data has been uploaded or approved by this scaffold.** Read the [sharing checklist](../../DATA_SHARING.md) before adding records.

Use one immutable version folder per dataset, for example `dataset_name_v1/`. It should contain:

- `DATASET_CARD.md`: fill the [template](../../templates/DATASET_CARD.md), including source rights, synthetic generation and review status.
- `manifest.json`: fill the [manifest example](../../templates/dataset_manifest.example.json) with real file hashes, record counts and split identities.
- Approved content only, such as `train.jsonl`, `validation.jsonl`, `calibration.jsonl` when applicable and `test.jsonl`.
- A documented schema/label map and generation/split scripts, either here or linked in the component folders.

The example filenames are a convention, not files already supplied. Do not fabricate empty datasets or mark an unreviewed dataset approved.

If data cannot be redistributed, include only its card and permitted metadata, a versioned authorized access link, checksums and regeneration/rebuild instructions. Do not copy the protected content into Git. Keep local raw/private data outside the checkout, or in the ignored `data/local/` area; ignoring files is not a privacy guarantee.
