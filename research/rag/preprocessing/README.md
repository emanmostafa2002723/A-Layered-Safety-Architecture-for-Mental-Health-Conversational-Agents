# Uploaded RAG chunker: offline source, not a deployment step

The uploaded [chunker.py](../../../nora/rag_project/scripts/chunker.py) lives in the application source tree. A static packaging review on 24 October 2026 records its exact 32,102 bytes and SHA-256 `7f8c5c45e01de106e6bed5f6cac313238f9ab6b1b8fd19849717a5cf6192d592` in the application source inventory.

This fixes a missing source-manifest entry. It is **not** an approval of generated knowledge content, a validation of the chunking algorithm, or evidence that rebuilding produces the selected runtime artifacts. The uploaded script itself is unchanged.

## Deployment boundary

- The selected web/pipeline source does not import this module. Serving NORA uses the separately checksum-verified processed RAG assets.
- Do not import or run this builder as part of server setup. It imports ML dependencies and creates `data/processed/` at module import time; execution can load/download an embedding model and overwrite processed files.
- Never run it against restored or hard-linked runtime assets. Writing through a hard link can also alter the older installation's underlying file.
- Any future rebuild belongs in a separate, explicitly chosen research workspace with authorized source documents and a new artifact manifest. Do not replace the pinned corpus/indexes without separate evaluation and integration review.

## Reproduction limitations noted in the static review

- Paths are relative to the current working directory, not to the script. The legacy docstring uses `chunker_v8.py`, while the uploaded filename is `chunker.py`.
- The script uses `cl100k_base` or an approximate word-count fallback; neither establishes the exact Mistral tokenizer count. Record the actual counting backend for any trial.
- Topic keywords are heuristic metadata, not input-crisis classifier predictions. Comments describing ten topics do not match the thirteen keys currently present.
- Long individual sentences are not forcibly split to the configured chunk-size limit. Treat those limits as targets, not validated hard bounds.
- The embedding revision and dependency environment are not pinned by this script alone. Its source documents and rebuilding environment have not been supplied or tested by this packaging review.

Use the [experiment card](../../templates/EXPERIMENT_CARD.md) and [data-sharing checklist](../../DATA_SHARING.md) before claiming a reproducible rebuild. The current review only parses syntax and inventories calls without importing the module or rebuilding an index.
