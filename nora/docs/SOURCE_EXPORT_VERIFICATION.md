# Source-only export verification — 20 October 2026

## Result

- **392 passed, 3 skipped**, 29.95 seconds on the available local Python environment.
- Three skips are the actual RAG-corpus retrieval probes. They run again after the verified corpus files are restored. Missing model/data assets were not replaced with placeholder inference.
- Synthetic account HTTP tests and shipped browser JavaScript contract tests passed. The connected interface serves the original approved artwork and uses authenticated real endpoints, not preview sample replies.
- Current input-boundary source imports in an isolated process without model files. Current Qwen output component initializes its rules/schemas without calling an inference service. The unapproved standalone-provider path also retains its abstain behavior.
- Asset linker tests cover read-only planning, exact hashes, missing mappings, safe paths, preserving existing mismatched targets and rollback of only newly created links on a simulated failure.
- No model tensor, trained adapter, knowledge-corpus text/index binary, private database, encryption key, environment or original model cache was copied into this source export.

The original full development copy recorded 711 passing tests/two environment-specific skips and a real-model HTTP smoke run. Those are historical development evidence, **not** a fresh learned-model benchmark of this source export or a Linux/server result. The export tests used the available local Python 3.13 environment; the declared deployment target remains Python 3.12.

## GitHub verification

On 20 October 2026, the submission workflow passed on Ubuntu with Python 3.12 and Node 22: **182 passed, 1 warning in 25.34 seconds**, plus the 217-file source integrity check. Both the push and pull-request jobs passed. The selected CI suite covers accounts, interface contracts, response policy, resources and server asset-linking behavior without downloading or loading models.

[Verified pull-request run](https://github.com/emanmostafa2002723/A-Layered-Safety-Architecture-for-Mental-Health-Conversational-Agents/actions/runs/39468400530). This is source-level CI evidence, not a claim that the university server has been migrated or the learned pipeline has been re-benchmarked there.

## Why the size is small

The source includes the application/UI, selected component source, prompts, configuration examples, manifests, tests and documentation. The runtime asset inventory contains exact names, byte sizes and SHA-256 values—not file content. Restoring those external assets is mandatory for inference. Qwen weights remain in the server's Ollama service and are checked by pinned digest at runtime.

The server inventory supplied by the user establishes:

- Phoenix V6 final-model adapter weights and its adapter/tokenizer configs match the current local version.
- Six processed RAG files in both reported server locations match the current local version.
- Mistral, embedding/reranker and Qwen files are present, but all required asset hashes still need the server linker check.
- Current crisis/boundary bundles were not found by the limited server scan. This does not prove absence outside its scan roots. The classifier-only packaging helper is available if a separate upload is required.
- About 16 GB free was reported after the original uploaded full-pipeline ZIP was removed. No further server deletion or cleanup is performed by this export.

## Source audit

`python scripts/audit_source.py` checks the reviewed source-file inventory, Python syntax, accidental model/private file inclusion, unexpected large files and a small set of common secret patterns. It is read-only. It is not a comprehensive secret scanner or license/privacy certification. `.gitignore` is a second line of defense, not proof of safe publication.

After restoring models, use `--allow-runtime-assets` for a server-side source audit, then `scripts/link_server_assets.py --check` for model file hashes. The API's existing hash checks and complete output gate remain active; this export does not relax runtime path checks.

## Remaining release checks

1. Review GitHub repository contents/history and visibility before publishing; no force push or overwrite.
2. Confirm team/source/image/model/data redistribution permissions. No project-wide open-source license has been selected automatically.
3. Linux/Python 3.12 source CI has passed; still review the existing university-server packages before any installation.
4. Restore and hash-check current assets, then run Qwen, RAG, crisis and full guarded smoke tests on the server.
5. Manually check the connected website in a browser. Browser automation was unavailable in the earlier interface work; DOM/fetch doubles are not visual/accessibility testing.
6. Keep deployment private/loopback. Public HTTPS, identity/recovery, operational privacy and clinical acceptance are not resolved by a GitHub upload.
