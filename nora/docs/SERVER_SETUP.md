# Server migration: small source, existing assets

## What is preserved

The older installation is `/home/jovyan/full_pipeline_server/Mental_Health_Chat_Full_Pipeline_ST01c_PhoenixV3_2026-10-14`. It is **not** the source of the current application code. Reuse only checksum-matching model/index files. Do not import its Python packages, copy its old response policy, or swap to its ST01c/V3 weights.

The latest source includes the connected NORA V2 website, Phoenix V6 selection, calibrated epoch-3 crisis integration, final input boundary, one-field Qwen RAG routing, conditional response guidance and full Qwen output checks. Training assets and experiment result folders are excluded. Some compatibility/experiment modules remain as import dependencies; they are not active backend selections.

## 1. Clone the code

Once the reviewed source has been pushed to the user's repository:

```bash
cd /home/jovyan
git clone https://github.com/emanmostafa2002723/A-Layered-Safety-Architecture-for-Mental-Health-Conversational-Agents.git nora-current
cd /home/jovyan/nora-current/nora
python scripts/audit_source.py
```

Do not run this clone over an existing directory. The current application lives in the repository's `nora/` subfolder; run the remaining commands there. A private repository requires your own GitHub authentication; never paste a token into shared terminal output or chat. No external application submodule is required.

## 2. Connect existing assets

```bash
cp config/server-assets.example.json config/server-assets.local.json
python scripts/link_server_assets.py --sources config/server-assets.local.json --check
```

`--check` is read-only. It hashes candidate files; this can take several minutes. It prints missing or mismatched files without importing old code or loading a model. The example includes the confirmed server paths for Phoenix V6 and RAG; the other model paths are candidates until hashes pass. Crisis/boundary source mappings remain `null` until the current asset bundle is placed on the server.

When all required files pass, explicitly apply:

```bash
python scripts/link_server_assets.py --sources config/server-assets.local.json --apply
```

This creates **hard links**, not duplicate weight copies or directory symlinks. It requires files and checkout to be on the same filesystem. Sources and permissions are not changed. A hard-linked weight file shares an inode with its original: do not edit either copy in place. Removing one path does not remove the data while another hard link exists. There is no automatic cross-filesystem copy fallback. All assets must verify before links are created; failed link creation rolls back only links created by that call.

**Do not delete the older installation yet.** We will first verify all new links and successful inference, and retain any needed rollback/config files. The separate ~1.03-GiB classifier weights and their small supporting files are not GitHub uploads.

## 3. Environment: inspect before installing

```bash
bash setup_linux.sh
.venv-api/bin/python scripts/check_dependencies.py
.venv-api/bin/python -m pip check
```

The setup creates a new project-local Python 3.12 venv with system packages visible. It does not install or download anything and refuses to overwrite an existing venv. Review the dependency output first. If packages are missing, use the runtime extra and observed-version constraints deliberately; Torch/CUDA selection must match the host, rather than triggering another multi-GB install blindly.

```bash
# Run only after reviewing which packages are missing and available storage.
.venv-api/bin/python -m pip install --no-cache-dir -e '.[runtime]' -r requirements/server-observed.txt
```

If the old venv contains compatible packages unavailable in the system environment, inspect it before choosing an alternative; do not point worker executables outside this project to bypass path checks.

## 4. Configure and verify

```bash
cp config/server-assets.env.example .env.server
.venv-api/bin/python scripts/link_server_assets.py --check
ollama list
.venv-api/bin/python run.py qwen-check
.venv-api/bin/python run.py preflight
.venv-api/bin/python run.py rag-check
.venv-api/bin/python run.py smoke --message 'Hi'
```

`qwen-check` checks the selected one-field router's prompt/model identity, without inference. `rag-check` exercises retrieval, and `smoke` runs a complete guarded turn. Hash/prompt checks, acceptance warnings and errors must not be bypassed. Preflight verifies the external-asset manifest and can take time reading the Mistral shards.

Local settings retain CPU Qwen and the tested small-GPU allocation. The A40 has more memory, but GPU placement of the router/output judge/classifiers must be benchmarked separately. Both router configuration and `config/local_output_guard.json` must agree on Qwen placement/context to avoid reload thrashing. Preserve all output checks.

## 5. Open the website privately

```bash
.venv-api/bin/python run.py web
```

NORA listens at `http://127.0.0.1:7861`. Use VS Code's authenticated port forwarding. Stop with Ctrl+C. Use a single model process, not simultaneous web/Gradio pipelines competing for GPU memory.

Optional component console:

```bash
.venv-api/bin/python run.py gradio
```

For Jupyter, choose the `.venv-api` interpreter and run the same commands in a terminal, or invoke `run.py` in a subprocess from your notebook. A public Colab tunnel is not an approved replacement for the local account/HTTPS security boundary.

## Private data

New account data is created only under `.runtime/private-accounts/`. Never upload that directory or its separate encryption key. The app remains a private testing candidate, not a public clinical release. Password recovery and public HTTPS deployment are not implemented by this migration.
