# NORA: connected website

> Historical guide from the complete local development folder. This source-only
> export does not contain its Windows environments, bundled Ollama or
> `START_NORA.ps1`. Use [SERVER_SETUP.md](SERVER_SETUP.md) and `run.py` for this
> checkout. The privacy/control descriptions below document the same account UI.

The final NORA V2 design is now the account website served by this project. It is not the simulated design preview. All code, images, model configuration and database controls are inside this pipeline folder.

## Open it on this Windows device

Stop any other pipeline/Gradio chat first: two model processes can exhaust the GPU.

Open PowerShell in this folder and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\START_NORA.ps1
```

Open **http://127.0.0.1:7861**. Keep the terminal running. Ctrl+C stops the website. The launcher starts/reuses the bundled local Qwen service; it does not download models. Ollama may remain running after the website stops. A different port is supported with `-Port 7862`.

Select **Sign in → Create account**, choose an alias and a password (12–128 characters), and confirm you are 18 or older. No email is required. Password recovery is not implemented in this private test; use a test password you can remember.

## What is connected

- Chat: original user text → existing input guardrails and retrieval → conditional response policy → Phoenix V6 generation → full Qwen output/final checks → approved public response. No fake chat replies and no unchecked token streaming.
- New chats, saved conversations and user isolation use the existing account service.
- My space: name, six avatars, avatar accessories/colour, theme, motion, text size, quiet mode, confirmed country and resource bookmarks are encrypted account preferences. They survive signing out/restarting. They are not model instructions or diagnoses.
- Memory and saved history are independent, OFF by default. Opt-in memory keeps limited dated user reports for 30 days, not complete transcripts. Saved history, if enabled, separately keeps conversations for at most 30 days.
- View/forget notes; delete individual chats; export your account data; sign out; delete account and data.
- Breathing/grounding/reflection activities run locally in the tab. Journal writing and mood check-ins are not sent to the AI or stored as memory. Videos open externally only on request; nothing autoplays.
- Support contacts come from the pipeline resource directory using a country only when explicitly confirmed. This does not represent new clinical/content approval of that directory.

Turning history OFF deletes saved transcripts and active chat context, not separate memory notes. Turning memory OFF deletes notes, not separately saved transcripts. Unsaved displayed messages have a bounded temporary tab cache; reloading/signing out clears it. No account password, session token, transcript or memory note is put in localStorage/sessionStorage.

The pipeline also uses bounded short-term RAM context (up to 12 recent turns, expiring after an hour of inactivity) to continue an active chat even when saved history is off. Signing out clears the browser view, not the entire model process. Delete the conversation entry to remove that chat's server context. The Conversations panel lets you manage/delete entries even with saved history off; entries contain basic dates and turn counts, not saved transcripts.

Public bundled images may be cached for one hour to avoid downloading them on every view change. Account responses, chats, memory, HTML and JavaScript remain no-store. No third-party scripts, web fonts, analytics, image services or video players are needed to render the interface.

Account aliases and basic timestamps/IDs are database metadata. Notes, transcripts and interface preferences are encrypted at rest; the encryption key is a separate local file, not external key management. The maximum 30-day retention applies to memory and saved chat content, **not** to account/profile preferences, which remain until changed/deleted.

## Files

```text
START_NORA.ps1                           Windows launcher
scripts/run_account_memory.py            Same-origin local app + existing pipeline
src/mental_health_pipeline/web_nora/      Final connected HTML/CSS/JS/images
src/mental_health_pipeline/account_api.py Authenticated API; no anonymous model endpoint
src/mental_health_pipeline/account_service.py  Owner-scoped pipeline calls
src/mental_health_pipeline/account_store.py    Encrypted account storage
src/mental_health_pipeline/account_preferences.py  Bounded UI settings
.runtime/private-accounts/               PRIVATE database + encryption key (never upload)
web/nora-design-v2/                      Separate visual preview; simulated chat only
```

Do not open the connected HTML with a file:// URL. It needs this server and account API. The previous diagnostic account screen remains at `/dev/account-console`; the Gradio testing console is unchanged.

## Linux later

After installing the existing project requirements and configuring local model/worker paths, start Qwen using the server's existing Ollama installation, then run from the project root:

```bash
.venv-api/bin/python scripts/run_account_memory.py --port 7861
```

It binds only to 127.0.0.1. Use an authenticated SSH/VS Code port forward for private testing. Do not expose this endpoint with `0.0.0.0`, a public tunnel or a Gradio share link. A public site requires an explicitly reviewed HTTPS/auth/session/deployment configuration; this change does not approve public clinical deployment.

## Verification

`tests/test_nora_integration.py` covers the packaged interface, strict preferences, encryption, owner isolation, country confirmation and checked-response API. `tests/nora_ui_contracts.cjs` executes the shipped scripts against DOM/fetch doubles, including late replies after logout. It is not browser layout/accessibility testing. `scripts/test_nora_full_pipeline.py` runs synthetic account requests over real loopback HTTP through the learned models and writes an evidence report. It never uses your real account database.

The full safety pipeline is retained, not certified perfect: model misclassification, response quality and false refusals remain possible. First requests can be slow during model loading. Inspect test evidence before claiming deployment readiness.
