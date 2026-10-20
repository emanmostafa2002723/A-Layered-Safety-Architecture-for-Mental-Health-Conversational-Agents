# Upload only this source folder

Repository supplied by the user:

https://github.com/emanmostafa2002723/A-Layered-Safety-Architecture-for-Mental-Health-Conversational-Agents

No remote changes are made by these preparation files. The original `Graduation Project` workspace is already a separate Git repository; **do not stage or publish it**. Use only the `NORA_GitHub_Source_2026-10-20` export. Never select the old full pipeline, model folders or private account directory in GitHub Desktop.

## Before pushing

1. Run `python scripts/audit_source.py` inside this export and review the included files.
2. Confirm repository visibility and team permission. Private is recommended until model/data/image attribution and source permissions are reviewed.
3. Inspect the existing GitHub repository. Do not force-push or replace its history. If it already has commits, clone it into a **separate new folder** and copy the reviewed source files into that clone, preserving its existing work. Inspect conflicts before committing.
4. If the remote is truly empty, initialize a dedicated repository **inside this source export**, not its parent. The working branch may be `codex/nora-source`; no repository or remote is initialized automatically by this package.
5. Review the staged file list. No model weights, processed RAG corpus, `.env.server`, `server-assets.local.json`, database, key, runtime directory or uploaded ZIP belongs in the commit.
6. Commit and push through your own GitHub sign-in. Never paste passwords/tokens into chat. No force push.

The small ZIP provided alongside this source folder is a convenience archive for extracting the source. Uploading that ZIP as a GitHub repository file does not create the normal browseable source tree; extract it first and commit its contents.

## After pushing

Use the server setup guide to clone the repository, verify/link the already present assets, restore only missing current classifier assets, check dependencies, then run the private website. Uploading source to GitHub is not approval to expose the app publicly.

The tracked model inventory is checksums and file names only. `requirements` installs software dependencies, **not** Phoenix/crisis/boundary trained weights or RAG text. Exact hashes must still pass on the server.
