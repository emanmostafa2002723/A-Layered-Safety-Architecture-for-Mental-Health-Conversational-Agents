# wellbeing-guardrails
Layered safety guardrails for the WellBeing Agent - Queen's University    Visibility: Private (keep your work private)

---

<div align="center">

<img src="nora/src/mental_health_pipeline/web_nora/assets/nora-logo.png" width="100" alt="NORA logo">

# NORA · A little room to be you.

**A mental-wellbeing companion with a layered safety architecture.**

Calm conversation · Conditional safety guidance · Retrieval-augmented support · User-controlled memory

**Private graduation-project integration · English-speaking adults**

</div>

## The current application

The latest NORA website and full-pipeline integration are in **[`nora/`](nora/)**.
This is a small source-only application: model checkpoints, knowledge-corpus
files, virtual environments, account databases and encryption keys are not
included. Existing server model assets must be verified before reuse.

| Experience | Architecture |
| --- | --- |
| Gentle web chat and customizable avatars | Phoenix V6 adapter on Mistral 7B |
| Optional grounding and reflection activities | Parallel crisis and boundary input assessment |
| Independent opt-in memory and saved history | Label-conditional response policy and semantic RAG |
| View, forget, export and delete your data | Complete Qwen output checking before response display |

### Start here

- **[Project overview and illustrated message flow](nora/README.md)**
- **[Server setup and existing-model reuse](nora/docs/SERVER_SETUP.md)**
- **[Source verification and remaining limits](nora/docs/SOURCE_EXPORT_VERIFICATION.md)**
- **[Model identities and required files](nora/config/runtime_assets.json)**
- **[Third-party notices](nora/THIRD_PARTY_NOTICES.md)**

Run application commands from the **`nora/` subfolder**, not the repository root.
No model weights need to be committed to GitHub.

## Existing WellBeing Agent reference

The original [`wellbeing_agent`](wellbeing_agent) submodule and its Git settings
are preserved. It is separate from the current NORA implementation; it is not
automatically imported or used as its runtime code. Cloning NORA does not
require recursively downloading that submodule.

> **Safety and release status:** NORA is an AI companion, not a human therapist,
> diagnostic service or emergency service. This integration supports private
> testing; it is not authorization for public clinical deployment. Passing
> software tests does not establish clinical effectiveness. Keep this work
> private until the team has reviewed source/data permissions and deployment.
