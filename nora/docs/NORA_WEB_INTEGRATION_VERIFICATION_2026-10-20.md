# Connected NORA: integration and latency evidence

The final NORA V2 interface is connected to the existing account service and full pipeline. The design-only preview is preserved separately. The Sites workflow was used locally; no public site, tunnel, Git push or hosted deployment was created.

## Software checks

- Final full regression run: **711 passed, 2 skipped, 72.38 seconds**. Skips: a test applying only when optional FastAPI is absent, and Windows symlink creation not permitted in the test environment.
- The new connected-site suite contributes 18 tests, including encryption, durable account preferences, isolation, strict input bounds, packaged assets, public-image caching versus private no-store, and approved-only response delivery.
- The shipped JavaScript ran against DOM/fetch doubles: actual account endpoints, escaped messages/sources, independent memory/history choices, failed profile saves, temporary chat switching, session expiry and discarding late responses after logout. These are software contracts, not browser rendering/accessibility checks.
- A local wheel build included all **11** connected-interface files (HTML, two scripts, CSS, seven images). Source packaging includes these files through `pyproject.toml`.

## Real learned-model requests over HTTP

Evidence: `reports/nora_connected_full_pipeline_20261020T162319Z.json`.

The test created only a synthetic account in a temporary encrypted database, served the actual NORA HTML/scripts over a real loopback HTTP socket, and called the real message API. Phoenix V6 ran on the local GPU with the selected crisis/boundary/RAG/response-policy stack and complete Qwen output checking. Profile/avatars persisted, a sleep note was recalled in a new chat, export worked, and account deletion invalidated the session. The test server and model pipeline were closed afterwards. It did not use the real account database.

| Case | Route | Pipeline seconds | HTTP round-trip seconds | HTTP/account overhead |
| --- | --- | ---: | ---: | ---: |
| Hi | normal_chat | 39.784 | 39.833 | 0.049 s |
| Trouble sleeping | normal_chat | 24.713 | 24.994 | 0.281 s |
| Sleep recall in a new chat | normal_chat | 8.757 | 8.796 | 0.039 s |
| Double prescribed medication dose | clinical_boundary_safe_completion | 15.922 | 15.956 | 0.034 s |

All four returned nonempty, released responses through the final gate. Greeting had no emergency block. Recall used the dated saved sleep report. The medication reply declined a personalized dose change. This tiny smoke set does **not** establish clinical safety or broad model quality.

The earlier report ending `161948Z` contains four successful chat cases but a failed cleanup request: the test harness omitted its required JSON header on DELETE. The browser client already supplied that header. The harness was corrected and the complete rerun above passed, including deletion. Keep the earlier report as an honest audit record, not a second complete acceptance run.

## Is the website the latency problem?

Not in these local API measurements. HTTP plus account/database handling added about **0.03–0.28 seconds** beyond the measured pipeline, approximately 0.1–1.2% of each final-run request. Browser JavaScript execution/painting and internet/Colab forwarding delay are **not** included: these were real HTTP requests, not an automated browser session. Regression tests ran concurrently, so these are functional smoke timings, not controlled frontend A/B benchmarks or percentile estimates.

Examples of the actual bottlenecks in that run:

- First greeting: input-guard startup/assessment 20.45 s; Qwen routing 7.30 s; generation 2.31 s; full final output gate 9.61 s.
- Warm sleep turn: input guards 0.36 s; routing 2.22 s; generation 9.63 s; final output gate 12.47 s.
- Generator preload was measured separately at 53.41 s. The earlier run's first message took 81.08 s, showing why cold-start timings must not be confused with steady-state frontend overhead.

Nested/parallel component rows overlap; do not sum every timing row.

### Lightweight frontend measures

- Plain HTML/CSS/JavaScript; no frontend framework runtime, web fonts, analytics, third-party scripts, embedded video player or extra AI calls for styling/settings.
- Seven local public images; resource photos/thumbnails lazy-load. Total connected source plus art is about 3.17 MiB, not a per-message payload.
- Public images cache for one hour. Account/API data, HTML and scripts stay no-store. External videos open only when requested.
- No polling loop or per-token browser/API chatter. One message request waits for the checked full response; safety checks are not bypassed for speed.

### Server/Colab later

More GPU memory can help the models, but merely moving the files does not move every component onto the GPU. The current local settings explicitly use **CPU Qwen** in both `.env.qwen-server` (`MHP_QWEN_NUM_GPU=0`) and `config/local_output_guard.json` (`num_gpu: 0`) to leave the 6 GB GPU available for Phoenix. Server GPU placement must be reviewed and tested with its actual VRAM. Keep models loaded in a persistent process and avoid running several model UIs simultaneously. No server speedup is claimed without a server benchmark.

## Remaining limits

- Local/private only, not approval for public clinical deployment. No password recovery or production HTTPS/session configuration was added.
- Browser automation was disabled in Work settings; the connected site's visual layout and browser accessibility were not re-tested automatically. Open it manually using `START_NORA.ps1` and inspect desktop/mobile, account dialogs and chat before a demo.
- Output checks remain fallible. This connection does not resolve all model factuality, routing or conversational-quality limitations documented elsewhere in the project.
- Public crisis-resource content and clinical release approvals remain separate from technical API integration.
