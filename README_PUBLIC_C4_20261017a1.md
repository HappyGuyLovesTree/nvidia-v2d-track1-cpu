# Public reconstruction candidate: C2/C3 recipes and explicit fresh replay

Date: 2026-10-17. Local candidate derived from frozen C3 `b433752cdd1828c34fa2ca9c328dd8b6ac2157e0` / tree `ec91ae800b38723b4ecfe79374b8169e7a70789f`. Original C2 `483f8d61690d7df229c57dca43750b23afcfbf57` and its history/config remain checkable. No remote repository or final public URL has been created. The actual new7 predictions and complete production view are still pending.

This candidate implements a complete public **fresh** path: obtain fixed assets/videos, prepare full original-frame PTS, create an independent session/empty baseline, run the preserved numerical stages under C3 supervision, export NPZ/mask/OBJ/provenance, and call the unchanged stock writer. It does not call `r12.py init/new-session/run`, use private historical parents, change the production root, or relabel newly computed values as historical C2/C3 artifacts.

## Environment, input and assets

The implemented preparation/asset/stock helpers and exact current-Mac24-distribution lock are described in [the preparation README](README_20261017a1.md). Its earlier statement that the inference adapter is pending describes that historical preparation snapshot; the current integration is the `tools/reproduce.py` and `v2d_public/fresh*` code described here. The Mac wheel lock and its evidence are unchanged by this integration. No new environment, model or video was downloaded or run for candidate preparation.

Use CPython3.12.7 and the pinned Mac arm64 lock. See `manifests/external_assets.json`, `input_roster.json` and `stock_kit.json` for official fixed revisions/paths/content hashes, 30 episodes,9877 required original frames,65 source runs and740780 sample keys. Model weights, videos and generated output never belong in this repository. Runtime paths and receipts are local workspace files. All source inputs can be obtained through the explicit `assets fetch/unpack` and `inputs pts` commands; no public result cache or private experiment folder is required for fresh recomputation.

```text
python tools/reproduce.py assets plan --manifest manifests/external_assets.json --roster manifests/input_roster.json --paths paths.example.json --workspace <workspace>
python tools/reproduce.py inputs plan --roster manifests/input_roster.json --paths paths.example.json --workspace <workspace>
```

After explicitly fetching or mapping the fixed model/video/kit files, run full PTS preparation over all30:

```text
python tools/reproduce.py inputs pts --roster manifests/input_roster.json --paths <local-paths.json> --workspace <workspace> --sample <fixed-kit>/data/track_1_sample_submission.parquet --ffprobe <ffprobe>
```

The generated index `r12_public_input_preparation_v1` contains immutable per-episode PTS and input references. Official independent bindings remain separate: only episode16 was historically independently checked; the other29 remain pending. The30 actual local content hashes/size/full-PTS/input chains have project receipt evidence, with no observed hash conflict. Hash equality to new official LFS metadata is not claimed. Under the coordinator's2026-10-17 PUB04 scope ruling, missing new metadata is not a prerequisite blocker for code publication or local stock packing. This does not pass full R12-AC22 or waive actual Kaggle account/rules/entry/quota/complete-pack/public-commit checks; any new content conflict stops the affected submission. Use the designated source evidence normalizer if new official metadata is obtained. This command is implemented; this task did not execute it on real videos.

## Fresh session and budget

```text
python tools/reproduce.py init --paths <local-paths.json> --workspace <workspace> --preparation <new-full-PTS-index.json> --minutes 150
python tools/reproduce.py status --root <returned-fresh-root>
```

Init requires a clean actual source commit, full passed PTS preparation, fixed numerical signature and external asset hashes. It creates a unique `fresh_runs/<id>` directory, a new `r12_public_fresh_run_v1`, a bounded `r12_public_fresh_session_v1`, an empty budget baseline and no call journal. It rejects nesting/resetting any historical production/fresh run. The baseline has `source=null`; original C1/C2 ledgers and internal SPEC/adoption documents are not dependencies.

The fresh adapter writes new, explicitly named PTS bridge records for the preserved numerical worker's input contract. They reference the actual new preparation and checker; they never reuse historical JSON hashes. Original frame IDs, stream time base, full PTS, required frames and source/gap boundaries remain intact.

Sessions have explicit wall timestamps and monotonic deadlines tied to this boot. Reuse the same fresh run for explicit renewal; it does not zero its current counts:

```text
python tools/reproduce.py new-session --root <fresh-root> --minutes 150
```

All reservations, failed/unknown calls, contiguous ordinals and retained/cumulative writer categories use the unchanged journal engine. C3's pure journal validator checks the current fresh totals against an empty ancestor. A pending call, resource latch, incomplete owner or failed/unknown supervision refuses continuation. It is not resolved by switching dates, creating another session or deleting a lock. Independent fresh experiments are separately labelled, never presented as continuation of the original production budget.

## Explicit inference and preserved numerics

```text
python tools/reproduce.py run --root <fresh-root> --episode 0 --execute-models
```

Select additional episodes explicitly, including all0—29 for a complete rebuild. Without `--execute-models`, official inference refuses. Each child also requires a matching session/code/episode/stage lease and the actual live parent controller PID/create-time/owner identity. Fixture runs cannot obtain model permission. No production registry or historical PID is signalled.

The controller uses frozen C3's identity-aware `execute_attempt` and `control_scope`; unknown child identity/refused cleanup does not become success. The worker invokes the original `r11_worker.stage_a/stage_b` numerical functions, preserving automatic boxes, low-score association,224 preprocessing, own-episode median, root/F/unit transform, source-local nearest parents, input-derived box geometry and two complete rig forwards per scored frame. Dense/OBJ construction is extracted from the preserved `r11_pack.stage_c` core, with its local L1 writer omitted; formal packing uses the stock CLI below. C2 and C3 recipe identities are in separate manifests. The current C3 numerical bodies are equal to C2; old23 have recipe=C2 and new7 recipe=frozenC3. All newly recomputed predictions use the actual fresh checkout commit as producer, never the historical cache producer.

OpenCV is the same GCD build as C2. The code requests `setNumThreads(1)`, while the frozen compatibility evidence queried8; this does **not** prove effective single-thread OpenCV execution. Recipe and worker records distinguish requested/reported/effective-unknown. Torch's separate getters reported1/1 in that zero-model reference check. ORT SessionOptions were1/1 and SEQUENTIAL, CPUExecutionProvider was available, and the original stageA explicitly checks the actual session providers when it runs. No real ORT model session or new7 worker was run by this candidate-preparation task. Do not extend the OpenCV observation to all runtimes.

`--scope fixture` init plus `run --fixture` is a separate model-free control fixture. It cannot be combined with model permission or exported as an official prediction view. It emits only fixture receipts; it does not fabricate real detector/HMR/rig results.

## Export, provenance and unchanged stock writer

After all30 actual fresh stages are complete and all supervision/owners/journals close:

```text
python tools/reproduce.py export --root <fresh-root>
python tools/reproduce.py pack run --paths <local-paths.json> --workspace <workspace> --episodes <export>/episodes --masks <export>/masks --provenance <export>/provenance.json --kit-manifest manifests/stock_kit.json --build-source <this-clean-checkout> --commit https://github.com/<verified-owner>/<verified-repo>/commit/<actual-checkout-sha40>
```

Export copies verified numerical files to a new view and records the actual fresh producer/tree, per-episode numerical recipe, video/required/artifact hashes, `imported=false` and actual pack build commit. Its provenance is generated after the source commit is frozen, outside Git; a source file does not guess or self-reference its future hash. The stock wrapper checks this actual checkout/URL, full official sample, masks, arrays, before/after input identity and exact output keys. It never patches `pack.write`. It does not verify remote ownership/access, Kaggle readiness or a leaderboard row; publishing/competition steps remain separate.

The original production submission path instead imports23 original C2 results and computes7 with the actual C3 producer. This candidate does not invent those pending7 outputs or a complete30 production view. Final mixed-source provenance and the actual public build URL must be filled from real results and audited before publication.

## Verification and limits

Zero-model fixtures cover the preparation tools and helper cleanup faults; fresh fixtures cover empty-baseline/session/PTS bridging, refusal of production nesting/model permission, guarded child supervision and dense/OBJ equivalence against the preserved C3 core. They do not measure new7 prediction quality or complete fresh30 numerical reproduction. No full fresh inference or cross-machine bitwise equality is claimed or added as a release gate. Existing wrong-person/pose/proxy quality failures, unknown true scale/contact and the historical C22 exit/reap gap remain disclosed.

Current source includes the real original history, full numerical code, public preparation/configuration, fresh controller/worker, dependencies/licenses and the stock entry. Final actual refs, external account/target, new7 provenance, public access and formal candidate generation remain pending.


This a1 documentation increment records the PUB04 evidence scope. It does not change declared input hashes, numerical source, model/thresholds, or the stage/session/budget implementation. The earlier6d7db8de4ca0a59be9f1b7f8a42a296ce43b37fa candidate remains a separate frozen review snapshot.
