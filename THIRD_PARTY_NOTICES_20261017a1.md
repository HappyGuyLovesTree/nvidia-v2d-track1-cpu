# License scope and dependencies

Date: 2026-10-17. The root Apache-2.0 text covers original project contributions only within the licensor's rights. It does not relicense external models, videos, metadata or third-party distributions.

The public PTS validation predicates are extracted/adapted from the project's frozen T011.05 `fetcher.py` (SHA256 `206c5699cc2fffa4fc82b451abf2b7032eb1cb28d4b49b8427d11ca987875773`), with explicit fresh paths/receipts and separate official bindings. They are project code, not downloaded model internals. The candidate retains the original C2/C3 numerical implementation and invokes it from the new fresh orchestration. Dense/OBJ construction is extracted from the preserved C3 stage core; its local L1 terminal writer is omitted in favour of the unchanged official writer.

Retained paired texts in `licenses/` cover OpenCV Zoo NanoDet (Apache-2.0), InstantHMR source and its own third-party notices (Apache-2.0 plus distinct weight notice), SAM student and mhr70 metadata (SAM License), MHR artifact (paired Apache-2.0), NVIDIA kit (upstream root source Apache-2.0/documentation CC-BY-4.0 plus nested Track1 NOTICE), and the two mesh wheel MIT texts. The external asset manifest supplies fixed upstream URLs/revisions/member hashes. No weight/video is redistributed.

The InstantHMR student remains SAM-licensed according to its paired source notice; its exact training-run provenance is unknown. Preserve SAM attribution for research/publication and use only the intended public forward interface. MHR rig artifact and SAM metadata are separate licensed entities. No blanket Apache claim is made over them.

`licenses/dependencies/<normalized-distribution>/` preserves actual installed license/NOTICE files for the24-package active Mac closure. The environment manifest maps every copied legal file to a hash. Distribution licenses remain individual; this preservation does not change their terms. Runtime wheel URLs/hashes refer to official PyPI metadata; packages and native binaries were not republished here. ffprobe remains an external native tool with its own build/license record.

All runtime downloads, input videos, ONNX/PT/PTH files, generated arrays/meshes/Parquet, private bindings, account observations and logs must remain outside the public source file set. The final integrated C4 still requires complete actual-ref content/permission review.
