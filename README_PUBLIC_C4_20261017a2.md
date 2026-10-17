# Fresh new-episode cutoff correction

Date: 2026-10-17. Incremental note for I12-PUB02. The complete public pipeline,
input evidence scope and existing limitations remain in
[the previous README](README_PUBLIC_C4_20261017a1.md).

An unfinished A stage requires the session's existing `new_episode_stop` gate.
The controller checks it before issuing a lease and again in the supervisor's
preflight before spawning. The worker checks it before entering A work, covering
a child that starts after the admission window closes. Equality with the
monotonic deadline refuses admission; wall-clock rollback does not extend it.
An already completed, validated A may continue B/C until the existing candidate
and hard stops. A failure or incomplete A does not qualify as completed A.

Run the focused zero-model tests from a clean source commit:

```sh
python -B -m unittest discover -s tests -p 'test_public_fresh_cutoff.py' -v
```

These tests use synthetic official admission metadata and intercepted numerical
stages. Fixture preparation runs a fake probe, never an actual video decoder.
They do not establish official input readiness, completed predictions, full
fresh30 inference or cross-machine numerical equality. Existing budget, unknown,
renewal, producer and production-resume contracts continue to apply.
