# knowledge/ — what innoferra-eval knows (cold-start map)

Durable, curated pages. Specs stay in `models/<name>/spec.yaml`; launch kits in `serving/<name>/`; this folder holds the
digests, decisions and gotchas that don't fit a YAML field. One page per model; findings dated inline.

| page | model | what it holds | updated |
|---|---|---|---|
| [minimax-m3.1.md](minimax-m3.1.md) | MiniMax-M3.1 (preview) | vendor-delta digest (numerics, DSpark, `reasoning_effort`), reference engine + build/launch method, node state, open questions | 2026-09-25 |
| [../docs/LEARNINGS.md](../docs/LEARNINGS.md) | M3, GLM-5.3 | measurement lessons, bypass-backend gotchas, cache-prewarm research verdict | 2026-09-22 |
| [../docs/MANUAL-MAP.md](../docs/MANUAL-MAP.md) | M3 | manual § → suite → threshold, plus what passes today | 2026-09-22 |
