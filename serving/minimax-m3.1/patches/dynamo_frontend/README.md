# Dynamo frontend patches (mounted read-only over the installed files by `dynamo/frontend.sh`)

- `utils.py` — Dynamo 1.5.0 `dynamo/frontend/utils.py` with `extract_mm_urls` collecting media parts from every message role.
  Upstream only walks `role == "user"`, but the MiniMax-M3 chat template renders a placeholder for every image in any role
  (tool results carry screenshots): mixed user+tool images made the worker raise "More 'IMAGE' tokens found than
  corresponding data provided" (500 / empty stream, 7 of 42 image requests in the 2026-09-26 replay) and tool-only images were
  silently dropped. Re-derive from the image if Dynamo is upgraded: `docker exec dyn-frontend cat .../dynamo/frontend/utils.py`.
