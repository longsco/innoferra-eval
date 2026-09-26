#!/usr/bin/env python3
"""Apply the innoferra DSpark-for-MiniMax-M3.1 patch set to a 0922-sglang source tree.

    python3 apply.py <path-to-0922-sglang>/python/sglang   [--check]

Idempotent: every edit is guarded by a marker. --check only reports what would change.
Edits:
  1. models/minimax_m3_dspark.py   (new)  DSparkMiniMaxDraftModel
  2. models/minimax_m3.py          target-side aux capture for DSpark (post-layer outputs incl. the last layer)
  3. models/minimax_m3_vl.py       set_dspark_layers_to_capture + aux -> logits processor on the VL wrapper
  4. arg_groups/speculative_hook.py  SGLANG_DSPARK_ALLOW_A2A=1 downgrades the "a2a must be none" rule to a warning
"""
import os, shutil, sys, py_compile

MARK = "# [innoferra-dspark-m31]"
root = sys.argv[1].rstrip("/")
check = "--check" in sys.argv
here = os.path.dirname(os.path.abspath(__file__))
changed = []


def patch(rel, pairs, must=True):
    p = os.path.join(root, rel)
    s = open(p).read()
    if MARK in s:
        print(f"  = {rel}: already patched"); return
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"anchor count {n} != 1 in {rel}:\n{old[:200]}")
        s = s.replace(old, new)
    if not check:
        open(p, "w").write(s); py_compile.compile(p, doraise=True)
    changed.append(rel); print(f"  + {rel}")


# 1. new draft model file
dst = os.path.join(root, "models", "minimax_m3_dspark.py")
if not os.path.exists(dst) or open(dst).read() != open(os.path.join(here, "minimax_m3_dspark.py")).read():
    if not check:
        shutil.copy(os.path.join(here, "minimax_m3_dspark.py"), dst); py_compile.compile(dst, doraise=True)
    changed.append("models/minimax_m3_dspark.py"); print("  + models/minimax_m3_dspark.py")
else:
    print("  = models/minimax_m3_dspark.py: up to date")

# 2. target model: capture post-layer outputs for DSpark
patch("models/minimax_m3.py", [
    (
        "        self.layers_to_capture = []\n\n    def get_input_embeddings(self) -> torch.Tensor:",
        f"        self.layers_to_capture = []\n"
        f"        {MARK} DSpark aux capture: flagged layers capture the previous layer's output in\n"
        f"        # prepare_attn (see forward); the LAST layer's output is captured after the loop.\n"
        f"        self.dspark_capture_last = False\n"
        f"        self.dspark_capturing = False\n\n    def get_input_embeddings(self) -> torch.Tensor:",
    ),
    (
        "        aux_hidden_states = []\n        if forward_batch.can_run_tbo:\n",
        "        aux_hidden_states = []\n        if forward_batch.can_run_tbo and not self.dspark_capturing:  # capture needs the per-layer loop\n",
    ),
    (
        "        if not self.pp_group.is_last_rank:\n"
        "            return PPProxyTensors(\n"
        "                {\"hidden_states\": hidden_states, \"residual\": residual}\n"
        "            )\n"
        "        if hidden_states.shape[0] != 0:\n"
        "            if residual is not None:\n"
        "                hidden_states, _ = self.norm(hidden_states, residual)\n",
        "        if not self.pp_group.is_last_rank:\n"
        "            return PPProxyTensors(\n"
        "                {\"hidden_states\": hidden_states, \"residual\": residual}\n"
        "            )\n"
        "        if self.dspark_capture_last and hidden_states.shape[0] != 0:\n"
        "            # output of the last decoder layer = the final norm's input (hidden + residual)\n"
        "            aux_hidden_states.append(\n"
        "                hidden_states + residual if residual is not None else hidden_states\n"
        "            )\n"
        "        if hidden_states.shape[0] != 0:\n"
        "            if residual is not None:\n"
        "                hidden_states, _ = self.norm(hidden_states, residual)\n",
    ),
    (
        "    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):\n"
        "        if not self.pp_group.is_last_rank:\n"
        "            return\n\n"
        "        self.capture_aux_hidden_states = True\n",
        "    def set_dspark_layers_to_capture(self, layer_ids: Optional[list[int]] = None):\n"
        "        if not self.pp_group.is_last_rank:\n"
        "            return\n"
        "        if not layer_ids:\n"
        "            raise ValueError(\"DSPARK requires explicit target layer ids for aux capture.\")\n"
        "        self.capture_aux_hidden_states = True\n"
        "        self.model.dspark_capturing = True\n"
        "        self.model.layers_to_capture = []\n"
        "        end = self.model.end_layer\n"
        "        for lid in sorted(int(x) for x in layer_ids):\n"
        "            if lid + 1 < end:\n"
        "                setattr(self.model.layers[lid + 1], \"_is_layer_to_capture\", True)\n"
        "                self.model.layers_to_capture.append(lid + 1)\n"
        "            elif lid + 1 == end:\n"
        "                self.model.dspark_capture_last = True\n"
        "            else:\n"
        "                raise ValueError(f\"DSPARK target layer id {lid} out of range (num layers {end}).\")\n\n"
        "    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):\n"
        "        if not self.pp_group.is_last_rank:\n"
        "            return\n\n"
        "        self.capture_aux_hidden_states = True\n",
    ),
])

# 3. VL wrapper (the arch actually served): delegate capture + pass aux to the logits processor
patch("models/minimax_m3_vl.py", [
    (
        "        if self.pp_group.is_last_rank and not get_embedding:\n"
        "            return self.logits_processor(\n"
        "                input_ids,\n"
        "                hidden_states,\n"
        "                self.lm_head,\n"
        "                forward_batch,\n"
        "            )\n"
        "        return hidden_states\n",
        f"        aux_hidden_states = None  {MARK}\n"
        "        if getattr(self, \"capture_aux_hidden_states\", False) and isinstance(hidden_states, tuple):\n"
        "            hidden_states, aux_hidden_states = hidden_states\n"
        "        if self.pp_group.is_last_rank and not get_embedding:\n"
        "            return self.logits_processor(\n"
        "                input_ids,\n"
        "                hidden_states,\n"
        "                self.lm_head,\n"
        "                forward_batch,\n"
        "                aux_hidden_states,\n"
        "            )\n"
        "        return hidden_states\n\n"
        "    def set_dspark_layers_to_capture(self, layer_ids):\n"
        "        if not self.pp_group.is_last_rank:\n"
        "            return\n"
        "        if not layer_ids:\n"
        "            raise ValueError(\"DSPARK requires explicit target layer ids for aux capture.\")\n"
        "        self.capture_aux_hidden_states = True\n"
        "        self.model.dspark_capturing = True\n"
        "        self.model.layers_to_capture = []\n"
        "        end = self.model.end_layer\n"
        "        for lid in sorted(int(x) for x in layer_ids):\n"
        "            if lid + 1 < end:\n"
        "                setattr(self.model.layers[lid + 1], \"_is_layer_to_capture\", True)\n"
        "                self.model.layers_to_capture.append(lid + 1)\n"
        "            elif lid + 1 == end:\n"
        "                self.model.dspark_capture_last = True\n"
        "            else:\n"
        "                raise ValueError(f\"DSPARK target layer id {lid} out of range (num layers {end}).\")\n",
    ),
])

# 4. arg rule: allow MegaMoE with DSpark+dp-attention when explicitly requested (dense draft)
patch("arg_groups/speculative_hook.py", [
    (
        "        if server_args.moe_a2a_backend != \"none\":\n"
        "            raise ValueError(\n"
        "                \"DSpark with dp attention only supports the built-in TP MoE \"\n"
        "                f\"(moe_a2a_backend='none'), got {server_args.moe_a2a_backend!r}.\"\n"
        "            )\n",
        "        if server_args.moe_a2a_backend != \"none\":\n"
        f"            {MARK} M3.1 NVFP4 experts only exist on MegaMoE; a DENSE draft never\n"
        "            # enters the MoE all-to-all, so allow it on explicit request.\n"
        "            import os as _os\n"
        "            if _os.environ.get(\"SGLANG_DSPARK_ALLOW_A2A\", \"0\") == \"1\":\n"
        "                logger.warning(\n"
        "                    \"DSpark with dp attention and moe_a2a_backend=%r allowed by \"\n"
        "                    \"SGLANG_DSPARK_ALLOW_A2A=1 (dense draft assumed).\",\n"
        "                    server_args.moe_a2a_backend,\n"
        "                )\n"
        "            else:\n"
        "                raise ValueError(\n"
        "                    \"DSpark with dp attention only supports the built-in TP MoE \"\n"
        "                    f\"(moe_a2a_backend='none'), got {server_args.moe_a2a_backend!r}.\"\n"
        "                )\n",
    ),
])

# 5. training-compatible attention: its extend path is varlen-generic (cu_seqlens/prefix_lens/_max_seqlen_q come
#    from backend._build_extend_metadata, which already knows TARGET_VERIFY), so let an explicit env waive the guard.
patch("layers/attention/minimax_sparse_backend.py", [
    (
        "            if runner.server_args.speculative_algorithm is not None:\n"
        "                raise ValueError(\n"
        "                    \"M3 training-compatible attention does not support speculative decoding\"\n"
        "                )\n",
        "            if runner.server_args.speculative_algorithm is not None:\n"
        f"                {MARK} verify == a varlen extend of speculative_num_draft_tokens per request\n"
        "                import os as _os\n"
        "                if _os.environ.get(\"SGLANG_M3_TRAINING_ALLOW_SPEC\", \"0\") == \"1\":\n"
        "                    logger.warning(\n"
        "                        \"M3 training-compatible attention with speculative decoding allowed by \"\n"
        "                        \"SGLANG_M3_TRAINING_ALLOW_SPEC=1 (TARGET_VERIFY served by the extend path).\"\n"
        "                    )\n"
        "                else:\n"
        "                    raise ValueError(\n"
        "                        \"M3 training-compatible attention does not support speculative decoding\"\n"
        "                    )\n",
    ),
])

print(("would change" if check else "changed") + ":", changed or "nothing")
