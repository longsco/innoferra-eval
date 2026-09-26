# DSpark draft model for MiniMax-M3.1 (innoferra port, 2026-09-25).
#
# The vendor drop `MiniMaxAI/MiniMax-M3.1-preview2-dspark-private/dspark/` declares
# architectures ["DSparkMiniMaxDraftModel"]: 5 dense MiniMax decoder layers fed by the
# concatenated target hidden states of layers `dspark_target_layer_ids` through
# `fc` + `hidden_norm`, a vanilla Markov head, and (ignored here, per the vendor) a
# confidence head. This file wires that checkpoint into the fork's DSpark worker
# (srt/speculative/dspark_components) the same way `models/dspark.py` does for the
# Qwen3 dense draft, but with the MiniMax layer stack (gemma norms, per-head qk-norm,
# partial rotary, swigluoai dense MLP) reused verbatim from `models/minimax_m3.py`.
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dspark import (
    build_confidence_head,
    build_markov_head,
    gather_and_crop_vocab,
)
from sglang.srt.models.minimax_m3 import (
    MiniMaxM3DecoderLayer,
    MiniMaxM3GemmaRMSNorm,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

_CKPT_PREFIX = "language_model.model.dspark."


class DSparkMiniMaxDraftModel(nn.Module):
    """MiniMax-M3.1 DSpark draft: `layers.{i}.decoder_layer` are plain dense M3 layers."""

    # Weight names in the checkpoint are `language_model.model.dspark.<x>`; the
    # decoder layers carry an extra `.decoder_layer` level.
    supports_fused_context_kv = False

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        text_config = getattr(config, "text_config", None) or config
        self.text_config = text_config

        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark MiniMax draft requires markov_rank > 0, got "
                f"{dspark_config.markov_rank}."
            )
        self.gamma = int(dspark_config.resolve_gamma(default=7))
        self.block_size = self.gamma
        if dspark_config.target_layer_ids is None:
            raise ValueError("DSpark MiniMax draft needs dspark_target_layer_ids.")
        self.target_layer_ids = [int(x) for x in dspark_config.target_layer_ids]
        self.num_context_features = len(self.target_layer_ids)

        hidden_size = int(text_config.hidden_size)
        num_layers = int(text_config.num_hidden_layers)
        eps = float(getattr(text_config, "rms_norm_eps", 1e-6))
        use_gemma_norm = bool(getattr(text_config, "use_gemma_norm", False))
        norm_cls = MiniMaxM3GemmaRMSNorm if use_gemma_norm else RMSNorm

        if getattr(text_config, "sparse_attention_config", None) is not None:
            raise ValueError(
                "DSpark MiniMax draft expects dense attention layers "
                "(sparse_attention_config must be null in dspark/config.json)."
            )
        moe_layer_freq = getattr(text_config, "moe_layer_freq", None)
        if moe_layer_freq is not None and any(int(x) != 0 for x in moe_layer_freq):
            raise ValueError(
                "DSpark MiniMax draft expects dense FFN layers "
                f"(moe_layer_freq={moe_layer_freq})."
            )

        self.layers = nn.ModuleList(
            [
                MiniMaxM3DecoderLayer(
                    config=text_config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}.decoder_layer", prefix),
                )
                for i in range(num_layers)
            ]
        )
        self.fc = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.hidden_norm = norm_cls(hidden_size, eps=eps)
        self.final_norm = norm_cls(hidden_size, eps=eps)

        head_cfg = SimpleNamespace(
            vocab_size=int(text_config.vocab_size),
            hidden_size=hidden_size,
            markov_rank=int(dspark_config.markov_rank),
            markov_head_type=dspark_config.markov_head_type or "vanilla",
            # Vendor (2026-09-25): "just Vanilla Markov Head and no confidence
            # head". The checkpoint ships confidence weights; they are skipped.
            enable_confidence_head=False,
        )
        self.markov_head = build_markov_head(head_cfg)
        self.confidence_head = build_confidence_head(head_cfg)

        self.embed_tokens: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None
        logger.info(
            "DSparkMiniMaxDraftModel: %d dense layers, gamma=%d, target_layer_ids=%s, "
            "fc %d->%d, markov_rank=%d, confidence_head=%s",
            num_layers,
            self.gamma,
            self.target_layer_ids,
            self.fc.in_features,
            self.fc.out_features,
            int(dspark_config.markov_rank),
            self.confidence_head is not None,
        )

    # ---- DSpark worker contract --------------------------------------------------
    @property
    def enable_confidence_head(self) -> bool:
        return self.confidence_head is not None

    def attach_shared_modules(self, *, embed_tokens: nn.Module, lm_head: nn.Module) -> None:
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        # Without --enable-dp-lm-head the target lm_head is vocab-sharded over the FULL TP group while, under
        # dp-attention, every rank runs a different draft batch, so the usual all-gather of local logits cannot
        # work. Assemble a full-vocab copy of the weight once (shape-consistent all-gather at init) and compute
        # draft logits locally. Cost: vocab x hidden bf16 = ~2.4 GB per rank for M3.1.
        self._full_lm_head_weight = None
        if get_tensor_model_parallel_world_size() > 1 and not get_parallel().enable_dp_lm_head:
            w = tensor_model_parallel_all_gather(lm_head.weight.data.contiguous(), dim=0)
            self._full_lm_head_weight = w[: int(lm_head.org_vocab_size)].contiguous()
            logger.info(
                "DSpark MiniMax draft: assembled a full-vocab lm_head copy %s (%s) for local draft logits",
                tuple(self._full_lm_head_weight.shape), self._full_lm_head_weight.dtype,
            )

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise ValueError(
                "DSparkMiniMaxDraftModel requires the target embed_tokens "
                "(call attach_shared_modules first)."
            )
        return self.embed_tokens(input_ids)

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return None

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DSpark MiniMax target_hidden feature dim mismatch: expected "
                f"[N, {expected}] (num_context_features={self.num_context_features}), "
                f"got {tuple(target_hidden.shape)}."
            )
        return self.hidden_norm(self.fc(target_hidden.to(self.fc.weight.dtype)))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        del get_embedding, pp_proxy_tensors
        if input_embeds is None:
            input_embeds = self.forward_embed(input_ids)
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )
        if hidden_states.shape[0] != 0:
            if residual is not None:
                hidden_states, _ = self.final_norm(hidden_states, residual)
            else:
                hidden_states = self.final_norm(hidden_states)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden_states)

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.lm_head is None:
            raise ValueError(
                "DSparkMiniMaxDraftModel requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        if self._full_lm_head_weight is not None:
            weight = self._full_lm_head_weight
            if hidden.dtype != weight.dtype:
                hidden = hidden.to(weight.dtype)
            return torch.matmul(hidden, weight.T), None
        weight = self.lm_head.weight
        if hidden.dtype != weight.dtype:
            hidden = hidden.to(weight.dtype)
        local_logits = torch.matmul(hidden, weight.T)
        return gather_and_crop_vocab(local_logits, self.lm_head), None

    # ---- context KV injection (target hidden -> draft KV cache) ------------------
    def _layer_ctx_kv(self, attn, ctx_hidden: torch.Tensor, positions: torch.Tensor):
        """K/V of one draft layer for already-committed context tokens, computed
        from the projected target hidden (the draft never re-reads the prompt)."""
        # Use the attention module's own prepare path (fused qk-norm + partial RoPE kernel when enabled) so the
        # injected context K matches bit-for-bit what the draft writes for its own block tokens.
        _, _, inner_state = attn.forward_prepare(
            positions=positions, hidden_states=ctx_hidden, forward_batch=None
        )
        _, k, v, _ = inner_state
        k = k.view(-1, attn.num_kv_heads, attn.head_dim)
        v = v.view(-1, attn.num_kv_heads, attn.head_dim)
        return k, v

    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        ctx_hidden = self.project_target_hidden(target_hidden)
        for layer in self.layers:
            attn = layer.self_attn
            k, v = self._layer_ctx_kv(attn, ctx_hidden, positions)
            if cache_loc_2d is not None and commit_lens is not None:
                pool.set_kv_buffer_prefix_valid(
                    attn.attn, cache_loc_2d, commit_lens, k, v,
                    attn.attn.k_scale, attn.attn.v_scale,
                )
            else:
                pool.set_kv_buffer(
                    attn.attn, cache_loc, k, v, attn.attn.k_scale, attn.attn.v_scale
                )

    def prune_to_ctx_kv_injection(self) -> None:
        self.markov_head = None
        self.confidence_head = None
        for layer in self.layers:
            layer.mlp = None
            layer.self_attn.o_proj = None
        torch.cuda.empty_cache()

    # ---- weights ----------------------------------------------------------------
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded, skipped = set(), []
        for name, loaded_weight in weights:
            if name.startswith(_CKPT_PREFIX):
                name = name[len(_CKPT_PREFIX):]
            name = name.replace(".decoder_layer.", ".")
            if name.startswith("confidence_head."):
                if self.confidence_head is None:
                    skipped.append(name)
                    continue
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(mapped)
                break
            else:
                if name not in params_dict:
                    skipped.append(name)
                    continue
                param = params_dict[name]
                if name == "fc.weight" and tuple(loaded_weight.shape) != tuple(param.shape):
                    raise ValueError(
                        f"DSpark MiniMax fc.weight shape mismatch: checkpoint "
                        f"{tuple(loaded_weight.shape)} vs model {tuple(param.shape)} "
                        f"(num_context_features={self.num_context_features})."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)
        missing = [
            n for n in params_dict
            if n not in loaded
            and not n.startswith("confidence_head.")
            # RadixAttention's optional per-layer KV scales are not in the checkpoint (default 1.0)
            and not n.endswith((".attn.k_scale", ".attn.v_scale", ".attn.q_scale", ".attn.idx_q_scale"))
        ]
        if skipped:
            logger.info("DSpark MiniMax draft: skipped %d checkpoint tensors (e.g. %s)",
                        len(skipped), skipped[:4])
        if missing:
            raise ValueError(
                f"DSpark MiniMax draft: {len(missing)} parameters not loaded, e.g. {missing[:6]}"
            )
        logger.info("DSpark MiniMax draft: loaded %d parameters", len(loaded))


EntryClass = [DSparkMiniMaxDraftModel]
