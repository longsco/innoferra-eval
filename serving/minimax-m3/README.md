# MiniMax-M3 on node 0008 (8x B300), 09-29

Weights: `nvidia/MiniMax-M3-NVFP4` (ModelOpt MIXED_PRECISION: routed experts NVFP4, attention / index / shared experts / dense MLP MXFP8;
250 GB, 88 shards) and draft `nvidia/MiniMax-M3-DSpark` (Qwen3DSparkModel, 6 sliding layers, window 1024, block 8, trained causal=false,
FP32, no confidence-head weights). Both public (MiniMax Community License / NVIDIA Open Model License), at `/data01/minimax31/m3/`.

Engine (research pass 09-28, three readers; all from code, not yet run): our fork tree `/data01/minimax31/src/0922-sglang-hicache`
mounted into `minimax-m31-sglang:demo-bef87f4` (stock v0.5.17 lacks the DSpark capture hook on the M3 VL wrapper).
4 engines x plain TP2 (no DP attention), `--quantization modelopt_mixed --moe-runner-backend flashinfer_trtllm_routed
--attention-backend trtllm_mha --kv-cache-dtype fp8_e4m3 --page-size 128 --dtype bfloat16`, DSpark with
`--speculative-draft-model-quantization unquant`, flashinfer draft attention, static ragged verify. None of the M3.1 envs.
Carried from M3.1: session-pinning gateway, 4 tokenizer workers, chunk 32768, 64 running per engine, mem 0.72, HiCache ratio 3.

`patch_dflash_bidir.py`: SGLang builds the draft's sliding layers causal and ignores `causal: false`; `SGLANG_DSPARK_BIDIR_SWA=1`
runs them bidirectionally in-block (the fix that lifted M3.1 decode 7-14%). A/B'd in the M3 phase.

`m3_phase.sh` (automatic, after the M3.1 inference-perf runs): A target-only canary, B DSpark causal vs bidirectional (accept on real
trace prompts), C 4 engines + HiCache block 8 vs 4 (10-min inference-perf), D the four inference-perf scenarios with the winner, then
frees the node for the M3.1 queue. Stops with "M3 PHASE FAILED" on any gate.

Production M3 for reference (not reproducible here): Dynamo 1.3 + in-house SGLang, 4 x TP2 per node, pure-NVFP4 checkpoint, DFlash2
draft block 4 window 2048 fa4, chunk 8192, 128 running, mem 0.8, HiCache 3.3, tc_piecewise prefill graphs, strict-affinity KV router.
