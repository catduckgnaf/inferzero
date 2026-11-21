# Plan: the Qwen improvements on Gemma 4

Written 2026-11-15. Gemma 4 (E4B, 12B, 31B dense; the 26B MoE) is left out of most of the speed work done for Qwen.
Each step below names the gate in the code that keeps Gemma out, what Gemma needs, and how it is checked.

Rules for every step: KL against f32 at or below 0.015 (laptop, `--kl-ref`); the Qwen models' logits bit-identical
before and after (`--dump-logits`, hashes of 4B / 27B / 35B / Flash Next); a row in `docs/before-after.md` and a
CHANGELOG entry. Order: the dense 12B and E4B first, then 31B, then the 26B MoE.

## 1. H8 (int8 activations) in float form

- Gate: `h8_float()` (`zero/engine_h8.h:53`, `!hp.gemma`) and `h8_auto_wanted()` (`zero/xe2/engine_xe2.h:29`).
- Needed: Gemma's extra norms (post-attention, post-FFN), its embedding scale and the soft-capped logits go
  through the H8 path; the per-layer embedding (E4B) stays f16.
- Check on the laptop: KL per model; prompt speed at 4K.

## 2. The XMX prompt GEMMs (follow from H8)

- B60 / B70: `gemm_xw` 32x64, with `ARC_GEMM_XW_PF` prefetch and `ARC_GEMM_XW_WGX=4`.
- 140T: `gemm_xw8` 32x16 with packing; the preset `p140_wanted()` (`engine_xe2.h:42`, `!hp.gemma`).
- A tile sweep on Gemma's shapes (`bench/gpu_sweep.ps1` / `.sh`, phases P and T): its FFN widths differ from Qwen's.
- Measured by users with the cards (issue #4).

## 3. Decode speed-ups

- `h8_decode` and `H8_NORM` once step 1 works.
- `ARC_GEMV_DB=1` (one window ahead) is for dense models: try it on the 12B / 31B.
- The decode attention split (`zero/engine.h:1529`, `!hp.gemma`): Gemma has head size 256 on global layers and 128
  (or 256) on the sliding ones; the split for the global layers first.
- `ARC_W8_PARTIAL_GB` (excludes MoE only): check that Gemma's dense models already take it.

## 4. XMX prompt attention

- Gates: `attnx_wanted()` and `attnfx_wanted()` (`engine_xe2.h:200, 288`): `!hp.gemma && !hp.swa_rings`.
- Needed: the sliding-window rings (ring offset and wrap in the K/V reads), Gemma's attention soft-cap if used, and
  the tile choices in `zero/engine_setup.h:245/260`.
- The biggest prompt gain at long context, and the hardest part: after steps 1-3.

## 5. Drafters

- `opt.mtp && !hp.gemma` (`zero/engine_load.h:562`, `zero/engine_slots.h:538`): Gemma's MTP drafters are listed in
  the catalog but take a path of their own. Check acceptance and speed with the shared path; the ring snapshot for
  tool calls (QUEUE item) before turning it on.

## 6. Several GPUs

- `zero/mgpu/engine_tp_shard.h:98` refuses Gemma. Needed: the shard split of Gemma's layers (sliding / global,
  per-layer embedding), its norms in `tp.cl`, then P2P and overlap as for Qwen.
- Correctness with `ARC_TP_SIM=1` on the laptop; speed only from users (1 / 2 / 3 / 4 GPUs).

## 7. The 26B MoE

- After the dense steps: the MoE parts (expert GEMMs, the 140T preset for MoE) as for the 35B.
