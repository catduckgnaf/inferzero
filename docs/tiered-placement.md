# Tiered placement: one model over every GPU, the CPU and the SSD (design, 2026-11-20)

The user's request (2026-11-20), for every model the engine runs, dense and MoE:

- fill the VRAM of the first card, then the second card's, then RAM, then the SSD (an A580 + A770 user; skipzx-lab's
  Arc Pro B60 24 GB + A770 16 GB, issue #58, is the first real box);
- the KV cache on the fastest card ("Battlemage looks faster");
- every card runs XMX with H8 where it has them, with **the same tuned switches it uses alone** ("all use optimized
  switches like now for each card");
- the CPU does work too.

Fixed by earlier decisions and kept here: prompt chunks of at most 4,096 tokens (Gemma 4 2,048; small-VRAM cards and
DeepSeek less); no extra resident copies of decoded weights (the minimal-cost rule: VRAM only for the placement itself);
nothing changes for a model that fits one card.

This extends the layer split of [arc-plan-2026-11-17.md](arc-plan-2026-11-17.md) (L1-L5) and uses what exists:
`TpEngine` and its shards (`zero/mgpu/`), the expert cache, RAM tier and expert file (`zero/xcache.h`,
`zero/expert_store.h`), `ARC_XCPU` and `ARC_HOST_CPU` (`zero/xe2/`), the layer-major prompt
(`forward_layer_major`).

## 1. The one rule: compute where the bytes are

Generation reads every weight it uses once per token. Copying weights between tiers per token costs more than anything
else: VRAM reads at ~450-600 GB/s on these cards, PCIe moves 16-32 GB/s, RAM gives a CPU ~40-90 GB/s, an SSD 3-7 GB/s.
So each tier computes its own share where it sits, and only small vectors cross:

| Tier | Holds | Computes generation | Computes prompts |
|---|---|---|---|
| GPU stages (fastest first in the plan, see 3) | whole layers: weights, KV cache / DeltaNet state, their experts' cache | that GPU, its own kernels | that GPU |
| RAM | rows of large matrices (dense), experts that miss the GPUs (MoE) | **the CPU, at the same time as the GPUs** (5) | the GPU of the stage, streaming the rows over PCIe once per prompt segment (layer-major) |
| SSD | MoE experts only (the expert file); Flash Next's n-gram table | read into the RAM tier / GPU cache on a miss | streamed once per segment |

A dense model never runs from the SSD: every token reads all of it (a 27B from a 6 GB/s SSD: ~2 s a token).

The time of a token is about the sum of the stages' times (they run one after the other), with the CPU's share
overlapped inside the stage it belongs to: `t = sum over GPU stages of max(bytes_gpu / bw_gpu, bytes_ram / bw_cpu) +
crossings x latency`. The plan minimizes that.

## 2. What is placed, per model family

| Family | The layer unit (moves whole between GPUs) | KV / state per layer | Pool placed apart from layers | Notes |
|---|---|---|---|---|
| Qwen3.5 / 3.8 dense (4B, 9B, 27B) | DeltaNet layers (3 in 4) and full-attention layers | DeltaNet: a fixed state; attention: KV growing with the context | – | embedding on stage 0; final norm, head and MTP head on the last stage |
| Qwen3.6 35B-A3B (MoE) | the dense part of each layer (attention / DeltaNet, norms, router, shared expert) | as above | the routed experts of each layer, cached per GPU of that layer | top-8 experts per token |
| Qwen3.8 Flash Next (qwen4exp) | each layer's mixer, QSA indexer, hyper-connection mixers, norms, router | QSA's KV and indexer keys on full-attention layers | routed experts (512, top-10); the PLE n-gram table (SSD / RAM) on stage 0 | the hidden state is 4 streams wide (hc): a crossing is 4 x n_embd |
| Gemma 4 (12B dense, 26B-A4B MoE, E2B / E4B) | each layer (sliding-window or global attention) | sliding-window rings (bounded) or global KV | experts of the 26B (gate_up fused) | f32 / f16 attention; two int8 passes (26B, 12B); the assistant drafter reads some layers' KV: those layers' stage runs the drafter |
| DeepSeek-V4.1 Flash | each layer's attention (compressed rows, sinks), Engram, router | compressed KV and the indexer | routed experts (MXFP4 / Q5_1 / Q8_0) | chunk 1,024 (its queries are 64 x 512 floats a row) |

Where the KV goes: a layer's KV lives on the GPU that computes the layer (attention reads it every token; elsewhere it
would cross PCIe twice per layer). So "KV on the fastest card" means the fastest card takes the most layers, and with
them the most KV and the long-context attention (section 3). The hybrid models have few full-attention layers (Qwen3.8:
16 of 64; Gemma 4: its global layers), which carry nearly all of the growing KV.

## 3. The plan at load

**3.1 Device profiles.** Each GPU is probed once and the result cached per device and driver (as the kernel cache is):
free VRAM after the driver's reserve; the single-token GEMV's GB/s on a large matrix of the model's main type; the
prompt GEMM's speed (the H8 / XMX path the card takes alone, a short timed GEMM); attention time at the configured
context (`attn_fx` on 16-lane XMX, A6 / `attn_q8` elsewhere); the links between GPUs and to host memory (`choose_sums`
already times them). The CPU: its RAM read speed with the CPU kernel (and AVX-512 / VNNI, NUMA nodes).

**3.2 Order.** The GPUs ordered by measured generation speed, the fastest last: the last stage holds the final norm,
the head (read every token), the MTP / drafter heads and the most layers, so the fastest card carries the most bytes,
the most KV and the head. Ties: the card with XMX attention and the most VRAM.

**3.3 Fill.** Each GPU gets a contiguous range of layers sized so that its weights, its share of the context (KV /
state at the configured context), its buffers at the prompt chunk and its slice of the expert cache fit; the fastest
gets layers first (from the end), then the next. If the model fits the fastest card alone, it runs there alone (no
split: one card is faster than any split for one conversation); `--split layer` still splits on request (two
conversations in flight, L4).

**3.4 The RAM share.** What does not fit any GPU stays in RAM as a **row share of the largest matrices** (FFN gate /
up / down, then the head's tail) spread over the layers of the stage whose GPU is closest in speed to the CPU, not as a
stage of its own: the CPU computes its rows while that GPU computes the rest of the same matrix (5.1), so no extra
crossing appears and the stage takes max(GPU, CPU). MoE: the experts that do not fit the GPUs' caches go to the RAM
tier (computed by the CPU on a miss, `ARC_XCPU`), then the expert file on the SSD.

**3.5 Experts (MoE).** Each GPU caches the experts of its own layers only (L1). Within a layer, the experts ranked by
use (R3's routing counts; Strata's profile of varied prompts) fill that GPU's cache first; the rest is the RAM tier,
then the SSD.

**3.6 The printed plan.** At load, one block in the log and the launcher: each device, its layers, its VRAM use, its
expected share of a token's time, the RAM share and the CPU's time, the experts per tier, and the expected tok/s; the
same as `--dry` without loading. Users see why a model is slow before they wait for it.

## 4. Each card runs its own tuned path

A shard is an `Engine` on its device with that device's defaults, exactly the ones it uses alone
([xe2-tuning.md](xe2-tuning.md)): a B60 its discrete-Battlemage set (H8 on `gemm_xw` WGX 4, 2D + prefetch 3 / 2,
SPLIT 34, H8_NORM, H8_GRF 128, GEMV_WAVE / DB, GDN_L1PF 2, `attn_fx`, MOE_GRF 128); an A770 the Alchemist set (the
140T preset: H8 on `gemm_xw8` 32 x 16 PACK 3 WGX 2 for chunks of 512+, SPLIT 17, H8_GRF 256, GVROWS 1 on Linux,
ARC_BATCH 64 on Windows, XSPLIT_TAB, the GEMV switches off); a B580 / B70 / 140V / B390 / Max 1550 likewise. Rules:

- every device-dependent default is a member of the shard's `Engine`, decided from its own device; no function-local
  static that the first shard to ask decides for all (one known: `engine_h8.h:126`, `ARC_H8_GRF`);
- process-wide switches that differ by card kind become per device (`ARC_BATCH`, `ARC_GVROWS`, the GEMV switches);
- a switch the user sets in the environment applies to every card, as today;
- each shard logs its own path at load ("GPU 1: XMX + H8 on gemm_xw8, the 140T preset"), and the check is that a
  card's lines in a split match the lines it prints alone.

## 5. The CPU works at the same time

**5.1 Dense rows.** The RAM share of a matrix (3.4) is computed by the CPU while the stage's GPU computes the other
rows; the GPU and CPU outputs meet in host-visible memory before the next kernel. Today `ARC_HOST_CPU` waits on the
host per matrix (`zero/xe2/engine_hostcpu.h`); this makes it concurrent. The split of rows comes from the probes, so
both finish together.

**5.2 Fewer bytes.** The CPU's form is int8 with one scale per 32 values (~1.1 bytes a weight), ~2.5x the reads of a
3-bit file. A 4-bit CPU form (~0.56 bytes a weight) or the native K-quant blocks the CPU decodes cheaply (Q4_K, Q5_K,
Q6_K) halves the RAM traffic; KL-checked.

**5.3 MoE.** `ARC_XCPU` as now (concurrent already), with the second pass of `--precise` (audit A2), AVX-512 / VNNI and
NUMA placement (X4-X6) where the CPU has them.

**5.4 Prompts** never run on the CPU: the stage's GPU streams the RAM rows over PCIe once per layer-major segment
(amortized over up to 16K rows; Flash Next and DeepSeek join the layer-major path first, K1).

## 6. Prompts over the stages

The layer-major prompt runs per stage: stage s takes a segment's chunks through its layers, then hands the segment's
hidden rows to stage s + 1 (one copy, `chunk x n_embd` floats, hc streams for Flash Next), and starts the next segment
(L3's pipelining): while the last stage finishes segment j, the first stage runs segment j + 1. Chunks stay at most
4,096 rows. Expert streaming and RAM-row streaming happen once per segment per stage.

## 6a. The user chooses what each device does (every combination)

Some users don't want their CPU busy (a desktop in use, a laptop on battery, a shared server), some want no SSD reads, some
want a card left out. Every part of the plan is a choice, each with a plain default and its cost printed in the plan:

| Choice | Values | What happens to the work it takes away |
|---|---|---|
| `--cpu-work` | `auto` (default: dense rows on discrete cards as `ARC_HOST_CPU` today; experts when the planner sees the CPU win, as `ARC_XCPU` measures) · `none` · `experts` · `dense` · `all` | `none`: the RAM rows and RAM-tier experts are read by the GPU over PCIe instead (slower per token, the CPU idle); the plan prints both speeds so the cost is visible |
| `--cpu-threads` | `auto` (the physical cores less one) · `N` | fewer threads: the CPU's share shrinks, the planner gives the GPUs more |
| `--gpus` | `auto` (every Arc GPU) · a list (`0,2`) · `N` | the cards left out hold nothing |
| `--split` | `auto` · `one` · `layer` · `tensor` | see 8 |
| `--ram-gb` | `auto` (free RAM less 16 GiB, as `ARC_XRAM_GB`) · `N` · `0` | `0`: no RAM tier; MoE misses go straight from the SSD to the GPU (or the CPU, if allowed, reads the SSD's records) |
| `--ssd` | `auto` · `off` | `off`: a MoE model whose experts don't fit VRAM + RAM is refused with the numbers, instead of streaming |
| `--kv-on` | `auto` (with its layers, the fastest card holding the most) · a GPU | a GPU named: the planner gives that card the full-attention layers (T5's opt-in test; extra crossings) |

All combinations are allowed. The planner (3) runs on the devices and tiers left, and the printed plan says what was
switched off and what it costs (expected tok/s with and without). Every choice is also an `ARC_*` variable and a
launcher option (O0-O6: a "CPU helps" toggle with experts / dense rows / both, and its expected effect on this PC), and
it is kept in presets.

## 7. Accuracy

- A layer split over GPUs of the same kind is one GPU's arithmetic in the same order: logits bit for bit
  (`--dump-logits`), as L1 states.
- Over different kinds, each card's own kernels (H8 on 16 or 8 lanes, its rounding) make its layers' sums: KL against
  the fastest card alone, token by token over hundreds of positions, at or below the model's limit with the same top
  token at 98 % or more (memory: kl-over-long-generations).
- The CPU's rows and experts: KL-checked as `ARC_HOST_CPU` / `ARC_XCPU` are today.
- The iGPU path (one GPU, the 125H) is untouched: placement only acts when a model does not fit one card or a split is
  asked for.

## 8. Defaults and switches

- `--split auto` (L5) picks: one card when the model fits the fastest card; the tiered layer split when it does not, or
  the cards differ, or a link is slow; the tensor split only where the cards match and the links are fast (4 B60s,
  stacks of one Max 1550 card, for prompts).
- `--split layer|tensor|one`, `--place-dry` (print the plan only), `ARC_PLACE_ORDER=0,1` (the order by hand),
  `ARC_PLACE_RAM_GB=N` (cap the RAM share).
- The launcher shows the printed plan and the expected tok/s next to Start (the options panel O0-O6 lists the split
  choice).

## 9. Work, in order

| # | What | Where | Check | Size |
|---|---|---|---|---|
| P0 | Per-device defaults: remove process-wide statics that depend on the device; per-device ARC_BATCH / GVROWS / GEMV switches; per-shard path logs | `zero/engine*.h`, `zero/xe2/`, `zero/mgpu/` | a B60 + A770 split's per-card lines equal each card alone | small |
| P1 | Device profiles + the planner + the printed plan (`--place-dry`) | `zero/mgpu/` (a `place.h`), `zero/engine_setup.h` | the plan for each catalog model on 1-4 cards of the testers' kinds, by hand against their memory | medium |
| P2 | Stages over layer ranges (L1) with the hand-off (L2), fastest last, KV with its layers | `zero/mgpu/` (`PpEngine`), `engine.h` / `engine_load.h` | `ARC_TP_SIM`: 2-3 stages on one GPU bit for bit (4B, 27B, 35B, Flash Next, Gemma 4 12B / 26B, DeepSeek) | large |
| P3 | MoE: each stage's expert cache, ranked by use; the RAM tier and SSD per stage | `zero/xcache.h`, `expert_store.h` | hit rates and tok/s against today's single-card streaming | medium |
| P4 | The RAM row share, concurrent CPU (5.1), the 4-bit CPU form (5.2) and the user's choices of 6a (--cpu-work none / experts / dense / all, --cpu-threads, --gpus, --ram-gb, --ssd, --kv-on) | `zero/xe2/engine_hostcpu.h`, the matmul dispatch | KL; tok/s on d-dorok's B580 (27B partly in RAM; llama.cpp 11.4, `ARC_HOST_CPU` 13.7) | medium |
| P5 | Prompts pipelined over the stages (L3), servers with two requests in flight (L4) | the prompt loop, `tools/server.cpp` | 4K / 16K / 64K prompts against one card and the tensor split | medium |
| P6 | `--split auto` and the launcher (L5, O0-O6) | `zero/mgpu/engine_tp.h`, `tools/server.cpp`, `scripts/zero-launcher.ps1`, `zero-launcher.sh` | the choice printed; presets keep it | small |
| P7 | Measure on real boxes: skipzx-lab (B60 + A770), hansong80 (2 x A770), manoreken2 (4 x B60; Xeon 6330), an A580 + A770 owner, d-dorok (B580), hibagus (Max 1550 stacks) | issues #58, #45, #4, #37, #48 | tok/s and KL against one card, the tensor split and RAM-only | – |

Each item gets a before-after row and a CHANGELOG entry; GPU-specific code stays in `zero/xe2/` and `zero/mgpu/`.
