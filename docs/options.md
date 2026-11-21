# Options you can turn on

The switches below are off unless the table says otherwise. Each changes speed or memory; the ones marked "same output" give the same text bit for bit, the others are checked against an f32 reference (KL). The launchers read this list from `scripts/options.tsv`; this page is made from that file (`powershell -File tools\gen_options_md.ps1`).

On Windows: `zero-server start -Opt <id>` turns one on (kept for later starts), `-NoOpt <id>` off, and `zero-server options` lists them with whether each is for your PC. On Linux: `./zero-launcher.sh --opt <id>` (`--opt <id>=<value>` for one that takes a value), `--no-opt <id>`, `--default-opt <id>`, `--options` to list them, or `o` in its menu; the same choices file as on Windows (`~/.inferzero/options.json`), also written into the systemd unit with `--install-service`. In Docker: `OPTIONS` in compose.yaml or kubernetes.yaml, e.g. `"copy_draft,xcpu=off,budget=12"` (exactly these at each start). By hand: set the environment variable before starting the server (PowerShell: `$env:NAME = "value"`, bash: `export NAME=value`), or add the flag to serverz's command line.

## CPU experts (`ARC_XCPU=1`)

- **What it does:** A MoE model's experts that are not on the card are computed by the CPU from RAM instead of being copied over PCIe.
- **Default:** on: discrete 16-lane cards (B-series, Max); off: Alchemist, iGPUs.
- **Best for:** b-series, multi-gpu, moe.
- **Cost:** none (RAM the expert tier already uses).
- **Output:** slightly different, checked against f32 (KL).
- **Measured:** a B60 on PCIe 3.0 x8, Flash Next generation 22.5 -> 27.3-28.2 tok/s; an A770 on x16: no gain; the top token against f32 a little lower than off (a B60: 241 -> 237 of 256).
- **More:** [docs/tiered-placement.md](tiered-placement.md), stable.

## CPU experts exact in precise mode (`ARC_XCPU_Q2=1`)

- **What it does:** With Flash Next's precise mode, the CPU's experts run both passes too (default: one pass, faster).
- **Default:** off.
- **Best for:** moe.
- **Cost:** the CPU experts 0.302 instead of 0.187 ms each.
- **Output:** slightly different, checked against f32 (KL).
- **Measured:** KL on 548 chat + 160 code positions 0.00076 / 0.00222 against one pass's 0.00126 / 0.00251; top token 702 against 700 of 708.
- **Note:** needs xcpu and fn_precise.
- **More:** [docs/tiered-placement.md](tiered-placement.md), stable.

## Decoded weights kept for short prompts (`ARC_140T_RES=1`)

- **What it does:** On the Arc 140T and Alchemist cards: short prompt chunks use resident decoded weights, long ones decode just in time.
- **Default:** off.
- **Best for:** alchemist, igpu-140t, dense.
- **Cost:** about 2x the weights' size in VRAM.
- **Output:** the same, bit for bit.
- **Measured:** not measured on a card yet.
- **Note:** needs room for the weights twice.
- **More:** [docs/xe2-tuning.md](xe2-tuning.md), experimental.

## Exact prompts on Lunar Lake / Panther Lake (`ARC_H8_MIN=100000`)

- **What it does:** Prompts read without the H8 rotation on iGPUs with 16-lane XMX (Arc 140V, B390): a little more exact, slower.
- **Default:** 256 (H8 from 256-token chunks).
- **Best for:** igpu-xe2, dense.
- **Cost:** 256-1,023-token prompts 1.4-1.8x slower.
- **Output:** slightly different, checked against f32 (KL).
- **Measured:** the 27B's top token 473 against 470 of 480 against f32 (the laptop, the B390 / 140V path).
- **More:** [docs/xe2-tuning.md](xe2-tuning.md), stable.

## H8 in two passes (exact prompts) (`ARC_H8_Q2=1`)

- **What it does:** Each rotated int8 prompt GEMM also runs on what its first rounding left out, so prompts are read as exactly as without H8, on any GPU where H8 runs (B-series, Alchemist, the 140T, Lunar Lake / Panther Lake).
- **Default:** off.
- **Best for:** b-series, alchemist, igpu-140t, igpu-xe2, dense.
- **Cost:** the H8 GEMMs take about twice as long: the laptop's 27B 4K prompt 115-120 -> 62.5 tok/s; a second activation buffer as big as the first.
- **Output:** slightly different, checked against f32 (KL).
- **Measured:** the 27B on 480 tokens against f32: KL 0.00112 -> 0.00061 (plain 0.00056), top token 467 -> 474 of 480 (the laptop).
- **More:** [docs/xe2-tuning.md](xe2-tuning.md), experimental.

## Flash Next precise mode (`--precise`)

- **What it does:** Flash Next with two int8 passes, f16 keys and values and the routers in BF16: for work that needs exactness.
- **Default:** off.
- **Best for:** moe.
- **Cost:** about half the prompt speed, some generation speed.
- **Output:** slightly different, checked against f32 (KL).
- **Measured:** KL against f32 under 0.015 (the default mode up to ~0.02 on text work).
- **More:** [docs/xe2-tuning.md](xe2-tuning.md), stable.

## Resident weights (`--budget <GiB>`)

- **What it does:** Keep the decoded weights on the card up to this many GiB, instead of decoding them for every prompt chunk.
- **Default:** the card's free memory, by the model.
- **Best for:** b-series, alchemist, dense.
- **Cost:** the given GiB of VRAM.
- **Output:** the same, bit for bit.
- **Measured:** a split 27B: the 512-token prompt 1,028 -> 1,481 tok/s, but the 4K prompt slower (manoreken2, #4).
- **More:** [docs/multi-gpu.md](multi-gpu.md), stable.

## Multi-GPU sums (`ARC_TP_P2P=1`)

- **What it does:** A split model's sums: 1 straight from GPU to GPU, 0 through host memory, prompt: direct for prompts only.
- **Default:** by the links: direct where it is faster.
- **Best for:** multi-gpu.
- **Cost:** none.
- **Output:** the same, bit for bit.
- **Measured:** 4 Max 1550 stacks: prompt only direct, 512-token prompt 2,137 -> 2,548 tok/s (hibagus, #48).
- **Note:** not with ARC_TP_OVERLAP.
- **More:** [docs/multi-gpu.md](multi-gpu.md), stable.

## Copy drafting (`--copy-draft`)

- **What it does:** With --mtp: where the last tokens appeared earlier in the conversation, the tokens that followed are drafted; the same text.
- **Default:** off.
- **Best for:** dense, moe.
- **Cost:** none.
- **Output:** the same, bit for bit.
- **Measured:** a code edit on the laptop: the 27B 17.5 -> 22.1 tok/s, the 4B 42.0 -> 58.5; summaries unchanged.
- **Note:** needs --mtp.
- **More:** [CHANGELOG.md](../CHANGELOG.md), experimental.

## Drafts from the common tokens (English and code) (`ARC_MTP_VOCAB=32768`)

- **What it does:** With --mtp: each draft reads only the 32,768 most common tokens' rows of the output head, not all 248,320; the model still checks every draft against the full vocabulary, so the text stays the same.
- **Default:** off.
- **Best for:** dense, moe.
- **Cost:** 69 MB of GPU memory (the 4B; more with a wider model); Chinese text drafts less well (the laptop's 4B 31.6 -> 26.7 tok/s), Indonesian about even.
- **Output:** the same, bit for bit.
- **Measured:** the laptop's 4B on a coding prompt 35.9-37.3 -> 44.3-45.8 tok/s, the 27B 10.9 -> 12.6; Indonesian 27.4 -> 28.9; Chinese 31.6 -> 26.7.
- **Note:** needs --mtp.
- **More:** [CHANGELOG.md](../CHANGELOG.md), experimental.

## Kernels sent in lists (`ARC_BATCH=64`)

- **What it does:** A token's kernels sent to the GPU in recorded lists instead of one by one.
- **Default:** 64 on Alchemist (Windows and Linux), off elsewhere.
- **Best for:** alchemist.
- **Cost:** none.
- **Output:** the same, bit for bit.
- **Measured:** an A750 on Linux (i915): the 4B 46.0 -> 66.6 tok/s, with MTP 68.4 -> 88.5 (hakangit, #72); an A770 on Linux (xe): no change (skipzx-lab, #45); two A770s on Windows 12.7 -> 21.6.
- **More:** [docs/xe2-tuning.md](xe2-tuning.md), stable.

## Gate and up in one launch (`ARC_GEMV_GU=1`)

- **What it does:** One token's FFN gate and up products in one kernel launch instead of two.
- **Default:** off.
- **Best for:** alchemist, b-series, dense.
- **Cost:** none.
- **Output:** the same, bit for bit.
- **Measured:** an A750: the 4B's generation +4.6 % alone, +2.5 % on top of ARC_BATCH (hakangit, #72).
- **More:** [docs/xe2-tuning.md](xe2-tuning.md), experimental.

## Multi-GPU by layers (`--split layer`)

- **What it does:** With --gpus 2 or more: each GPU holds a range of the layers instead of a share of every layer, the rows passed from GPU to GPU; for a model one card cannot hold, or links too slow for the tensor split. --tensor-split a,b,... sets each GPU's share (GPU 0 holds the last layers).
- **Default:** tensor (each layer's matrices over the GPUs).
- **Best for:** multi-gpu, dense.
- **Cost:** about one card's speed for one request (a token visits each GPU in turn).
- **Output:** the same, bit for bit.
- **Measured:** bit for bit against one GPU (ARC_TP_SIM, the 4B and 27B, 2 and 3 GPUs); an A770 pair not measured yet.
- **Note:** needs --gpus 2 or more; Qwen dense only; no --mtp, images, saved prompts or sessions yet.
- **More:** [docs/multi-gpu.md](multi-gpu.md), experimental.
