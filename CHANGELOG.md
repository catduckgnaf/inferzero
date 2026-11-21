# Changelog

## 2026-11-21: model downloads resume correctly after a cut

- **A cut download could grow past its size and start over from nothing** (`scripts\hf-get.ps1`, the Linux
  launcher's downloads). A 16.67 GB file grew to 17.8 GB after a connection reset and was downloaded again in full.
  Two causes, both fixed:
  - curl's own `--retry` with `-C -` resumed from the offset it found at its start, so bytes were written twice. The
    scripts now make the tries themselves: up to 6, each resuming from the `.part` file's length measured just before.
  - Two downloads of the same file at once (two windows, or the launcher and a script) appended to one `.part`. Now a
    lock file (`<file>.part.lock`, `flock` on Linux) refuses the second one with a message.
- Checked with a forced cut: curl killed mid-file at a limited rate. The same run resumed, the file's SHA-256 matched,
  and a second run started at the same time was refused. Checked in Windows PowerShell 5.1, PowerShell 7 and bash.

## 2026-11-21: the B60's own defaults (prompt +4-6 %, generation +2-3 %), and the sweep's verdict

- **A Pro B60, B580 or B50 (the B60's die) now has its own prompt GEMM tile:** 64 x 32 with 8 column sub-groups a
  work-group, prefetching 1 chunk ahead. A B60's 27B prompt: 512 / 4K / 32K tokens 1,109.9 / 1,872.5 / 1,619.3 ->
  1,171.8 / 1,954.2 / 1,680.0 tok/s, the same logits (skipzx-lab, #57).
- **Its generation** picks the one-token GEMVs' rows by type again, not by the GPU's thread count (the B70's rule):
  +3.5 % and +2.4 % on two B60s (23.29 -> 24.10, 23.36 -> 23.91 tok/s), the same logits (skipzx-lab #57, MaineTim
  #19).
- The B70, the B65 and every other GPU keep their defaults: the engine tells the B60's die apart by its PCI device
  ID (BMG-G21, 0xE200-0xE21F). Release .21.17 went by the thread count, which also caught a Pro B65 (the B70's die
  with 160 EUs); .21.18 corrects that.
- **`bench/gpu_sweep` phase X named the wrong winner:** a form had to beat the best one so far by 2 %, so a later,
  faster form was not named (on skipzx-lab's B60 it named x1 while x4 was faster at all three contexts). Now the bar
  is 2 % over the defaults, and the fastest form past it wins.
- **`--mtp` on a model file without the MTP block** now says so at load, instead of decoding plainly in silence.
  Unsloth's Qwen3.6-35B-A3B-UD-Q4_K_M.gguf has none; a B390 owner saw `--mtp` run at plain speed with no MTP line
  (#75).

## 2026-11-21: the layer split in the server (`serverz --gpus N --split layer`)

- **`serverz --gpus N --split layer`** (and `--tensor-split a,b,...`) serves a model split by layers, as runz runs
  it. The server's text chat works on it as on the tensor split: prompts, generation, `--slots` with their batched
  steps.
  - `--ctx auto` sizes every GPU's stage on that GPU and takes the smallest context that all of them fit.
  - The server drives the last stage, and each call that changes a conversation runs on the earlier stages first.
- **The GPUs' order changed, in runz too:** GPU 0 now holds the last layers and the head, and GPU 1 the first layers
  (the server's threads work on GPU 0). `--tensor-split` shares are in GPU order, so the first share is GPU 0's (the
  last layers).
- A split that fails to load (not a Qwen dense model, not enough memory) loads on one GPU instead, as the tensor split
  does. A GPU failing in the middle of a call leaves the split broken until the server restarts, and `/health` says so.
- The launchers, `zero-server -Opt split_layer` and Docker's `OPTIONS=split_layer` get it as the option "Multi-GPU by
  layers".
- The same answers as one GPU: under `ARC_TP_SIM`, the 4B's answers to two prompts were the same text for 2 and 3
  stages, one at a time and two at once (batched steps), and runz's logits stayed byte for byte one GPU's.
- The options catalog now marks a value the user gives in angle brackets (`budget`: `<GiB>`). A word there is the
  value that turns the option on (`split_layer`: `layer`). Before, any word read as "needs a value", so `-Opt
  split_layer` would have been refused.
- Not yet: saved prompts and sessions (off with either split), images, `--mtp`, MoE models.

## 2026-11-21: split a model over GPUs by layers (`--split layer`, runz, first form)

- **`runz --gpus N --split layer`** gives each GPU whole layers instead of a part of every layer. GPU 0 embeds the
  token and runs its layers, then hands its rows on to the next GPU; the last GPU runs the norm and the head. There
  are no sums after every layer.
  - It's for slow links, mixed cards, and models one card can't hold.
  - For a single request it isn't faster than one card: a token visits the GPUs in turn.
  - `--split tensor`, the split by matrices with sums per layer, stays the default and the fast form on matching,
    well-linked cards.
- `--tensor-split a,b,...` sets each GPU's share of the layers.
- The same logits bit for bit as one GPU: checked under `ARC_TP_SIM` byte for byte, with the 4B and the 27B, 2 and 3
  stages, short and long prompts, and both prompt paths.
- First form: runz only, Qwen's dense models, no `--mtp`. The hand-off goes through host memory. The server, MoE
  models, drafting and GPU-to-GPU copies are next.

## 2026-11-21: the full GEMM sweep for 16-lane GPUs (B-series, Lunar Lake, Panther Lake)

- **`xw_peak --all` now sweeps the 16-lane prompt GEMM too** (`gemm_xw`, which B-series cards, the 140V, the B390 and
  a Max run). It picks the GPU's own kernel, and `--lanes 16` asks for it.
  - The forms: tiles of 16 / 32 / 64 tokens by 16-128 columns, 1 / 2 / 4 / 8 column sub-groups, with and without the
    lock.
  - Where the GPU has 2D block loads, each 2D form too: the loads only, the loads and C, and the loads and C rounded
    apart.
  - Every form is checked bit for bit against the plain kernel, in the same stages as the 8-lane sweep.
- `bench/gpu_sweep` phase X runs it on 16-lane cards, where it was skipped; its winners go into the model as the
  engine's own switches.
- Checked on the laptop with the DPAS emulated: all 44 forms that build without a spill gave the plain kernel's
  results bit for bit (gate, up + SwiGLU, a split-K down). The 8-lane sweep is unchanged. Speed needs a 16-lane card.

## 2026-11-21: the context a "does not fit" warning suggests now fits the first time

- **Fix (issue #45):** following the warning's `--ctx N fits` on a 16 GB A770 took three steps with `--mtp` (92,752 ->
  73,728 -> 56,320 -> 53,248) and two without.
  - On a discrete card, a context too big for the card makes the engine keep some of the weights in host memory
    (read over PCIe), so the GPU's total falls by much less than the context does.
  - The suggestion now counts those weights too: it's the largest context with every weight on the GPU, where
    nothing goes to host memory. The warning also says how much of the weights is in host memory.
  - runz and the server alike. Unchanged where no weights are in host memory (the laptop: the same 55,296).
- `ARC_FIT_PARTS=1` (runz): the memory count the check reads, by part.

## 2026-11-21: faster drafting for English and code (opt-in)

- **`ARC_MTP_VOCAB=32768`** (catalog option `mtp_vocab`): with `--mtp`, each drafted token reads only the 32,768 most
  common tokens' rows of the output head instead of all 248,320. The model still checks every draft against the full
  vocabulary, so the text is exactly the plain run's. A token outside the list is just never drafted. The idea is
  from FR-Spec (Zhao et al., ACL 2025); no code is taken.
  - On the laptop: the 4B on a coding prompt 36-37 -> 44-46 tok/s (+21-23 %), the 27B 10.9 -> 12.6 (+15 %),
    Indonesian +5 %.
  - **Chinese drafts less well: 31.6 -> 26.7 tok/s.** That's why it stays off by default.
  - It needs 69 MB more GPU memory on the 4B, more for a wider model.
- The token list (`scripts/draft-vocab/vocab-248320.txt`, for Qwen3.5 / 3.6 / 3.8) is ranked over this repository's
  files and the model's own answers in Indonesian, Chinese and Spanish. `tools/make_draft_vocab.ps1` makes it again.
- `bench/gpu_sweep.ps1` (Windows) lost its G, P, C and M tables and then stopped at phase K with "The specified
  wildcard character pattern is not valid" (manoreken2's B60, #4): the functions' printed tables went into the
  variables that were meant to hold their results. Fixed; the bash sweep was right.

## 2026-11-21: H8 in two passes, for exact prompts (opt-in)

- **`ARC_H8_Q2=1`** (catalog option `h8_q2`; `zero-server -Opt h8_q2`, `./zero-launcher.sh --opt h8_q2`): H8 reads
  a prompt in two int8 passes instead of one. The first pass's rounding error is quantized and multiplied through
  the same weights, then added.
  - H8's own rounding doubled the KL against f32 on every GPU where it runs. With the second pass it's back at the
    plain path's level: the laptop's 27B over 480 tokens, KL 0.00112 -> 0.00061 (plain 0.00056), and the same top
    token as f32 at 474 of 480 positions instead of 467.
  - The cost: the H8 GEMMs take about twice as long. On the laptop, whose GEMMs are most of the prompt, the 4K prompt
    goes from 115-120 to 62.5 tok/s. It also needs a second activation buffer as large as the first.
  - It's for work that needs the exact answer; the default stays as it was, byte for byte.

## 2026-11-21: after an update into a new folder, the launcher offers to point the shortcut and PATH there

- When inferzero is unpacked into a new folder, the desktop shortcut and the PATH entry (for `zero-server` in a
  terminal) still opened the old folder. Now the Windows launcher notices this when its window opens, and asks once
  whether to point them to the folder it runs from. A "No" is remembered for that folder.
- The launcher review's other high findings were checked against the code: the window fits the screen's work area,
  the labels are plain words, the network option warns and shows the LAN address, serverz's default port is 8090,
  `serverz --help` is short (with `--help all`), and zero-server lists every model. Still open: downloads in the
  main window instead of a dialog.

## 2026-11-21: the speed and accuracy options in Docker

- **`OPTIONS` in `docker/compose.yaml` and `docker/kubernetes.yaml`** chooses the options of `docs/options.md` for the
  container. It's a comma or space list: `copy_draft` turns one on, `xcpu=off` turns one off, `budget=12` gives one a
  value. Exactly these apply at each start, starting from the defaults; nothing is kept in the data volume, so an
  option removed from the list is off again at the next start.
- The entrypoint passes them to the launcher's new `--no-saved-options` (only the options given, nothing saved), and
  serverz's log names them.

## 2026-11-21: the server's log names the options in effect

- At start, serverz writes one line naming the speed and accuracy options in effect, by their titles:
  `options: CPU experts: off; Resident weights (3); Copy drafting; Kernels sent in lists`, or `none (the defaults)`.
  - It names any option set, however it was chosen: either launcher, `zero-server -Opt`, Docker, or by hand.
  - The catalog (`scripts/options.tsv`) is read from the package.
  - A value the catalog doesn't name shows as `(NAME=value)`.

## 2026-11-21: the speed and accuracy options in the Linux launcher

- **`zero-launcher.sh` has the options** (those in `scripts/options.tsv`, as on Windows):
  - `--options` lists each one: what it's set to now, its default, what it does, the PCs it's for, and its cost.
  - `--opt <id>` turns one on (`--opt budget=12` for one that takes a value), `--no-opt <id>` off, and
    `--default-opt <id>` back to the default. Each takes a comma list.
  - `o` in the menu changes them one by one.
- **The same choices file as zero-server on Windows** (`~/.inferzero/options.json`), kept for later starts. The
  launcher puts each choice into serverz's environment or onto its command line, says them in words at start, and
  `--install-service` writes them into the systemd unit.
- Bash and the system's tools only, no jq or Python. `tests/test_launcher_options.sh` (21 cases) runs in CI. Both
  PowerShells read the file the launcher writes.

## 2026-11-21: the A770's leaner prompt GEMM loop, on by default

- **`ARC_GEMM_XW_PTR` is on by default on Alchemist cards** (A770, A750, A580; `=0` turns it off). It was opt-in in
  2026.11.21.07.
  - skipzx-lab's A770 (#45): the 27B's 4K prompt went 1,432 -> 1,528 tok/s (+6.7 %, the same in two rounds). The
    three FFN GEMMs are 7-14 % faster. xw_peak went from 45 % of the card's peak to 49 %.
  - The logits are the same bit for bit on the card.
- iGPUs (the 140T) keep it opt-in until measured.
- The same A770's driver has no `cl_intel_subgroup_buffer_prefetch`, so DeltaNet's L1 prefetch can't run on
  Alchemist. That isn't a build-guard bug.

## 2026-11-21: 4 drafts a step on Alchemist cards

- **Alchemist (A770, A750, A580) drafts 4 tokens a step by default with `--mtp`,** where it kept the 27B's 7.
  - skipzx-lab's A770 (#45), the 27B on a coding question: plain 22.02 tok/s, 7 drafts 40.75, **4 drafts 46.08
    (+13 %)**. 2 to 6 drafts: 42.06 / 45.56 / 46.08 / 45.11 / 43.74.
  - Every count gave the plain run's text.
  - The A770's verify pass costs more than a B-series card's (84 ms against a B60's 54), so later drafts pay less
    there.
- B-series cards keep 3, and iGPUs keep their counts. `--mtp-drafts N` still overrides.

## 2026-11-21: a context too big for the GPU is reported the same way with and without drafting, by runz and the server

- **The server checked nothing when `--ctx` was given by hand** (issue #45): a context past the GPU's share paged
  without a word. skipzx-lab's 16 GB A770 ran the 27B at 92K with `--mtp` at 11 tok/s. Now it warns, as runz does:
  how much it needs, the share, and the largest `--ctx` that fits. The context stays the one asked for. `--ctx auto`
  already counted drafting and is unchanged.
- **Both programs name the context that fits, counted the same way for every mode:**
  - everything the engine holds but the context, plus the context's own bytes at that length;
  - with `--mtp`, the MTP block, its KV rows and the draft buffers are in that count, so drafting gets a smaller
    context than plain, never a pass.
  - On the laptop: the 27B at 230K, plain "--ctx 219136 fits" (19.89 GiB, no warning; 1,024 more warns), with
    `--mtp` 182,272.
  - On an A770's share (15.11 GiB, imitated with `ARC_DRIVER_RESERVE_GB`), `--chunk 1024`: plain fits to 78,848,
    `--mtp` to 55,296, so `--mtp` at 64K pages as well.

## 2026-11-21: a leaner loop for the A770's prompt GEMM (opt-in)

- `ARC_GEMM_XW_PTR=1` (Alchemist's big tile): the copy of each chunk into local memory has its addresses set once
  before the loop, and each step adds a constant. Before, every unit rebuilt a 64-bit address from the chunk index
  and tested whether it was real, inside a branch.
  - The kernel-loop audit had found the GEMM's inner loop bound by issuing its own instructions: 410 for 33 DPAS on
    an A770 (12.4 each), which caps it near 65 % of the card's peak. With the switch: 184 (5.6 each), no spill.
  - The same bytes to the same places: the same results bit for bit. xw_peak compares it with the default, and the
    27B's logits through the engine are byte for byte the same (the laptop, DPAS emulated).
  - Its speed on an A770 is not measured yet: asked on #45. If it wins there, it becomes Alchemist's default.
  - The start-up line says when it's on.

## 2026-11-21: batching on one Alchemist card under Linux, as the last release meant

- **Fix (issue #72):** release 2026.11.21.05 meant to send an Alchemist card's kernels in recorded lists under Linux
  too (`ARC_BATCH=64` by default), but on one GPU it didn't: GPU 0's default was still set only under Windows, so
  only a split model's other GPUs got it.
  - hakangit's A750 (i915) on .21.05 with no variable: MTP 68.2 tok/s (4B) / 55.3 (9B), the same as `ARC_BATCH=0`.
    With `ARC_BATCH=64`: 88.4 / 68.5, which is now the default. 128 gives the same (88.2 / 68.4).
  - Every drafting run gave the plain run's tokens. Windows is unchanged: the same line, now on both.

## 2026-11-21: the kernel loops counted for any GPU

- `tools\kernel_isa.ps1` compiles for any Intel GPU now (`-Device dg2-g10`, `bmg-g31`, `ptl-h` ...; `mtl` as
  before). It also counts DPAS, local-memory and 2D block messages, and with `-Loops` each loop's body, biggest first,
  with its instructions per DPAS or DP4a.
- What it found (docs/xe2-tuning.md section 16, the kernel-loop audit's step L2):
  - **The A770's prompt GEMM is bound by issuing its own loop:** 410 instructions for 33 DPAS (12.4 each), which caps
    it near 65 % of peak, as hansong80's `xw_peak --probe` measured. Each global load recomputes its 64-bit address,
    about 20 scalar instructions. A leaner loop is the next opt-in form (QUEUE).
  - **The B390's GEMM loop uses plain loads:** 1,323 instructions for 128 DPAS, against 317 with the 2D block loads,
    which are on by default only on discrete cards. Asked on #1 and #74 to measure them (`ARC_GEMM_XW_2D=1`).
  - gdn_recur's L1 prefetch is absent from the A770's offline build: its guard needs an extension that ocloc doesn't
    define for dg2. To check on a card.

## 2026-11-21: the chat template checks without Python; no .py left in the repository

- `tests\test_chat_golden.ps1` replaces `tests/test_chat_render.py` and `tests/test_chat_jinja.py`. It renders the
  same cases with `bin\chat_render.exe` and compares each rendering with a committed golden, byte for byte:
  - the native Qwen rendering, 7 cases with and without Qwen3.8's form (14);
  - minja on the three Gemma 4 templates in `tests\data`, 9 cases each (27).

  `tests\run_tests.ps1` runs it, so it no longer skips where Python or Jinja2 is missing.
- The goldens are what Jinja2 rendered, made without running Python: both .py checks passed on 2026-11-10, every case
  identical, and nothing they render has changed since. The templates, `src\json.h` and minja are untouched, and the
  two later changes to the renderer are a conversation that opens with a tool message (no case does) and an error
  message moved to the log. The test says how to update a golden after a change.
- Checked in Windows PowerShell 5.1 and 7: all 41 identical. A golden changed by one letter is reported, with the
  byte where it differs.
- With this, the repository holds no Python at all. The packages already held none.

## 2026-11-21: the tiny DeepSeek-V4.1 test model made in C++

- `tools\make_tiny_deepseek41.cpp` replaces `tools/make_tiny_deepseek41.py`. It writes the same keys, tensors,
  shapes and types, with the same options (`--seed --nu --heads --real --unsloth --q5_1-down`), and each encoder
  rounds as the .py's did. Build it with `.\build.ps1 tools\make_tiny_deepseek41.cpp`.
  - Its random numbers come from its own generator, not NumPy's, so a seed gives other weights than the .py's did.
  - Checked against `tools\ref_forward` (100 random tokens, mean over every position, prompt / token by token):
    - every expert used (`--nu 16`): KL 0.0027 / 0.0028 (the .py's model: 0.0032 / 0.0033);
    - Unsloth's types: 0.0044 / 0.0044 (0.0052 / 0.0052);
    - Q5_1 down experts: 0.0029 / 0.0029;
    - 4 of 16 experts: 0.055 / 0.054 (router near-ties flip experts, as before: 0.003-0.084).
  - `--real` (4.4 GB, experts made and written one at a time) loads and generates.
- `tools/gemma4v_ref.py` is removed: `tools\gemma4v_ref.cpp` already did all it did, and more (the 26B's
  encoder).

## 2026-11-21: the expert-cache simulators in C++

- `bench\cache_sim.cpp` replaces `zero/cache_sim.py` (routes recorded with `ARC_DUMP_ROUTES`), and
  `bench\xtrace_sim.cpp` replaces `bench/xtrace_sim.py` (traces recorded with `ARC_XTRACE`). Build them with
  `.\build.ps1 bench\cache_sim.cpp bench\xtrace_sim.cpp`; they also build with g++ on Linux.
  - The same policies, options and tables as the .py files. xtrace_sim breaks ties the same way. cache_sim's `lfu`
    now picks the smallest (layer, expert) among equally used victims; the .py took whichever its hash set listed
    first, so on ties its lfu rows could differ a little.
  - Checked on hand-made traces whose counts were worked out by hand: every policy of both tools gives the expected
    misses. Then on the laptop's Flash Next (256-token prompt, 48 tokens): no policy beats the engine's LRU (slru
    +0.3 to +5.1 %, fewest uses +0.1 to +5.5 %), and Belady's bound is -42.7 %, as the earlier study found.

## 2026-11-21: the prompt benchmark in PowerShell

- `bench\prompt_bench.ps1` replaces `bench/prompt_bench.py`: the same request (one landing-page task on every model,
  a fixed seed, reasoning off), the same saved answer and page, and the same summary record in
  `bench\results\landing\summary.jsonl`, which `bench\review_data.ps1` reads.
  - The .py could hang for good: the serverz that zero-server started in the background held its captured output
    open. The .ps1 waits for zero-server's own process and reads its output from files.
  - Checked in Windows PowerShell 5.1 and PowerShell 7 on the laptop's 4B (64 tokens): the same answer, tokens and
    draft counts in both.

## 2026-11-21: streamed MoE right on every GPU; Alchemist sends kernels in lists on Linux too

- **Fix (issue #74):** on a Panther Lake B390 under Linux, Flash Next generated broken text while its expert cache
  was cold: stray CJK characters, words cut in the middle (KL against f32 0.88 mean, 10 of 13 top tokens). It was
  the race of #56: a token's missed experts read their address row while the host was still writing it.
  - The fix of #56 (the row copied to the GPU first, `ARC_XSPLIT_TAB=1`) was the default only on Alchemist. It is
    the default on every GPU now. With it the reporter measured KL 0.0075 and 13 of 13 top tokens, the same as with
    no split at all.
  - The same results bit for bit where the race didn't strike (the laptop's Flash Next, 64 tokens after a 3,584-token
    prompt). It costs one more host wait per split layer: about 4 % of generation on the laptop, whose experts stream
    from its SSD.
- **Alchemist cards (A770, A750, A580) under Linux send a token's kernels in recorded lists**, as under Windows since
  2026-11-18.
  - An A750 with the i915 driver: the 4B's generation 46.0 -> 66.3 tok/s (128 per list: 66.6), with MTP 68.4 -> 88.5
    (hakangit, #72).
  - An A770 with the xe driver: no change either way (skipzx-lab, #45).
  - The same commands in the same order, so the same results. The one-row matrix-vector rule for unbatched Alchemist
    no longer applies there (the same sums, other rows a sub-group).
- `AGENTS.md`: no Python anywhere in the repository; every `.ps1` works in Windows PowerShell 5.1 and 7.

## 2026-11-21: every script reads text the same in Windows PowerShell 5.1 and PowerShell 7

- 5.1 read a UTF-8 file without a BOM as the ANSI code page; 7 reads it as UTF-8. So a non-ASCII character in a
  model path, a preset or a setting came out as 2-3 wrong characters under 5.1. The prompts `bench\gpu_sweep.ps1`
  and `bench\website_bench.ps1` build from README.md and USAGE.md came out differently too.
  - Every `Get-Content` of a text file now says `-Encoding UTF8` (59 lines in 13 scripts).
  - Those two benchmarks' prompts are now the same in both. Under 5.1 they differ from before, so a 5.1 run from
    before this is not comparable with one after it; 7's are unchanged.
- `tests\test_ps_compat.ps1` (in run_tests) parses every script with both, and refuses a `Get-Content` without
  `-Encoding`.

## 2026-11-21: the launcher's speed and accuracy options

- The Windows launcher's Server options card has a section "Speed and accuracy options": each option of the catalog
  with a default / on / off choice, what it does, whether its output stays the same, what it costs, and whether it is
  for this PC (one that is not is greyed, with the reason, not hidden).
- Start passes the options you changed to `zero-server`, which keeps them for later starts; `-DefaultOpt <id>`
  puts one back to its default.

## 2026-11-21: zero-server turns the speed and accuracy options on and off

- `zero-server start -Opt copy_draft` turns an option of the catalog on, `-NoOpt xcpu` off (an option that is on
  by default, off). `-Opt budget=12` gives one its value; several go with commas.
- The choices are kept for later starts (`%USERPROFILE%\.inferzero\options.json`); the command line, the dry run
  and the start lines say which are on. A change restarts a running server.
- `zero-server options` lists every option: its state, what it does, its cost, whether its output is the same or
  checked, and whether it is for this PC, from the PC's graphics.

## 2026-11-21: one list of the options you can turn on

- `scripts/options.tsv` lists the opt-in switches people may want, ten for now, each checked against the code:
  - what each does and its default;
  - which PCs it is for and what it costs;
  - whether the output stays the same;
  - what it measured, and on whose card.
- The launchers will read it for their options screens (next).
- `docs/options.md` is the same list for reading, made from that file (`powershell -File tools\gen_options_md.ps1`).

## 2026-11-21: expert files carry a checksum per layer

- A new expert file records each layer's checksum in its header, in room the layout already keeps (marked "SUMS"):
  the layers' bytes and offsets, and the header's version, are as before.
- **Nothing is rebuilt**, in either direction: existing expert files load as before (they just have no sums to
  check), and older releases load the new files as they are (the first form, a version 2 header, would have made them
  rebuild it; caught in review before a release).
- `ARC_EXPERTS_CHECK=1` reads a file with sums back at load and names the first layer that differs from what was
  written. `ARC_XBUILD_VERIFY=1` does the same right after a build.
- Tested on a small model:
  - an old file loaded unchanged, and an older release loaded a new file unchanged;
  - a new one has the same layer bytes;
  - a build resumed after a stop matches one built in one go;
  - a flipped byte is caught, with its layer named.
- The package check (S8) now reads both files through the guarded copy, and the stall watch covers it.

## 2026-11-21: a package's first part read in large pieces

- Making a package (`serverz --pack`) first writes every tensor but the experts. Those were written straight out of
  the mapped model file, so each page was read on its own as it was reached, which is slow on a USB disk and hung a
  drive that struggles with that pattern.
- Now each 64 MB piece is copied into a buffer while the next one is read ahead in large requests, and the copy has
  S4's protection (a page that can't be read is read again).
- The package is the same bit for bit: the test model's package and expert file byte-identical the old way, the new
  way, and with read faults forced. `ARC_XBUILD_PREFETCH=0`: as before.
- **A package is checked before it's used**: its GGUF is read back and compared with the source (its keys, its
  tensor list, every kept tensor's bytes). One that differs is never put in place. A write gone wrong on a bad drive
  then shows at once, not as a broken model later. `ARC_PACK_CHECK=0` skips it.

## 2026-11-21: less host work per generated token; an expert file's build survives a bad read

- **Each one-token matrix-vector product keeps its kernel** after its first launch. Before, every launch put together
  the kernel's name from ~6 strings and looked it up, ~330 times a token. On Windows those allocations share one
  lock, which a split model's GPU threads wait on. The rest of the fix for split generation (issue #4): the same
  kernels, logits byte-identical (the 4B, the 4B over 2 simulated GPUs, the 27B, Flash Next). No change on one GPU on
  the laptop.
- **A page of the model that can't be read while the expert file is written** (a drive error, a file changed under
  the build) ended the process before: Windows' in-page error, Linux's SIGBUS.
  - Now those bytes are read again with ordinary reads, up to 3 tries. If that fails, the build stops with the file
    and offset, and its finished layers are kept.
  - A SIGBUS anywhere else still reaches the crash log as before.
  - Tested: three copies made to fail on purpose, the expert file the same bit for bit; on Linux, a truncated
    mapping's pages come back as errors.

## 2026-11-21: a fix for slow generation over several GPUs (to be confirmed on 4 B60s)

- Generation split over several GPUs had become no faster than one card: the 27B over 4 Pro B60s took 48.8 ms a
  token (18.6 tok/s), against 18.4 ms in early November (issue #4).
- The likely cause: two checks added on the path of every matrix-vector product asked the GPU driver for its
  properties each time, ~660 driver calls a token on each GPU. They also read environment variables, which on Windows
  take a lock all threads share. A split model's GPUs, one thread each, then waited on each other inside the driver.
- These answers never change, so they are read once now: the GPU's kind and thread count, the GEMV's read-ahead
  choice, and five switches read per layer.
- The same results bit for bit (the 4B, the 4B over 2 simulated GPUs, the 27B). One GPU is not slower or faster on the
  laptop. The 4 B60s will show whether this was all of it.
- Expert files: a build kept after a stop (S2) now says its size and how to remove it, and the start-up names one
  left more than 7 days (`ARC_XTMP_DAYS`).

## 2026-11-21: an expert file's build goes on where it stopped

- Writing a MoE model's expert file (once per model: Flash Next 52 GB, tens of minutes) began again from zero after
  any stop. Now it goes on from the next layer, whether the stop was Ctrl+C, a stopped container, a power cut or a
  full disk.
  - After each layer the file is flushed to the drive, and a small `.tmp.done` file records where the next layer
    starts.
  - The next start takes it up only when that record and the `.tmp`'s header match this model's build exactly;
    anything else is removed and built from zero, as before.
- The finished file is the same bit for bit: a test model's expert file stopped after 2 of its 10 layers and finished
  at the next start, byte-identical to one built in one go.
- `ARC_XBUILD_RESUME=0` turns it off. (`ARC_XBUILD_TEST_STOP=N` stops a build after N layers, for tests.)

## 2026-11-21: a test for Flash Next's stray words: f16 keys alone, and a counter

- **`ARC_FN_KV16=1`** (a test switch): Flash Next's default mode with one part of precise mode, the attention's K / V
  cache in f16 and its scores in f32. Everything else stays as in the default mode (one pass, the same experts).
  It tests whether the int8 keys cause the stray words of other languages in Japanese answers (issue #4).
  - The laptop, a 3,584-token prose prompt, 128 tokens, against precise mode: KL 0.0045 mean / 0.0178 max over the
    first 24 tokens (the top token the same up to there), the default 0.0076 / 0.0211 over 17.
  - Generation 5.18 -> 4.69 tok/s on the laptop. Unset, nothing changes.
- **`bench\stray_words.ps1`** counts them in answer files, per answer and per 1,000 characters: characters of
  scripts Japanese doesn't use (Hangul, Hebrew, Cyrillic, Greek, Arabic, Thai, Devanagari), and Latin words of 4+
  letters inside Japanese sentences.

## 2026-11-21: copy drafting (opt-in): code edits and other repeated text up to 39 % faster, the same output

- **`--copy-draft`** (runz and serverz, with `--mtp`; or `ARC_COPY_DRAFT=1`): when the last 4-8 tokens appeared
  earlier in the conversation, the tokens that followed them there become the drafts. Otherwise the MTP block drafts
  as before.
- The model checks every draft as it checks MTP's, so the text stays the same as without it.
- A copied step skips the MTP block's own drafting, which is most of the gain where the answer repeats the context:
  - a code edit (give back a function with one change), 256 tokens: the 4B 42.0 -> 58.5 tok/s, the 27B 17.5 -> 22.1;
  - a summary, where little repeats: unchanged within the noise (the 4B 29.4 / 30.2, the 27B 9.7 / 9.3).
- `ARC_COPY_NGRAM` / `ARC_COPY_MIN` set the longest and shortest match (default 8 / 4). Shorter matches made the
  summary slower (2 tokens: 27 % of their drafts right).

## 2026-11-21: long MoE prompts decode each matrix once (Flash Next, the 35B)

- A prompt longer than a chunk goes through the model layer by layer. Each matrix is decoded into a staging slot
  once and used for every chunk.
- But the shared expert's gate / up / down, the router, DeltaNet's a / b and Flash Next's indexer all used one slot
  together. They took it from each other, so they were decoded again in every chunk: in Flash Next, ~242 decodes a
  chunk.
- Now each has a slot of its own. On the laptop, at 4 chunks of 512, the decodes fell from 1,124 to 692. On the
  laptop the prompt speed stays within its noise (decoding is a few % of its time); on a B70, decoding was 8-13 % of
  Flash Next's prompt time.
- The same results bit for bit (Flash Next, the 27B, the 4B). About 9 MB more VRAM on Flash Next; none on dense
  models.

## 2026-11-21: the whole sweep: every form of the 8-lane XMX GEMM, at every context, and it goes on after a stop

- **`bin\xw_peak.exe --all`** builds every form of the A-series / 140T prompt GEMM (`gemm_xw8`) with both operands
  packed, about 600: tiles, work-groups, the double buffer, the epilogue and local memory stages. It runs in stages:
  - stage 0 drops the forms that spill or do not build;
  - stage 1 times every survivor quickly on the 27B's six GEMM shapes, an untimed launch first;
  - stage 2 times the best 10 again and checks them bit for bit;
  - stage 3 tries the best 3 with the row order and the prefetch.
- It shows a progress bar with the time left. Each part can be chosen (`--family`, `--tm`, `--nc`, `--wg`, `--cps`,
  `--db`, `--epi2`, `--ms`, `--stages`, `--limit`, `--top`). `--env-out` writes each winner as the engine's switches.
- **`bench\gpu_sweep.ps1` / `.sh`, phase X** (on by default, 8-lane XMX GPUs only) runs it, then each winner in the
  model at each context with the logits compared.
- **`-Contexts`** (default `512,4096,32768,max`; max is the longest of 262K / 131K / 64K that runs) also sets phase
  C's lengths.
- **A stopped sweep goes on where it stopped** (each finished run saved under `%USERPROFILE%\.inferzero\gpu-sweep`;
  `-Fresh` starts again).
- **Engine, opt-in:**
  - `ARC_GEMM_XW_SLM_TILE` takes any TMxCOLS of 16 / 32 / 64. A shape over 64 KB of local memory is refused, and the
    default is used.
  - `ARC_GEMM_XW_DB=1` works on every tile.
  - The start-up line names the big tile in use.
- Unset, nothing changes. The new shapes give byte-identical logits (the laptop, the A770 preset emulated).

## 2026-11-21: Alchemist: switches for bigger work-groups through local memory, for a sweep

- The A770's big tile (32 x 32, 4 x 4 sub-groups through local memory) can now take other shapes, all opt-in, for a
  sweep on the cards:
  - `ARC_GEMM_XW_SLM_TILE=32x32 / 64x16 / 32x16` sets the sub-group tile;
  - `ARC_GEMM_XW_SLM_WG="X,Y"` sets column x token sub-groups, 2, 4 or 8 each, at most 32 (8 columns is new);
  - `ARC_GEMM_XW_SLM_CPS=1 / 2 / 4` sets how many 64-value chunks one barrier covers (`gemm_xw8 -DXS_CPS`).
- Each prompt GEMM takes the chosen shape where it divides the matrix and the tokens, and otherwise the fallback tiles
  as before. Unset, nothing changes.
- On DG2 at 256 registers (ocloc), these build without a spill: 32 x 32 at 4x4 / 4x8 / 8x4, 64 x 16 at 4x4 / 8x4 /
  8x2, and 32 x 16 at 8x4, with CPS 1. With CPS 2, only 32 x 16 8x4 (at 32 x 32 it spills 3.1-3.6 KB). xw_peak has a
  row for each.
- The same results bit for bit: xw_peak with the DPAS emulated, every row identical, and the laptop's 27B with the
  A770 preset emulated, logits byte-identical in four of the shapes.
- The XMX speed check's log line names the kernel it timed: a model in the 140T / Alchemist preset is not judged by it
  (issue #71).

## 2026-11-21: confirmed on cards: the A770's default, ARC_XCPU run to run, MTP's gain

- **The A770's new default** (b443eff), skipzx-lab at 150 W with no switches: the 27B 4K prompt 1,490.5 tok/s, 16K
  1,369.2; with `ARC_GEMM_XW_BIG=0` 1,326.0, so the big tile is +12.4 % on its own. The fallback tiles run at 128
  tokens and the big one at 512, as the log says.
- **ARC_XCPU** on a B60 (Flash Next, a ~4,200-token prompt then 128 tokens, each run from the same `.lru`): three
  runs byte-identical, the same 5,691 experts on the CPU each time, 26.6-27.1 tok/s.
- **MTP on a real prompt** (the 27B, a coding question in the chat template, 256 tokens): the A770 22.4 -> 48.6 tok/s
  (+117 %), the B60 23.3 -> 54.3 (+133 %). On the site and in the README.

## 2026-11-21: Q4_1, Q5_0 and Q8_1 models load (decoded on the CPU, kept as Q8_0)

- GGUFs whose weights are Q4_1, Q5_0 or Q8_1 now load. These types have no GPU kernel here yet, so their matrices are
  decoded on the CPU at load and kept as Q8_0, with one line in the log per type; before, loading stopped with
  "unsupported weight type". Q8_0 has finer steps than those types, so little is lost. The conversion runs 1,024
  rows at a time, never a whole matrix of floats in RAM (a vocab-sized output head would have been 5 GB; the
  Researcher's review).
- Checked on the laptop with Qwen3.5 4B, converted by llama.cpp's llama-quantize from its UD-Q4_K_XL:
  - each against its own f32 reference, all 480 positions (corrected 2026-11-21: the first figure, 0.00188, was one
    position): Q4_1 KL 0.00128 (max 0.0245), the same top token 473 / 480; Q5_0 0.00103 (max 0.0118), 472 / 480;
    so the conversion to Q8_0 costs almost nothing;
  - against the original UD-Q4_K_XL's reference: Q4_1 KL 0.042 and Q5_0 0.020, mostly llama-quantize re-quantizing
    an already 4-bit model;
  - a test of each decoder against the source tensor found and fixed a wrong sign in Q5_0's (correlation 0.9991
    after);
  - Q8_1 is converted per block too (Q8_0 takes its own scale), so it is close but not exact.
- Q8_K, TQ1_0 and TQ2_0 are known now, so a file holding them opens; a weight of theirs still stops with a message
  naming the type.
- The types that loaded before are unchanged: Qwen3.5 4B and the 27B give byte-identical logits before and after.

## 2026-11-21: bench/gpu_sweep waits for other GPU runs, and warms drafting with the question

- `gpu_sweep.ps1` / `.sh` wait at start, up to 30 minutes, while another runz, serverz or xw_peak is running. Two
  programs on one GPU made a whole phase's numbers wrong: on the laptop, MTP measured 2-2.5x slower than plain, when it
  is 80 % faster.
- The untimed warm-up of a drafting setting (`--mtp`, `--dflash`) now runs the question itself. `--bench-pp` decodes
  plain, so MTP's verification kernels were first built inside a timed run.

## 2026-11-21: Alchemist: the 27B's prompt +15.7 % (a 32 x 32 tile through local memory by default)

- **Measured** by skipzx-lab (an A770 held to 150 W, Linux, issue #45, main b9f011c, the 27B IQ3_S, a 45 s pause
  before each run). Every prompt GEMM at 32 x 32 with 4 column sub-groups, its operands through local memory with 4
  token sub-groups a work-group, and the epilogue a row at a time:
  - the 4K prompt 1,286.0 -> 1,488.4 tok/s (+15.7 %), 16K 1,191.6 -> 1,358.1 (+14.0 %);
  - the GEMMs 2,168 -> 1,690 ms, every one faster, the small ones too (ssm_out 164.2 -> 114.6, o 55.0 -> 38.3);
  - xw_peak 116.7 TOPS, 50 % of this card's measured 232, against 40 % before.
- **Now the default on Alchemist** (A770 / A750 / A580) wherever that tile fits: both operands packed, N a multiple
  of 128, and the tokens a multiple of 128. Elsewhere, short prompts for example, a GEMM keeps the per-shape tiles of
  before: 32 x 32 without local memory is far slower there (548.8 tok/s in the same round). It is decided per GEMM, so
  a split model's GPU decides for its own slice. The row-at-a-time epilogue (`ARC_GEMM_XW_EPI2`) is on by default on
  Alchemist for every tile (+2.8 % on its own). The same results bit for bit: the laptop's 27B with the A770 preset
  emulated, logits byte-identical at 1,024 tokens (the big tile) and at 600 (the fallback), and checked at start.
  `ARC_GEMM_XW_BIG=0` and `ARC_GEMM_XW_EPI2=0` turn them off. The Arc 140T keeps its preset until measured.

## 2026-11-21: expert-file and package builds stop when the source drive stops answering (S1); B3, B5

- **A stall watchdog (S1):** writing an expert file or a package reads a whole GGUF, and some drives hang under it (a
  D: drive did on 2026-11-19), which used to hang the build for good. Now:
  - when no data comes from the source for 60 s, the log names the drive and advises copying the model to an internal
    SSD;
  - the build stops at its next check with a clear error, removes its `.tmp`, and a server stays up;
  - only a read that never returns (3x the limit) ends the process, with exit code 4. The next start removes a
    leftover `.tmp`, as before.
  - Packages are written in 64 MB pieces, each counted as progress, so a slow USB disk is never taken for a hung one.
    `ARC_XBUILD_STALL_S=N` sets the limit; 0 waits forever. Checked by a forced stall (`ARC_XBUILD_STALL_TEST=1`): the
    error, then nothing left in the destination.
- **B3:** the 4 alignment bytes in each layer entry of the expert file's header are now written as zeros. Before, they
  were whatever the compiler left, so two builds could differ by them. Same size and layout: files already written
  read as before. Checked: packing Gemma 4 26B gave an expert file byte-identical (SHA-256) to the one already there.
- **B5:** a failed write of the expert folder's `source.txt` is now reported. Only the cleanup of orphaned folders
  reads it.
- B4 / S18, paths with any characters and longer than 260 on Windows, were already done: every exe embeds a UTF-8
  code page and long-path manifest (e82fa74).
- The start-up GEMM self-check (W3) now runs once per engine, not once per process, so each GPU of a split over unlike
  cards is checked (the Researcher's review).

## 2026-11-21: each encoder says what its GEMMs run on

- The first time an image or audio encoder runs, the log says what its matrix products run on, for example `the vision
  encoder (Qwen3-VL): int8 GEMMs, two passes, on the prompt GEMM (XMX)`, or `(DP4a)`, or `f16 GEMMs (vit_gemm_f16, no
  XMX)`. That covers the Qwen3-VL and Gemma 4 vision encoders, the Qwen3 and Gemma 4 audio encoders, and Whisper. On a
  card owner's log it answers whether the encoders already use the matrix engines, before work starts on putting them
  there (QUEUE E0). Results are unchanged.

## 2026-11-21: a start-up check of the GPU's matrix kernels; the launcher's button to Intel's driver page

- **A start-up self-check** for a bad driver whose version looks fine: runz and serverz put a small Q8_0 matrix (64
  x 1,024 random values) and a few tokens through the GPU's prompt GEMM, and compare with the CPU's result of the
  same int8 values. A relative error past 1e-3, or a value that is not a number, gives the same warning as an old
  driver: in the log, in the web UI's bar and in the launcher. It never refuses to start. It runs once per process and
  takes a few milliseconds. The XMX kernels were already checked against this one at start, bit for bit, so this also
  covers them. On the laptop (32.0.101.9175) the error is 1.7e-07. `ARC_SELF_CHECK=0` skips it; `=2` shows the
  warning as if it had failed (a test).
- **The Windows launcher** shows a bar with a button to Intel's driver page when the server reports an old driver or
  a failed check, once per server start.

## 2026-11-21: ARC_XCPU's choice of experts no longer depends on timing; guarded waits; the RAM tier rule per GPU

- **ARC_XCPU** (on by default on discrete 16-lane cards) computes on the CPU the experts that are missing from the
  GPU's cache, and copies them into the cache in the background. Which experts the CPU takes depended on two kinds of
  timing, so a temp-0 output could differ between two runs with the same start:
  - the background copies were committed whenever they had finished (a query at every layer). Now each one is
    committed at a fixed point: at its layer in the next token, after waiting for it, which it has nearly always
    finished by then;
  - near the RAM tier's cap, the main thread and the prefetch threads raced for its last records. Now the main
    thread takes the records for each prefetch, in order, before its thread starts, and the threads take none.
- **Checked on the laptop** (Flash Next, ARC_XCPU forced on, a 2 GiB tier, an 8 GiB cache, 256 tokens and 64
  generated, every run from the same saved cache state): three runs byte-identical, 116 experts on the CPU each time.
  The code before this was also identical there (115 each): on an iGPU the copies finish too quickly for the timing to
  show. The race was seen on B-series cards over PCIe, where it is to be confirmed. With ARC_XCPU on, the results
  differ from before by which expert the CPU takes (one in 116 here), within the KL measured for ARC_XCPU. Without it,
  nothing changes.
- Note for anyone comparing runs: the expert cache saves its contents at the end of a run (the model's `.lru` file)
  and fills from it at the next start, so two runs start from different caches unless that file is the same.
- The waits for the background copies honour a split model's abort, as the other waits do: a GPU that fails ends
  the run with an error instead of hanging the other GPUs (the Researcher's review).
- The start-up fill's RAM tier rule (`ARC_XRAM_FILL`) compares against each GPU's own share of the experts on a split
  model, not the whole file (the Researcher's review). One GPU: unchanged.

## 2026-11-21: `ARC_XHIST=1`, the prompt's missing experts by tokens; `ARC_ATTN_X8=1` only where XMX passed its checks

- `ARC_XHIST=1` (diagnostics): at the end of a run with streamed experts, the prompt's experts that were not in the
  cache when their layer began, counted by how many of the chunk's tokens route to each (1, 2, 3-4, ... 129+), with
  their GB. It is the first measurement for computing thinly routed missing experts on the CPU. It changes no result
  (the laptop's Flash Next: logits byte-identical); there, a 256-token prompt's 5,929 misses had 1,900 experts with one token.
- `ARC_ATTN_X8=1` on an iGPU (the Arc 140T) now needs XMX to have passed its start-time checks, as the other XMX
  attention does: a 140T whose check failed, e.g. with an old driver, keeps `attn_q8`. Discrete cards unchanged.

## 2026-11-21: help texts and docs for PRs #67 / #68; the Max defaults' switches read once

- `runz --help`, `serverz --help`, docs/server.md and docs/models.md describe the borrowed MTP block for dense
  Qwen3.5 / 3.8 fine-tunes, and say that pruned MoE models with 16 to 256 experts load.
- The Data Center GPU Max defaults' environment checks (`ARC_GVROWS`, `ARC_GEMM_XW_PF` / `_PFD`, `ARC_GEMM_XW_WGX`,
  `ARC_MOE_WY`) are read once, not at every GEMV / GEMM / MoE launch. The same results on every GPU (the laptop's 27B
  at 4,096 tokens and Flash Next: logits byte-identical).
- `ARC_GEMM_XW_SLM` and `ARC_GEMM_XW_EPI2` say at start when they are on (skipzx-lab, #45).

## 2026-11-21: `ARC_GEMM_XW_EPI2` (opt-in, 8-lane XMX): the prompt GEMM's epilogue a row at a time

- **Why:** skipzx-lab's A770 (issue #45, 150 W, main a5deeff) measured `ARC_GEMM_XW_SLM`.
  - The local-memory loop lost on the 32 x 16 GEMMs: the 27B's 4K prompt 1,288 -> 1,210 tok/s at `=1`.
  - It won only on the 16 x 32 ones at `=8` (ssm_out / o / k / v -8 to -9 %).
  - In `xw_peak`, the 32 x 32 tile through local memory was the fastest down GEMM (7.81 against 8.38 ms) but lost
    badly on up + SwiGLU (12.67 against 8.13). The plain 32 x 32 tile showed the same gap in the model (up 898
    against gate 636 ms, where 32 x 16 has 551 against 489).
  - The difference is the epilogue: a row's C was read and written in 32-byte pieces spread over the tile, and the
    accumulate mode was decided for every value.
- **New:** `ARC_GEMM_XW_EPI2=1`, `gemm_xw8.cl -DXW_EPI2`.
  - A row's columns are read and written in one block message, and the SwiGLU mode is decided once.
  - Store and add keep the add under its per-value condition. In a loop of its own, `-cl-mad-enable` fused the
    product into the add, and split-K sums differed by an ulp (caught by `xw_peak --emu`).
  - ocloc dg2-g10 at 256 registers, no spill: 32 x 16 with the read-ahead loop 239 -> 175 memory messages, 32 x 32
    525 -> 269, 32 x 32 through local memory 457 -> 201.
  - The same operations per value: bit for bit, `xw_peak --emu` on the 125H (store, SwiGLU, split-K: identical),
    checked at start.
- `bench/xw_peak`: `e2` variants of the default, of 32 x 32 with 2 column sub-groups and of 32 x 32 through local
  memory. Speed to be measured on an A770.

## 2026-11-21: pruned MoE models, and borrowed MTP heads for dense fine-tunes (PRs #67, #68 by triblax)

- **MoE models with 16 to 256 experts** load, not only 256. A pruned model whose expert count is not a multiple of
  16 (e.g. 184 of the 35B-A3B's 256) routes with its last lanes masked. Models whose count is a multiple of 16 run
  the same kernels as before, token for token (Flash Next on the laptop: logits byte-identical). Flash Next keeps its
  512.
- **`--mtp-model` for dense Qwen3.5 / 3.8 fine-tunes** shipped without their own MTP block: the base model's GGUF
  lends its block (e.g. a 27B fine-tune with the Qwen3.8 27B's). Drafts are verified, so the output stays the
  fine-tune's own. A head is taken only when it matches the model: the same width, layer count, expert count, heads,
  head sizes, FFN width and vocabulary. The last four checks were added at the merge, and a head that doesn't match
  is refused with a message.

## 2026-11-21: Data Center GPU Max (Xe-HPC): its own defaults, from hibagus's sweeps

- A Data Center GPU Max (Ponte Vecchio: device IDs 0x0BD0-0x0BDB, or the name) now takes the settings hibagus's
  one-stack Max 1550 sweep found faster (issue #48, 2026-11-20, main 03c310a). In every case the logits were the same
  as with the defaults:
  - the 27B's generation: one GEMV row a sub-group, as `ARC_GVROWS=1` (27.77 -> 29.70 tok/s, +6.9 %);
  - the 27B's prompt: `ARC_H8_NORM=0`, `ARC_H8_REUSE=0`, `ARC_GEMM_XW_PF=2`, `ARC_GEMM_XW_WGX=8` (alone +5.9 / +4.6 /
    +2.9 / +2.1 %; together 4K 2,645 -> 2,910 tok/s, +10.0 %, 16K 2,129 -> 2,306, +8.3 %);
  - the 35B-A3B's generation: `ARC_MOE_GR_GU=4`, `ARC_MOE_GR_DN=8` (81.6 -> 88.0 tok/s, +7.9 %);
  - the 35B-A3B's prompt: `ARC_MOE_GRF=256`, `ARC_MOE_CG=4` (4K 4,665 -> 5,138 tok/s, +10.1 %; 16K 3,994 -> 4,348).
- Each of these set by hand decides as before. Other GPUs are unchanged: the laptop's 27B at 4,096 tokens gives the
  same logits byte for byte. Each GPU of a split model decides for its own. Two of the pairs were measured together
  only with a value next to these (the combined 27B prompt run with `ARC_GEMM_XW_PFD=8`, the 35B prompt with
  `ARC_MOE_CG=8`), so a clean run on the new defaults is asked for.

## 2026-11-21: build.sh builds bench/xw_peak (`./build.sh xw_peak`)

- The 8-lane XMX GEMM probe now builds on Linux too, so Alchemist owners there can measure `ARC_GEMM_XW_SLM`'s
  variants in a minute. It is not in the default list: `./build.sh` builds what it built before.

## 2026-11-21: `ARC_GEMM_XW_SLM` (opt-in, 8-lane XMX): the prompt GEMM's operands through local memory

- **Why:** an A770's H8 prompt GEMMs take 1.5-2x a Pro B60's (skipzx-lab, issue #45: the 27B's 4K prompt 1,289
  against 1,866 tok/s) although it measures more int8 XMX (232 against 196 TOPS); its GEMMs run at ~28 % of that.
  The Arc 140T's `xw_peak --probe` (issue #5) reached 85 % with the loop's operands already in L1, so they arrive
  late. No tile, column sub-group, prefetch or lock setting closed it.
- **New:** `ARC_GEMM_XW_SLM=1` (or `=2` / `=4` / `=8`: token sub-groups a work-group, `=1` means 4).
  - `gemm_xw8.cl -DXW_SLM`: all a work-group's sub-groups copy its packed chunk (A and B) into local memory together,
    one block read each, and double-buffer it, so the next chunk's reads go out before this chunk's DPAS. Every byte
    is read from L2 once a work-group instead of once a sub-group.
  - Only where both operands are packed (the Alchemist / 140T preset) and the token tiles divide; elsewhere the
    kernel as before.
  - The same integer sums in the same order: bit for bit. Checked at start (a new check shape that runs it), and with
    `xw_peak --emu` on the 125H (store, fused SwiGLU and split-K shapes: identical).
  - ocloc dg2-g10 at 256 registers, 32 x 16 tiles with 4 x 4 sub-groups: no spill, 24 KB of local memory, 249
    memory messages against 299 for the default; 32 x 32 with 4 x 4: no spill, 32 KB.
- `bench/xw_peak`: the engine's Alchemist default (`A770 32x16 w4 DB`) and three `slm` variants, to measure on an
  A770 in a minute. Speed to be measured (asked on #45).

## 2026-11-21: streamed MoE in a server: later prompts up to 44 % faster (the start-up fill out of the RAM tier by default)

- **Measured** by teodortomas (a Pro B65 32 GB, Linux, issue #47, main 8c02980): Flash Next IQ3_S (52 GB of experts),
  a 26.4 GiB RAM tier, serverz, three 3.4K-token prompts in a row:
  - fill in the tier: 22.83 / 22.01 / 20.62 s;
  - with `ARC_XRAM_FILL=0`: 27.0 / 13.0 / 11.5 s, so the second and third prompts are 41 / 44 % faster.
  - With it, the SSD reads 13 GB a later request instead of 26-28, the tier serves ~12,000 records a request against
    ~6,000, and the generation's cache hits went from 80-83 % to 89-90 %.
  - The first request (27.0 s) was on a fresh build; that run's kernels were built during it.
- **Now the default** where the RAM tier and the GPU's expert cache together cannot hold every expert: the cache's
  fill at start goes to the GPU only, and the tier fills with what the cache does not hold. Elsewhere, including any
  GPU without a tier (iGPUs), nothing changes. The same results bit for bit (the laptop's Flash Next with a 2 GiB tier
  forced: logits byte-identical with the fill in the tier, out of it, and at the default). `ARC_XRAM_FILL=1` puts the
  fill in the tier as before, `=0` never.

## 2026-11-21: `ARC_XRAM_FILL=0` (opt-in): the expert cache's fill at start kept out of the RAM tier

- **Why:** on a B65 (teodortomas, issue #47) the second and later 3.4K-token Flash Next prompts in one serverz still
  read 26-28 GB from the SSD each (19.7-21.0 s). A request touched ~17,300 experts that were not in the GPU's cache
  when used; the 25.3 GiB RAM tier holds ~10,400 of the 2.44 MB records. If the tier held none of what the GPU cache
  holds, ~6,900 would be left for the SSD (~17 GB), but ~11,500 were read. The tier keeps the first records read and
  never replaces any, and the GPU cache's fill at start went into it too, so part of it held copies of what the cache
  already had.
- **New:** with `ARC_XRAM_FILL=0` that fill goes to the GPU cache only, and the tier fills with the records read
  later. The same data either way: the same results (the laptop's Flash Next with a 2 GiB tier forced, 256 tokens
  and 8 generated: logits byte-identical). Speed to be measured on a discrete card; the most it can save is the
  ~11 GB above, ~4-5 s of a 21 s prompt.

## 2026-11-21: self audit: an attention kernel's guard, split models' GEMM tiles, help texts

- **attn_split** (the long-prompt attention over key splits) builds only where its work-group divides the K / V tile,
  as attn_q8 does. The `#error` that kept bad sizes out was dropped for issue #70, so a build with such sizes would
  have loaded only part of each tile, with no error. No model shipped today has such sizes (head dims 64-512 all
  divide), so nothing changed for them: the same kernels as before where the sizes fit.
- **Alchemist's tile per GEMM shape** is no longer the default on split models. A shard's GEMMs are sliced while the
  rule's n_embd is not, so with 3 or 4 A770s the 27B's gate / up / down took the 16 x 32 tile that was measured
  slower for them on one card. Split models go back to the one tile they had before 2026-11-20 (the same results;
  `ARC_GEMM_XW_SHAPE=1` still turns it on).
- **ARC_XCPU's default** is decided once per engine, not with a driver call on every MoE layer of every token.
  The help texts now say where it is on: any discrete GPU without 8-lane sub-groups, a Data Center Max included. They
  also say that results move run to run with it on, so bit-for-bit checks set `ARC_XCPU=0`.
- `serverz --help` lists the discrete defaults `runz --help` already did (`ARC_GEMM_XW_GU`, `_SHAPE`, `_DB`). Stale
  comments that still called these switches opt-in are fixed. The docs now say that `ARC_GEMM_XW_NC_E=2` turns the
  per-shape tiles off. serverz's new expert line logs each of its lines with the log's prefix.

## 2026-11-21: serverz logs each request's expert reads (streamed MoE)

- After each request of a model whose experts stream from the SSD, serverz logs runz's expert line: hits and misses,
  the GB read from the SSD and how fast, and the RAM tier's size and the records taken from it. The numbers count from
  the server's start, so a request's own reads are the difference to the line before. Asked for by issue #47: on a
  B65, later prompts in one server were barely faster than the first (21.6 / 20.7 / 19.9 s), and serverz printed
  nothing that showed why.

## 2026-11-21: Alchemist: `ARC_GEMM_XW_DB=2` (opt-in), the read-ahead loop unrolled for every tile

- **Why:** with the reads one step ahead (`=1`, the default on the 32 x 16 tiles), hansong80's A770 (#45) lost on
  the 16 x 32 GEMMs (ssm_out / o +137-139 %). ocloc (2026.1, dg2-g10, 256 registers) shows no register spill in that
  loop at either tile. What changed is the unrolling: the plain loop is unrolled over 4 chunks (65 DPAS in the
  binary), the read-ahead loop is not (18). Unrolling it as it stands spills (16 x 32: 5,952 bytes at 2 chunks, 5,696
  at 4).
- **New:** `=2` reads the next chunk's weights into the same registers, a column group's right after its last DPAS of
  the chunk, so the loop unrolls over 4 chunks without a spill (76 DPAS at 16 x 32, 72 at 32 x 16). It is on every
  tile, 16 x 32 included. The same DPAS in the same order: the same results bit for bit (the laptop's 27B with the
  A770 preset emulated, 1,024 tokens: logits byte-identical for `=0`, `=1` and `=2`). The default is unchanged; its
  speed is to be measured on an A770.

## 2026-11-21: Alchemist: the 32 x 16 prompt GEMMs read one step ahead by default (`ARC_GEMM_XW_DB`)

- **Measured** by hansong80 (an A770 16 GB, issue #45, main 2a78721, the 27B's 4K prompt, two rounds): with the reads
  one step ahead, every GEMM on the 32 x 16 tile got faster (down 646.9 -> 583.0 ms, -10 %; qkv / z -8 %; q -7 %; up
  -6 %; gate -4 %), and every one on the per-shape 16 x 32 tile slower (ssm_out 169.7 -> 405.2 ms, o 55.3 -> 131.0,
  k / v +72-82 %): -4.9 % in all with it on every tile.
- **Now:** only the 32-token tiles take it, and on Alchemist that is the default; the 16 x 32 tiles keep the plain
  loop. From the gains above, about 150 ms of a 3.56 s prompt (+4 %), to be confirmed on the whole prompt. The same
  results bit for bit (checked at start; the laptop's 27B with the preset emulated: byte-identical logits).
  `ARC_GEMM_XW_DB=0` turns it off; `=1` turns it on for 32-token tiles on any 8-lane GPU (the 140T included).

## 2026-11-20: a fix for B-series cards and Lunar / Panther Lake: an image no longer ends the server (issue #70)

- **What happened:** on a GPU with 16-lane sub-groups (the B-series, the 140V / B390) the first chat request with an
  image ended serverz: building the Qwen3-VL vision encoder's attention kernel failed on an `#error` in another kernel.
  The ops kernels build each kernel together with the parts before it, so the vision kernel's build also compiled
  `attn_q8` with the vision heads' size (72) and a work-group that does not divide its tile (kemnitzs, a Pro B70).
- **The fix:** `attn_q8` is compiled only where its work-group divides the tile; in a build for another kernel it is
  left out. The text path is unchanged: the 27B's logits byte for byte as before.
- **Checked** on the laptop with the 16-lane kernels forced (`ARC_NO_SG8=1`, as a B70 takes them): before, the same
  `#error` and the server gone; after, the 27B with its mmproj describes a test image right ("a red circle, a blue
  square, and the black text HELLO 42") and the server stays up.

## 2026-11-20: `bench/gpu_sweep.ps1` / `.sh`: one sweep for every Intel GPU, in place of eleven

- Every switch a user can change, measured on one GPU with a real model: generation and the prompt one setting at a
  time (smallest to largest), each setting's best value checked against the defaults' logits, the best combination
  built greedily (only each setting's best value, kept when faster), drafting (MTP 2-7, DFlash2, texts against plain),
  the XMX tile per GEMM kind, where the time goes by kernel, and the MoE settings with `ARC_XCPU`. An untimed warm-up
  the first time a model meets a set of switches, so no timed run includes a kernel build; progress on the console.
- Removed: gemm_sweep, b60_sweep (2), decode_try (2), xw_sweep, xw_pergemm (2), xehpc_sweep, igpu_sweep, igpu_grf_sweep
  (`f33_sweep` stays for the chunked DeltaNet's KL checks). Their results stay in the docs and issues.

## 2026-11-20: Alchemist / Arc 140T: the prompt GEMM's loads one step ahead, opt-in (`ARC_GEMM_XW_DB=1`)

- **Why:** the user asked for the A770's slow GEMMs to reach the ratio its other kernels already have against a B60
  (1.15-1.25x, against 1.5-2x for the H8 GEMMs). In gemm_xw8's packed loop every read is waited on right where it is
  used: a token group's activations just before its DPAS, a chunk's weights at the chunk's start. With 256 registers
  an Xe-core runs half its threads, so little else covers that wait.
- **What:** `ARC_GEMM_XW_DB=1` (gemm_xw8.cl -DXW_DB, where both operands are packed): the next token group's
  activations are read before this group's DPAS, and at a chunk's last group the next chunk's weights too. The same
  DPAS in the same order, so the same results bit for bit (checked at start; the laptop's 27B, the preset emulated:
  byte-identical logits). ocloc dg2 at 32 x 16 WGX 4 / 16 x 32 WGX 2: no spills, no reads left between the loop's DPAS.
- **Opt-in** until an A770 / 140T measures it; it is the first of three steps toward that ratio (next: the down GEMM's
  split-K pieces in one launch, then the 128-register form if the registers allow).

## 2026-11-20: Alchemist: a tile per prompt GEMM by default; Flash Next keeps 4,096-token chunks on split models

- **A tile per GEMM shape on Alchemist (A770 / A750 / A580):** `bench/xw_pergemm` on two A770s 16 GB (issue #45:
  skipzx-lab at 150 W, hansong80 at 225 W; the 27B IQ3_S at 4K and 16K) gave the same split: 32 x 16 with 4 column
  sub-groups a work-group for the big GEMMs (down, up, gate, qkv, q, z: 1-9 % faster each), 16 x 32 for those with
  N <= n_embd and K <= 8,192 (ssm_out, o, k, v: 7-27 % faster). The GEMMs' time with each at its best: -3.5 to -6.6 %
  against the one default tile. The same sums in each column: checked at start (with a long-K shape for the big
  tile), and the laptop's 27B gives byte-identical logits with the preset emulated. `ARC_GEMM_XW_SHAPE=0` keeps one
  tile; setting `ARC_GEMM_XW_TM` / `_NC` / `_WGX` does too. The Arc 140T keeps its preset's tile until measured.
- **Flash Next on a split model keeps 4,096-token prompt chunks on any card:** on two A770s 16 GB (hansong80, #45,
  .20.03) 2,048 against 4,096 gave the prompt 466 against 527 tok/s and generation 12.2-12.8 against 13.1-13.4. A chunk's
  sums go across the split (192 at 2,048, 96 at 4,096), so the smaller chunk pays them twice; one 16 GB card keeps
  2,048 (14-19 % faster generation there).

## 2026-11-20: bench/xw_pergemm.ps1 / .sh: which XMX tile each of a model's prompt GEMMs wants

- For every tile setting, the real model's prompt (`runz --bench-pp N --n 0 --prof`) and each GEMM kind's GPU time
  (gemm_down / up / gate / qkv / o / ssm_out / z ...): two rounds in alternating order after an untimed warm-up, the
  fastest setting per GEMM, and the total if each GEMM took its own. 8-lane XMX (Alchemist, the 140T): 32 x 16, 32 x 32,
  16 x 32, WGX 4, and with all settings 16 x 64, 64 x 16, WGX 1, 32 x 32 WGX 4; 16-lane (B-series): 32 x 64, 32 x 32,
  32 x 96, 32 x 128, and 16 x 128, 64 x 64, WGX 1 / 2. Every setting gives the same results (checked at start). It decides
  whether the engine should pick the tile per GEMM (`ARC_GEMM_XW_NC_E` does it for the n_embd-wide ones by hand).

## 2026-11-20: Alchemist / Arc 140T: wider tiles for the n_embd-wide H8 GEMMs, opt-in (`ARC_GEMM_XW_NC_E=2`)

- **Why:** skipzx-lab's A770 profile (issue #45) has every H8 GEMM 1.5-2x a B60's although the A770 measures more int8
  XMX, and the three whose N is n_embd (down, o, ssm_out: K up to 17,408 against N 5,120) the most, ~2x. At gemm_xw8's
  32 x 16 tile each activation chunk serves 16 columns, so a long K's activations are read again for every 16 (the
  B-series' 32 x 64 serves 64). ocloc for dg2 (the A770): gemm_xw8 32 x 16 has 303 sends for 68 DPAS and 80 syncs
  between its first and last DPAS, against gemm_xw 32 x 64 on bmg's 93 sends for 128 DPAS and 30 syncs.
- **What:** `ARC_GEMM_XW_NC_E=2` runs those n_embd-wide GEMMs at 32 x 32 (two column groups a sub-group) and leaves the
  others at 32 x 16. Each column's sum is the same: checked at start with an n_embd-wide shape, and the laptop (the
  preset emulated, a 1,024-token prompt of the 27B) gives byte-identical logits with and without it.
- **Opt-in** until an A770 / 140T measures it; skipzx-lab's sweep set one tile for every GEMM (32 x 32 for all:
  1,004 tok/s against 1,217), which would hide a win on these three alone.

## 2026-11-20: the CPU computes MoE experts missing from the GPU cache by default on B-series cards (`ARC_XCPU`)

- **Why:** a Pro B60 on PCIe 3.0 x8 (skipzx-lab, issue #57, a62be02), Flash Next IQ3_S, a ~4,200-token prompt then 256
  tokens, three pairs: 22.66 / 22.59 / 22.25 tok/s off against 26.33 / 26.34 / 26.11 on (+17 %), ~13,340 experts on
  the CPU at 0.10-0.11 ms each. The user: make it the default.
- **Where:** discrete Battlemage cards (the B-series: B580, Pro B50 / B60 / B65 / B70). Not on Alchemist (an A770 on
  PCIe 3.0 x16 measured no gain, 20.0 / 20.6 -> 18.9 / 20.1) and not on iGPUs (they read RAM directly). `ARC_XCPU=0`
  turns it off, `=1` on, anywhere. Split models only with the test switch, as before.
- **What changes in the answers:** the CPU's sums are not the GPU's bit for bit (KL-checked: over 708 positions the
  top token 701 -> 695, 98.2 %; skipzx-lab's generation 241 -> 237 / 256), and which experts the CPU takes follows the
  background fills' timing, so a temp-0 answer can differ slightly between runs until the fills are committed at fixed
  points (queued). `ARC_XCPU=0` gives the reproducible path.
- **A timing trap:** `--bench-pp N --n 128` in the default mode showed it slower on the B60 (early stalls of
  190-340 ms a step); a real prompt then generation is the measure. With the GGUF on a USB disk the gain fell to ~10 %.

## 2026-11-20: Flash Next's 4,096-token prompt chunks from 20 GiB on discrete cards (16 GB cards keep 2,048)

- **Why:** on a 16 GB A770 (skipzx-lab, issue #57, Linux, held to 150 W) the bigger chunk gained the prompt only
  3-6 % (16K 517 -> 535 tok/s) and cost generation after it 12-16 % (15.1 -> 13.3 tok/s after 16K, 15.2 -> 12.8
  after 32K), because its buffers take room from the expert cache (16 -> 14 % of the experts). On a 24 GB B60 it gained
  the prompt 10 % for 2 % of generation. The user: "keep the faster one".
- **What:** Flash Next takes 4,096 on discrete cards with 20 GiB or more (the B60, B65, B70) and 2,048 below (the
  A770 / A750 16 GB, Pro B50 16 GB, B580 12 GB). iGPUs keep the 15 GiB line. `--chunk` still overrides it.
- **Accuracy:** the same at every chunk size (skipzx-lab's KL against f32, 256 positions: the B60 0.00740 at 4,096,
  0.00897 at 2,048; the A770 0.00774 at both).

## 2026-11-20: the FFN's gate and up in one launch by default on discrete cards (`ARC_GEMM_XW_GU`)

- A Pro B60 under Linux (skipzx-lab, issue #57, main a62be02): the 27B's 4K / 16K prompt 1,805.8 / 1,764.9 ->
  1,865.9 / 1,831.6 tok/s (+3.3 / +3.8 %), the logits of a 4K code prompt byte for byte the default's. So it is the
  default on discrete cards with 16-lane XMX (the B-series); `ARC_GEMM_XW_GU=0` turns it off. iGPUs with 16-lane XMX
  (the 140V / B390) keep it opt-in until measured; the laptop's logits are byte for byte as before.
- Also measured by skipzx-lab, nothing changed for it: `ARC_ATTN_X8=1` on an A770 (150 W) is slower than `attn_q8`
  (27B 4K 1,217 -> 1,160 tok/s, 16K 1,135 -> 923; its best work-group, 1 x 1, still 1.6x attn_q8's time): it stays
  opt-in. The A770's gemm_xw8 tile, group and prefetch sweep: nothing beats the defaults (every set bit-identical).

## 2026-11-20: precise mode's CPU experts in one pass by default; the old-driver warning in the web UI and launchers; ARC_BATCH per GPU

- **ARC_XCPU in `--precise`: one pass on the CPU by default** (the user's choice). With `ARC_XCPU=1` the CPU's
  experts run one int8 pass, while the GPU's experts and everything else keep both. Measured (Flash Next, token by
  token against f32, 548 chat + 160 code positions): KL 0.00126 / 0.00251, top token 700 / 708 (98.9 %), 0.187 ms an
  expert. `ARC_XCPU_Q2=1` selects both passes on the CPU (A2's form: 0.00076 / 0.00222, 702 / 708, 0.302 ms). This
  replaces the `ARC_XCPU_Q2=0` of the entry below, whose default was two passes. With ARC_XCPU unset nothing changes.
- **Run to run:** which experts the CPU takes depends on when the background fills finish (fill_commit only polls
  them), so ARC_XCPU's logits differ between runs of the same prompt. With one pass that shows: 6 runs of the code
  tokens KL 0.00168-0.00519 (top token 157-158 / 160), 3 of the chat 0.00084-0.00152 (540-543 / 548); the worst
  pair 697 / 708 = 98.4 %, all far under precise's 0.015. Two passes stay at 0.00222-0.00224 (158 / 160).
- **The old-driver warning, shown (W2):** serverz's `/props` carries `driver_warning` when the Intel graphics driver
  is older than 32.0.101.8976. The web UI shows it in a bar above the chat (with a link to Intel's driver page; hidden
  until the page is reloaded), the Windows launcher in its status line and details, `zero-server status` in yellow,
  and `zero-server doctor` as a FAIL line (from the driver's own version, the server running or not). Windows only,
  as the check itself; Linux reports its driver differently.
- **ARC_BATCH per GPU (tiered placement P0):** the recorded-list default (64 on Alchemist under Windows) is now each
  GPU's own, from its own kind, instead of GPU 0's for the whole process, so a later split over unlike cards gives each
  card its setting. Today's splits are of one kind: nothing changes for them or for one GPU.

## 2026-11-20: `ARC_XCPU_Q2=0`, precise mode with a faster CPU (opt-in); ARC_XCPU's top token over 708 positions

- **`ARC_XCPU_Q2=0`** (opt-in): in Flash Next's precise mode the CPU's experts run one int8 pass, as before A2, while
  the GPU's experts and everything else keep both. Unset, the CPU runs both passes (A2), since precise is chosen for
  exactness. Flash Next IQ3_S, token by token against f32, the Japanese chat (548 positions) and 160 code tokens, a
  4 GiB cache with the RAM tier:

  | Precise mode | KL (chat / code) | top token | CPU, ms an expert |
  |---|---|---|---|
  | CPU off | 0.00075 / 0.00221 | 702 / 708 | - |
  | CPU on, two passes (default) | 0.00076 / 0.00222 | 702 / 708 | 0.302 |
  | CPU on, one pass (`ARC_XCPU_Q2=0`; before A2) | 0.00126 / 0.00251 | 700 / 708 | 0.187 |

  All within precise's 0.015 and above 98 %. The laptop's iGPU gains no speed from ARC_XCPU (it reads RAM directly);
  a discrete card on a slow link does (skipzx-lab's B60: Flash Next 22.5 -> 27.3-28.2), so the speed is asked there.
- **ARC_XCPU in the default mode** over the same 708 positions: top token 701 off, 695 on (98.2 %, the rule is 98 %);
  KL 0.00220 -> 0.00263 (chat), 0.00827 -> 0.00379 (code). Every flip is a near tie of the f32 reference (top-two
  margins 0.0004-0.21 in log-prob). The 156 / 160 of A2's row was those 160 code tokens alone.
- With `ARC_XCPU_Q2` unset: logits byte for byte as after A2 (precise, ARC_XCPU=0).

## 2026-11-20: the FFN's gate and up GEMMs in one launch on 16-lane XMX, opt-in (`ARC_GEMM_XW_GU=1`)

- **Why:** on a Pro B70 the 27B's 4K prompt takes 1,378 ms (2,962-2,972 tok/s); 3,000 needs 1,365. `gemm_up` takes
  219 ms against `gemm_gate`'s 170 on the same shape: the gate GEMM writes its f32 result to `big_` and the up GEMM's
  fused SwiGLU reads it back, 285 MB a layer each way (18 GB per 4K prompt).
- **What:** `gemm_xw.cl -DXW_GU`: one launch computes both. A sub-group's first half of column groups reads the gate
  weights, the second half the up weights of the same 32 columns, and its epilogue writes silu(gate) * up. The gate
  value is computed as the gate GEMM stores it and the product as the fused SwiGLU computes it, so the results are the
  same bit for bit. ocloc for a B70 (32 x 64, 2D loads and stores, prefetch, WGX 4, 256 registers): the same 128 DPAS
  a chunk, first-to-last DPAS 285 -> 237 instructions, the kernel 4,683 -> 2,253 instructions, sends 93 -> 70, no
  spills.
- **Opt-in** on 16-lane XMX (discrete Battlemage, the 140V / B390); checked at start against the two launches on
  random data, and off with a line if it differs. Without the switch nothing changes.
- **Checked here** (the laptop, `gemm_xw`'s DPAS emulated): the 27B's 480-token prompt + 16 tokens, logits byte for
  byte with and without it. The speed is for a B70 owner to measure (with and without resident weights).
- **Not built:** moving part of `h8_act`'s rotation into this epilogue. In H8's stage order only the two lowest index
  bits could move there and stay bit-identical; that is a few ms at most, less than it would add to the epilogue.

## 2026-11-20: the expert cache's background fills: four races and a leak (self audit A3-A6)

These are the copies `ARC_XCPU=1` queues so that a token's CPU-computed experts enter the GPU cache for the next tokens.
- **A3:** `shrink()` (a launch that ran out of memory) could free a 1 GiB piece of the cache while a fill was still
  copying into it, and the fill's commit then set a slot past the shortened arrays. Fills now finish and commit first.
- **A4:** a fill could take the slot a deep prefetch was writing on its thread. No fills are queued while one runs.
  Fills are only a head start for later tokens; the CPU computes the miss either way.
- **A5:** the destructor left the fills' copy list and its 512 events behind at every model switch. It now waits for
  the fills and destroys both.
- **A6:** an error while queuing a fill left its slots outside every list and its experts marked pending. They are
  now given back before the error goes on.
- Checked: `ARC_XCPU=0`, Flash Next's logits byte for byte as before (64 tokens). `ARC_XCPU=1`: 128 tokens,
  13,521 experts on the CPU and as many fills, none dropped. The out-of-memory and model-switch paths were not
  run here.

## 2026-11-20: packing and expert-file builds: a moved source, two processes, the stop guard (self audit A7-A10)

- **A moved or copied source (A7):** packing a model again from its new place (copied from D: to C:, say) was
  refused as "made from another file". Now the package's tensor table (every tensor's name, type and shape) is
  compared with the file's: the same tensors are taken as the same model, with a line saying so; a model of the same
  name made otherwise is still refused.
- **Two processes building the same file (A9):** a build now holds a lock beside its `.tmp` (`plat::BuildLock`:
  Windows opens it unshared and deleted on close, Linux uses `flock`). A second process that finds the lock held stops
  with "another inferzero is writing ... now" instead of removing the `.tmp` the first is still writing. The line
  about an old `.tmp` now says "removed" only when it was.
- **The stop guard (A8):** guards are counted, so two builds on two threads restore the Ctrl+C / SIGTERM handlers in
  the right order; a signal the process ignores (SIG_IGN) stays ignored and no longer stops a build.
- **Docker (A10):** `compose.yaml` and `kubernetes.yaml` no longer say SIGTERM handling is "not yet".
- Inference is untouched: no kernel or engine path changed.

## 2026-11-20: ARC_XCPU in precise mode runs both passes, and works on packaged models (self audit A2)

- **Precise mode (`--precise`):** the GPU's experts run two int8 passes there (the activations and their rounding
  residual), but the CPU's experts under `ARC_XCPU=1` ran one. Now the CPU does both, as the GPU's fused kernels do:
  gate / up against the activations and their residual, the two sums added; h quantized with its residual
  (`quant_act32_q2`, as `moe_quant`); the down rows against both. Flash Next, 160 code tokens token by token against
  f32: KL 0.00221 with the CPU off, 0.00222 with it computing ~10,700 experts (top token 158 / 160 both). Two passes
  cost the CPU 0.301 ms an expert against 0.176 in the default mode.
- **Packaged models:** a launcher package keeps the experts in its `.experts` file only, so its GGUF's expert tensors
  have no data, and `ARC_XCPU=1` refused every such layer: it did nothing on a packaged model. The CPU only ever
  computes experts whose record is in the RAM tier (where records stay once read), so it now reads those alone and no
  longer needs the GGUF's copy.
- With `ARC_XCPU=0` (the default) nothing changes: Flash Next's logits byte for byte as before, default and precise.

## 2026-11-20: the prompt attention as flash attention on 8-lane XMX, opt-in (A770 / A750, Arc 140T)

- **Why:** the A770's 27B 4K prompt is ~1,223 tok/s against a B60's ~1,726, and attention is 13.6 % of its 16K
  prompt: Alchemist and the 140T ran `attn_q8` on DP4a, since `attn_fx` (Battlemage) needs 16-lane DPAS.
- **What:** `zero/kernels/attn_x8.cl`, the 8-lane form of `attn_fx`: 8 query tokens a sub-group, 8 x 8 x 32 int8
  DPAS, K / V straight from the int8 cache, `attn_fx`'s online softmax and P quantization per 32 keys. The scores are
  computed transposed (keys as the DPAS rows, one token a lane), so K is read in full cache lines and the softmax needs
  no reductions; one 8 x 8 transpose a span turns P into the A operand. Heads of 256 dims run on two sub-groups that
  add their integer partial scores through SLM. ocloc (`dg2`, `arl-h`), the 27B's shape: no spills, 1,459 / 1,446
  instructions and 72 memory messages per 32-key span.
- **Opt-in:** `ARC_ATTN_X8=1` (on GPUs with 8-lane XMX), checked at start against `attn_q8` like `attn_fx`;
  `ARC_ATTN_X8=2` emulates the DPAS with DP4a on any GPU with 8-lane sub-groups (tests). Owners of an A770 / 140T:
  to measure (`--bench-pp 4096` / `16384` / `65536` with `--prof`).
- **Accuracy** (the laptop, DPAS emulated), against `attn_q8`: the 27B over 4 x 512 tokens of USAGE.md KL 0.00058
  (max 0.0128, same top token 98.53 %), one 4,096-token chunk 0.00056 (max 0.0146, 98.83 %); the 4B 0.00078
  (max 0.0176, 98.33 %). Against f32 (the 27B, 480 positions): 0.00056 (`attn_q8`) -> 0.00060, max
  0.0133 -> 0.0101, top token 469 -> 476 of 480.
- Without the switch every GPU runs as before: the laptop's 4B and 27B logits bit for bit (480 tokens of USAGE.md + 16).

## 2026-11-20: 4,096-token chunks for MoE models fully on the card too; the driver warning's threshold; a warm-up fix

- **The 35B-A3B (and other MoE models) with every expert on the GPU: chunk 2,048 -> 4,096** (the user: "Yes, 4096
  for all", "gemma4 2048 then"). Gemma 4 keeps 2,048. Speed and KL to be measured on a B70 / B60 (asked; the laptop
  streams the 35B's experts, so it can't run this case): not in a release until that's back.
- **The old-driver warning** (self audit F1): from 32.0.101.8976, the oldest driver users measured right
  (manoreken2's B60s; a1ex4's 140V ran 8991), instead of 9033, which warned those. Checked with `ARC_TEST_DRIVER`:
  8976 and 8991 silent, 31.0.101.5328 warns.
- **The warm-up's 2-8-row passes** read past their token list when a context was under 20 tokens (self audit F4): the
  list now holds at least 8.

## 2026-11-20: `bench/xehpc_sweep.sh`: every kernel and tile setting swept on one GPU

- For the Data Center GPU Max 1550 (issue #48), whose settings were never tuned (the defaults are the B-series ones), and any discrete card: each generation and prompt setting changed alone from the defaults, from the smallest value to the largest (prompt chunks at most 4,096), a winner at +2 %; each winner's logits then hashed against the defaults (`same`, or `differs`: a KL check before it could be a default), and a phase's winners run together at tg128 / pp512 / pp4096 / pp16384. `--moe-model` adds the MoE settings, `--quick` takes fewer values. Linux, bash only. Written by the Researcher session; not run on a GPU yet.

## 2026-11-20: a warning when Windows has put an old Intel graphics driver in place

- **Why:** a Windows Update put a February 2024 Intel driver (31.0.101.5328) in place of 32.0.101.9150 on the laptop.
  On it, Flash Next's logits were wrong (KL against f32 1.5-5, against 0.0022 on 32.0.101.9175) and a run ended in a
  device lost, with nothing to say why.
- **What:** on Windows, runz and serverz compare the driver with 32.0.101.9033 (the oldest docs/install.md lists) and,
  when it is older, log once: the version, that answers may be wrong, and to install Intel's current driver. They still
  start. (`ARC_TEST_DRIVER=x.y.z.w` checks a given version, a test of the warning.) The launcher and web UI showing it,
  and a start-up self-check of the GPU's results, are next (QUEUE).

## 2026-11-20: prompt chunks of 4,096 tokens for Flash Next and the streamed MoE models

- **The user's choice:** 4,096 is the prompt chunk, and the largest. Flash Next goes 2,048 -> 4,096 on GPUs with 15 GiB
  or more (smaller cards keep 2,048); the other streamed MoE models (the 35B-A3B when its experts are streamed) 8,192 ->
  4,096. Gemma 4 keeps 2,048, which was faster for it.
- **Flash Next:** a Pro B70 (joshdev-de, issue #27): 4K / 16K prompt 1,150 / 1,110 -> 1,257 / 1,216 tok/s (+9 %), its
  expert cache 23.35 -> 22.53 GiB; the laptop (driver 32.0.101.9150): 4K 156 -> 164 tok/s, generation after it 5.3 ->
  5.8. KL against f32 on 3,584 tokens the same at both sizes (default 0.00894, precise 0.00583; driver 32.0.101.9175).
- **The streamed 35B-A3B:** from the laptop's earlier sweep, a 32K prompt 604 -> 533 tok/s (-12 %), its expert cache
  77 -> 81 %, generation at 128K 12.6 -> 12.8 tok/s. `--chunk 8192` gives the old speed back.

## 2026-11-20: a fix for Lunar Lake / Panther Lake: wrong answers after a short follow-up with a resident model

- **The bug** (self audit, A1; releases 2026.11.19.05-.07): on iGPUs with 16-lane XMX (Arc 140V, Arc B390), a
  model whose decoded weights are kept on the GPU (a 4B, maybe a 9B) has them converted to H8's rotated form the
  first time a prompt of 256+ tokens runs. A later chunk under the H8 threshold (a short follow-up message, a long
  prompt's tail) then took the plain path over those rotated weights: wrong logits. Models whose weights are decoded
  just in time (the 27B there) were not affected.
- **The fix:** weights kept on the GPU always take H8 there, as before release .05; the threshold (`ARC_H8_MIN`)
  applies only to weights decoded just in time. Not run on a 140V or B390 here (the laptop has no XMX); the owners
  are asked to check a 4B with a long prompt and then a short one.

## 2026-11-20: the first requests after an update don't wait for the 2-8-row kernels (one longer warm-up instead)

- **What** (joshdev-de, PR #65): the warm-up also runs passes of 2 to 8 rows, which short prompts, short follow-up
  turns and `serverz --slots` use. A Pro B70's `serverz --slots 4`, 27B, after an update: the first request 35.2 ->
  4.0 s, the first round of 4 at once 19.3 -> 9.0 s.
- **The cost:** the one-time warm-up after each update takes longer, 10.9 -> 85.6 s on the B70 with the driver's
  compiler cache off (~75 s more, once); later starts skip it as before. Bit-identical: the warm-up's output is
  thrown away.

## 2026-11-20: the first answers after an update no longer wait for kernels: the warm-up covers MTP heads and switches

- **An MTP head in a file of its own is warmed over every row count** (joshdev-de, PR #62). The warm-up rejected every
  draft, so the MTP block only ever ran on one row there; Flash Next's Q8_0 head (`mtp-*.gguf`, its own matrices and
  experts) then compiled 11 kernels in the first real answer. A Pro B70, first MTP answer after an update: Flash Next
  5.0 -> 56.1 tok/s, the 35B-A3B 11.4 -> 131.3. Installs warm up once more after this release.
- **A run with other `ARC_*` switches, or another MTP head, warms its own kernels** (joshdev-de, PR #63). The warm-up's
  marker ignored them, so such a run skipped the warm-up and compiled in generation: a B70's 27B with
  `ARC_GEMV_WAVE_MT=0`, first run 18.1 -> 64.8 tok/s. Each combination warms up once.
- Both bit-identical: the warm-up's output is thrown away (`--dump-logits` the same before and after on the B70's
  27B, 35B-A3B and Flash Next, and here on the 4B with and without MTP: both builds on the same driver, 31.0.101.5328,
  which Windows Update had installed; the identity holds, the speeds of that morning don't).

## 2026-11-20: Alchemist: the decoded weights kept for short prompts, beside the 140T preset (`ARC_140T_RES=1`, opt-in)

- **Why:** on Alchemist the 140T preset keeps no decoded copy of the weights, so a prompt chunk under 512 tokens
  decodes every weight first (~45-55 ms on an A770's 4B at any length).
- **What** (skipzx-lab, PR #61): with `ARC_140T_RES=1`, where the decoded copy fits it stays for chunks under 768
  tokens (the DP4a GEMM), and the preset still takes longer chunks from the raw weights. An A770 on Linux, best of 3:
  the 4B's 128-token prompt 1,161 -> 2,030 tok/s, 512 2,744 -> 3,170, 4K unchanged (5,025 / 5,006); the 9B (with
  `--budget 14`) 684 -> 1,487 at 128. Generation the same. The cost is the copy: the 4B's GPU buffers 3.9 -> 7.7 GiB,
  the 9B's 6.6 -> 13.9, less room for the context, so it stays opt-in.
- **Accuracy:** each chunk takes one of the two existing paths, bit for bit (17- and 640-token prompts equal to the
  resident path, 3,584 tokens equal to the default). KL against f32 on 512 tokens goes down, as the short chunks read
  the plain int8 weights instead of the rotated ones: the 4B 0.00197 -> 0.00102 (top token 490 -> 501 / 512).

## 2026-11-20: `serverz --gpus N` says what a split model leaves out

- With `--gpus 2`, `--mmproj` was accepted and then dropped without a word (`/props`: vision false; catduckgnaf,
  issue #59). The server now logs that images are not read with a split model yet, and that drafting is off.

## 2026-11-19: split models: the prompt's sums direct by themselves where the direct links are a little slower

- **Measured:** hibagus's 4 stacks of Max 1550s (issue #48), the 27B, 5 rounds of the three settings: host memory
  (the timed choice there) 512-token prompt 2,204-2,239 tok/s, generation 47.0-47.2; `ARC_TP_P2P=1` 2,666-2,688 and
  44.8-45.1; `ARC_TP_P2P=prompt` 2,676-2,697 and 46.8-47.3: both best ones.
- **What:** when the links' timing finds the direct path at 0.5-0.8 of host memory's speed (hibagus: 31 against 46
  GB/s), a split model now takes `=prompt`'s split by itself: the prompt's sums straight from GPU to GPU, a generated
  token's through host memory. At 0.8 or more both go direct, below 0.5 both through host memory, as before; the
  load line says which. The logits are the same on every path.
- **Confirmed on release .07** (hibagus, 5 rounds, the default against `=prompt`): the default picked the split by
  itself every time (31.0-31.1 against 45.6-46.6 GB/s), prompt 2,670-2,686 tok/s and generation 45.9-47.2, the same
  as `=prompt`; the two stacks of one card still go direct for both.

## 2026-11-19: Lunar Lake / Panther Lake: H8 from 256 tokens, not 1,024 (a correction to release .05)

- **Why:** a1ex4's crossover runs on the Arc 140V (issue #60, release .05, `--bench-pp` best of 3) put it at ~140
  tokens: H8 / without it 128 tokens 86 / 89-93 tok/s, 256 156 / 96, 512 252 / 176, 1K 339 / 185, 4K 384 / 170. The
  1,024 of release .05 came from one cold prompt each and made prompts of 256-1,023 tokens 1.4-1.8x slower.
- **What:** `ARC_H8_MIN`'s default on iGPUs with 16-lane XMX is now 256: only chunks under 256 tokens skip H8.
- **Accuracy:** on a 480-token prompt, KL against f32 0.00117 (max 0.029, the top token 470 / 480) with H8 against
  0.00057 (473 / 480) without, both far inside 0.015; generation never runs H8. `ARC_H8_MIN=100000` keeps every
  prompt off H8 for those who want the second.

## 2026-11-19: `ARC_TP_P2P=prompt`: a split model's prompt sums direct, its generated tokens' through host memory

- **Why:** on hibagus's 4 stacks of Data Center GPU Max 1550s (Xe Link between the cards, issue #48), the links'
  timing (31 GB/s direct across cards against 46 through host memory) rightly puts generated tokens' sums in host
  memory (a step 19.7-20.0 ms against 20.4-20.9 direct), but the prompt was 19 % faster direct (512 tokens 2,137-2,161
  -> 2,548-2,573 tok/s): one switch chose for both.
- **What:** `ARC_TP_P2P=prompt` sends the prompt chunks' sums (the reduce-scatter, 64 rows and more) straight from
  GPU to GPU and keeps the one-step sums of generated tokens in host memory. Opt-in: `=1` / `=0` and the timed
  choice are unchanged.
- **Checked:** the laptop (`ARC_TP_SIM`, the 4B over 2): `=prompt`, `=1` and `=0` give byte for byte the same logits.
  hibagus is asked to time it.

## 2026-11-19: Lunar Lake and Panther Lake laptops: short prompts without H8 (`ARC_H8_MIN`)

- **Why:** on a1ex4's Core Ultra 7 258V (Arc 140V, issue #60), the 27B on a 480-token prompt ran at 126 tok/s with
  H8 (the Hadamard-rotated int8 prompt GEMMs) and 147 without, and H8 cost accuracy there: KL against f32 0.00117
  (max 0.029, the top token 470 / 480, under 98 %) against 0.00057 (max 0.007, 473 / 480). A 128-token prompt spent
  70 % of its 1.5 s decoding weights for H8. At 4K, H8 stays well ahead (444 against 324 tok/s).
- **What:** on iGPUs with 16-lane XMX (Arc 140V, Arc B390), prompt chunks shorter than 1,024 tokens take the plain
  XMX GEMM; longer ones keep H8. `ARC_H8_MIN=N` moves the threshold (0: H8 at every length, as before); `ARC_H8=1`
  forces H8 everywhere. Discrete cards, the Arc 140T's preset and the 125H (no XMX) are unchanged.
- **Checked:** not on a 140V or B390 here: the numbers above are a1ex4's runs with H8 on and off, which this
  threshold picks between by chunk size. On the laptop's 125H nothing changes (no H8 there). The owners are asked
  for the crossover.

## 2026-11-19: `ARC_XCPU=1` for DeepSeek-V4.1 (opt-in)

- **What:** the CPU now also computes DeepSeek-V4.1's experts missing from the GPU's cache: kernels for MXFP4 (the
  experts of Unsloth's UD-Q8_K_XL), Q5_1 (some of its down projections) and Q8_0, and DeepSeek's clamped SwiGLU
  (`zero/xe2/cpu_experts.h`, `zero/xe2/engine_xcpu.h`). Each kernel gives the exact dequantized values' sums to
  ~1e-8 of the sum of its terms, in the GGUF layout and the expert file's (a test of random rows of 2,304 and 5,120
  values).
- **Checked on the laptop** with a DeepSeek-V4.1 of the real widths and Unsloth's types (random weights,
  `tools/make_tiny_deepseek41.py --real --unsloth --q5_1-down`), its experts streamed through a 0.75 GiB cache with
  the RAM tier on: the CPU computed 1,067 of the generated tokens' experts at 0.135 ms each, the copies in the
  background raised the cache's hits from 25 % to 84 %, and KL against the f32 reference over 160 tokens was 0.0572
  against 0.0578 without it (that model's own level: random routers with many near ties). No faster on the iGPU,
  which reads RAM directly (44.5 against 47.0 tok/s): it is for cards on a PCIe link. Off, the logits are byte for
  byte as before.
- **One model over several GPUs: with `ARC_XSPLIT_TP=1` too** (a test switch). It lets a split model run a token's
  cached experts before its missed ones arrive, which `ARC_XCPU` builds on; that was off for split models since logits
  changed from run to run with it (QUEUE F30). On the laptop (`ARC_TP_SIM`) F30 did not come back: Flash Next over 2
  with 3 GiB caches gave the same logits run to run and the same as with it off, with the expert table in host memory
  or its GPU copy; the DeepSeek model over 2 the same KL with it on (0.05696), and 0.05735 with `ARC_XCPU=1` as well.
  Real cards will say whether it holds there.

## 2026-11-19: `runz --bench-batch` checks batched decoding against each conversation alone on one GPU too

- **What:** `runz --bench-batch N` (several conversations' next tokens in one pass, as `serverz --slots` does) only
  timed it on one GPU; the split-model form also checked the logits. Now both do: 8 greedy steps of every slot in
  one pass against each slot alone, compared byte for byte; where they differ, the number of rows, the largest
  difference and how often the top token agrees. It also says so when a model has no batched decoding.
- **The laptop's iGPU, 4 slots, all bit-identical:** together against one slot alone, 4B 58.8 against 20.8 tok/s
  (2.83x), 27B 17.0 against 5.3 (3.19x), Flash Next IQ3_S (experts streamed) 15.1 against 8.2 (1.84x), Gemma 4 26B
  38.9 against 19.2 (2.02x). Split over 2 / 3 / 4 "GPUs" on the one iGPU (`ARC_TP_SIM=1`, correctness only): the 4B,
  the 27B and Flash Next bit-identical at every count (the 27B's 4 slots 3.18x / 3.15x / 3.04x; Gemma 4 cannot be
  split yet). No discrete card has run it yet: the testers are asked to.

## 2026-11-19: A770: a 35B-A3B token's missed experts read their table from the GPU (issue #56)

- **On an A770 the 35B-A3B wrote corrupted text when its experts were streamed** (skipzx-lab, issue #56, Linux): its
  logits changed from run to run from the 6th-12th token on, `ARC_XSPLIT=0` gave the right text, and a B60 with
  the same binary was right either way. A token whose experts are partly missing runs the cached ones first, while
  the others are copied in (`ARC_XSPLIT`); the missed experts' entries in the expert table, which lives in host
  memory, are written just after the cached experts' kernels read the same row. The missed experts' kernels now read
  a copy of that finished row in GPU memory (`ARC_XSPLIT_TAB`, `zero/engine_moe.h`): the same addresses, so the same
  results bit for bit, for one host wait more per split layer.
- **Default on Alchemist** (8-lane discrete cards); Battlemage and the iGPUs keep the row in host memory, as before.
  `ARC_XSPLIT_TAB=1` / `0` force it on / off; `2` only adds the wait, keeping the row in host memory (a test of which
  part matters).
- **Checked:** not on an A770 yet (skipzx-lab is asked to). On the laptop with it forced, Flash Next IQ3_S streamed
  (the split runs at 87 % hits): logits with `=1` and `=2` byte for byte equal to `=0`, the speed the same (6.9-7.0
  tok/s).

## 2026-11-19: pack and expert-file failsafes: a same-named package, Ctrl+C / SIGTERM, a full disk, sleep, stale .tmp

- **A package of the same name made from another file is no longer overwritten** (B2). Two folders' GGUFs of the
  same file name make the same package folder; `--pack` now reads its `package-source.txt` and refuses, naming both
  files, when it records another one. A folder that records none is packed as before, with a line saying so.
- **Ctrl+C and SIGTERM stop a pack or an expert-file build cleanly** (D1): between two batches, the half-written
  `.tmp` removed, then the signal does what it did before. `docker compose stop` and a pod's deletion send SIGTERM;
  their 60 s grace periods (2026-11-19) now let that cleanup run instead of leaving tens of GB for the next start.
- **A drive without room is refused before anything is written** (S3): the expert file and the package GGUF each
  check their drive's free space (their size + 256 MB), naming both numbers; before, a full disk stopped a build
  after tens of minutes.
- **A `.tmp` left by a build that was killed** (a power cut, SIGKILL) is removed first, with a line giving its size
  (S7).
- **Windows does not sleep while a pack or an expert-file build runs** (S11, `SetThreadExecutionState`).
- **Linux: the source pages a build has copied leave the process at once** (B6, `madvise(MADV_DONTNEED)` per batch
  and per package tensor), instead of staying mapped until the whole GGUF was read, which counted against a
  container's memory limit. Windows already trimmed them per layer.
- **Checked:** Gemma 4 26B Q4_0 packed three times: the expert file and the package GGUF byte for byte the same as
  before these changes (SHA-256); stale `.tmp` files removed and logged; another source under the same name refused,
  the package unchanged. Flash Next IQ3_S (83.6 GB) packed through it in 809 s. Linux: compiled, not run.

## 2026-11-19: Flash Next on a card in a slow slot: the experts missing from the GPU computed by the CPU (`ARC_XCPU=1`, opt-in)

- **Why:** a Pro B60 in a PCIe 3.0 x8 slot spent ~17 ms of each ~44 ms Flash Next token waiting for experts missing
  from its cache: each one is ~2.3 MB copied over the link, ~0.36 ms (skipzx-lab, pull request #55).
- **What:** with `ARC_XCPU=1`, a generated token's missing experts whose records are already in RAM are computed by
  the CPU while the GPU runs the cached ones. The CPU uses the GPU's own int8 inputs, and its results go straight into
  the rows the GPU sums, with no wait for them. The missing experts are still copied into the cache in the background
  for the next tokens; without that the hit rate fell from ~90 % to 38 %. An expert not in RAM yet goes to the GPU as
  before. The CPU code (`zero/xe2/cpu_experts.h`, `zero/xe2/engine_xcpu.h`) is AVX2, for IQ2_S, IQ3_XXS, IQ3_S,
  IQ4_XS, IQ4_NL and Q2_0; a CPU without AVX2, or another type, keeps the GPU path. `ARC_XCPU_THREADS=n` sets its threads (half the logical CPUs by
  default). Not for Gemma 4, DeepSeek-V4.1 or one model over several GPUs.
- **Speed:** that B60, Flash Next IQ3_S, 4K prompt then 256 tokens: 22.5 -> 27.3-28.2 tok/s (+22 %). An A770 in a
  PCIe 3.0 x16 slot gains nothing (Flash Next Coder IQ1_M: 20.0-20.6 -> 18.9-20.1), so it stays opt-in.
- **Accuracy:** not bit-identical when on (the CPU adds in another order): KL over 480 generated tokens 0.0134-0.0157,
  against 0.0141 off. Off, the logits are the same as before, byte for byte.

## 2026-11-19: `serverz --pack` refuses a package as its source

- **`serverz --pack <a package's .gguf>` could destroy the package** (self review, B1). A package's expert tensors
  have no data of their own, so the new GGUF lacked them, nothing was built, and the final rename replaced the
  package's own GGUF with it. On Linux that succeeds even while the file is mapped, and if the original GGUF was
  already removed, the model was lost. Only `zero-server.ps1` refused it; `serverz --pack` and the Linux launcher did
  not. It now stops at once with "is already a package: pack the original GGUF instead", before anything is written
  (checked on the laptop's Flash Next IQ4_XS package: exit 1, the package untouched).

## 2026-11-19: the expert file written from source rows read in order (`--pack`, a MoE model's first load)

- **Why:** packing bartowski's Flash Next IQ4_XS from an external USB disk ran at ~17-80 MB/s, and that disk hangs
  under scattered reads. The expert file's build (`zero/expert_store.h`) converted 32 experts at once, each thread
  reading its expert's rows from the mapped GGUF itself: ~96 small reads at a time across three tensors.
- **What:** a batch's 32 experts are one contiguous range of rows in each expert tensor (gate, up, down; Gemma 4's
  gate_up holds both halves). The build now copies those ranges out of the mapping in one pass, prefetching the next
  batch's ranges while this one converts (`plat::prefetch_mapped`: `PrefetchVirtualMemory`, `madvise` on Linux), then
  converts from RAM on the same threads as before. RAM: one batch of source rows (~85 MB for Flash Next IQ4_XS),
  freed after each layer; if it cannot be had, the build reads in place as before. `ARC_XBUILD_POOL=0`: the old reads.
- **Checked:** `serverz --pack` of Gemma 4 26B Q4_0 from the laptop's SSD, main's build against this one: the expert
  file (12.33 GB) and the package GGUF byte-identical (SHA-256). Time 120 -> 102 s (the second run from a warmer
  cache), then with both warm 102 -> 73 s. The same code writes a MoE model's expert file at its first load, so that
  is faster too.

## 2026-11-19: a 12 GB card's 27B nearly twice as fast: the weights in RAM computed by the CPU, by default

- **`ARC_HOST_CPU` is on by default on discrete cards.** When a model doesn't fit the card (the 27B on a 12 GB B580),
  the matrices kept in RAM are now computed by the CPU for each generated token, instead of being read by the GPU
  over PCIe. d-dorok's B580 (PCIe 3.0, an i9-9900KF), the 27B, two rounds each:
  - 0.49 GiB in RAM (`ARC_HOST_KEEP_GB=0.75`): 7.11 -> 13.70 tok/s (a step 140.6 -> 72.6 ms);
  - the default room (0.85 GiB in RAM): 4.54 -> 11.88;
  - a 4K context (1.02 GiB in RAM): 3.90 -> 10.27.

  The prompt is the same or faster (the 512 prompt 836 -> 1,022 tok/s). llama.cpp SYCL on the same card: 11.4. The
  cost is RAM: ~1.1 bytes per weight of those matrices (~0.55 GiB at 0.49 GiB in RAM). The KL is the GPU path's or
  lower (the laptop's 4B: 0.0014-0.0017 against 0.0034). iGPUs keep it off: their GPU reads RAM at full speed.
  `ARC_HOST_CPU=0` turns it off; `=1` forces it. Issue #37.
- **Measured, not changed:** `ARC_BATCH=64` on that B580 under Linux, the 4B 93.57 -> 92.87 and the 9B 56.80 -> 56.47
  tok/s. It stays an Alchemist-on-Windows default. The 9B's GEMVs on that card: with the reads alone
  (`ARC_GV_PROBE=1`) as slow as the real kernels, with the decode alone (`=2`) 20-30 % faster. Its access pattern is
  the limit (plan item G6).

## 2026-11-19: `runz --kl-ref` shows where the top token flips (`ARC_KL_FLIPS=1`)

- **With `ARC_KL_FLIPS=1`, every KL path of `runz --kl-ref` prints each position whose top token is not the
  reference's**: the reference's top two tokens, their margin in log-probability, and where our pick stands in the
  reference. It works on one GPU and with `--gpus`, for prompts and `--kl-decode`. It shows whether a missed top token
  is a near-tie or a real miss. Flash Next IQ3_S on the 3,584-token prose prompt, the laptop: default 239 / 256 (KL
  0.0091, max 0.063), precise 240 / 256 (0.0060, max 0.069); most flips have margins under 0.1 (e.g. 11.75 % against
  11.74 %), so precise mode's ~15-bit activations do not fix them.
- **Compact f32 references for `--kl-ref`** (`bench/prompts/refs/`, `.izkl`): each position's top 512 log-probs and the
  probability left over, ~1 MB a prompt instead of ~250 MB, so they ship in the repository and the packages. The top
  token and the margins are exact; the KL is ~2 % low (the rest taken as one bucket: Flash Next's prose 0.00894
  against the full reference's 0.00914). `ARC_KL_PACK=out.izkl runz <model> --kl-ref full.bin` makes one. The first:
  Flash Next GSQ-RCO IQ3_S on `qwen-prose-3584.ids.txt`, so a card owner can measure Flash Next's accuracy without
  the hours of the 125B's CPU reference.

## 2026-11-19: Alchemist's codebook GEMVs one row a sub-group

- **The A770 / A750 / A580 run the one-token GEMV of the codebook and IQ4 / Q6 types with one row per sub-group**
  (ARC_GVROWS=1's choice; 2 or 4 before, by type). An A770, the 27B's `--prof-gen` over 32 tokens: `gemm_down`
  395.7 -> 321.2 ms, gate 284.4 -> 273.0, up 280.5 -> 272.7, qkv 109.7 -> 104.4; 2 rows and 4 rows were slower
  (TheUniKian, issue #45). **Only where the kernels are not batched** (Alchemist on Linux): a second A770, two
  rounds, unbatched 12.62 -> 13.20 tok/s (+4.8 %), with `ARC_BATCH=64` (the Windows default) 21.40 -> 21.19 (-0.6 %).
  With batching the step is not bound by the kernels' time, and the profiled kernel times (which run unbatched)
  overstated it (hansong80). So on Windows the A770 keeps the rows it had. Bit-identical: a row's sums do not
  depend on its sub-group (the laptop's 8-lane GEMV gives the 27B's logits unchanged with ARC_GVROWS=1). iGPUs and the
  B-series are unchanged. `ARC_GVROWS=N` still sets it.
- **Measured on the same A770, not changed yet:** its XMX probe with the data in L1 reaches 37 % of the 231 TOPS peak
  with the 32 x 16 tile its preset uses (38 % with the DPAS alone), and 54 % with 32 x 32. Its FFN prompt GEMMs run at
  the same 37 %. But in the model 32 x 32 is slower: the 27B 4K 1,227 -> 965 tok/s (TheUniKian), and the full
  sweep (hansong80) has 32 x 16 near the best at 4K / 16K (WGX 4 +0.8 %), 16 x 16 at 128 registers +3.9 % at 512
  tokens only. The tile stays. At 16K, attention is 13.6 % of the A770's prompt: the 8-lane XMX attention (A6) is
  worth more there than any tile.

## 2026-11-19: `runz --bench-pp` warms the generation kernels up before timing them

- **`--bench-pp`'s `tg` line no longer includes the generation kernels' first-use cost.** Its warm-up ran a prompt
  pass only, so the first generated token compiled the one-token kernels inside the timed run. On a dense model that
  was tens of ms. On the 35B it was seconds: a B580's step 0 took 4,225 ms against a 12.6 ms median, tg128 21.5 tok/s
  for a ~79 tok/s steady state (tenken, issue #53). The bench now runs the server's warm-up first. The laptop's 35B:
  the first step 55.8 ms against a 51.9 median. Earlier MoE `tg` figures from `--bench-pp` read low by that one step;
  the `tg steps` median was right all along.

## 2026-11-19: serverz listens on 8090 by default, as everything else

- **`serverz` started without `--port` now listens on 8090** (8080 before), the port zero-server, both launchers,
  Docker, the README and USAGE already used. The launchers always pass the port, so they are unchanged; only a bare
  `bin\serverz.exe -m ...` moves. `--port 8080` keeps the old one. `scripts\remote-tunnel.ps1` falls back to 8090
  too; docs/server.md and zero-pi's help say 8090. (The launcher review of 2026-11-19, docs/launcher-research.md
  section 7.)
- zero-server's help lists the `g31` shortcut (the server knew it), and says `g26` is a file `pull` does not fetch.
- **`serverz --help` is a short page:** the ten options most starts use, three examples, and where the rest is.
  `serverz --help all` (or `--help-all`) prints the whole text as before (136 lines). Started without a model it
  prints the short page and exits 1, as before.
- **The Windows launcher (`zero-launcher.cmd`):**
  - The window fits the screen: its size is clamped to the work area, so at 150 % on a 1080p or 1200p laptop Start /
    Stop no longer sit below the edge.
  - Ticking "reachable from other computers" shows the addresses other computers use and that the server has no
    password.
  - Plain labels: the command-line names (`-NoMtp`, `-NoStream`, `-Precise`, `-Whisper`, `-Asr`, `--no-swap`) moved
    into tooltips, "Id (Pi /model)" became "Model id", "Resident W8 weights" became "Decoded weights kept".
  - MTP drafts and Accuracy moved to the advanced view; the test-only "f32 attention" left the window (a preset
    that sets it still does).
  - Precise mode's tooltip has Flash Next's current default KL (0.0091 on prose, not 0.0105).

## 2026-11-19: the wide-tile GEMM's 2D stores kept where a compiler fuses their add

- **`gemm_xw`'s output in 2D blocks is no longer dropped when it misses the start check by a rounding** (hibagus,
  PR #51). On a Data Center GPU Max 1550 (driver 25.18) the 2D form failed in the add and split-K modes only, by
  1-6 ulp. That driver's compiler fuses the 2D epilogue's `product + old` into one multiply-add, and gemm.cl's not.
  The check now tries the 2D form once more with the product rounded before the add (`-DXW_C_RND`), and keeps it if
  it then matches gemm.cl bit for bit. Max 1550, the 27B: 4K prompt 2,460 -> 2,641 tok/s on one stack, 512 tokens
  1,748 -> 1,939, `--gpus 2` 3,260 -> 3,375; logits identical. Where the 2D form passed already, nothing changes.
  The Arc B390 and Pro B65 logs failed the same two modes, so they may gain too. `ARC_GEMM_XW_DIAG=1` prints the
  first differences and their size; `ARC_GEMM_XW_C_RND=1` from the start, `=0` never.

## 2026-11-19: `ARC_HOST_CPU=1` (opt-in): weights kept in RAM computed by the CPU, not read over PCIe

- **For cards too small for the whole model** (the 27B on a 12 GB B580): the matrices that stay in host memory are
  read by the GPU over PCIe on every generated token. On d-dorok's B580 (PCIe 3.0, 0.49 GiB in RAM) that gives 7.1
  tok/s, where llama.cpp runs the layers it cannot fit on the CPU and makes 11.4 (issue #37). With `ARC_HOST_CPU=1`
  the CPU computes those matrices for one token instead (`zero/xe2/engine_hostcpu.h`). Each is converted once at
  load to int8 with a scale per 32 values, ~1.1 bytes a weight of extra RAM. Its GEMV runs on the CPU's cores
  against the GPU's int8 activations; only the vectors cross the bus. One kernel for every weight type: AVX2,
  SSSE3 (Ivy Bridge, older AMD, built with `build.sh` there), or plain C.
- **Accuracy:** KL-checked, not bit-identical. The laptop's 4B token by token against f32: 0.00138 / 0.00156 /
  0.00174 with 1 / 2 / 4 GiB on the CPU, against 0.00336 all on the GPU. The 27B with 0.5 GiB gives the same 48 greedy
  tokens.
- **Speed:** to be measured on a B580. On the laptop the GPU reads host memory at full speed, so the CPU can only be
  slower there (27B 5.03 -> 4.22 tok/s). Prompts and 2-8 row passes (MTP) still read those weights on the GPU.

## 2026-11-19: Flash Next's default mode at KL under 0.01

- **Flash Next keeps its hyper-connection mixer rows in BF16 by default**, as the file has them, instead of
  requantizing them to Q8_0 (the user's choice). The default mode's KL against f32 on the laptop: 160 code tokens
  0.0153 -> 0.0046 (same top token 95.0 -> 96.9 %), token by token 0.0283 -> 0.0083 (95.6 -> 98.8 %), 3,584 tokens of
  prose 0.0105 -> 0.0091. The maxima: 0.555 -> 0.126, 2.533 -> 0.189, 0.161 -> 0.063. The same top token is at
  98.8 % token by token, but 96.9 % on the 160 code tokens' prompt and 93.4 % on the prose, under the 98 % rule: the
  prose's misses are near-ties where llama.cpp disagrees too, and the code prompt's are what F3 / F5 are for. The cost: 0.73 GiB more read each token, taken from the expert cache (14.50 -> 13.76 GiB on
  the laptop); a 2,048 prompt 138 -> 116 tok/s, generation 7.7-8.0 -> 7.4 tok/s here. On a B70 the extra read is about
  3 % of a step. F2 (the mixer GEMMs in BF16 on XMX) is meant to take the prompt cost back on discrete cards. `ARC_FN_HCBF16=0` gives
  the Q8_0 rows of before. Precise mode is unchanged (KL 0.0017 / 0.0022 / 0.0060). `--ctx auto` works Flash Next's
  context out again (its key now says which rows); other models keep theirs.

## 2026-11-19: Linux builds for CPUs without AVX2

- **`build.sh` on a CPU without AVX2 / FMA builds for that CPU** (`-march=native`) and says so, instead of a build
  whose start-up check stopped it (ThatGuy-Bob, issue #50: an E5-2697 v2, Ivy Bridge, with an A770 / A750). The
  engine uses no AVX2 code of its own; its CPU side works on what the compiler gives it. `src/cpu_check.h` on Linux
  now checks the instructions the build was compiled for, not AVX2 always. `INFERZERO_MARCH=<arch>` picks another
  target. The release packages are unchanged (AVX2 / FMA / F16C).

## 2026-11-19: setup's 35B 2-bit GSQ install fixed

- **`setup.cmd` with `35b-gsq` failed after its download** ("Cannot find drive. A drive with the name '@{key=4b; ...'",
  gee456, issue #49). setup kept the models folder in `$arcModels`. PowerShell names ignore case, so the shared model
  list's `$ArcModels`, loaded just after, replaced it, and the step before the conversion made a folder named after
  the list. The folder now has a name of its own. The download is kept, so running setup again converts it.

## 2026-11-19: self audit of 2026-11-18's changes

- **`ARC_HOST_KEEP_GB` decides again whether a dense model is tight on a discrete card**, as before 2026-11-18. The
  automatic room had left it only the share of the weights kept in RAM, while "tight" was always decided by the old
  2.5 GiB line. So `=4` with a model under that line kept every weight on the card, and `=0.5` with a model just over
  it gave 1,024-token chunks and nothing in RAM. Without the switch nothing changes.
- **Resident weights' 4096-token prompt chunks now reach 12 GB cards.** The line was 12 GiB, and a "12 GB" card
  reports ~11.3 GiB (a B580: 11.33), so only 16 GB cards got them. It is now 11 GiB. iGPUs are unchanged.
- **Security:** a page of another local server could send a chat request naming a saved audio file. Whisper wrote
  that file as text before the check that refuses saved files to such pages, so the transcript reached the page in
  the answer. The check now runs before Whisper.
- **serverz:** a package made while the server runs (`zero-server pack`) is found by name. The model-list fingerprint
  did not look in the packages folder, so the request failed until a restart. The fingerprint also reads the model
  folders only as deep as the scan does: an extra folder such as a drive's root was walked whole, under the server's
  lock.
- **Flash Next `-DHC_UPQ`:** the last work-group of a superblock now reads the others' values behind device-scope
  release and acquire fences. Before, only work-group fences ordered them, and the other work-groups may run on other
  Xe-cores. The logits are the same bit for bit on the laptop. `ARC_HCF_ROWS=64` is refused (16 is used): its rows
  took the whole 64 KiB of local memory.
- **Image encoders:** the Qwen3-VL position table's rank is checked before it is read. Gemma 4 and Qwen3-VL patch
  weights are checked against 3 x patch x patch, and Qwen3-VL's second patch weight against the first. The f16 fallback
  of 2026-11-18 covers every model that sets `hp.qwen3`, so every dense Qwen3-VL model unless `ARC_Q3_Q2=0`. Its
  comment said only Flash Next `--precise` and a qwen3 model given such a projector.
- **Launcher:** the Get models dialog counts the memory of an Arc Pro B65 and of the laptops' Arc A-series (A370M to
  A770M). Their names did not match, so only the RAM was counted.
- **Small:** the IQ tables' GPU buffer is made once. A second kernel build (H8 off after its checks) leaked the first
  until the engine ended.
- **Text:** the help (runz, serverz, zero-pi) and docs/server.md say 3 MTP drafts for the 27B on B-series cards, as
  the code does since 2026-11-18. Comments on `ARC_H8_NORM` (on by default on discrete cards) and on
  `ARC_TP_EVSUM=1` corrected: under that switch a context overflow loads the split model again.
- Checked: the 4B and 27B logits bit-identical to the build before (`--dump-logits`, 64 tokens); Flash Next too;
  the Linux build.

## 2026-11-19: `--gpus`: the host staging buffers sized at load

- **A split model no longer fails at its first long prompt for want of pinned host memory** (hibagus, PR #48). Each
  GPU's staging buffer (the bounce buffer for host copies) started at 4 MB and grew on the first prompt with a
  larger input, on every shard's thread at the same moment. With 4 Data Center GPU Max 1550 stacks, that
  `zeMemAllocHost` failed now and then (7 of 56 runs of a 512-token 27B prompt) with ~1 TB of RAM free. The buffers
  are now sized at load to a prompt chunk's input, one GPU at a time (80 MB a GPU for the 27B at chunk 4096): 0 of
  40 failed. If that allocation fails, the load throws and the server loads the model on one GPU, as for the shards'
  other allocations. The results are the same (a bounce buffer). The RAM is the same as after a first long prompt
  before. `ARC_TP_STAGE_PRE=0`: as before.

## 2026-11-19: the `--ctx auto` memory line's "kept free", an Arc Pro B65 on the site, Docker tested

- **serverz's `memory (--ctx auto):` line showed 1.17 GiB kept free**, a figure of an older plan, while the plan keeps
  `ARC_DRIVER_RESERVE_GB` free (0.5 GiB by default). So its "left" read 0.67 GiB too low: -0.65 on a B580 that had
  ~0.02 left (tenken, issue #40). The context chosen was right; only the line (and `/metrics`' `keep_free`) was wrong.
- **Docker:** the image from `docker/` ran on Unraid with an Arc B580, without `ipc: host` or `privileged` (tenken,
  issue #40). The Kubernetes file is still untested.
- **Site:** an Arc Pro B65 (32 GB, Linux, release .08; teodortomas, issue #47): the 27B 4K prompt 1,735 tok/s,
  generation 21.6, 46.7 with MTP.

## 2026-11-18: Alchemist on Windows: kernels sent in recorded lists, the 27B's generation +58-70 %

- **`ARC_BATCH=64` by default on Alchemist (A770 / A750 / A580) on Windows.** Each kernel appended to the GPU's queue
  on its own cost 33-44 us there (`replay_probe`), against ~2.5 us in a recorded list. The 27B makes 837 launches a
  token, so ~19 ms of a 73 ms step was the GPU waiting. Two A770s, 27B generation, `--bench-pp 128 --n 64`:
  12.67 -> 21.58 tok/s (78.6 -> 46.1 ms a step, hansong80) and 14.33 -> 22.66 (69.4 -> 43.8 ms, TheUniKian, issue
  #45). The logits are the same bit for bit at 0, 16, 64 and 256. `ARC_BATCH=0` turns it off. Other GPUs are
  unchanged: a Pro B60 gained nothing from it (2026-11-12). The 35B-A3B on an A770: 24.87 -> 42.00 tok/s, the same
  logits (hansong80).
- **MTP on top of it:** a code answer of 95 tokens, 21.71 / 23.06 tok/s plain, 30.14 / 31.31 with MTP (50 % of
  the drafts accepted, 3.06 tokens a pass). In `--bench-pp`'s synthetic prompt few drafts are accepted, so MTP
  loses there (21.32 plain, 20.47 MTP): that is the prompt, not the batching (855 launches and 5 host waits a token,
  2.4 us a launch). `runz --bench-pp` now says so when it is given `--mtp`: its generation line never drafts.
- **Measured:** an Arc 140T's 27B 4K prompt 405 tok/s on release .07's defaults, `ARC_GEMM_XW_LOCK=1` 387 (stays
  off). An A770's `ARC_H8_NORM` off / on: 1,210 / 1,223 tok/s, it stays on.

## 2026-11-18: `--npu on` warns where it costs the GPU its H8 prompt path

- **A line at load when `--npu on` turns off H8 on a GPU where H8 is the default** (discrete cards, the B390, the 140T
  preset): the GPU alone is likely faster there. An Arc B390's 27B 4K prompt ran 613-628 tok/s alone and 193-330
  with the NPU at every share from 0.12 to 0.5 (gee456, issue #31; docs/xe2-tuning.md, section 15). A log line only:
  nothing else changes.

## 2026-11-18: IQ3T reads ahead, Flash Next precise mode's 32-row fix, DeepSeek Engram as llama.cpp

- **`ARC_IQ3S_TILE=1` v2 (opt-in):** the planar IQ3_S GEMV now issues every row's loads of a pair before decoding
  any of them. With `GEMV_PF` / `GEMV_DB` (on by default on discrete cards) it also loads the next pair while this
  one decodes, as the default path does. The first form waited for each row's loads in turn: a Pro B70's 27B was
  35.24 -> 32.7 tok/s (issue #27). Bit-identical on the laptop's 27B: generation, MTP, H8, the 140T preset, and
  with `ARC_GEMV_PF=1` / `ARC_GEMV_DB=1` forced on. The laptop's speed is unchanged (5.52 against 5.46-5.48 tok/s,
  no read-ahead there); the B70 decides.
- **Flash Next, `ARC_HCF_ROWS=32` / `64` now exact:** the f32 mixer GEMM's store loop wrote only the first 16 rows
  of a work-group. Now 32 rows are bit-identical to 16. The default stays 16: on the laptop, precise mode's 2,048
  prompt was 91.6 / 90.4 tok/s at 16 and 75.7 / 75.1 at 32.
- **DeepSeek-V4.1's Engram hash as llama.cpp's:** once a shift reaches the sequence start or a token outside the
  map, every longer shift uses the pad id too (before, only that one shift did). The f32 reference
  (`tools/ref_forward`) changed the same way. The tiny test model, 300 tokens, against the reference: KL 0.00208.
  Text without such a token inside it gets the same rows as before.
- **H8 turned off after its kernels were built** (an XMX check failed, or the 140T preset without XMX): the
  float copy of the activations is dropped and the kernels are built again without it. The laptop's 27B with
  `ARC_140T=1` gives logits bit-identical to the default.
- Measured, not kept: `ARC_XPRED` (the next layer's experts guessed from this layer's input and read from the SSD
  early). Flash Next on the laptop went 8.39 / 8.20 -> 7.43 / 7.40 tok/s: 27 % of the guesses were used, and the
  hit rate fell from 94.4 to 91.3 %. Removed.

## 2026-11-18: Flash Next's mixer quantizes its own output, one launch fewer per mixer

- **`hc_up1` quantizes the block input itself** (`-DHC_UPQ`), instead of a separate `quant_rows` launch after it: 97
  launches fewer a Flash Next token (48 layers x 2 mixers + the last). Each 256-value superblock is quantized by the
  last of its 16 work-groups to finish (an atomic count, nobody waits), with the same `quant_store16` on the same
  floats, padding rows included. Bit-identical: the default and precise modes' generation and MTP's 2-8 row verify
  passes, `--dump-logits` with and without. The laptop, one pair of runs: 6.08 -> 7.75 tok/s generation (this
  laptop's noise is large; a card's measurement decides). `ARC_HC_UPQ=0`: the separate launch.
- Test switches for measuring Flash Next's error part by part (the plan's F1), all off unless set:
  `ARC_FN_HCBF16=1` (the HC mixers' rows as the file has them), `ARC_FN_AB32=1` (DeltaNet's alpha / beta in f32),
  `ARC_HCF_ROWS=16 / 32 / 64` (the rows per work-group of the f32 mixer GEMM in precise mode; 32: its time 3.74 ->
  2.30 s of a 2,048-token prompt, the prompt 84.5 -> 88.2 tok/s, but not bit-identical, so 16 stays until its KL is
  measured).
- runz `--mtp`: the per-phase time line (`MTP time`).

## 2026-11-18: `runz --mtp` warms up the verification kernels first, and says where an MTP step's time goes

- **The first `runz --mtp` run after an update no longer compiles inside the timed generation.** The server already
  built every draft count's drafting and verification kernels before serving (`warm_up_once`); `runz` warmed up
  one plain token only. So the first MTP run after a build looked several times slower than plain: the laptop's
  35B 4.6 tok/s with MTP against 21.3 plain, ~35 after; Flash Next 1.94 against 4.42, 7.74 after. A Pro B70's 27B
  ran 8.4-8.7 tok/s in its first MTP round (issue #27). Its Flash Next drop from 38.55 to 5.44 with MTP was most
  likely the same. Logits bit-identical (the 35B with `--mtp`, before and after).
- **A new line after an MTP run:** `MTP time: drafting X ms, verification Y ms (Z ms a pass), verify_done W ms`.
- Measured on the laptop with the kernels built: the 35B (every expert cached) 21.2 tok/s plain, 32.7-35.3 with
  MTP (a verify pass of ~6 rows 85-90 ms, ~1.9 plain steps); Flash Next 6.71 plain, 7.74 with MTP. The 2-8 row expert
  kernels already read each distinct expert once for all the rows of a pass (the plan's G2 "union read").

## 2026-11-18: discrete cards: the norm and the H8 rotation in one kernel, 3 MTP drafts for the 27B

- **`ARC_H8_NORM` on by default on discrete cards** (it was on in the 140T preset only). A Pro B70's 27B 4K prompt
  2,890 -> 2,972 tok/s: `h8_act` 88.4 -> 60.3 ms, the norm 69.7 -> 60.0 ms, hashes equal (joshdev-de, issue #27,
  7368768). The B60's no gain of 11-14 was before the 128-register default. `ARC_H8_NORM=0` turns it off. iGPUs
  unchanged.
- **3 MTP drafts a step for the 27B on discrete cards** (4 before): the same B70, a code prompt, drafts 2 / 3 / 4
  58.2 / 59.8 / 57.5 tok/s, accepted 75.0 / 70.6 / 60.4 %. The 4th draft is accepted too seldom to pay for its
  verification. `--mtp-drafts N` still sets it.
- Measured, not changed: `ARC_GDN_L1PF=8` no gain (2,893); `ARC_IQ3S_TILE=1` slower on the B70 (35.24 -> 32.7 tok/s
  generation, every FFN GEMV slower), so it stays off. The B70's `gpu_bench`: memory read 603 GB/s, INT8 XMX
  366.6 of 367 TOPS.

## 2026-11-18: a request naming a model the server doesn't list no longer waits for a rescan

- **serverz:** a request whose `model` is not in the list (a benchmark's or a client's own name, `gpt-4o`) rescanned
  every model once a minute, and the request waited for it. The rescan parsed every GGUF header again and, for a
  model that nearly fills the card, repeated its trial loads. A benchmark saw seconds of apparent prompt time now
  and then (ThatGuy-Bob, issue #39). Now the folders are only listed (paths and sizes), and the models are scanned
  again only when that changed (a download, a file added, removed or packed). The model list's context figures are
  kept for the process. The laptop, 4B, a `gpt-4o` request: first 7.9 s, one a minute later 1.67 s; now 0.49 and
  0.36 s, as any request.

## 2026-11-18: every doc and help text brought up to date; a guide for contributors

- **CONTRIBUTING.md (new):** where the speed is still missing on each card, measured (the Arc 140T's prompt GEMM at
  ~33 % of its XMX peak, a B70's prompt GEMMs and generation at ~62 %), how to measure a card with one script, how to
  build, and the rules a change is checked against. Issue templates for hardware results and bugs.
- **Help texts:** runz, serverz, gpu_bench, xw_peak, zero-server.ps1 and zero-pi.ps1 checked option by option against
  their parsers: the Alchemist and B390 defaults, `--chunk`, `--mtp-drafts` (2..7, 4 on B-series cards), `--gpus`
  for DeepSeek-V4.1, `--expert-ram-gb`'s caps on split models, `--mtp-model`; precise mode's KL as measured last
  (0.0105 -> 0.0060 at 3.5K tokens; serverz and the scripts still said 0.0186 -> 0.0127, a measurement before PR #21).
- **User docs** (README, USAGE, features, install, server, models, remote access, the website's text): Alchemist and
  the B580 among the supported cards with their results, the B60 / B70 / 140T numbers of the latest releases, the
  four-engine web search, the Docker and Kubernetes files (untested), BF16 MoE files refused, Flash Next's Q8_0 MTP
  head and `--mtp-model`, the 35b-gsq conversion without Python.
- **Hardware docs:** docs/xe2-tuning.md opens with a table of each switch's default per GPU kind (B-series, B390 /
  Lunar Lake, 140T, Alchemist, the 125H), read from the code, with the switches it lacked; docs/multi-gpu.md a switch
  table and the RAM tier's limits; docs/architecture.md's source layout and AGENTS.md's map rebuilt against the tree.
- `runz`: the pipelined sums' count printed with the right format (a compiler warning).
## 2026-11-18: `ARC_IQ3S_TILE=1`, opt-in: the 27B's IQ3_S rows repacked for the one-token GEMV

- **For a B70 generation A/B** (plan G5, docs/arc-plan-2026-11-17.md). A dense matrix's IQ3_S rows are repacked once
  at load, in place, as planes in pairs of superblocks: the same bytes in another order, no extra VRAM or RAM. The
  one-token GEMV then reads each lane's bytes with block reads and no local window. Every other reader knows the
  layout (`decode_w8_sb` / `_sl`, `h8r_decode`, `gemv_mt`, `gemv_gu`, the NPU's range decode). Experts are never
  repacked.
- Bit-identical: on and off, and off against the build before (the laptop's 4B and 27B, `--dump-logits`). The
  laptop is bandwidth-bound: generation 4.13 / 4.49 tok/s off, 4.34 / 4.57 on, alternated, within its noise. The B70
  is decode-bound, where the repack is meant to help (its FFN GEMVs at ~62 % of the bandwidth).

## 2026-11-18: the model list no longer logs "--ctx auto" as if the model were loaded again

- **`serverz --ctx N`:** the model list (`/v1/models`, read when the web UI opens) works out every model's
  automatic context for the picker, and logged it as "--ctx auto: ... tokens", which read as the loaded model's
  context being replaced. It was not: the model kept its `--ctx` (`/props` n_ctx 8000 after the list and a chat,
  `-c 8000`). Those lines now say "model list: ... would get ..." and are debug-level (ThatGuy-Bob, issue #39).

## 2026-11-18: the rest of the review's fixes

- **A hostile PDF** (which `web_fetch` can bring in from a page) no longer fills memory through its fonts. A
  ToUnicode entry keeps at most 32 UTF-16 units, each font's maps and the text they expand to are charged to the
  document's work budget, and once that is spent a font builds no maps. One code mapped to 1 MB, repeated in a 1 KB
  content stream, gave a TB of text; a thousand fonts sharing one `/W` array built ~50 GB of maps.
- **The server:**
  - a page of another local server (allowed the model API) can no longer name the user's chats, transcripts or
    saved files in a chat request (`file` parts, `/files/` references), whose text the answer would show it;
  - a missing `/files/<id>` reads the chat folders again at most every 2 s (any web page could load it in a loop);
  - an MCP server started through a `.cmd` / `.bat` gets an argument with `& | < > ^ ( )` quoted
    (`npx mcp-remote https://h/sse?a=1&b=2` ran `b=2` as a second command);
  - Anthropic streaming reports the prompt's tokens in `message_delta` (it said 0);
  - an HTTP body of many small chunks, and slow headers, no longer cost quadratic time;
  - an audio upload named without an extension no longer takes the whole name as its format.
- **Files and formats:**
  - a GGUF tensor must be whole blocks in every row, as ggml requires;
  - Whisper's `model.bin`: every variable's shape is checked against its bytes, and the f16 copies no longer read
    64 bytes past the variable; conv1's padding is checked against the mel bins;
  - Gemma 4 and Qwen3-VL image encoders, Gemma 4 audio: the position tables, patch weights, output projection and
    depthwise convolution are checked against the widths the kernels use;
  - a Word template unpacks to 512 MB at most; an audio file's gaps add an hour of silence at most;
  - a saved conversation with a negative position is refused.
- **Chat template (Qwen):** a conversation that starts with a tool message opens its user turn.
- **The Qwen3-VL image encoder** runs in f16 where the engine runs two int8 passes of its own (Flash Next
  `--precise` with `ARC_VIT_Q2=1`, a qwen3 text model with a Qwen3-VL mmproj), which wrote past its buffer.
- **Test switches** clamped to what the buffers hold: `ARC_ATTN_DEC_SG`, `ARC_QWEN_DS_S`, `ARC_MOE_GR_GU`,
  `ARC_MOE_GR_DN`, `ARC_MOE_GR_DN_S`. The multi-GPU sum counter is 64-bit.
- Checked: the 4B and 27B bit-identical to the build before; Flash Next loads and answers; the 4B's image encoder
  and Whisper large-v3 load and answer; `test_doc_read` passes; a foreign page's file part 403, its plain chat 200.

## 2026-11-18: the automatic room beside a dense model's weights, measured on a B580

- **A dense model that nearly fills a discrete card keeps fewer of its weights in RAM.** d-dorok swept
  `ARC_HOST_KEEP_GB` with the 27B on an Arc B580 12 GB at 4K (issue #37): room 2.5 GiB (the old rule) 2.24 GiB of
  weights in RAM, 1.94 tok/s generation; 1.25: 3.98; 1.0: 5.03; 0.75: 7.08 tok/s, prompt 1,606 -> 1,877 tok/s,
  no error; 0.5: the prompt fell to 701 tok/s (the card over its memory). Its buffers were 1.56 GiB.
  - The estimate now takes the KV cache as allocated (Qwen: int8 K and V with their scales, a third of the 6 bytes
    a value it counted);
  - and it fills the card's whole reported memory: the driver's 0.5 GiB reserve is not kept a second time (the
    card had 0.3 GiB more than it reports, and the estimate is ~0.15 GiB high).
  - For that run: ~0.95 GiB of the weights in RAM, between the measured 1.25 and 1.0 points (~4-5 tok/s), 0.7 GiB
    short of where the prompt fell. `ARC_HOST_KEEP_GB=0.75` is the measured best for 4K. Whether a model counts as
    tight is unchanged.
- **Linux launcher:** Ctrl+C stops the server again. A background command of a non-interactive shell starts with
  SIGINT ignored, so the INT the launcher passed on never stopped `serverz`; it now passes TERM (d-dorok).

## 2026-11-18: fixes from a review of the kernels, the failsafes and the headers

A review of every kernel, the multi-GPU failsafes and the `.h` files, each finding checked in the code first. The
iGPU path is bit-identical to the build before (4B and 27B, `--dump-logits`, every prompt and generated position).

- **A dense model that nearly fills a discrete card** (today's automatic room, `plan_tight`):
  - whether it counts as tight is decided by the old rule again. The new, smaller room was worked out for a tight
    model's buffers (1,024-token chunks), but deciding by it let a model just under the old line come out not
    tight: every weight on the card and 4,096-token chunks, out of memory. It also changed with the context, so the
    server's `--ctx auto` probe and the real load could disagree;
  - the room now only says how many of a tight model's weights go to RAM. It also counts the just-in-time staging
    of every weight role, the layer-major residual, the asked `--chunk`, the MTP check's state, and the server's
    three state snapshots per slot. For the 27B on a 12 GB card at 4K, `runz` keeps ~0.4 GiB more of the weights on
    the card than the old 2.5 GiB rule; `serverz` keeps what the old rule kept, which is what it allocates.
- **Multi-GPU, `ARC_TP_EVSUM=1` (opt-in):** a shard that fails releases its events at once. Before, GPU 0, whose
  waits have no abort, stayed parked on them until the time limit, and the model faulted. A model held after a
  fault releases them too, so the server's other users of GPU 0 (speech to text) are not left behind them. A
  failed pass after such a release always loads the model again.
- **The gemm_xw start check:** a GPU that hangs during it is reported as a hang, as in the other checks, not as
  "the kernel differs", which turned gemm_xw off for the whole process.
- **The XMX start check** frees its buffers on every way out (they stayed for the process).
- **Flash Next's sparse attention:** a query whose selected blocks hold none of its keys gets 0, not NaN.
- **Two-pass experts (Gemma 4, Flash Next `--precise`):**
  - `ARC_G4_ONE=1` on an expert model no longer builds a broken kernel or doubles gate and up;
  - a prompt chunk that `moe_rg` cannot take (`ARC_MOE_TM`, `ARC_MOE_RG_MIN`, an expert width not a multiple of
    64) now runs both passes through the fused per-tile kernels, not one pass silently.
- **A malformed model file** now gives an error instead of a crash: Flash Next PLE hash columns past the table, a
  vision or audio encoder with 0 heads.
- **The server:**
  - special tokens in a bare string inside a content array are made plain text, as in a text part;
  - a `tool_choice` naming a tool that is not in `tools` asks for any tool. Before, no call could pass, so every
    token scanned the whole vocabulary until the context was full;
  - a stop sequence is found when a long token runs past it;
  - tool-call ids carry their own request's number (two requests at once on `--slots` gave the same ids);
  - `/sessions` (clear, save) only from this computer;
  - a downloaded file's name on Linux has its quotes and control bytes replaced.
- **Documents and the web:**
  - an XLSX cell keeps at most 32,767 bytes of text, and a sheet 64 MB. One long shared string used by many cells
    was copied into each of them, many GB from a small file;
  - turning a web page into text is no longer quadratic on many nested `<form>`, `<nav>` or `<svg>` tags.
- **The tokenizer:** a UTF-8 lead byte without its continuation bytes is one invalid byte (U+FFFD), as llama.cpp
  does, so the ASCII after it stays text.

## 2026-11-18: Docker and Kubernetes files (not tested yet)

- **`docker/`:** a `Dockerfile` (Ubuntu 24.04, Intel's GPU runtime, a release's Linux package), a `compose.yaml`
  (`docker compose up -d`: the GPU passed through, the models and compiled kernels kept on the host) and a
  `kubernetes.yaml` (the GPU from Intel's device plugin, a volume for the cache). The container starts the launcher
  with `MODEL`, `GPUS` and `SERVER_ARGS`; it downloads the model on the first start. See docs/install.md, Linux, step 6.

## 2026-11-18: a dense model that nearly fills a discrete card keeps more of its weights on it

- **The memory kept free for the buffers and the context is now worked out from the model and the context asked**,
  not a fixed 2.5 GiB (sized for a 32K context): the activations of a prompt chunk, the just-in-time decode's
  copies, the context's KV caches, the server's reserve. What does not fit then goes to RAM, as before. An Arc B580
  12 GB ran the 27B at a 4,240-token context with 2.24 GiB of its weights in RAM and ~1.5 GiB of its card unused:
  every token read those over PCIe 3.0, 1.94 tok/s (d-dorok, issue #37). Never more room than the old rule, so a
  32K context keeps what it had. `ARC_HOST_KEEP_GB=N` still sets it by hand. iGPUs are unchanged.

## 2026-11-18: a split model's generation sums without host round trips (opt-in)

- **`ARC_TP_EVSUM=1`: the sums of a model split over several GPUs meet on the GPUs during generation** (and drafts,
  under 64 rows), not on the host. Each GPU's copy of its part signals an event, and each GPU waits on every GPU's
  events of that sum before adding the parts: no wait for the GPU and no barrier on the host per sum, the host
  queueing ahead. Two Arc A770s under Windows spent 73.5 ms of an 81 ms token in 128 such sums, ~0.57 ms each for
  ~20 KB a part: a round trip each, not bandwidth (hansong80, issue #35). The same kernels on the same parts in the
  same order: the logits are the same bit for bit. Once per 512 sums each GPU waits and the shards meet, so the
  events can be reused (resetting each one just before its reuse let a shard ahead pass on a stale signal: found
  and fixed in testing). Prompt chunks keep the reduce-scatter; after event sums, a GPU waits and the shards meet
  before it. A shard that fails gets its events signalled from the host, so no GPU stays blocked. Off by default
  until two real GPUs measure it.
- **`ARC_TP_SIM` puts its shards on the GPU's compute queues in turn** (one queue for all before): a shard waiting on
  another's event would otherwise hold up the queue that event needs. The same logits as before (the old and new
  builds 901D5D90... three times each). With fewer queues than shards, `ARC_TP_EVSUM` stays off under it, and says so.

## 2026-11-18: memory bandwidth measured closer to a fast card's peak

- **The bandwidth runz compares generation against ("% of this GPU's measured N GB/s") is the best of two read
  patterns over 1 GiB** (256 MB where that does not fit), not one pattern over 256 MB. An Arc A770 measured 273-293
  GB/s against 560 on paper, so its generation looked at 52 % of the card when it read about a quarter of it
  (issue #35): at its speed a 256 MB read lasts ~0.5 ms, near a Windows launch's own cost, and each sub-group had
  one read in flight. The second pattern keeps 2 KB in flight per sub-group, the sub-groups side by side; the
  laptop's iGPU still reads fastest with the first (72-73 GB/s against 58-60), so both are timed.
- **`gpu_bench`'s memory row tries three ways** (16-byte loads, the same with 4 times the work-groups, sub-group block
  reads with 2 KB in flight) and shows each and the best: on an A770 the 16-byte loads gave 406 GB/s of 560.

## 2026-11-18: the launcher's model list counts a discrete card's own memory

- **Fixed: on a PC with a discrete Arc card, the launcher's "Get models" dialog refused models larger than the RAM**:
  an Arc A770 16 GB with 16 GB of RAM could not tick the 27B, the 35B-A3B or Flash Next (hhunaid, issue #34). Their RAM
  figure is a laptop's, whose iGPU's memory is that RAM; a discrete card holds the weights in its own. The dialog now
  adds the card's memory (from the registry's 64-bit size; discrete A-series and B-series cards by name, not iGPUs,
  whose figure there is their share of the RAM) and says "This PC has 16 GB of RAM and a GPU with 16 GB of its own".
  Laptops are unchanged.

## 2026-11-18: `gpu_bench` says the clock each test ran at

- **`bin\gpu_bench.exe` reads the GPU's actual clock and throttle reasons while each test runs** (Level Zero Sysman,
  sampled every 5 ms) and gives each row's share at that clock too: `FP32 (vector) ... 46%   at 1,100 MHz: 96%, held by
  average power`. An Arc A770 measured FP32 at 46 % of paper, XMX at 73 % (issue #35), where the same kernels give the
  laptop 98 %; the compiler makes the same code for both, so the clock under load is the likely reason, which the
  bench could not show. Where the driver gives no Sysman, the rows read as before. The laptop: 2,199-2,200 MHz in
  every test, FP32 98 % at its clock.

## 2026-11-18: Alchemist cards take the Arc 140T preset for small models too

- **On Arc A-series cards a dense Qwen model whose decoded weights would fit now runs the Arc 140T preset** (weights
  decoded just in time into H8 on XMX) instead of keeping them resident for DP4a, unless `--budget` is given. An Arc
  A770's Qwen3.5 4B 4K prompt 3,600 -> 4,821 tok/s (+34 %) with 3.98 GiB of buffers against 7.40, generation the same
  (37.8 / 38.2 tok/s; hansong80, issue #35, release .02). On the laptop, the 4B through the emulated preset against its
  resident DP4a path: KL 0.0039 over 2 x 4,096 tokens (perplexity 11.3685 / 11.3651). The log says so; `--budget 9`
  (or any value) keeps the old choice. The 140T iGPU and the B-series are unchanged.

## 2026-11-18: Alchemist cards split the long H8 GEMMs as the Arc 140T does

- **The Arc A-series (Alchemist) cuts the H8 prompt GEMMs over 8,192 values long into pieces of 17 superblocks**, as
  iGPUs and the Arc 140T do (34 before, the B-series' figure). An Arc A770's Qwen3.8 27B 4K prompt 1,150 -> 1,223 tok/s,
  its `gemm_down` 823 -> 623 ms (TheUniKian, issue #36, release .08; one piece: 1,155). Another order of the pieces'
  sums: the laptop's emulated preset, 2 x 4,096 tokens of the 27B, 17 against 34 pieces: KL 0.00110 (max 0.150, the
  same top token 98.5 %). `ARC_GEMM_SPLIT=34` as before. The B-series and the laptop are unchanged.

## 2026-11-18: the 2D prompt GEMM's fallback goes to the plain form

- **A failed check of the 2D prompt GEMM now runs the plain form**, not the 2D loads alone: on the B60 under Windows
  that fails the full form's check (release .07's fix), the loads alone took the 27B's 4K prompt to 1,705 tok/s
  against the plain form's 1,726, with equal logits (manoreken2, issue #4, release .08). `ARC_GEMM_XW_2D=2` still
  asks for the loads alone. A B60 under Linux passes the full form (MaineTim, issue #19: 1,635-1,644 tok/s on the
  27B UD-Q5_K_XL, the same logits with and without it), as the B70 does.

## 2026-11-18: the first Arc A770 runs; resident weights in longer prompt chunks on discrete cards

- **A model whose decoded weights stay resident now runs prompts in 4,096-token chunks on discrete cards of 12 GB or
  more**, as the others do (512 before, a figure tuned on the laptop's iGPU, which keeps it). With resident weights
  nothing is decoded per chunk, so a longer chunk only means fewer passes over them: an Arc A770's Qwen3.5 4B 4K
  prompt 3,211-3,219 -> 3,594-3,598 tok/s (hansong80, issue #35, the first A-series report). On the laptop the 4B's
  perplexity over 2 x 4,096 tokens is the same with either chunk (KL 0.00000); the buffers take 0.6 GiB more, hence
  the 12 GB. `--chunk 512` as before.
- **The log says why the Arc 140T preset is off when the weights stay resident** ("--budget 0 ... turns it on"): on
  the A770 the 4B kept its decoded weights and so ran DP4a, while the 9B, decoded just in time, took the preset
  (2,969-3,017 tok/s against 2,082-2,099 without it).

## 2026-11-17: two Arc 140T results become its preset's defaults

- **The Arc 140T preset's H8 GEMM in work-groups of 2 column sub-groups (`ARC_GEMM_XW_WGX=2`)** and **the norm and the
  rotation of its rows in one kernel (`ARC_H8_NORM=1`)**, both on by default there now. A Core Ultra 9 285H's Arc
  140T, Qwen3.8 27B IQ3_S 4K prompt, best of 3 (PascalSoftwares, issue #5, release .07): 368.3 tok/s as it was,
  390.0 with `WGX=2` (`WGX=4` 383.2, `WGX=4 LOCK=1` 365.8), 382.1 with `ARC_H8_NORM=1` (logits identical; `h8_act`
  658 -> 459 ms, `rmsnorm_quant`'s 456 ms -> `rmsnorm_h8r` 306). Both together are not measured yet. The laptop's
  emulated 140T preset gives the same logits as before. `ARC_GEMM_XW_WGX=1` / `ARC_H8_NORM=0` turn them off; other
  GPUs keep their defaults (the preset also covers Alchemist cards, unmeasured).

## 2026-11-17: a BF16 MoE model refused before its expert file is written

- **A runz / serverz older than its kernels now says so:** since the H8 rewrite, `h8.cl` needs `rot.cl` in front of
  it, which only the new engine adds. An older binary with the new kernel files stopped with "use of undeclared
  identifier 'rot_pack'" (MaineTim, issue #19); now the build error says to rebuild both.
- **Fixed: a Flash Next GGUF in BF16 wrote its 257.7 GB expert file first, then stopped** with "unsupported weight
  type BF16" (manoreken2, issue #4). The experts run in quantized types only; their types are now checked before
  the expert file is written, and the message says so (a Q8_0 GGUF is close to the unquantized model).
- Flash Next's 640-wide shared expert in F32 / F16 / BF16 is requantized to Q8_0 as every other float matrix is
  (it was refused).

## 2026-11-17: the 2D prompt GEMM on drivers where its epilogue differs (a B60 under Windows)

- **Fixed: on a Pro B60 under Windows, release 2026.11.17.06's 2D prompt GEMM failed its start check**, so the card
  ran the plain form (correctly, but without the gain), and the log asked for a report (manoreken2, issue #4). Only
  the modes that add C's old value to the product differed (the split-K pieces and the add mode): that driver's
  compiler fuses `product + old` into one multiply-add differently in the 2D epilogue than in the reference kernel.
  The stores and the fused SwiGLU, which reads C the same way, were right, and so were the 2D loads. Now a failed
  check drops C's 2D form first and keeps the loads in 2D blocks, where the loop's message savings are (16 -> 9 a
  64-value chunk); the epilogue is then the reference's code. Only if that differs too does it run the plain form.
  `ARC_GEMM_XW_2D=2` asks for the loads only. A B70 under Linux, which passed the full form's check, is unchanged.
- The log no longer says "needs 16-lane XMX with 2D block loads" after such a fallback.

## 2026-11-17: the B-series' H8 prompt GEMM in 2D messages by default, with a deeper prefetch

- **`ARC_GEMM_XW_2D` on by default on discrete cards** (the B-series), with the prefetch of the weights and the
  activations 2 chunks ahead when neither `ARC_GEMM_XW_PF` nor `ARC_GEMM_XW_PFD` is set. A Pro B70's Qwen3.8 27B IQ3_S
  4K prompt: 2,768.1 tok/s as it was, 2,853.2 with the 2D form, 2,896.7 with the deeper prefetch added (+4.6 %; `gemm_up`
  247.0 -> 218.8 ms, `gemm_down` 190.5 -> 176.5, `gemm_gate` 179.5 -> 169.9). The logits' hashes are equal (joshdev-de,
  issue #27, main 1a7ac94, best of 3). The same prefetch without the 2D form was slower (2,711.4), so the plain form
  keeps the weights 1 chunk ahead. ocloc `bmg` / `ptl`: no spills (sends 77 -> 93 with the deeper prefetch).
  `ARC_GEMM_XW_2D=0` turns it off. iGPUs (Lunar / Panther Lake's B390) keep it opt-in until one measures it.
- **The Gemma 4 drafter's loader checks its matrices' shapes** as the main model's does (`gemv_weight` through `mat()`):
  a malformed drafter file is refused instead of read out of bounds (from a review by Gemini).

## 2026-11-17: the B-series' and Panther Lake's H8 prompt GEMM in fewer messages (opt-in)

- **`ARC_GEMM_XW_2D=1`:** `gemm_xw` (the 32 x 64 H8 GEMM of 16-lane XMX GPUs) moves the same bytes into the same DPAS
  operands with fewer memory messages: a chunk's 32 activation rows in one 2D read instead of four, a column group's
  1 KB of weights in one 2D read instead of two block reads, and the output 8 rows x 16 columns a message instead of a
  row a message. ocloc `bmg` / `ptl`: send instructions 489 -> 77, instructions 7,957 -> 6,866, no spills; the loop's
  messages a chunk 16 -> 9, the epilogue's 128 -> 16. Bit-identical by construction and checked at start like every
  shape; a failed check keeps `gemm_xw` without it. Not for the 32 x 96 tiles (they spill with it). The laptop has no
  2D block loads, so a card has to measure it.

## 2026-11-17: two B70 results become defaults on discrete cards; one test switch stays off

- **`ARC_H8_GRF=128` by default on discrete cards with 16-lane registers (B-series):** a Pro B70's Qwen3.8 27B 4K
  prompt 2,725 -> 2,772 tok/s (`h8_decode` 61.9 -> 43.7 ms, `h8_act` 95.0 -> 88.5 ms), the same logits (joshdev-de,
  issue #27, release 2026.11.17.03). The B60 is the same chip. Lunar / Panther Lake keep 256 until a B390 measures it;
  `ARC_H8_GRF=256` turns it off.
- **`ARC_MOE_GEMM_PAD=1` by default on discrete cards with 16-lane XMX:** Flash Next's expert gate / up through decode +
  the XMX GEMM, a Pro B70's 4K prompt 1,182 -> 1,414 tok/s (+20 %; `moe_gateup` 1,504 -> 506 ms). The laptop's KL against
  the per-tile kernel was 0.00000. iGPUs and 8-lane GPUs keep the per-tile kernel; `ARC_MOE_GEMM_PAD=0` turns it off.
- **`ARC_MOE_RG_XMX=1` stays off:** the same B70's 35B-A3B was slower with it (4K 5,177 -> 4,371, 16K 5,435 -> 4,465 tok/s),
  though bit-identical: its expert GEMM from the raw rows is slower on XMX than decode + the tiled GEMM.
- The laptop's logits are unchanged (both defaults are for discrete cards only).

## 2026-11-17: safety checks from a review (Gemini, checked against the code)

- **The GEMMs' output alignment:** `matmul1` refuses a row stride or column offset that is not a multiple of 4 floats.
  The kernels write C 16 floats at a time with block writes, which need 16-byte addresses; a misaligned one would
  write wrong places without an error. Every model's shapes are multiples of 16 today.
- **H8's int32 sums:** `h8_matmul` refuses a piece of K over 133,000 values (a split's whole dot product is one int32
  sum, at most 127 x 127 x K). The models' K is 17-22K and split-K cuts it further.
- **`ARC_MOE_RG_XMX=1`'s first-use check** (discrete cards, opt-in): the 128 MiB copy it compares with is outside the
  memory plan, and on a nearly full card its allocation ended the process. Now it keeps DP4a for that type and says
  so; the copy is freed on every way out.
- **Multi-GPU prompt sums (`ARC_TP_PIPE`):** the pieces' counters were `long`, 32-bit on Windows, and grow for the
  life of a load: a server running prompts without a break could wrap them in ~4-33 days, then a shard could pass a
  wait early (wrong sums) or wait forever. Now 64-bit.
- **`clrt.h`'s kernel-argument cache:** a free is counted before `zeMemFree`, not after, so no other thread can get
  the same address and set it on a kernel before the count moves (no case of it is known; a one-line hardening).
- `gemm.cl`: the SwiGLU gate value is `gate`, not `g` (which also names the column group).
- Not changed, checked: the epilogue's block reads cannot be widened (rows are `ldc` apart); `native_exp` would move
  every model's logits for a negligible cost; `moe_scan` at 512 experts, sub-buffer alignment (our `clCreateSubBuffer`
  is a pointer offset over Level Zero), `asc_transpose` twice and `ARC_GDN_PF` on the 140T (spills) were not issues.
- Checked: logits identical bit for bit before and after on Qwen3.5 4B, Gemma 4 E4B and Qwen3.8 27B, and the 4B over
  2 GPUs (`ARC_TP_SIM=1`).


## 2026-11-17: L1 column sharing for the Arc 140T's prompt GEMM, its full form (opt-in)

(Release 2026.11.17.02 carried a first form of these two switches on 8 lanes, the section below; this one replaces
it: the packed 32 x 32 tile with them, the prefetch shared out, the narrower-tile fallback, `xw_peak`'s flags. The
switches' names and defaults are the same.)

- **`ARC_GEMM_XW_WGX=2 / 4` and `ARC_GEMM_XW_LOCK=1` on 8 lanes (`gemm_xw8`, the Arc 140T):** as the 16-lane kernel's
  (+3.7 % on a B70), 2 or 4 column sub-groups a work-group read the same activation chunk while it is in L1, and the
  lock holds the work-group in step around each 64-value chunk (a split barrier). With the activations' prefetch
  (`ARC_GEMM_XW_PF=2 / 3`) each column sub-group prefetches only its share of the token groups. The 140T's
  `xw_peak --probe` reached 85 % of the XMX peak with its operands in L1, against ~33 % in the engine; this tests
  whether sharing brings them there. Off by default: the 140T's default is unchanged.
- **Packed 32 x 32 tiles with column sharing:** `ARC_GEMM_XW_NC=2` keeps the packed operands (`ARC_GEMM_XW_PACK`,
  on in the 140T preset) when `ARC_GEMM_XW_WGX` is 2 or 4 (alone it was slower: 28 % of the peak against 33 %). A
  width that is not a multiple of 32 runs one column group a sub-group, so a packed input always reaches `gemm_xw8`
  (the packed orders do not depend on the tile width).
- **`bin\xw_peak.exe --wgx 2|4` and `--lock`:** every variant again with the column sub-groups and / or the lock, in
  the same run, compared bit for bit with the plain default.
- Checked on the laptop with DPAS emulated: every `xw_peak` variant with `--wgx 2`, `--wgx 4 --lock` and `--lock`
  bit-identical; `runz --dump-logits` (1,753 tokens in 512-token chunks) on the 4B and 27B the same SHA-256 as before
  with the switches off, and on the 27B in the emulated 140T preset (`ARC_140T=1 ARC_XMX_TRY=2`, which runs it on
  `gemm_xw8`) the same with WGX 2 / 4, the lock, 32 x 32 with WGX and the activations' prefetch with WGX; the
  start-time check passed in each. ocloc `arl-h`, 256 registers:
  no spills in any combination (32 x 16 and packed 32 x 32, WGX 1 / 2 / 4, with and without the lock and prefetch).

## 2026-11-17: H8's rotation rewritten, spill-free; three opt-in switches for the 140T, Alchemist and the Xe2 / Xe3 GPUs

- **The rotation in one file of our own, `zero/kernels/rot.cl`.** The Walsh-Hadamard transform of H8's blocks (its
  sign hash, the stages in registers and across lanes, the int8 rounding and the scale) was written three times, in
  `h8.cl`, `decode.cl` (`h8r_decode`) and `ops.cl` (`rmsnorm_h8r`). The engine now puts `rot.cl` in front of those three
  kernels' sources (`engine_setup.h kernel()`), and the H8R kernels are rewritten around it. The arithmetic is the same
  in the same order: every H8 path's logits are the same bit for bit (below). The `NOTICE` file no longer lists the H8
  code.
- **No register spills on the 125H, the 140T, the 285H and Alchemist.** ocloc for `mtl` / `arl-h` (Xe-LPG, where a
  16-lane float takes 2 registers) showed three H8 kernels spilling 1 KB at 256 registers: the packed weight writer
  (`h8r_weights -DXW_PACKW_OUT`, the 140T preset's), `h8r_decode` for the codebook types and `rmsnorm_h8r`. Their
  addresses are now one base and small offsets (a block's region of the weights is 4 x HB uints from its first; the
  packed order keeps each 16 words in the same 256), so the unrolled loops hold no 64-bit arithmetic. Now spill-free:
  every `h8r_*` kernel and `h8r_decode` on `bmg`, `mtl`, `arl-h` and `ptl` (150 of 152 builds); `rmsnorm_h8r` keeps its
  1 KB on Xe-LPG (opt-in there). Instructions down 10-43 % (`arl-h`: the packed writer 4,633 -> 2,661,
  `h8r_decode` IQ3_S 3,434 -> 2,868, `h8r_colmax` 2,732 -> 2,239).
- **`ARC_H8_GRF=128` (opt-in):** the rotation kernels at 128 registers, twice the threads per core, where a register
  holds 16 floats (Xe2 / Xe3: the B-series, Lunar / Panther Lake); ignored on Xe-LPG, which spills 8-16 KB there.
  ocloc `bmg` / `ptl`: no spills at 128. The same logits (checked on the laptop with the 128-register builds forced,
  `ARC_H8_GRF=-128`, a test setting). Its speed: the B60 / B70 / B390 owners to measure.
- **`ARC_GEMM_XW_WGX` / `ARC_GEMM_XW_LOCK` on 8 lanes (opt-in):** the 140T's and Alchemist's `gemm_xw8` with 2 / 4 / 8
  column sub-groups per work-group, which then share their activations in L1, as the 16-lane `gemm_xw` has done since
  2026-11-15 (a Pro B70 +3.7 %). The 140T's `xw_peak --probe` reached 85 % of the XMX peak with the operands in L1,
  ~33 % in the engine. Bit-identical (checked at start for every shape used); ocloc `arl-h`: no spills. Off by
  default on 8 lanes; `bench\xw_sweep.ps1 -Group` now runs there too.
- **`ARC_H8_NORM=1` now works in the 140T preset:** `rmsnorm_h8r` writes `gemm_xw8`'s packed order (`-DXW_PACKA_OUT`,
  as `h8r_act`). Before, it fell back to `rmsnorm_quant` wherever the packed activations were used, so the fused norm
  never ran on a 140T. Bit-identical.

## 2026-11-17: the launcher's icons in the website's colours

- The tray icon, the window's icon and the splash's logo draw the inferzero mark in `site/icon.svg`'s colours (lit
  `#0078d4`, dim `#c9d3de`; on dark `#59a7e3` / `#2b3440`), so it is one icon in the browser, the web UI and the
  launcher. Before: the Windows accent colour, with half-transparent dim tiles in the tray. The tray icon lights up
  while a model runs (the Z in the icon's blue, with the green dot) and is grey when stopped; it follows the
  taskbar's light or dark setting.

## 2026-11-17: the B390 keeps up to 16 GB of decoded weights when its GPU share is large

- **Why:** gee456's B390 (128 GB of RAM, a 114.5 GiB GPU share, issue #1) ran the 27B's 4K prompt at 583-592 tok/s
  with 16 GB of decoded weights kept resident, against 540-552 without (release .07, the path that is the default
  since 2026.11.17.01). iGPUs had no resident decoded weights by default, because on an Arc 140T with 64 GB the
  automatic size (26 GiB) made its prompt slower (340-357 -> 176 tok/s): on an iGPU that memory is the system's RAM.
- **What:** an iGPU with 16-lane XMX (passed at start) and a GPU share of 64 GiB or more keeps up to 16 GiB of decoded
  weights resident (`ARC_W8_PARTIAL_GB=auto`, `zero/engine_load.h`). The 140T (8 lanes; a 64 GB machine's share is
  ~32 GiB) and iGPUs without XMX are unchanged; `ARC_W8_PARTIAL_GB=0` turns it off, a number of GB sets it. A resident
  copy is decoded by the same code as the just-in-time one: the same logits.

## 2026-11-17: the B390's fast prompt path by default, Flash Next's Q8_0 MTP head, `--mtp-model` in the server

- **Panther Lake's Arc B390 (and Lunar Lake): H8 on the wide-tile XMX GEMM and flash attention by default.** gee456's
  runs on release .07 (issue #1), the 27B IQ3_S 4K prompt: as before 258-268 tok/s; H8 on `gemm_xw` 546-560; plus the
  flash attention 540-552 (attention 377 -> 160 ms); plus 16 GB of decoded weights resident 583-592. KL against the
  plain XMX path 0.00175 (same top token 98.1 %, two 4,096-token chunks of USAGE.md). iGPUs with 16-lane XMX now take
  H8 (`h8_auto_wanted`, `gemm_xw` follows it) and `attn_fx` where XMX passed its start-time checks; they keep one
  column group, 1-wide work-groups, weights decoded just in time (RAM is the GPU's memory), and the GEMV defaults (the
  discrete cards' GEMV rules made generation slower there: 5.81 -> 4.40 tok/s). `ARC_H8=0` / `ARC_ATTN_FX=0` turn them
  off. The 125H (no XMX) and the Arc 140T (8 lanes, its own preset) are unchanged: the laptop's 27B logits the same
  SHA-256.
- **Flash Next's MTP head in Q8_0:** SC117's `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` stopped the load with "unexpected
  expert tensor blk.48.ffn_down_exps.weight" (gee456, issue #31): rows of 640 values were allowed for some 32-value
  block types and Q4_0, not Q8_0. Every 32-value block type is allowed now (`is_per32_type`; the padding with zero
  blocks was already generic). The Q4_0 head checked again on the laptop: it loads and drafts (74 % accepted).
- **`serverz --mtp-model FILE`:** as runz's. Without it the server only took an `mtp-*.gguf` next to the model. A head
  or drafter that is not the loaded model's (after a model switch) is skipped with a line (`mtp_model_soft`; runz keeps
  the error); the `--ctx auto` cache key includes it (a Q8_0 head pins about twice a Q4_0 one's GPU memory).
- **`ARC_MOE_GEMM_PAD=1` (opt-in, the plan's X5):** Flash Next's expert gate / up of prompt chunks of 1,152+ tokens
  through decode + the int8 GEMM (XMX on discrete cards) instead of the per-tile DP4a kernel, its down projection
  staying on `moe_rg`. Bit-identical to the per-tile kernel (Flash Next, 2 x 2,048 tokens of USAGE.md: KL 0.00000,
  perplexity the same). On the laptop (no XMX) slower: 4K prompt 181-185 -> 161-176 tok/s, the decode of every expert
  (`decode_w8` 0.45 -> 4.77 s) costing more than the GEMM saves (`moe_gateup` 6.74 -> 3.14 s); a B70 to measure. Not
  for Gemma 4 26B (GELU experts).
- **`ARC_MOE_RG_XMX=1` (opt-in, the plan's X1): the expert GEMMs of prompt chunks on XMX from the raw expert rows.**
  `moe_rg` (the int8 GEMM that decodes each expert row in registers, no `decode_w8` pass) takes each 64-value chunk's
  dot products as int8 DPAS for the types with one scale per 32 values (IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XXS and the per-32
  float types): a lane's decoded row is DPAS's weight operand as it is, the tile's 16 tokens two row groups of 8, the
  float steps after the integer sums unchanged. It replaces decode + GEMM where that ran (the 35B and other unpadded
  MoE models, from 320 tokens a chunk instead of 1,152) and gives Flash Next's down projection, on `moe_rg` already,
  the XMX form; Flash Next's gate / up stay where they were (`moe_rg` there is not the per-tile kernel's results: KL
  0.0181). Each expert type is checked at its first use: the DP4a kernel writes the output, the XMX one a copy, compared
  byte for byte; a difference keeps DP4a for that type, with a line (`zero/xe2/engine_moe_xw.h`). Laptop, DPAS
  emulated (`ARC_MOE_RG_XEMU=1`): IQ3_XXS, IQ4_NL, Q2_0, IQ3_S and IQ4_XS the same as DP4a bit for bit, and Flash
  Next's logits with the switch the same SHA-256 as without. Speed: a B60 / B70 to measure.
- **Self audit before committing (fixed):** the server's own head test refused Gemma 4 drafters and was looser than
  the engine's; a dry run (`--ctx auto`) never counted H8's buffers when H8 came from the default (`h8_auto_` or the
  140T preset: XMX is not checked in a dry run); the iGPU XMX attention defaults now wait for XMX to pass its checks.
- **The release packages** leave out the Markdown notes (only CHANGELOG.md, README.md and USAGE.md stay: the
  benchmarks and the KL commands read USAGE.md, the tests README.md) and the test PDFs (`tests/test_doc_read.cpp`
  skips their checks without them: 26 ok, 2 skipped; 40 ok in the repository).
## 2026-11-17: a split model's RAM tier at most 128 GiB in all; H8 measured on Flash Next, not taken

- **The RAM tier:** manoreken2's 4 Pro B60s (issue #4, release .16): the default 32 GiB per GPU ran as measured
  (DeepSeek-V4.1 Q8_K_XL, 512-token prompt 24.0 tok/s, generation 8.32), while 48 and 64 GiB per GPU (192 / 256 GiB
  pinned in all) stopped after loading. The per-GPU size is now also capped at 128 GiB divided by the GPUs
  (`zero/engine_stream.h`), so more than 4 GPUs cannot cross what ran; 4 or fewer are unchanged.
- **H8 on Flash Next:** faster on the B60 (8K prompt, alternated: 515.7 -> 541.2 and 550.8 -> 610.2 tok/s), but
  less accurate than its limit allows: KL against the default path 0.0256 (max 1.22, the same top token 93.3 %; laptop,
  2 x 2,048 tokens of USAGE.md), on top of the default's own ~0.017 against f32. It stays opt-in (`ARC_H8=1`). Flash
  Next's H8 rotates the int8 activations (its quantization kernels keep no f32 copy for H8); the f32 copy, as for the
  Qwen models, is what would have to come first.
## 2026-11-16: a split model keeps experts in RAM again, up to 32 GiB per GPU (DeepSeek-V4.1 over 4 B60s)

- **Why:** since 2026-11-16 (release .05) a model split over several GPUs kept no experts in RAM, after an unbounded
  tier (a quarter of the free RAM per GPU) ran DeepSeek-V4.1-Flash UD-Q8_K_XL over 4 Pro B60s out of memory.
  manoreken2's sweep of the tier's size on the same machine (1 TB of RAM, issue #4, release .10), per GPU 0 / 8 / 16 /
  32 GiB: 512-token prompt 14.4 / 14.5 / 15.9 / 23.9 tok/s, generation 3.55 / 6.69 / 8.51 / 9.25 tok/s (the wait at
  the sums per token 117 / 46 / 30 / 24 ms), each size without an error.
- **What:** a split model's RAM tier is at most 32 GiB per GPU and at most half the RAM free beyond 16 GiB shared by
  the GPUs (`zero/engine_stream.h`). `ARC_XRAM_GB=0` turns it off, `=N` sets N GiB per GPU. One GPU is unchanged.
  The values are the same: the experts come from RAM instead of the SSD.
- **Also confirmed today (joshdev-de's B70, issue #27, release .15):** `ARC_MOE_GRF=128` gives the same logits as 256
  (the same SHA-256), and the 35B's generation is 95-98 tok/s once the kernels are cached (the 3.6 s first step of
  the earlier run was the kernels being compiled). The B60's sweep (manoreken2) shows `ARC_MOE_GRF=128` neutral there
  (35B 3,891 against 3,907-3,912 tok/s) and nothing faster than the defaults on the 27B and 35B.
## 2026-11-16: the Windows launcher stays in the tray when minimized (issue #33)

- **Why:** minimizing the launcher closed it, and the tray icon disappeared with it (PascalSoftwares, release
  2026.11.16.12). The window was shown with `ShowDialog`, and hiding a form shown that way ends its message loop: the
  Resize handler's `Hide` (minimize to the tray) made `ShowDialog` return, and the script ended. Windows PowerShell 5.1,
  a test form: the loop returned 0.02 s after the hide.
- **What:** `scripts/zero-launcher.ps1` runs its window with `Application.Run`, whose loop goes on while the window is
  hidden: minimized, it waits in the tray, and the tray's Show the window brings it back. The server was not affected
  (a process of its own). Checked: the test form stays 3 s after the hide; the launcher's `-SelfTest` opens, lists the
  models and closes as before.
## 2026-11-16: two defaults from users' runs: the MoE GEMMs' register file on discrete cards, the 140T's prefetch

- **`ARC_MOE_GRF=128` by default on discrete cards with 16-lane XMX:** a Pro B70's Qwen3.6 35B-A3B 8K prompt 5,377.3 /
  5,338.2 -> 5,612.5 tok/s, +4.4-5.1 % (joshdev-de, issue #27, release 2026.11.16.12; the only switch of his sweep
  past 3 %). The same kernel source with more threads per Xe-core; iGPUs keep 256 registers (the 128 build spilled
  there). `ARC_MOE_GRF=256`: as before.
- **`ARC_GEMM_XW_PF=1` by default in the Arc 140T preset** (also Alchemist cards): a 140T's 27B 8K prompt 359.0 /
  359.3 -> 366.4 / 366.9 tok/s, +2 % (PascalSoftwares, issue #5, release .12); `2` and `3` were slower (341.9 /
  338.5), two chunks ahead the same as one (366.8). `ARC_GEMM_XW_PF=0`: off.
- **Checked:** the laptop takes neither (the 27B's logits the same SHA-256); the 140T preset with DPAS emulated, now
  with the prefetch by default, the same SHA-256 as without it.
- The same B70 sweep found nothing on the 27B (2,727 tok/s at 8K; every tile and group switch slower), and its 35B
  profile shows where the MoE prompt goes next: the expert GEMMs 39 % (`moe_gateup` + `moe_down`), `gdn_recur` 18.5 %,
  and `decode_w8` 11.6 % at 4K, the experts' per-chunk decode (QUEUE: experts kept in their decoded form).
## 2026-11-16: XMX prompt attention on iGPUs with 16-lane XMX too (Panther Lake's Arc B390, Lunar Lake)

- **Why:** gee456's Arc B390 ran `bench\f33_sweep.ps1` and the website benchmark on release 2026.11.16.07 (issue
  #31). The prompt attention on XMX (`ARC_ATTN_XMX`, on by default on discrete cards since 2026-11-10, off on iGPUs
  until one measured it) at 65K tokens: Qwen3.5 4B 950 -> 1,079 tok/s, Qwen3.8 27B 206 -> 215, Qwen3.6 35B 795 ->
  904; at 512-16K within about 1 % either way; logits bit-identical.
- **What:** on by default on iGPUs with 16-lane XMX as well (`zero/xe2/engine_xe2.h attnx_wanted`); `ARC_ATTN_XMX=0`
  turns it off. The 125H (no XMX) and the Arc 140T (8 lanes) never take it: their logits unchanged (the 27B's, the
  same SHA-256 on the laptop).
- The same sweep: the chunked DeltaNet forms (`ARC_GDN_CHUNK=1/2/3`) were slower on the B390 at every length; they
  stay off.
## 2026-11-16: the Arc 140T's wide-tile GEMM can prefetch its next chunk (opt-in, `ARC_GEMM_XW_PF`)

- **Why:** the 140T's `xw_peak --probe` (PascalSoftwares, issue #5) shows its XMX loop reaching 85 % of the int8 peak
  at 32 x 32 tiles, and the loads alone ~128 bytes a clock per Xe-core, when the data is in the first-level cache;
  the engine's GEMM runs at ~33 %. So the core can load fast enough: the operands are not in L1 when the loop wants
  them. The same day's sweep of the engine's other switches on the 140T found nothing beyond 1 % (27B and Flash
  Next), and `ARC_GEMV_WAVE=1` no gain in generation (6.05 against 5.97 tok/s).
- **What:** `gemm_xw8` (8 lanes: the 140T, Alchemist cards) takes the B-series cards' `ARC_GEMM_XW_PF`: 1 prefetches
  the next chunk's weights, 2 its activations (packed layout only), 3 both, with the portable `prefetch()` built-in
  (`zero/kernels/gemm_xw8.cl`, `zero/xe2/engine_xw.h`). Off on 8 lanes unless set; 16 lanes unchanged.
- **Checked:** bit-identical (laptop, the 140T preset with DPAS emulated, 27B, `--dump-logits`: off / 1 / 3 the same
  SHA-256). Speed on a 140T: asked for on issue #5.
## 2026-11-16: a dense model larger than a discrete card keeps part of its weights in RAM (an Arc B580 for the 27B)

- **Why:** a B580 owner wants Qwen3.8 27B, whose weights (about 11.7 GB for IQ3_S) do not fit a 12 GB card with any
  context. The engine already kept part of a dense model's weights in host memory on an iGPU whose share it nearly
  fills (Gemma 4 31B), but only on iGPUs.
- **What:** on a discrete card, the weights that would leave less than 2.5 GiB of the card's memory for the working
  buffers and the context go to host memory (pinned, read by the card over PCIe), at most the RAM free at the start
  less 4 GiB (`Engine::plan_tight`, `zero/engine_sessions.h`; `ARC_HOST_KEEP_GB` sets the room kept,
  `ARC_HOST_WEIGHTS_GB=N` the amount, 0 none). Prompt chunks are 1,024 tokens, as on a tight iGPU. The same kernels
  read the same bytes: the logits are bit-identical (checked on the laptop with weights forced into host memory).
- **Speed:** the prompt reads each host-resident matrix once per chunk, so it loses little; generation reads them
  on every token over PCIe (a B580 is PCIe 4.0 x8, ~13 GB/s, against 456 GB/s for its own memory), so it slows with
  the share kept there. MTP drafts help: a verification reads the weights once for several tokens. Not measured on a
  card yet.
- **The launcher and `--list-models`:** the model list's memory now has `host_weights` (bytes), and the launcher's
  fit line says how much of the weights is in RAM and that generation is slower for it.
- Models that fit are unchanged: the plan only acts when the weights would leave less than 2.5 GiB free.
- **Found in an audit the same evening, fixed before release:** the server's `--ctx auto` cache (`m26`) would have
  kept a "does not fit" from before for that very model; the room kept now also holds the encoders' and speech to
  text's reserve and any context or slot count past one 32K slot (estimated from the model's shape), where before
  speech to text left a B580 no context and `--ctx 65536` ran out of the card's memory; the MTP block's layer counts
  only when it is loaded; and the model list shows the 1,024-token chunk such a model takes.

## 2026-11-16: H8 does not fit Gemma 4 12B (measured, not built)

- One H8 pass instead of Gemma's two int8 passes would have taken the 12B's 4K prompt from ~142 tok/s toward the
  one-pass ~230, but its activations do not survive one int8 rounding, rotated or not: KL against f32 on 480 tokens
  of USAGE.md, two passes 0.00001; one pass 0.064; one H8 pass from the f32 rows 0.155 (engine); the f32 reference
  with H8's rounding simulated (`REF_HAD`) 0.549 with one scale per token row, 0.267 per 256 values. The engine change
  that tried it was not kept. Left to try: H8 in two passes (the rotated rows and their residual), which would let
  discrete cards run Gemma's two passes on the wide-tile XMX GEMM.
## 2026-11-16: the resident W8 default on discrete cards leaves room for speech to text and the third snapshot

- **Why (an audit of the last two days' engine changes):** `ARC_W8_PARTIAL_GB=auto` keeps decoded weights resident
  in the room an engine sees left. Two things it did not see:
  - a Qwen3-ASR engine loaded in the server next to the chat model asks for no resident W8 (`w8_budget_gb = 0`), but
    auto W8 still ran for it, counting only its own memory: with a large chat model on a B60 / B70 and an ASR GGUF
    other than Q8_0, its load could run out of GPU memory (the shipped Q8_0 file has nothing to decode);
  - the server keeps 3 recurrent-state snapshots, the room counted 2: the third, made at the first chat, took ~150
    MiB of the 27B's 0.5 GiB margin.
- **What:** auto W8 is skipped for an engine with `w8_budget_gb <= 0` (the ASR engine, `runz --budget 0`), and its
  room counts the third snapshot (`zero/engine_load.h`). `snapshot_bytes()` itself, which the `--ctx auto` plans and
  the expert cache use, is unchanged, so contexts stay as they were. On the 27B about 150 MiB fewer resident weights.
  The values are the same: a resident copy and a just-in-time one are made by the same `decode_w8`.
- Not changed: with resident W8 at 262K, loading Whisper / ASR while the chat model is loaded is refused by the
  free-memory check, as it should be; loading it first, then the model, leaves it room.
## 2026-11-16: the Arc 140T preset on Alchemist cards (A770, A750, A580) by default

- **Why:** the Alchemist A-series has the 140T's 8-lane XMX (the 140T's Xe-LPG Plus cores are Alchemist's with XMX),
  but the 140T preset (H8 on `gemm_xw8`, 357 tok/s on the 27B's 4K prompt) was for iGPUs only, and the discrete
  cards' path needs 16-lane XMX. So an A770 took the plain 8-lane `gemm_x8` or DP4a, whichever won a start-time
  speed check, never H8 on the wide-tile GEMM (on a 140T `gemm_x8` was 1.6x slower than DP4a, issue #5, and the
  preset 1.2x faster than DP4a).
- **What:** discrete cards with 8-lane DPAS take the preset by default, for the models it covers on the 140T (Qwen
  dense), with the weights decoded just in time as there (`ARC_W8_PARTIAL_GB=auto` skips them;
  `zero/xe2/engine_xe2.h`, `zero/engine_load.h`). `ARC_140T=0`: the path before. The start-time checks still turn it
  off if its kernels differ or do not build. Unlike on the 140T, the H8 GEMM's split-K keeps the discrete cards' 34
  superblocks a piece (KL measured with both).
- **Not measured:** no one has run inferzero on an A-series card yet; A770 / A750 / A580 owners, a run with and
  without `ARC_140T=0` is welcome. iGPUs and the B-series are unchanged: the 125H's logits bit-identical before and
  after (Qwen3.5 4B, `--dump-logits`, prompt + 24 tokens).
## 2026-11-16: web search over four engines at once, merged as SearXNG does

- **Why:** the web UI's search asked Brave Search and, only when Brave gave nothing, Bing. A captcha or a rate limit
  left one engine's results or none. SearXNG (a Python program) gets better results by asking many engines and
  merging them; this does the same inside the server, with no new library and nothing to install.
- **What:** `web_search` asks Brave Search, Bing, DuckDuckGo (its HTML page) and Wikipedia (its API) at the same time,
  12 s at most each, and merges the results by page (the same address without `www.`, a trailing `/` or a fragment
  counts once). The ranking is SearXNG's: a page found by k engines at positions p1..pk scores k x sum(weight / p),
  Wikipedia at half weight (its full-text search often matches off-topic articles); the longer snippet is kept.
  `"engine"` still takes `brave` or `bing` alone, and now `duckduckgo`, `wikipedia` or a list (`bing,wikipedia`).
- **Bing:** its address now sets the market (`setlang=en-US&cc=US&mkt=en-US`). Without it Bing answered from some
  hosts with Office and Microsoft 365 pages whatever the query, which the old fallback passed on too.
- Not added: Mojeek (refuses automated queries with HTTP 403).
- Checked: the parsers and the merge on fixed pages (`tests/test_web_tools.cpp`); live on Windows (all four engines
  answered) and on Linux (Bing and Wikipedia; Brave and DuckDuckGo had started refusing this machine after the test
  runs, HTTP 429 and their bot check, and the merged search still answered from the other two). Not measured: the
  server's endpoint end to end (a user's server was running on the GPU).

## 2026-11-16: the Windows launcher with more than 2^31 bytes in a number (132 GB of RAM, issue #31)

- **Why:** with 132 GB of RAM the launcher stopped at its status update: "Cannot convert argument "val2", with value:
  "122933022720", for "Max" to type "System.Int32"" (gee456, issue #31). Windows PowerShell picks `[math]::Max` /
  `Min`'s Int32 overload for an integer literal next to a large number, and the GPU's memory share in bytes (half of
  132 GB) does not fit.
- **What:** those calls take doubles now: the GPU memory bar of `scripts/zero-launcher.ps1`, the download progress
  of `scripts/launcher-downloads.ps1` and `launcher-firstrun.ps1` (downloads past 2 GiB), and the free-space need of
  `scripts/zero-server.ps1`. Checked in Windows PowerShell 5.1 with the reported value: the old line fails as
  reported, the new one gives 50 %.
- `clinfo --all`: also every extension the device reports, one per line (a new GPU's features: an Arc B390's,
  Xe3, were asked for on issue #31).
## 2026-11-16: a split model keeps no experts in RAM by default (DeepSeek-V4.1 over 4 GPUs)

- **Why:** DeepSeek-V4.1-Flash UD-Q8_K_XL over manoreken2's 4 Pro B60s (1 TB of RAM, issue #4) ran out of GPU memory
  at its first prompt in every run: `runz --gpus 4` and serverz, with the expert cache at its default, 9 GB or 4 GB,
  and `--ctx auto`, while each card had only 13.5-20.8 of its 23 GiB in use. Each GPU's RAM tier (the experts read
  from the SSD kept in pinned host memory, a quarter of the free RAM each, ~240 GiB) was what grew at the first
  prompt. With `ARC_XRAM_GB=0` serverz answered: a 2,560-token prompt at 20.9 tok/s.
- **What:** a model split over several GPUs keeps no RAM tier unless `ARC_XRAM_GB=N` gives it N GiB per GPU, with a
  line saying so (`zero/engine_stream.h`). One GPU is unchanged. The values are the same: the experts come from the
  SSD instead of RAM. The size a split model can keep safely is still to be measured.
## 2026-11-16: `xw_peak --probe`, where the Arc 140T's prompt GEMM loses its time

- The 140T's best prompt GEMM (packed 32 x 16 tiles) runs at ~33 % of its 76.8 int8 TOPS, and larger tiles were
  slower although they need fewer loads per DPAS. `bin\xw_peak.exe --probe` times the GEMM's packed loop on data that
  stays in the first-level cache, for 32 x 16, 32 x 32, 16 x 32 and 16 x 16, three ways: loads and DPAS, loads only
  (bytes a clock per Xe-core), DPAS only. It says whether the loop is short of load bandwidth (then ~50 % is the most
  this kind of kernel gives) or waits on latency (then software pipelining is next). docs/xe2-tuning.md section 12.
- Laptop (125H, DPAS emulated): builds and runs; the loads alone reach ~84 B/clk per Xe-core there. A bench tool
  only: the engine is unchanged.

## 2026-11-16: `ref_forward REF_CMP`: the block-by-block comparison of the engine's dump, without a script

- `REF_CMP=engine.bin` next to `REF_DUMP=ref.bin` makes `tools/ref_forward` compare its residual streams with the
  engine's (`runz ARC_DUMP_ROWS=engine.bin`, the same layout) after the pass: per half-layer (DeepSeek-V4.1) or per
  layer, the relative L2 distance and the largest difference, so the first block where the engine leaves the
  reference shows at once. Before, the comparison was a Python snippet in an issue comment; nothing a user runs needs
  Python. Checked on the tiny synthetic DeepSeek-V4.1 (`make_tiny_deepseek41.py`): 20 half-layers, 0.4 % -> 2.7 %.
- The help of every packaged program is up to date (commit `2233bd6`): every option with its real default, `--help` /
  `-h` in the 11 tools that had none, Linux paths in the Linux build.
- The audit reports (`docs/*audit-2026*.md`, 15 files) are removed; this changelog keeps their findings.
## 2026-11-16: the GEMVs' rows per sub-group from the GPU's thread count on discrete cards (joshdev-de)

- **Why:** on a Pro B70 the 27B's codebook GEMVs read 340-450 GB/s where Q4_K's read ~530. Part of the gap was the
  rows per sub-group, chosen by weight type: they left the GPU's last round of sub-groups mostly empty. A B70 runs
  2,048 sub-groups at once; its 5,120-row matrices at 2 rows made 2,560 sub-groups (1.25 rounds), its 17,408-row gate /
  up at 4 rows 4,352 (2.1 rounds). Found and written by joshdev-de (issue #27, branch `b70-gemv-wave`), after
  oneDNN's kernel selector, which prices the full rounds and the part-filled last one.
- **What** (`zero/xe2/engine_xe2.h`; the thread count T from Level Zero, `zrt::hw_threads` in `zero/clrt.h`):
  - one token (`gemv`, `gemv_v`): T < N <= 4T rows take 4 rows a sub-group (one round), N > 4T one row (many rounds,
    the last one's waste small); N <= T and the output heads as before (`xe2_gemv_rows`);
  - 2-8 tokens (MTP verification, `gemv_mt`, `gemv_mt_v`): N > T rows take 3, 4 or 5 rows, whichever fills the last
    round best; where 4 rows fit one round, 4 below 6 tokens and 3 from 6 (`xe2_gemv_mt_rows`). On the 27B: `qkv` 5
    rows (2,048 sub-groups, one full round), `q` and gate / up 3. N <= T, the heads, Q6_K and the fused two-pass GEMV
    as before. 6 rows or more were slower everywhere.
- **Speed (the B70, 312 W, 27B IQ3_S):** plain generation 32.3 -> 35.2 tok/s (+9 %, median step 30.8 -> 28.2 ms);
  Q4_K_M 26.0 -> 27.8; MTP with 4 drafts (the default) 66.0 -> 70.6 through serverz (+7 %). The prompt is unchanged.
- **Accuracy:** bit-identical: a row's sums don't depend on which sub-group computes it (`--dump-logits` the same,
  plain and with 2, 4 and 7 drafts; KL 0.00146 either way).
- On discrete cards only; iGPUs unchanged (the laptop bit for bit). `ARC_GEMV_WAVE=0` turns both off,
  `ARC_GEMV_WAVE_MT=0` the 2-8-token part. A B60 (1,280 threads) gets other counts: not measured there yet.
- Tried and dropped: splitting K inside the work-group (oneDNN's few-token kernels). Only the 5,120-row matrices
  gained, below 6 tokens (~2 % of the GPU time), and the sums' order changes, so verification would no longer match
  plain decoding bit for bit.
## 2026-11-16: resident decoded weights in the room the context leaves, on by default on discrete cards

- **Why:** joshdev-de measured `ARC_W8_PARTIAL_GB=auto` on a Pro B70 (issue #27, 27B IQ3_S): the 4K prompt 2,490 ->
  2,694 tok/s at runz's 4K context (+8.1 %, 15.7 GiB resident, `h8_decode` 159 -> 64 ms) and 2,477 -> 2,578 at the
  server's 262K context (+4.1 %, 7.4 GiB); the logits bit-identical, `--ctx auto` the same 262,144 tokens.
- **What:** on discrete cards a model whose decoded weights don't all fit keeps as many layer matrices' W8 resident
  as the room after the model, its buffers and its context allows, without the switch (`zero/engine_load.h`).
  `ARC_W8_PARTIAL_GB=0` turns it off, a number keeps that many GB as before; iGPUs keep their weights just in time
  (a 140T was half as fast with it) unless a number is given; split models and MoE models as before.
- **The server's memory line** (`memory (--ctx auto)`) has a `resident W8` term now, taken from `left`: it said
  7.33 GiB left with 7.43 GiB of it in use by the resident copy (joshdev-de).
## 2026-11-15: `ARC_W8_PARTIAL_GB=auto` for discrete cards only

- On an Arc 140T (a Core Ultra 9 285H with 64 GB, PascalSoftwares, issue #5) `auto` kept 26.2 GiB of the 27B's
  decoded weights resident, 39.5 GiB on the GPU in all, and the 4K prompt fell from 340-357 to 176 tok/s: there the
  GPU's memory is the RAM Windows runs in. On iGPUs `auto` is now ignored, with a line saying so; a number of GB
  (`ARC_W8_PARTIAL_GB=4`) still works there. Discrete cards are unchanged.
- The same run's generation sweep (`decode_try.ps1`): no GEMV switch gains on the 140T (plain 6.05 tok/s; the read
  is 84 % of the measured 79 GB/s, ~74 % of the DDR5-5600's 89.6 GB/s). The website's 140T prompt bar is that run's
  357 tok/s (release 2026.11.15.11).
## 2026-11-15: `ARC_W8_PARTIAL_GB=auto`: resident decoded weights in the room the context leaves (opt-in)

- **Why:** a Pro B70 runs the 27B with its weights decoded again for every prompt (`W8 just-in-time`: raw + W8 are
  ~39 GB, more than its 32 GB) while only 13.3 GiB of the card is in use at a 4K context; `h8_decode` is 159 ms of
  its 4K prompt. By hand, 16 GB of resident W8 (`ARC_W8_PARTIAL_GB=16`) took that prompt 2,217 -> 2,358 tok/s
  (joshdev-de, issue #27).
- **What:** `ARC_W8_PARTIAL_GB=auto` (`w8_part_auto`, `zero/engine_load.h`) gives the layer matrices a resident W8
  copy, in load order, once the model, its buffers and its context are on the GPU, while the GPU share less the
  driver's reserve, the server's snapshots, the encoders' room and 0.5 GiB more leaves room. The automatic context
  (`--ctx auto`) is sized as before: its dry load takes none. A number (`ARC_W8_PARTIAL_GB=16`) works as before.
- **Laptop** (4B with `--budget 0`, 1,798-token prompt): first token 3,119 ms just in time, 2,140 ms with `auto`
  (176 matrices, 3.57 GiB), 2,096 ms fully resident. Logits identical in all three, and with a partial 1.5 GB; serverz
  with `--ctx auto` keeps its 262,144 tokens.
- Opt-in until a B70 and a B60 measure it.
## 2026-11-15: 4 MTP drafts a step by default for the 27B on discrete cards

- **Why:** joshdev-de measured the 27B's MTP drafts on a Pro B70 (issue #27, release 2026.11.15.09): through
  `serverz --mtp`, greedy, three 400-token answers, 4 drafts gave code 64.9, prose 53.4 and json 79.1 tok/s against
  57.9 / 49.6 / 79.1 with the default 7 (mean +6 %, code +12 %). Each draft past 3 adds ~6-7 ms of verification a
  pass on that card, and the later drafts are accepted less (`runz`: 73 / 60 / 92 % at 4, 61 / 53 / 86 % at 7).
- **What:** the default is 4 where it was 7 (the 27B) on discrete cards (`xe2_mtp_drafts`, `zero/xe2/engine_xe2.h`).
  iGPUs keep 7 (the laptop: 3 -> 7 drafts 12.6 -> 14.7 tok/s), other models keep theirs, DFlash2 keeps its 7, and
  `--mtp-drafts N` overrides it. The text is the same: speculative decoding keeps the main model's tokens.
- Also measured there: `ARC_GDN_CHUNK=3` (the chunked DeltaNet on XMX) is slower than `gdn_recur` on the B70 (295
  against 244 ms, the 4K prompt -3 %), so it stays off (docs/xe2-tuning.md).
## 2026-11-15: the prompt attention as flash attention on XMX on discrete cards (`attn_fx`, PR #29)

- **Why:** at long context the attention is most of a discrete card's prompt time: a Pro B70's 27B at 64K spent
  ~27 s of ~51 s in `attn_q8` (joshdev-de, issue #27).
- **What:** a new kernel, `zero/kernels/attn_fx.cl`, from joshdev-de's PR #29 (run on his B70). Each sub-group takes 8 query tokens, so every int8 DPAS has all 8 rows (`attn_q8`: one token x up to 4
  heads); the keys come in spans of 32, read straight from the int8 K / V cache with the next span prefetched into
  L1; 32 sub-groups a work-group. The same operands, online softmax, cache and output as `attn_q8`.
- **Where:** on by default on discrete cards with 16-lane XMX, for the plain causal prompt attention only. Sliding
  windows, images, Flash Next's sparse masks, Gemma and iGPUs keep `attn_q8`. Checked at start against `attn_q8` on
  random data; over 2e-2 of the output range, or a build failure, and `attn_q8` is used. `ARC_ATTN_FX=0` turns it
  off.
- **Speed (the B70, 27B IQ3_S):** attention 117 ms -> 45 ms at 4K, 1,706 -> 625 ms at 16K, 26.9 -> 9.6 s at 64K; the
  prompt +4 % / +17 % / +51 % (1,284 -> 1,943 tok/s at 64K).
- **Accuracy:** not bit-identical (P is quantized per 32 keys instead of 16). KL against the plain path 0.00152 ->
  0.00146 (4 chunks of 512), 0.00141 -> 0.00138 (one 16K chunk); greedy text the same, with and without MTP.
- Measured on one B70 with the 27B only; B60s and other head dims to confirm. The laptop's iGPU path is bit-identical.
## 2026-11-15: the H8 GEMM's wider work-groups and the GEMV's read-ahead on by default on discrete cards

- **Why:** joshdev-de measured both opt-in switches of 2026-11-15 on a Pro B70 (issue #27, `6466672`, 27B IQ3_S,
  312 W), each in alternated rounds, with a bit check of every setting.
- **`ARC_GEMM_XW_WGX=4`** (the H8 GEMM in work-groups of 4 column sub-groups): the 4K prompt 2,311 -> 2,396 tok/s
  (+3.7 % in both rounds; `gemm_down` 215 -> 191 ms). `2` gave +2.6 %; `LOCK=1` nothing alone and less with `WGX`;
  `8` -1.7 %. The default on discrete cards now (`zero/xe2/engine_xw.h`); `ARC_GEMM_XW_WGX=1` is the old one.
- **`ARC_GEMV_DB=1`** (the single-token GEMV one window ahead instead of the prefetch): generation 31.9 -> 32.3 tok/s
  (+1.4 % in all three rounds). The default for dense models on discrete cards now (`zero/xe2/engine_xe2.h`); MoE
  models keep the prefetch, whose +11 % on a B60's 35B was not measured against it. `ARC_GEMV_DB=0` is the old one.
- Logits identical on the B70 for every setting; iGPUs unchanged (the laptop's 4B bit for bit).
## 2026-11-15: a split model's sum waits count from the last expert read from the SSD

- **Why:** manoreken2's DeepSeek-V4.1-Flash over 4 Pro B60s (issue #4, release 2026.11.15.02) stopped in the cold
  first prompt of `runz --gpus 4 --bench-pp 512`: "GPU 1 did not bring a piece of its sums within 120 s (a GPU
  hangs?)". Nothing hung: every shard was reading its experts from the SSD at once (one B60 alone takes 47.7 s for
  that first run), and the failsafe counted from the start of the wait.
- **What:** `ExpertStore::read_ssd` counts every finished expert read in a process-wide counter
  (`g_expert_reads_done`, `zero/io_beat.h`), and the split model's waits (`TpComm::barrier`, `wait_all`) start their
  120-s limit (`ARC_TP_TIMEOUT`) again whenever it moved. A GPU that really hangs is still caught once the reads stop.
- Logits bit-identical (laptop, the 4B over 2 shards with `ARC_TP_SIM=1`, `--dump-logits` before and after).
## 2026-11-15: the single-token GEMV reads one window ahead, opt-in (after a look at urakozz's decode GEMV)

- **Why:** on a Pro B70 urakozz's plain 27B decode streams ~88 % of his measured bandwidth (int4 g64 weights,
  lane per output column, split-K for the 5,120-row matrices). Ours is at ~67 %, its FFN GEMVs at ~62 % (issues #12,
  #27). His loads stay in flight while he dequantizes. Ours, in `gemv`, issues each row's window reads and waits for
  them before decoding.
- **What:** `ARC_GEMV_DB=1` (any GPU): `gemv` (IQ3_S, Q4_K, Q5_K, Q8_0, ...) loads the next window (the next row's,
  or the next step's first row) into registers before it decodes the current one, in place of `ARC_GEMV_PF`. The
  same bytes, the same arithmetic: bit-identical. Off by default until a B-series card measures it.
- **Also:** the `ARC_GV_PROBE=1 / 2` measurement now covers every type on the `gemv` path, not only IQ3_S.
- **Checked** (laptop): logits bit-identical (the 4B and the 27B, prompt and generated tokens). The speed is the
  same within noise (4B 17.5-20.2 -> 18.6-20.3 tok/s, four alternated runs; 27B 4.70-4.77 -> 4.65-4.84). The probes
  say why: the laptop's Q4_K GEMVs are memory-bound (skipping the decode is no faster), and its IQ3_S ones are close
  to decode-bound (the decode alone, from cache, is nearly as slow as the streamed GEMV). By that ratio, on a B70
  (about 8x the bandwidth for ~4.6-5.8x the vector compute, worked out) the codebook decode, not the access
  pattern, would cap those GEMVs at roughly 60-80 % of bandwidth, around the ~62 % measured. Builds for Battlemage (ocloc `bmg`) without spills. Comparison:
  docs/xe2-tuning.md section 10.

## 2026-11-15: the wide-tile H8 GEMM in larger work-groups, opt-in (after a comparison with urakozz's engine)

- **Why:** on a Pro B70 our prompt GEMMs run at ~62 % of the XMX int8 peak, urakozz's at 72-81 %
  (urakozz/intel-arc-inference-server, Apache-2.0, `src/kernels/prefill/pf_int8.cl`). Its work-group is 32 sub-groups
  over 256 tokens x 128 columns, kept in step each 64-value chunk by a split barrier, so the work-group shares both its
  activations and its weights in L1. Ours stacks 8 sub-groups on the tokens only (256 x 64, not in step), so every
  64-column block reads the same activation rows again.
- **What** (`gemm_xw.cl`, `zero/xe2/engine_xw.h`, 16-lane XMX only):
  - `ARC_GEMM_XW_WGX=2 / 4 / 8`: that many column sub-groups per work-group beside the token ones, each sub-group
    computing what it did before. Where a matrix's column blocks do not divide, the widest count that does; fewer
    token sub-groups where the work-group would not fit a Xe-core (32 threads at 256 registers).
  - `ARC_GEMM_XW_LOCK=1`: a split barrier around each 64-value chunk.
  - With `ARC_GEMM_XW_PF`, the column sub-groups share the activation prefetch.
  - Off by default until a B-series card measures it: `.\bench\xw_sweep.ps1 -Model <gguf> -Group`.
- **Checked** (laptop, DPAS emulated: `ARC_H8=1 ARC_GEMM_XW_TRY=1 ARC_GEMM_XW_LANES=16`, Qwen3.5 4B, a 703-token
  prompt): logits identical to the default for WGX 2 / 4, LOCK, WGX 4 + LOCK, WGX 8 + LOCK and WGX 2 + PF 3; the
  start-up check, which now has a shape of two whole work-groups of column blocks, passed each time. All variants
  build for Battlemage (ocloc `bmg-g21`, 256 registers, 2D loads). Speed: not measured (no XMX here). Without the
  switches the kernel compiles as before; the iGPU's default path does not run it.
- **Also found:** urakozz's DeltaNet is not ahead of ours. His `gdn_scan` row is the last of the chunked form's ten
  launches. His six recurrence kernels take ~155 ms per 2,048 tokens (his probe of 2026-11-10), ~310 ms at 4K, against
  `gdn_recur`'s 263 ms on a B70 (issue #27).

## 2026-11-15: no Python in the release packages: the 35b-gsq conversion and the AutoRound import in C++

The packages hold no `.py` file any more, and no step of setup, the launchers, the downloads or the tests a user
runs needs Python, on Windows as on Linux.
- **`bin\convert_gsq.exe`** (`tools/convert_gsq.cpp`) replaces `tools/convert_gsq.py` for `setup -Models 35b-gsq`;
  setup downloads the checkpoint with `scripts\hf-get.ps1` (curl) instead of `huggingface_hub`. On fake GSQ
  checkpoints, one with the real 4B vocabulary (109.5 MB out), it writes the same GGUF as the Python script, byte for
  byte, and refuses the same bad inputs with the same messages. `A = -exp(A_log)` follows NumPy's float32 `exp`
  kernel (AVX2 + FMA), checked against NumPy over all 2^32 float32 inputs: 0 differences.
- **`bin\import_autoround_experts.exe`** (`tools/import_autoround_experts.cpp`) replaces
  `tools/import_autoround_experts.py` for `setup -Models flash-ar4`: byte-identical expert files on fake checkpoints
  (3 layers x 64 experts, 535 MB out; 2 x 4), the same checks (SHA-256 of every shard, the pinned revision, every
  code and scale decoded again) and refusals; about 1-3 s against ~10 s for the Python on the same data (the
  shards hashed ahead on two threads, the experts converted on every core). A failed run removes its partial file.
- **`bench\report.ps1`** replaces `bench\report.py` (`bench\run_all.ps1`'s tables, the same output byte for byte on
  `bench\results\results.jsonl`); **`tests\test_tools.ps1`** replaces `tests\test_tools.py` (the server API check of
  `tests\run_tests.ps1`: tool call, then prefix reuse; checked against serverz with the 4B).
- **The packages** (`scripts/make-package.ps1`, `scripts/make-package-linux.sh`) copy every tracked file but the
  `.py` ones; the Windows package adds the two converters to `bin\`. The developers' Python scripts (the Jinja2
  template checks, simulators, table generators) stay in the repository; `run_tests.ps1` skips the two Jinja2 checks
  where they are absent.
- Docs: `docs/install.md` (the download table with `get-models.ps1` / `hf-get.ps1`, the GSQ conversion by hand),
  `docs/autoround.md`, setup's help, the catalog's texts. CI builds the converters.
- The engine and the kernels are unchanged: the same logits on every GPU.
## 2026-11-15: no Python on Linux either: the launcher, its downloads, tests and the benchmarks in bash

inferzero is a standalone engine: nothing a user runs to install or use it needs Python (the README's promise for
Windows). The Linux launcher and the benchmark ports of the last two days broke that; they are bash now.
- **`zero-launcher.sh`** is the whole launcher (bash and the system's tools: curl, sha256sum, coreutils), with every
  feature and fix of the Python one it replaces (`scripts/zero_launcher.py`, deleted): `--check`, `--list`,
  `--download`, `--model`, the port check, the readiness wait, `--install-service` with `--service-name` /
  `--replace-service` (an existing unit never overwritten, refused before any download, a file system without hard
  links handled, `$` kept in `Environment=`).
- **Downloads with curl** into the Hugging Face cache layout (`snapshots/<commit>/<file>`, `refs/main`), resumed from
  `.part`, each file's size and, for LFS files, its SHA-256 taken from the repository's own headers (`X-Repo-Commit`,
  `X-Linked-Size`, `X-Linked-Etag`: no JSON to parse), the free space checked first, `HF_TOKEN` for gated
  repositories. `scripts/hf-download.py` (unused now) is deleted.
- **The benchmarks:** `bench/website_bench.sh`, `bench/b60_sweep.sh`, `bench/decode_try.sh`, the Linux twins of the
  `.ps1` files (the same runs, records and printed output); the Python ports `bench/*.py` of 2026-11-15 are deleted.
- **Tests:** `tests/test_zero_launcher.sh` (13 cases, as the Python tests) and `tests/check_catalog_sync.sh` (the
  launcher's catalog against `scripts/model-catalog.ps1`) in CI's `linux-package` job, which also syntax-checks the
  four bash scripts.
- Checked in WSL2 (Ubuntu 24.04): the launcher's tests 13 / 13, the catalog 8 / 8; a download of a plain and an LFS
  file into a scratch cache (SHA-256 checked; a second run finds both there); the 4B started from the cache, `/health`,
  the web UI, a chat answer, a busy port refused, SIGTERM ending the server; `decode_try.sh` against the real `runz`,
  every row parsed.
- Still Python, and optional on Windows: the `35b-gsq` conversion and the `flash-ar4` AutoRound import (NumPy);
  both C++ since the entry above.

## 2026-11-15: a self audit and a DeepSeek audit of the day's changes; `website_bench` and `ref_forward` on Linux

Changes since release 2026.11.14.13 reviewed twice, separately, by DeepSeek-V4.1 through Pi and by a second reviewer.
Both found the `gdn_recur` prefetch paths bit-identical, in bounds, and the iGPU path unchanged. Fixed from their
findings:
- **`bench/b60_sweep.py` / `.ps1`:** the H8 baseline's split-K was still 17 after the default became 34, so the sweep
  re-ran the default as a "change" and never measured 17. Its base is now 34 on discrete cards; `--quick` measures 34
  too (both reviewers).
- **`h8_splits` (`zero/engine_h8.h`):** cached the first GPU's iGPU / card status, so in a split over an iGPU and a
  card one of them got the other's default; now only the variable is cached. `ARC_GEMM_SPLIT` is clamped to 0..4096
  in its three readers (a huge value overflowed the piece count and left the output unwritten).
- **The launcher:** `--install-service` refuses an existing unit before downloading a model, not after; a file system
  without hard links falls back to an exclusive create instead of a traceback; `Environment=` lines keep `$` (systemd
  expands it in `ExecStart=` only). Three tests added (`tests/test_zero_launcher.py`: 13).
- **Docs:** `ARC_GEMM_SPLIT`'s new default is for the H8 GEMMs only; `ARC_GDN_L1PF=1` is off and `ARC_GDN_PF` >= 2
  makes it do nothing; each B70's L1PF result is attributed to its card; the site's header named the old release; the
  README and site's B70 numbers are now the Linux card's 2,315 tok/s (issue #27).
- **Linux:** `tools/ref_forward.cpp` called `_fseeki64` (Windows only): now `plat::fseek64`, so the f32 reference builds
  with g++. `build.sh` and the Linux package now hold `ref_forward` and `gpu_bench` too.
- **`bench/website_bench.py`:** the website benchmark in Python for Linux and Windows (the same phases, records and
  tables as the `.ps1`), its default models now also Gemma 4 31B and DeepSeek-V4.1-Flash IQ2_XS.
- Not fixed, noted: the split-K 34 default's accuracy was checked against the plain path (KL 0.00152), not yet against
  the f32 reference; asked for on a B70.

## 2026-11-15: the H8 GEMMs' split-K in larger pieces on discrete cards (+3 % on a B70's 27B prompt)

- `ARC_GEMM_SPLIT` defaults to 34 superblocks for the H8 GEMMs on discrete cards (was 17; the plain W8 and the
  x16 GEMMs keep 17, so models without H8 there, the MoE ones and Gemma 4, do not change): the 27B's FFN down projection (K = 17,408)
  runs in 2 split-K pieces instead of 4. A Pro B70's 4K prompt +3.1-3.2 % in three interleaved rounds (2,244 ->
  2,315 tok/s; `gemm_down` 253 -> 214 ms). Not bit-identical (each piece is rounded on its own): KL against the
  plain path 0.00145 -> 0.00152 (max 0.039 -> 0.052), same top token 97.55 -> 98.14 % (joshdev-de, issue #27).
  iGPUs keep 17 and their bits; `ARC_GEMM_SPLIT=17` restores the old default.

## 2026-11-15: `runz --kl-ref` with `--feed-step` scores the right positions (issue #23, chumbleyjl)

- With `--feed-step N` and a reference of every position, `runz` scored the last piece's rows against the
  reference's first positions, then stopped at `logits_at: row 15 is not one of the 15 the last pass left`: the first
  readable position was counted from 0, not from the last piece's start. Now it is, and a reference that covers
  positions no longer readable says which ones are (`ARC_KL_FROM=N` with `ref_forward REF_ALL_FROM=N`); a reference
  of the wrong size says so too (before: nothing printed). Tooling only: no engine change.

## 2026-11-15: the DeltaNet recurrence's L1 prefetch on by default on discrete cards (+1.4 % on a B70's 27B prompt)

- `ARC_GDN_L1PF=2` is the default on discrete cards (`zero/xe2/engine_xe2.h` `xe2_gdn_l1pf`): a Pro B70's 27B 4K
  prompt, `gdn_recur` 293 -> 263 ms, 2,216 -> 2,247 tok/s, the logits identical over 18,566 tokens (joshdev-de,
  issue #27). `ARC_GDN_L1PF=0` turns it off. iGPUs: unchanged. `ARC_GDN_PF=2` stays off (slower on the B70).

## 2026-11-15: the GPU sweeps in Python, for Linux and Windows alike (issue #27)

- `bench/b60_sweep.py` and `bench/decode_try.py`: the same runs, order, variables, best-of and 3 % rules, KL and
  token checks and printed output as the `.ps1` versions, standard library only (`bin/runz` on Linux,
  `bin/runz.exe` on Windows). Linux owners of discrete cards could not run the `.ps1` sweeps (joshdev-de asked for
  one script for both systems). The `.ps1` files stay, with a line that the two must stay the same. Checked:
  `decode_try.py` on the laptop's 4B, every row parsed as the `.ps1` prints it.

## 2026-11-15: the DeltaNet recurrence's loads further ahead, opt-in (`ARC_GDN_PF`, `ARC_GDN_L1PF`; the B70)

- A B70's 4K prompt on the 27B spent 292 ms in `gdn_recur`, as a B60 did (289 ms), with 60 % more Xe-cores (issue
  #27, joshdev-de): each token's step waits for its q / k / v loads, about 1 us of global memory latency, against
  ~0.2 us of arithmetic, and the kernel asked for them only one token ahead.
- **`ARC_GDN_PF=2`** (up to 8): the operands of that many tokens in registers. On a B70's 16-lane build no spill at
  2, 15 KB of spill at 4 (ocloc, Battlemage), so 2 is the one to try there. On the laptop's 8-lane build it spills
  already at 2: `gdn_recur` 238 -> 1,394 ms (4B, 4K prompt).
- **`ARC_GDN_L1PF=D`** (e.g. 4, 8, 16): token t+D's k / q / v rows prefetched into L1 with
  `intel_sub_group_block_prefetch`, no registers; only where the driver has `cl_intel_subgroup_buffer_prefetch`
  (Battlemage yes, no spill; Meteor Lake no, where the switch does nothing).
- Both keep the arithmetic and its order: logits bit-identical (4B, 3,584 tokens: `ARC_GDN_PF=2` and `=4` against the
  default). Off by default; the B70 owners measure them.

## 2026-11-14: Linux service installation preserves existing units

- `--install-service` refuses to overwrite an existing unit unless
  `--replace-service` is explicit. `--service-name` supports a separate instance.
- Unit writes are atomic, do not follow existing symlinks, and reject line
  breaks in cache environment values. Ten model-free tests cover these cases.

## 2026-11-14: builds with GCC 16 (issue #27, joshdev-de)

- GCC 16's libstdc++ no longer brings in `<climits>` through other headers, so `INT_MAX` was undeclared in
  `zero/engine_drafters.h`, `zero/mgpu/engine_tp.h`, `tools/server.cpp` and `tools/server_chat.h` (Fedora 45).
  `#include <climits>` in `zero/engine.h` and `tools/server.cpp`. No code change: Windows (MSVC) and g++ 13 build
  as before.

## 2026-11-14: a Linux package and a terminal launcher

- **`inferzero-<version>-linux-x64.tar.gz`** (`scripts/make-package-linux.sh`): the tracked files plus `bin/runz`,
  `bin/serverz` and `bin/mgpu_probe` built on Ubuntu 24.04 (g++ 13), for Ubuntu 24.04 and newer. Needs Intel's GPU
  runtime, libgomp and Python 3; [docs/install.md](docs/install.md#linux) has the steps. `./build.sh` builds the
  same programs from the sources.
- **`./zero-launcher.sh`** (`scripts/zero_launcher.py`, standard library only): a terminal menu with the Windows
  launcher's steps. It checks the system (`--check`: CPU, Level Zero, OpenCL, libgomp, Intel's driver, the GPU node
  and its `render` group, RAM), lists the catalog (the same models as `scripts/model-catalog.ps1`, kept equal by
  `tests/check_catalog_sync.py` in CI), downloads with `scripts/hf-download.py`, starts `serverz` and opens the web UI
  once it answers. `--model KEY|FILE`, `--download`, `--port`, `--host`, `--gpus`, `--install-service` (a systemd
  user unit). Not offered there: Flash Next's AutoRound import and the speech models.
- **CI**: a `linux-package` job builds the package on Ubuntu 24.04 and runs what needs no GPU.
- **The website and the README** say what runs on Linux.
- **Reviewed before release** by DeepSeek-V4.1 through Pi and by a second reviewer, separately; fixed from their
  findings:
  - the wrapper changed folder, so a relative `--model` path, and a symlinked launcher, broke;
  - "downloaded" looked only at `refs/main`'s snapshot, so a model could be fetched again;
  - no port check, so a server already on 8090 answered for the new one;
  - `--host ::` was never reported ready (the server's IPv6 socket is IPv6-only);
  - a Gemma drafter's copy was not atomic;
  - the free-space check ignored what was already downloaded;
  - the browser shared the server's process group and closed with Ctrl+C;
  - SIGINT sent to the launcher alone did not reach the server;
  - the systemd unit: no cache variables, no line-break check, `XDG_CONFIG_HOME`;
  - `$XDG_CACHE_HOME` was not honoured (the engine does; `hf-download.py` too, on Linux only);
  - libgomp not checked;
  - `-ldl` for glibc before 2.34;
  - the archive held the builder's uid;
  - the package cleanup could delete other folders in `dist/`;
  - `--release` without a value was ignored;
  - a rebuild in a package kept its `PREBUILT.txt`;
  - CI piped `--help` into `head` without `pipefail`.
  Also found by the self audit: Ctrl+C was sent to the server twice, a proxy could catch the health check, the
  exit code of a server killed by a signal, and WSL2's `/dev/dxg` not counted as a GPU.
- Windows: nothing changes (no engine, kernel or Windows script touched, except `hf-download.py`'s Linux-only cache
  path).

## 2026-11-14: an AutoRound package refuses an expert file from another checkpoint (PR #25, chumbleyjl)

- The package's `inferzero.autoround.revision` (the checkpoint's 40-digit SHA) is checked against the expert file's
  fingerprint, which the importer sets to the SHA's first 16 hex digits (src/autoround_revision.h): an expert file of
  the same shapes from another import used to load without a word and decode other weights. Both loaders check it,
  the engine (zero/expert_store.h `valid`) and tools/ref_forward. A malformed or missing revision is refused too;
  packages without AR4_128 experts and plain models are not affected. Found by the pair audit of 2026-11-14.
- `tests/test_autoround_revision.cpp` (no model, no GPU): 23 checks, in CI.
- Checked by chumbleyjl on the B70: the matching package's logits bit-identical before and after. Checked here: see
  the merge's commit.

## 2026-11-14: Linux: AutoRound packages open, expert reads into host memory, split models freed in turn (PR #24, chumbleyjl)

- **The expert file's layer table compared field by field** (zero/expert_store.h `LayerInfo::operator==`), not with
  `memcmp`: its 4 padding bytes after the three types are not set on GCC, so a matching file could be refused (the
  AutoRound package on chumbleyjl's B70) or a plain model's expert file written again. The file layout is unchanged.
  `tests/test_expert_layout.cpp`.
- **Linux: an expert read into Level Zero host memory that fails with `EFAULT`** (`O_DIRECT` cannot pin that memory
  on some drivers, though the CPU can write it) is read through a 4 KB-aligned buffer of the read slot, then copied
  (src/platform.h `AsyncFile`). It goes with this audit's realignment of short `O_DIRECT` reads: the staged bytes are
  copied before the read advances. `tests/test_platform_read.cpp`, built and run by a new Linux job in CI.
- **Linux: a split model's shards are freed one at a time**, each on its own thread (zero/mgpu/engine_tp.h
  `shutdown`): Intel's Linux runtime crashed while two shards freed their kernels at once (B70, `ARC_TP_SIM`). With
  this audit's frees: through pointers taken out first, one limit each, and a free given up on does not stop the
  others'. Windows frees them together as before.
- Checked by chumbleyjl on the B70 (Linux, GCC 15): the AutoRound package opens and runs; the two-shard run ends
  normally. Checked here: see the merge's commit (logits bit-identical on Windows, the tests on both systems).

## 2026-11-14: serverz keeps the startup model's context when it loads it again (PR #26, chumbleyjl)

- A model started with an explicit `--ctx` (262,144 on chumbleyjl's B70) came back at its automatic context (131,072)
  when it was loaded again: after `/v1/server/unload` and a request naming it, or when a failed switch to another
  model restored it. Both paths set `--ctx auto` for every model; now only for models other than the startup file,
  as the startup `--dflash` drafter already was (tools/server.cpp). Other models still choose their own context; a
  startup `--ctx auto` stays automatic. No engine or kernel change. Checked on the B70 (Linux): 262,144 after both
  paths, the same reply to the same request before and after.

## 2026-11-14: Q5_1 weights on the GPU (Flash Next UD-Q4_K_XL's down experts)

- Unsloth's UD-Q4_K_XL Qwen3.8 Flash Next keeps its down experts in Q5_1, which the engine refused ("unexpected expert
  tensor blk.0.ffn_down_exps.weight"; PR #20 had added only the CPU reference decoder; issue #23, chumbleyjl).
- Q5_1 (w = d q + m, q = 0..31) stays in its GGUF layout on the GPU and is decoded per 32-block into the per-32 int8
  form every kernel already reads. That form has no offset, so each block is requantized with its own scale
  max(|m|, |m + 31 d|) / 127: at most half a step off, ~0.4 % of the block's largest value, as Q4_K / Q5_K are
  requantized for the min-free GEMM. Prompt and generation decode through the same function and see the same values.
  Rows of K = 640 are padded with zero blocks as the other small-block types' (src/quants.h, zero/kernels/decode.cl).
- Checked on the tiny DeepSeek-V4.1 (`tools/make_tiny_deepseek41.py --q5_1-down`, every expert used) against
  tools/ref_forward, which decodes Q5_1 exactly: KL 0.00363 prompt / 0.00362 token by token against the Q8_0
  control's 0.00340 / 0.00343. Not checked here: Flash Next itself (the 640-wide rows; its file is not on the
  laptop): its KL against f32 is the next step, on a machine that has it.

## 2026-11-14: a pair audit (DeepSeek-V4.1 through Pi and a second reviewer) and its fixes

Six parts reviewed twice by reviewers that did not see each other's results: the engine since the morning's audit, the
tools and scripts, the contributor pull requests, the Linux port, the web UI and the multi-GPU failsafes.
docs/audit-2026-11-14-pair.md lists every finding, who found it, the verdict and the
fix. The logits stay bit-identical; nothing found changes a model's results.

- **Multi-GPU failsafes:**
  - **Unloading a split model:**
    - after a failure, a shard's free that took over 10 s was given up on, and its thread then wrote into the freed
      model;
    - without one, a GPU that stopped while freeing hung the unload;
    - now one limit for the frees, and a thread given up on touches nothing of the model when its job ends.
  - **Waits:** the overlap's sum thread and the expert prefetch's copies now have a limit.
  - **The link test:** a timed copy that does not finish keeps its buffers.
  - **The direct sums' inboxes** no longer take the first pass's sum buffers' room.
  - **Smaller:** `~XCache` throws nothing; a leaked model's caches no longer shadow GPU 0's.
- **Precise mode and AR4:**
  - precise mode is per engine;
  - the system-prompt cache and saved conversations are off in precise mode, where the server had deleted files the
    model saved in its fast mode;
  - the `--ctx auto` key records precise mode;
  - a Flash Next shape other than `hc_n × n_embd = 10240` is refused.
- **DeepSeek-V4.1:**
  - `ARC_TP_ATT_QSPLIT=1` no longer applies to it;
  - a `candidate_block_size` of 0 is refused.
- **Scripts:**
  - `setup -Yes` was reset by the AutoRound script (its functions moved to `scripts/autoround.ps1`);
  - deleting the shards resolved symlinks against the wrong folder;
  - free-space checks broke UNC and relative cache paths;
  - `HOME="~"` looped at server start.
- **Linux:**
  - the terminal ignored its timeout while output kept coming;
  - file dates came out around 1453;
  - ffmpeg allowed every protocol on untrusted bytes;
  - an MCP child that stopped reading hung its stop;
  - PDF page counts were read from anywhere in pdfinfo's output;
  - FIFOs blocked a model's open.
- **Web UI:**
  - a deleted chat came back through its queued messages;
  - messages typed during Compact were never sent;
  - the folder picker failed on Linux;
  - images a tool showed split turns;
  - Enter during an IME composition sent the message;
  - and smaller ones.
- **Left on purpose:**
  - the layer table's padding compare (pull request #24 fixes it; both reviewers judge it safe to merge);
  - the Windows half of the MCP stop deadlock;
  - Linux children outliving a killed server (documented).

## 2026-11-14: Flash Next with Intel's AutoRound INT4 experts in one step (`setup -Models flash-ar4`)

- **`scripts/get-autoround.ps1`** (and `setup -Models flash-ar4`): Intel's AutoRound checkpoint is safetensors, so it
  is imported: the expert shards at the importer's pinned revision (~78 GB; the index says which hold experts, the
  102 GB shard without any is skipped), `model-info.json` from the Hugging Face API (every file's size and SHA-256,
  which the importer checks), `tools/import_autoround_experts.py`, Unsloth's UD-Q4_K_XL (~111 GB) for the rest and
  its image projector, then `bin/import_autoround_pack.exe` (now in the release package) writes the package where
  the server lists it. The AutoRound shards are deleted afterwards when the user agrees (`-DeleteSource`).
  `scripts/hf-get.ps1`: `-Revision` for a pinned commit (refs/main left as it was).
- **Precise mode by itself** (zero/engine.h `FnPreciseHold`): a model with AR4_128 experts turns Flash Next's precise
  mode on while it is loaded, without `--precise`; unloaded, it is off again, so the GSQ Flash Next loaded after it
  in the same server runs in its default mode. `engine_flashnext.h`'s mixer mode is no longer cached once per process.
- **The model list** (tools/server_catalog.h): an AutoRound package keeps the UD-Q4_K_XL it was made from listed (it
  is another model; a package made with `--pack` still hides its source); its alias `qwen3.8-flash-next-autoround`,
  shortcut `flash-ar4` (zero-pi too). Tested with metadata-only GGUFs in a temporary data folder against release
  2026.11.14.10, which hid the UD-Q4_K_XL.
- Checked here: the pinned download and `model-info.json` against the API, the importer's checks on them, the shard
  list from the real index (15 files), the scripts parse. Not run here: the full download and import (~290 GB of
  disk); to be run by a tester.

## 2026-11-14: the per-type lists in one place; the AutoRound files tidied

- **One list per kind of type** instead of a copy in every place that needs it (adding MXFP4, MXFP8 and AR4_128 meant
  editing about ten lists, and pull request #22's merge conflicted in six of them): `is_per32_type()` in
  `src/quants.h` (the per-32-scale types, used by `w8_variant`, `gpu_regrouped` and `moe_rg_type`), with its kernel
  twin `PER32`, and `VB_TYPE` in `kernels/decode.cl` (the types with a vector decoder, used by `gemv_v` and the
  experts' `MOE_VB`).
- `kernels/ar4.cl` folded into `kernels/decode.cl`: one `AR4_128` definition next to the other type ids (it was in
  both), and `bench/gemv_bench` / `bench/moe_bench`, which read `decode.cl` alone, now get `ar4_unpack` too.
- `tools/import_autoround_pack.cpp`, `tests/test_autoround_decode.cpp` and `tests/test_autoround_package.cpp`
  rewritten in the project's style (the same checks; usage and exit codes explained, a message on each failure).
- Logits bit-identical to release 2026.11.14.10 (Qwen3.5 4B, Qwen3.8 27B, Qwen3.6 35B-A3B, Flash Next IQ3_S); the
  AR4 decode test passes; the expert kernels build for AR4_128, Q4_0, IQ4_XS and MXFP4.

## 2026-11-14: a split model's GPU out of memory shrinks its expert cache; AR4 experts on iGPUs

- **Out of device memory on a split model** (zero/xcache.h, zero/engine_stream.h): a kernel launch or allocation that
  runs out of memory on one of the GPUs now shrinks that GPU's expert cache and retries, as on one GPU. A split had
  no such hook, so the first prompt of DeepSeek-V4.1 over 4 B60s at 262K ended serverz (layer 27, `ds_expand`, with
  the cache planned to 0.01 GiB of the limit; issue #4). Tested with `ARC_FAKE_OOM=400`: Flash Next over 2 GPUs
  (`ARC_TP_SIM`), both caches 5.99 -> 5.00 GiB, the run finished and its logits are bit-identical to the run without
  it. One GPU unchanged.
- **The hook after a review by DeepSeek-V4.1 (through Pi):** one hook installed once (`std::call_once`) and never
  cleared, which finds the cache of the failing thread's GPU (a destroyed cache could clear another engine's hook;
  shards loading together wrote it at once); an ARC_BATCH flush of another GPU's list no longer shrinks from the
  wrong thread; `shrink()` does not re-enter when its own copies run out of memory. `ARC_FAKE_OOM=400`, Flash Next:
  split over 2 and one GPU, the caches shrink, logits bit-identical to the runs without it. CI builds and runs
  `tests/test_autoround_blocks.cpp`.
- **AR4_128 experts on iGPUs** (zero/engine_stream.h): the iGPU refusal removed (Xe-LPG / Xe-LPG+ / Xe3 have strong
  int4 DP4a paths). On the laptop's Xe-LPG: `tests/test_autoround_decode` on synthetic AR4 data (4.9 M weights),
  GPU decode bit-identical to the CPU decoder; the expert kernels (`moe_gateup_q`, `moe_down_q`, `moe_rg`,
  `decode_w8_sb` / `_sl`, `gemv_v`) build for AR4. A whole AR4 model on an iGPU is still to be run (the import does
  not fit the laptop's SSD). Still precise Flash Next only.

## 2026-11-14: AutoRound INT4 experts (AR4_128, pull request #22 by chumbleyjl)

- An experimental private AR4_128 importer (`tools/import_autoround_experts.py`, `tools/import_autoround_pack.cpp`),
  CPU/reference and generic native decode plumbing (`ar4_unpack` in `kernels/decode.cl`), plus source/package checks. Existing model
  defaults kept. Downstream AutoRound-hybrid benchmarks and accuracy limits are separate from upstream-branch
  validation; see docs/autoround.md.
- Merged with main's MXFP4 / MXFP8 types (both kept). AR4 experts also run on a split model (`--gpus`): each GPU
  streams its experts as on one GPU; and on iGPUs (the laptop: AR4 GPU decode bit-identical to the CPU decoder on
  4.9 M synthetic weights, the expert kernels build for AR4). Still precise Flash Next only.
- Logits bit-identical to release 2026.11.14.10 for the other models (laptop: Qwen3.5 4B, Qwen3.8 27B, Qwen3.6
  35B-A3B, Flash Next IQ3_S; prompt + 8 generated tokens). The AR4 path's GPU KL on this branch is asked of the PR's
  author (B70), one GPU and split.

## 2026-11-14: `h8_decode` reads each superblock once and covers the per-32 types; part of the decoded weights resident (opt-in)

- **`h8_decode`** (H8's decode + rotation in one pass, the B60 prefill plan's step 13.3): a work-group's 16 rows of
  superblocks are read once, in row order, into local memory, and each work-group does 4 rotation blocks (its codebook
  copied once for them). Before, each lane loaded a whole superblock into registers to decode one or two of its eight
  32-blocks. The laptop's 27B, 2K prompt, `ARC_H8=1`, interleaved runs: 1,180 -> 876 ms (the kernel alone; against
  release 2026.11.14.09 in back-to-back pairs 924 / 938 -> 872 / 871 ms). `ARC_H8_DEC_PRIV=1`: the old loads;
  `ARC_H8_DEC_R=1 / 2 / 4 / 8`: blocks per work-group.
- It now also takes the per-32 float types (Q8_0, Q4_0, IQ4_NL, Q2_0, MXFP4), which kept two passes (the 4B's Q8_0
  matrices: 33 -> 57 fused decodes per 2K prompt). The requantized types (Q2_K / Q4_K / Q5_K) keep two passes: their
  requantization repeated in the fused kernel gave ~2 values in a million 1 apart (the rounding of decode_w8's division
  and multiply-adds under `-cl-mad-enable`), so they are not bit-identical there.
- **`ARC_W8_PARTIAL_GB=x`** (step 13.6, opt-in): with the weights decoded just in time, the layer matrices loaded first
  keep their decoded copy resident up to x GB, so a prompt chunk decodes only the rest. The laptop's 27B with 4 GB (62
  of 400 matrices): 512-token prompt 98.1 -> 101.2 / 102.6 tok/s. A B60's best size is to be measured (issue #4).
- Logits bit-identical: H8 against the two-pass path, the old kernel and release 2026.11.14.09 (27B, 4B with
  `--budget 0`), the partial residency against none (27B, with and without H8), the default paths against .09.

## 2026-11-14: DeepSeek-V4.1's context memory counted right; the split overlap with resident weights

- **DeepSeek-V4.1's context memory** (`--ctx auto`, the memory lines): counted by its own caches (the KV-source
  layers' compressed rows and index keys, the indexer's score buffers, zero/engine_ds41.h) instead of the Qwen
  formula, which took 10.1 GiB at 262K for ~1.0 GiB real, so `--ctx auto` chose a context far below what fits. Tiny
  model, 4K -> 131K: 0.625 GiB counted, 0.62 measured.
- **A split model's expert cache leaves room for the sum buffers** its first pass allocates (~max_chunk x n_embd x
  10 bytes; only a dry load counted them). DeepSeek over 4 B60s at 262K had its cache planned to 0.01 GiB of the
  limit and ran out of device memory at the first prompt (manoreken2, issue #4). Whether this was the whole cause
  is for his rerun to show: the GPU's own count still had ~2.2 GiB free at the error.
- **`ARC_TP_OVERLAP=1` on a dense split model with resident weights**: its prompts longer than a chunk now go
  layer by layer, so the sums overlap the next chunk's work; before, chunk-by-chunk passes left it nothing to do
  (the 27B over 4 B60s at 64K: 1,169 tok/s off, 1,172 on). `ARC_TP_LM=1`: layer by layer without the overlap.
  Logits bit-identical either way (4B over 2, `ARC_TP_SIM`, 3,584 tokens); the default and one GPU unchanged.
  Speed: the B60s to measure.

## 2026-11-14: the XMX prompt GEMM prefetches its weights a chunk ahead by default on B60 / B70 (+16 % prompt)

- `ARC_GEMM_XW_PF` now defaults to 1 (the weights) and `ARC_GEMM_XW_PFD` to 1 (one chunk ahead) on 16-lane XMX cards.
  A Pro B60's `bench\xw_sweep.ps1` (manoreken2, issue #4, release 2026.11.14.05), the 27B IQ3_S: 4K prompt 1,321.7 ->
  1,529.0 tok/s, 512 tokens 737.3 -> 867.2, logits identical (the prefetch changes no arithmetic). `=0` turns it off.
  iGPUs, the 140T's 8-lane kernel and generation: unchanged.

## 2026-11-14: DeepSeek-V4.1-Flash over several GPUs (`--gpus N`)

- Its routed experts are split over the GPUs (each caches and computes its own range, as Flash Next's), everything
  else whole on every GPU: the attention (one kv-head), the indexer, the hyper-connection mixing, the shared expert's
  part on GPU 0; the Engram rows are read on the host by each. The MoE's partial outputs are summed into the block
  output (hco_) before the mixing. The experts in RAM are one pinned pool every GPU reads from.
- Checked on the tiny model with Unsloth's types (MXFP4 experts, every expert used) against tools/ref_forward,
  `ARC_TP_SIM`: KL 0.0052 / 0.0052 on 1 GPU, 0.0053 / 0.0052 on 2, 0.0053 / 0.0052 on 3, 0.0051 / 0.0052 on 4
  (prompt / generation); at the real widths (--real --unsloth: 384 experts, top 6, MXFP4) 0.066 on 1 GPU, 0.067 on 2 and on 4 (the random router's near-tied experts; the split adds nothing). The 4B split bit-identical to release 2026.11.14.08.
  Speed: not measured (manoreken2's 4 B60s, issue #4).

## 2026-11-14: Flash Next's long-prompt KL on prose; a prose prompt for long-context checks

- `bench/prompts/qwen-prose-3584.ids.txt` (3,584 tokens of docs/lessons-intel-arc-igpu.md): Flash Next KL over its last
  256 positions with QSA on, against tools/ref_forward: default 0.0105, precise 0.0060 (main with PR #21; release
  2026.11.14.07 the same). The code prompt's 0.18 comes from four digit / identifier positions where the engine,
  ref_forward and llama.cpp b12292 each predict a different token (bench/prompts/README.md).

## 2026-11-14: a split GGUF whose first part holds metadata only and ends without padding is read (issue #4)

- `src/gguf.h`: a part with no tensors may end right after its header (Unsloth's DeepSeek-V4.1-Flash UD-Q8_K_XL part 1
  of 9, 5.8 MB of metadata). It was refused as "gguf: truncated", so runz could not open the model at all.
- `scripts\hf-get.ps1` takes a folder of a repository and downloads every file in it, a split model's parts whole
  (`hf-get.ps1 unsloth/DeepSeek-V4.1-Flash-1017-GGUF UD-Q8_K_XL`: its 9 parts, 508 GB; no merging needed). Before
  downloading it checks the drive's free space against what is left to fetch, and at the end prints the runz command
  with the model's first part.
- `runz --kl-ref` with `ARC_KL_SHOW=x`: each position above x also prints both sides' probability of the prompt's next
  token and their top tokens (which side the text agrees with).

## 2026-11-14: DeepSeek-V4.1-Flash in Unsloth's UD-Q8_K_XL types: MXFP4 experts, MXFP8 Engram (issue #4)

- **MXFP4 weights** (ggml type 39: 32 E2M1 codes and one E8M0 scale): decoded on the GPU into the per-32 int8 form
  the GEMMs read, exactly (the codes 0..12 times 2^(e - 128); src/quants.h, kernels/decode.cl, decode_moe.cl), so
  routed experts in MXFP4 run on the MoE paths, the expert file and the expert cache as IQ4_NL ones do.
- **MXFP8 rows** (type 43 of Unsloth's llama.cpp fork: 32 E4M3 codes and one E8M0 scale; upstream ggml's 43 is plain
  F8_E4M3): decoded on the CPU, where the engine reads the Engram table's rows.
- Unsloth's file (`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`, UD-Q8_K_XL, 508 GB in 9 parts): experts MXFP4, every
  other matrix BF16 (taken as Q8_0 at load, as before), the router F32, the Engram table MXFP8; the metadata keys as
  the ISTA-DASLab files'. Checked on the tiny model with these types (`tools/make_tiny_deepseek41.py --unsloth`): KL
  against tools/ref_forward 0.0052 prompt / 0.0052 generation with every expert used; with 4 of 16 experts 0.070
  (the same model in Q8_0 0.084: the random router's near-ties); the expert cache at 19 % and no RAM tier gives the
  same logits bit for bit. Every other model's logits identical to release 2026.11.14.06. Not run yet: the real model
  (it needs a tester with ~600 GB of RAM).

## 2026-11-14: a split model takes the direct GPU-to-GPU sums at equal link speeds too (issue #17)

- **Multi-GPU (`--gpus`):** the sums go straight from GPU to GPU when the slowest direct copy is at least 0.8x the
  host link (it was 1.1x). Two Pro B60s on Linux with 14.1 GB/s direct and 14.2 through host memory (issue #17):
  the 27B's 4K prompt 1,384-1,442 -> 1,578-1,630 tok/s, a 4,210-token prompt 1,293 -> 1,515, generation 36.1 -> 37.8,
  the logits identical. Each GPU's kernels read their own memory instead of the other GPU's parts across PCIe.
  Windows (no peer access) and one GPU: unchanged.

## 2026-11-13: Flash Next's precise mode keeps its source values; QSA in float attention (PR #21, chumbleyjl)

- **Precise mode** (`--precise`, `ARC_FN_PRECISE=1`): the mixers' source values kept (F32 where BF16 cannot hold
  them exactly; zero/engine_load.h); DeltaNet's alpha / beta projections in F32 on one GPU, out of the Q8 qkv / z
  stack (engine_qwen.h); float attention with the QSA masks, its short decode calls split over key ranges
  (kernels/ops_attn.cl, ops_flashnext.cl). The batched changes are Flash Next's only; `--attn-f32` in the default
  mode is still refused. docs/flash-precise-repair-2026-11-13.md.
- Submitted as a draft: GPU and model checks on the rebased code, and speed and memory before and after, were to come
  before the merge. Checked here before the merge (laptop, 2026-11-14): precise KL on 160 code tokens 0.00455 /
  0.02056 -> 0.00173 / 0.00221 (prompt / generation), the default mode and the other models bit-identical to release
  2026.11.14.07 (docs/before-after.md).

## 2026-11-13: the CPU reference decodes Q5_1 rows (PR #20, chumbleyjl)

- `dequant_row` (src/quants.h) decodes Q5_1 in blocks of 32 values, rows of K = 640 included. Before: rows of fewer
  than 256 values were left unwritten, and rows of 256 or more threw an unsupported-type error.
- `tests/test_quant_rows.cpp`: a model-free regression, in CI.
- CPU only: the GPU refused Q5_1 then, so this fixed the f32 reference, not the KL of a model main could run. Since
  2026-11-14 the GPU takes Q5_1 too (the entry "Q5_1 weights on the GPU" above).

## 2026-11-14: a split model's sums straight from GPU to GPU where the links allow it; `Connection: close` honoured

- **Multi-GPU (`--gpus`):** the engine times every link at start: each GPU to host memory and back and, when the
  driver reports peer access (Linux), each GPU to every other. Both go to the log. When direct copies are faster, every
  GPU gets an inbox in its own memory that the others' parts and sums are copied straight into: one PCIe crossing
  instead of two, and the kernels read local memory. The logits are the same bit for bit
  (`ARC_TP_SIM`, forced on: 2 and 3 shards, against release 2026.11.14.04). `ARC_TP_P2P=0` / `=1` forces it off / on.
  Windows' B60s report no peer access: unchanged there. Speed on Linux B60s: not measured yet (issue #17); measured
  later the same day, see the 0.8x entry above (4K prompt 1,442 -> 1,630 tok/s).
- **Server:** a request with `Connection: close`, or HTTP/1.0 without `keep-alive`, gets `Connection: close` and the
  connection ends after the reply. Before, the server answered `keep-alive` and kept the connection waiting for up to
  600 s.

## 2026-11-14: the Linux port audited as a whole; its findings fixed (issue #11)

- **Audit:** Pi with DeepSeek read every Linux code path of the port, statically (docs/audit-2026-11-14-linux.md): no
  Windows change; every fix of the first audit in place.
- **Fixed on Linux:** models in the Hugging Face cache are found where `hf download` puts them (`~/.cache/huggingface`,
  or `$XDG_CACHE_HOME`), without `HF_HOME` (before: the model list was empty on a stock install); the expert file's
  writer survives a short `write(2)`; a child process (the terminal's shell, an MCP server, ffmpeg) gets only its own
  three files, not the server's log; the terminal's session state is safe with two commands at once and with a command
  that runs `set --`; a filesystem that takes `O_DIRECT` only at open is read cached; smaller ones (an error number,
  `PATH` when none is set, `runz`'s process scan, the probes' kernel-cache paths, `-rdynamic` in the build line).
- **Checked:** Linux (WSL, the 4B) and Windows give the same server replies; the terminal and project-folder tests on
  both; Windows' CI build and unit tests; the 4B's logits on Windows bit-identical to release 2026.11.14.01.
- **Found while testing, left for later (both systems, older than the port):** the HTTP server keeps a connection open
  when the request says `Connection: close`.

## 2026-11-14: DeepSeek's audit of the day's changes (docs/audit-2026-11-14.md): no high or medium findings; 5 small fixes

- `clEnqueueMapBuffer` bounds- and null-checked; unmap writes back only maps for writing; `logit_bias` ids past
  INT_MAX ignored (they wrapped to another token); `rmsnorm_h8r`'s limits as build errors; a plain norm ends the fused
  norm's rotation mark; `tests/test_pbkdf2.cpp` in CI. Logits identical to release 2026.11.14.03 (default, the fused
  norm, the 140T packing emulated).

## 2026-11-14: the Arc 140T's prompt GEMM reads its operands in the DPAS order (issue #5)

- **Why:** PascalSoftwares' `xw_peak` on a Core Ultra 9 285H (2026-11-13): `gemm_xw8` 32 x 16 with both operands in
  the DPAS operand order ("packAW") ran the 27B's FFN shapes at 25.0 TOPS against the default's 23.4 (4,096 tokens,
  +7 %) and 24.5 against 21.6 (512, +13 %), bit-identical. Only `xw_peak` made those layouts (on the CPU); the engine
  wrote the plain order. His model sweep found the default 32 x 16 already the best tile (329.7 tok/s at 4K).
- **What:** `ARC_GEMM_XW_PACK=3` (the 140T preset's default; `1` activations, `2` weights, `0` off): the kernels that
  make the GEMM's inputs write them in that order themselves, with no repacking pass: `h8r_act` the activations,
  `h8r_decode` (or `h8r_weights` on a weight's first decode) the staged weights, and `gemm_xw8` reads them with
  `-DXW_PACKA -DXW_PACKW`. Only where `gemm_xw8` takes the GEMM (one column group a sub-group, its shapes); resident
  weights stay plain; a staged weight records its order and is decoded again for a GEMM that needs the other one; the
  fused norm (`ARC_H8_NORM`) steps aside. The start-up check also tests each packed form against gemm.cl.
- **Checked** (laptop, the 140T preset with DPAS emulated, `ARC_140T=1 ARC_XMX_TRY=2`): the 27B's logits with packing
  3 / 1 / 2 identical to 0, and the packed kernels built (`ARC_KLIST`); the 4B with weights decoded just in time the
  same; the default path identical to release 2026.11.14.02. Expected on a 140T: the 27B's prompt +4-5 % at 4K,
  ~+7 % at 512 (its GEMMs are ~72 % of the prompt); the owner to measure. Review: no defect; packing limited to one
  column group a sub-group (with wider tiles a packed input could reach a GEMM gemm_xw8 declines).

## 2026-11-14: the kernels are compiled for the GPU they run on (issue #11)

- **Why:** the engine runs its kernels through Level Zero, which only Intel's driver has, but compiles them with
  OpenCL and loads the result as a native binary. OpenCL used the first GPU of the first platform that had one. A
  machine can list others' OpenCL first: Mesa's rusticl on Linux (a `clinfo` of 2 Pro B60s on issue #11 shows it next to
  Intel's platform), Microsoft's OpenCLOn12 on Windows (a B70 on issue #12 has it), or an Intel iGPU before an Arc card.
  Then the kernels would be compiled for the wrong device: the load fails, or the card's features (8-lane sub-groups,
  XMX, 2D loads) are read from the other one.
- **What** (`zero/clrt.h`): Level Zero starts first; OpenCL then takes Intel's platforms before others' (rusticl,
  OpenCLOn12, Clover, PoCL last), and on them the GPU with Level Zero GPU 0's PCI device id, else its name, else the
  first Intel GPU, else the first GPU as before.
- **Checked** (laptop, one Arc iGPU, Intel's GPU and CPU platforms): the same device as before, logits bit-identical to
  release 2026.11.14.02; `serverz`, `mgpu_probe`, `gpu_bench` start as before. The cases it fixes need such machines;
  the review traced them in the code.

## 2026-11-14: the norm and the H8 rotation of its rows in one kernel, opt-in (B60 prefill plan 13.5)

- **Why:** with H8 (the default on Arc B-series cards) each layer input was normalized by `rmsnorm_quant`, which also
  wrote an f32 copy of the rows, and then `h8r_act` read that copy back to rotate it: on a Pro B60's 27B 4K prompt
  232 ms of 3,051 (docs/xe2-tuning.md section 13).
- **What:** `ARC_H8_NORM=1` runs the norm of the DeltaNet, attention and FFN inputs inside the H8Share scope of the
  GEMMs that read them, as one kernel (`ops.cl rmsnorm_h8r`): `rmsnorm_quant`'s body unchanged (the same work-group
  and reduction, the same int8 rows), the f32 values in local memory instead of the copy, and `h8r_act`'s rotation
  made from them. The copy's write and read and 128 of a prompt's 256 rotation launches are gone. Off by default
  until a B60 measures it.
- **Checked** (laptop, `ARC_H8=1`): logits bit-identical with and without it (the 4B over chunks of 512 with a padded
  last chunk, the 27B, the 4B split over 2 with `ARC_TP_SIM`); without it the default and H8 paths identical to
  release 2026.11.14.01. The 27B's 2,048-token prompt: norm + rotation 687-694 ms -> 498-545 ms of GPU time (the
  fused norm 267-270 ms against `rmsnorm_quant`'s 327); the whole prompt within noise on the laptop, whose GEMMs
  dominate. Review: no defect; a size check added (a row past ~15,000 values would not fit in local memory).

## 2026-11-13 (evening): serverz runs on Linux (step 3 of the port, issue #11)

- **What:** `serverz` builds and runs on Linux (checked in WSL with the 4B on the Arc iGPU). Images, audio and PDF
  pages go through `ffmpeg`, `ffprobe` and poppler when installed (`src/platform_proc.h` runs them with time and size
  limits, in a process group), with a message naming the package when not. The terminal tool runs bash with the same
  chat sessions; MCP servers through `posix_spawn`; the remote access password hash in C++ (`src/pbkdf2.h`, the same
  hashes as Windows' CNG); a crash report from the crash signals. The web UI words the terminal for bash when
  `/props` says `"platform": "linux"`. `web_view` and the recorder stay Windows-only for now.
- **Checked:** the server test gives the same replies in WSL as on Windows, generated text included; the terminal,
  project folders and the media functions tested on both systems (docs/linux-port.md, Progress). Windows: every CI
  target builds, its unit tests pass, the server's replies as release 2026.11.13.09's.
- **Audited:** by Pi with DeepSeek (docs/audit-2026-11-13b.md): no Windows change found; its findings fixed before the
  commit, among them a PDF page error that could end an OCR batch, a wait that could outlive its deadline, the project
  folder checks on Linux, and the crash handler's safety. A stereo file's mix is the channels' average, as on Windows.

## 2026-11-13: the wide-tile GEMM prefetches its next chunks, opt-in (B60 prefill plan 13.2)

- **Why:** on a Pro B60 the 27B's prompt GEMMs run at ~52 % of the XMX peak (~103 of ~197 TOPS), and the tile
  sweep found no better shape than 32 x 64 (docs/xe2-tuning.md section 13): what is left is feeding the engines.
- **What:** `ARC_GEMM_XW_PF=1` prefetches into L1 the weights of the 64-value chunk `ARC_GEMM_XW_PFD` (1 / 2 / 4,
  default 2) chunks ahead, one 2D prefetch per column group (its 1 KB chunk as a 64-byte-wide surface of 16 rows:
  a quarter of the messages two buffer prefetches made); `=2` the activations (2D loads only), `=3` both. ocloc
  `bmg`: no spills, 128 DPAS a superblock as before, +16 messages a superblock for each part. Off by default;
  `bench\xw_sweep.ps1 -Prefetch` tries the nine settings.
- **Checked:** the start-up check passes for each setting; with DPAS emulated (16 lanes, H8) the 4B's logits are
  identical to gemm.cl's at PF 1 (PFD 1 / 2 / 4) and 3; the default path identical to release 2026.11.13.11.
  Without 2D loads (or emulated) only the weights' part is built, and the log says so (found by the review).

## 2026-11-13: a split model keeps its decoded weights when its own share fits `--budget` (B60 prefill plan 13.1)

- **Why:** `plan_weights()` compared the whole model's raw + decoded W8 weights with `--budget`, also on a split, so
  the 27B (~11 GB raw + ~26 GB W8) decoded its weights for every prompt chunk on 2-4 B60s, though a shard of 2 needs
  ~5.7 + ~13 GB. At a 512-token prompt that decode is ~40 % of one B60's prompt time (docs/xe2-tuning.md section 13).
- **What:** each layer matrix counts by the share a shard holds (`tp_weight_share`, `zero/mgpu/engine_tp_shard.h`,
  the same rules as the split itself), rank 0's share on every shard: on an uneven split (the 27B over 3: k-heads
  6 / 6 / 4) the shards must decide alike, or the split's layout check fails it over to one GPU (found by the
  review). Under `ARC_TP_SIM` the budget is divided by the shards, which all sit on the one GPU. The default budget
  (9 GB) is unchanged: the 27B over 2 B60s keeps its weights with `--budget 20`, over 4 with `--budget 10`. One GPU:
  the same sum as before.
- **Checked** (laptop, `ARC_TP_SIM`): the 27B's decision from serverz's `--ctx auto` dry probe; a split's resident
  weights against just-in-time (`--budget 0`), the 4B over 2 / 3 shards, with and without `ARC_H8=1`: logits
  bit-identical; the uneven 27B split over 3 at a budget between its shards' shares loads and runs; one GPU
  bit-identical to release 2026.11.13.11. Speed: the B60s to measure.

## 2026-11-13: Flash Next's PLE history after a snapshot restore (chumbleyjl, PR #15)

- **The bug:** Flash Next's n-gram embedding (PLE) needs the two token ids before each position; the GPU keeps them in
  the PLE layer's state, the host a copy that `forward()` refreshes (`ple_sync`). `verify()` (speculative decoding)
  embeds without that refresh, so a verification right after restoring a snapshot used the ids of the continuation
  that was dropped. On a native Linux B70 a replayed reasoning answer drifted at character 344.
- **The fix:** both restores refresh the host's ids from the restored state: `restore_host()` (the PR) and
  `restore(slot)`, the device snapshot, which had the same gap (added to the PR; the server uses both). `runz
  --snap-test` checks a verification straight after each restore against a cold one, bit for bit, for a Flash Next
  model with MTP.
- **Checked:** the owner's B70 runs: restored and cold continuations bit-identical (two restore paths, 256 tokens
  each). Laptop: other models unchanged (the 4B's `--snap-test` and logits as before; the change runs only with PLE).

## 2026-11-13: expert records copied without their pool's padding (chumbleyjl, PR #14)

- **The bug:** a pool's slots are padded to its largest record. With the RAM tier (discrete cards with free RAM,
  `ARC_XRAM_GB`), a record is copied to the GPU from where it sits in RAM, packed at its layer's own size, and the
  warm-up, demand and deep-prefetch copies read a whole padded slot from it: past the record, past a 1 GiB piece for
  the last one. On a native Linux Arc Pro B70 with a 32 GiB tier that crashed Flash Next; on the laptop it read
  readable memory without a crash.
- **The fix:** every copy takes the layer's record size (`zero/xcache.h`; the deep-prefetch copy added to the PR).
  No kernel or weight change; the padding is never read.
- **Checked:** the 35B-A3B forced into padded pools, RAM tier off and on, the same logits as before bit for bit
  (docs/before-after.md); the B70 owner's runs with 32 / 64 GiB tiers complete.

## 2026-11-13: the peer-access probe on Linux; a newer compute runtime in WSL

- **`mgpu_probe` builds on Linux** (`zero/mgpu_probe.cpp`: the library lookup and page memory through `src/platform.h`),
  so whether Arc cards copy to each other directly on Linux (they cannot on Windows) can be measured by an owner of two
  or more cards on native Linux. WSL cannot answer it: one GPU on the laptop, and ~77 us per GPU round trip there
  against 4.7 us on Windows (the same probe, the same machine), which is also why WSL generates ~15 % slower.
- **WSL's compute runtime 26.40** (Intel's packages for Ubuntu 24.04, in place of Ubuntu's 23.43): the 4B's logits on
  Linux bit-identical between the two runtimes, KL 0.00090 against the f32 reference. Windows' build unchanged
  (mgpu_probe rebuilt, the same output shape). docs/linux-port.md, Progress.

## 2026-11-13: runz runs on Linux (WSL2, the laptop's Arc iGPU)

- **What:** `zero/runz.cpp` and the engine build with g++ 13.3 on Ubuntu 24.04 and run on the iGPU through Ubuntu's
  compute runtime in WSL2. The engine's last Windows calls went behind `src/platform.h` (free RAM, 64-bit seeks, ids,
  environment, path separators) or `#ifdef _WIN32` (the CPU check, power throttling, DXGI's budget, the process list).
- **Measured:** the 4B gives the same 32 greedy tokens as on Windows, 14.7 tok/s (Windows 16.6-19.1), and KL 0.00090
  against its f32 reference (Windows 0.00100; the limit 0.015). Not bit-identical to Windows (another kernel compiler).
- **Windows unchanged:** bit-identical to release 2026.11.13.09 (the 4B, 32 tokens; the 35B-A3B, 16 tokens).
  docs/linux-port.md (Progress) has the build line and what is not tried yet; docs/before-after.md a row.

## 2026-11-13: the web tools on Linux through libcurl (Linux port, issue #11)

- **What:** `tools/web_tools.h` (web_fetch, web_search, the downloads behind them) has a Linux half on libcurl,
  loaded at run time (`src/curl_min.h`, no curl headers needed to build; `libcurl.so.4` from a standard Ubuntu
  install). The same rules as on Windows: public addresses only (each address checked before libcurl connects to
  it), web ports, http(s), redirects by hand, no proxy, the same limits and flags.
- **Checked:** one test program built on Windows (WinHTTP) and Linux (libcurl in WSL) gives the same results for 18
  cases, the refusals included; the Windows serverz's web_fetch / web_search identical to release 2026.11.13.09
  (the Windows code is the same; its Latin-1 conversion moved into a function both halves use).

## 2026-11-13: the HTTP server runs on Linux (WSL), its first Linux build

- `tests/test_http.cpp` (the HTTP server with its test client) builds with g++ 13.3 on Ubuntu 24.04 in WSL, with
  `-Wall -Wextra` and no warnings, and passes all its cases there. The files step 1 of the port moved behind the
  platform layer pass a Linux syntax check. Fixed on the way: the log (`src/log.h`) used `<io.h>`, `localtime_s` and
  `_isatty`, now behind `#ifdef _WIN32` with `localtime_r` / `isatty` on Linux. Windows unchanged (the same lines
  under `_WIN32`; runz, serverz and test_http rebuilt, test_http passes). docs/linux-port.md, Progress.

## 2026-11-13: `zero-server -Host` / `--host`, as serverz's own option

- **`zero-server start -Host 0.0.0.0`** (also `--host 0.0.0.0`, `--host=0.0.0.0`, `-host:0.0.0.0`): the listen
  address, before only `-ListenHost` (PowerShell keeps `$Host` for itself, so it is an alias of `-ListenHost`, which
  still works). `--port=8091` too. Before, `--host 0.0.0.0` stopped with "unknown option --host: did you mean -Port?".
- The "did you mean" hint knows the aliases and prefers a name with the same first letter on a tie (`--hots`:
  `-Host`, not `-Slots`). Checked with `-DryRun` through `zero-server.cmd`: each form gives serverz `--host 0.0.0.0`,
  the port forms `--port 8091`, and the `=` forms count as given options (restarts keep them). USAGE.md shows both.

## 2026-11-13: the NPU on the website; `bench\npu_try.ps1` for newer NPUs

- **Website:** a card for the NPU next to the GPU (`--npu on`): Qwen3.8 27B 4K prompt 133 -> 153-162 tok/s, Gemma 4
  12B +10 % on a Core Ultra 5 125H; mixture-of-experts and small models leave it off; newer NPUs untested.
- **`bench\npu_try.ps1 -Model <dense gguf>`:** the prompt with the NPU off, on, and with its three shares scaled
  together (x0.3 / 0.5 / 0.7 / 1.3), the time the GPU waited for it, and the KL against NPU-off over two 4,096-token
  perplexity chunks. The shares were set on the 125H, whose GPU does ~8.6 int8 TOPS; a Lunar Lake or Panther Lake
  GPU with XMX is several times faster and likely wants a smaller share. The 125H's 27B at 4K: off 128.4 tok/s, on
  151.4 (+18 %), x0.3 / 0.5 / 0.7 / 1.3 129.3 / 137.9 / 133.5 / 125.8; KL 0.0012-0.0034 (limit 0.015).

## 2026-11-13: Qwen3.8 Flash Next over several GPUs (`--gpus N`)

- **Why:** over 4 Pro B60s (96 GB) Flash Next's 52 GB of experts fit in the GPUs' memory instead of streaming from
  the SSD (manoreken2, issue #4). `--gpus` refused it until now.
- **How it is split** (docs/multi-gpu.md, "Flash Next"): the 512 experts by expert as the 35B-A3B's (128 per GPU over
  4, each GPU streaming and caching only its own); DeltaNet by k-head; attention by kv-head over 2 GPUs, whole on
  every GPU over 3-4; the hyper-connection mixers, PLE, the QSA indexer and the shared expert whole on every GPU.
  Each block's output is summed over the GPUs into the block-output buffer, then added to the 4 residual streams as
  on one GPU.
- **Accuracy** (laptop, `ARC_TP_SIM`, KL against the f32 reference on 450 tokens, prompt / decode): one GPU 0.0168 /
  0.0151, over 2-4 GPUs 0.0140-0.0176; with `ARC_FN_PRECISE=1` one GPU 0.0093 / 0.0096, split 0.0083-0.0110. One GPU
  and the existing splits (27B, 35B-A3B) give the same logits bit for bit as before.
- **runz:** `--kl-ref` with `--gpus` measures the prompt path too (before: the tokens one by one only).
- Speed on several cards is to be measured by their owners.

## 2026-11-13: the HTTP server's sockets behind the platform layer (Linux port, issue #11)

- **What:** `src/http.h` (the server's HTTP layer) no longer calls Winsock or Win32 directly: its sockets go through
  a new `src/platform_net.h` (socket setup, accept, send / receive with time limits, the poll that sees a client
  leave, shutdown, the library's start), its media files (`/files/<id>`, Whisper's recordings, with Range) through
  `plat::SeqFile` in `src/platform.h`. Each has a POSIX half, so the file can build on Linux.
- **Windows unchanged:** every call is the one made before, with the same arguments. Checked with the 4B against
  release 2026.11.13.09's serverz: `/health`, the web UI, `/v1/models`, keep-alive, HEAD, a plain and a streaming
  chat (the same text), a stream the client closes after five events (the next request answered at once), two
  clients at once, a chunked request body, a 3 MB file in full and by four ranges (416 included); `test_http` passes.
  No speed or logits change (the engine is untouched). Not compiled on Linux yet. docs/linux-port.md, Progress.
- **Also on the platform layer:** the web_view proxy's sockets (`tools/web_view.h`) and `tests/test_http.cpp`'s
  client. web_view checked against release 2026.11.13.09's serverz: the same replies for an https and an http page
  and a refused port.

## 2026-11-13: a fuller GEMV sweep for generation on discrete cards (`bench\decode_try.ps1`)

- **Why:** a Pro B70 generates the 27B at 32.2 tok/s, 30.8 ms a token, where reading its 11.3 GB of weights at the
  measured 547 GB/s would take ~20.7 ms (issue #12): ~10 ms a token goes somewhere else.
- **New rows:** `ARC_GVROWS=8`, `ARC_GV4=0`, the IQ3_S path at 1 to 8 rows, and for MoE models the verification's
  experts' down GEMV at 1 / 2 / 8 rows (`ARC_MOE_GR_DN_S`) and `ARC_MOE_GRF=128`. **Generation after a long prompt**
  (`-Ctx`, 16,384 by default): plain and the generation attention's switches (`ARC_ATTN_DEC_GRF=128`,
  `ARC_ATTN_DEC_SG=256 / 1024`, `ARC_ATTN_DEC_SMAX=128 / 512`), with a per-kernel profile there too.
- Run on the laptop with the 4B (`-N 32 -Ctx 2048`): every row runs, the bit-identical switches give the same
  tokens as plain. The B60 / B70 owners to measure.

## 2026-11-13: gpu_bench's FP16 vector test fixed; a 32-wide FP16 row on Xe2 (issue #12)

- **Why:** a Pro B70 measured FP16 (vector) at 21.4 of 45.9 TFLOPS on paper (47 %), a B580 the same share
  (friskens, issue #12).
- **What ocloc shows** (bmg-g21): a 16-lane sub-group's f16 multiply-add compiles to `mad (16|M0) :hf`, half of
  Xe2's 64-byte register, so it runs at the FP32 rate; a 32-lane one compiles to `mad (32|M0) :hf`, the whole
  register, the packed form. Also, the test started every lane from the same values, and ocloc then made the chains
  scalar (`(W) mad (1|M0) :hf`): the values now differ per lane.
- **`bin\gpu_bench.exe`:** the FP16 row with per-lane values, and a new row "FP16 (vector, 32-wide)" on Xe2 / Xe3
  (on Xe-LPG, 32-byte registers, 16 lanes already fill one: the laptop's 125H 7.66 of 7.88 TFLOPS, 97 %). The B70's
  number for the new row is to come.

## 2026-11-13: wider H8 tiles to sweep (B60, Arc 140T); the multi-GPU sum overlap merged

- **Tiles beyond 32 x 64, opt-in, for a sweep** (`bench\xw_sweep.ps1 -Wide`, docs/xe2-tuning.md section 11): on
  16-lane cards `ARC_GEMM_XW_NC=6` (32 x 96: a width's first 96-column blocks, the rest 64 wide from a column offset)
  and `8`, `ARC_GEMM_XW_TM=64`; on the 140T `ARC_GEMM_XW_NC=4` and `ARC_GEMM_XW_TM=64`, and six more `xw_peak`
  variants. ocloc at 256 registers: on a B60 32 x 96 spills ~2 bytes and is the one wider tile that loads less per
  DPAS; 32 x 128 and 64 x 64 spill heavily. Defaults unchanged; every shape checked bit for bit (start-up check, and
  the 4B's logits with DPAS emulated).
- **Merged: the multi-GPU FFN sums beside the next chunk's work** (`ARC_TP_OVERLAP=1`, opt-in; the entry of
  2026-11-12 below). After the merge: one GPU identical to release 2026.11.13.06, and the 27B split over 2 / 3
  simulated shards identical to that release's with the switch off and on.

## 2026-11-13 (morning): split models with the wide tiles measured; Flash Next over several GPUs queued

- **A split 27B on Pro B60s with the wide-tile H8 GEMM** (release 2026.11.13.05, issue #4): the 4K prompt over 2 GPUs
  1,368.8 -> 1,596.6 tok/s, over 4 1,734.0 -> 1,887.6; website, docs/multi-gpu.md.
- **Queued: Flash Next (qwen4exp) over several GPUs** (asked for in issue #4); refused with a message today.

## 2026-11-13 (morning): the Linux port, Whisper's model file through the platform layer

- **`plat::FileMap::open_fs`** (`src/platform.h`): a file opened by a `std::filesystem::path` (CreateFileW on Windows);
  Whisper's model.bin (`src/whisper.h`) is mapped through it. Windows: the same calls and messages as before. Checked
  against release 2026.11.13.05: a synthesized 8-s clip transcribed with large-v3 gives the same text and JSON (and the
  same speed, 1.6-1.7 s), a missing model the same error; runz's logits identical. Left (as before): the model path
  goes through a narrow string on its way, so a folder name outside the code page still fails.

## 2026-11-13 (dawn): the wide-tile H8 GEMM on by default on discrete cards (a B60: 27B 4K prompt +21 %)

- **`ARC_GEMM_XW` on by default where H8 is** (the Qwen dense models on discrete cards with 16-lane XMX; it was on only
  in the Arc 140T preset): the H8 prompt GEMMs with 32 x 64 tiles per sub-group (`gemm_xw.cl`, QUEUE 33.4), checked
  bit for bit at start. A Pro B60's sweep (issue #4, manoreken2, `bench\xw_sweep.ps1`), Qwen3.8 27B: 512 / 4K / 16K
  prompts 691.1 / 1,066.5 / 978.9 -> **716.8 / 1,291.7 / 1,169.5 tok/s**, logits identical; 32 x 32 tiles 739.5 /
  1,267.5 / 1,147.6, 16 x 64 and 32 x 16 slower than without. `ARC_GEMM_XW=0`: as before. The laptop's path unchanged
  (runz's logits identical to release 2026.11.13.04). The run of 2026-11-12 that kept it off (the default faster at every
  size) was on a slowed card: its default's passes fell from 486.6 to 421.3 tok/s and it measured 316 GB/s against 433
  (docs/xe2-tuning.md section 11).
- **The chunked DeltaNet on XMX stays off** (`ARC_GDN_CHUNK=3`, QUEUE 33.1): on the same B60 the 27B's 4K prompt 918.3
  tok/s against 1,087.1 (`gdn_chunk_x` 886 ms against `gdn_recur`'s 290), KL 0.00102 against the sequential kernel;
  the vector forms 886-893 tok/s.

## 2026-11-13 (before dawn): the Linux port, the expert file's writing through the platform layer

- **`plat::DirectWriter`, `page_alloc` / `page_free`, `trim_working_set`, `replace_file`, `remove_file`,
  `last_error`** (`src/platform.h`): the expert file's writing (`ExpertStore::build`, zero/expert_store.h) uses them,
  so expert_store.h has no Windows calls left. Windows: the same calls as before (an unbuffered write-through file,
  VirtualAlloc, SetProcessWorkingSetSize, MoveFileExA). Linux: O_DIRECT writes (plain where a filesystem has none),
  mmap, rename. Checked against release 2026.11.13.03: two test models' expert files written again by both builds
  byte-identical (33.4 MB each, no .tmp left), and the logits identical for the 35B-A3B, Flash Next and the 4B.

## 2026-11-13 (night): the Linux port, the expert file's reads through the platform layer; B60 numbers

- **`plat::AsyncFile`** (`src/platform.h`): a file read without the OS file cache, reads in flight on numbered slots;
  the expert store (`zero/expert_store.h`) reads its experts through it. Windows: the same calls as before (unbuffered
  overlapped reads, an event per slot). Linux, a first form: `O_DIRECT` with `pread` (buffered where a filesystem has
  no `O_DIRECT`), `io_uring` later. Checked against release 2026.11.13.02: logits identical for the 35B-A3B (experts
  cached) and Flash Next (34.9 GB of experts read from the SSD in the run, 4.5-6.0 GB/s against 5.4), and the 4B.
- **A Pro B60's numbers** for the rotation reuse and the swiglu skip (issue #4): the 27B's 4K prompt 1,024.5 -> 1,080.9
  tok/s (+5.5 %); website and README updated.

## 2026-11-13 (early morning): the Linux port, model files through the platform layer

- **`plat::FileMap`** (`src/platform.h`, docs/linux-port.md step 1): a file mapped read-only, in three steps (open,
  size, map). GGUF model files (`src/gguf.h`, the parts of a split model too) and the DFlash2 drafter's safetensors
  (`zero/engine_drafters.h`) use it; on Linux it is `open` / `fstat` / `mmap`. On Windows the same calls, checks and
  messages as before: runz's logits identical to release 2026.11.13.01 for the 4B, a split GGUF (Flash Next, 2 parts)
  and the 27B with its DFlash2 drafter (the same greedy text); a missing and an empty file give the same errors.

## 2026-11-13 (night): the Linux port's first piece, a platform layer

- **`src/platform.h`** (docs/linux-port.md, step 1): the GPU runtimes' loading (`ze_loader.dll` / `OpenCL.dll`,
  on Linux `libze_loader.so.1` / `libOpenCL.so.1`) and the user's home folder (`%USERPROFILE%`, on Linux `$HOME`)
  behind one small interface; `zero/ze_min.h`, `src/cl_min.h`, `zero/clrt.h` and `src/app_dir.h` call it. On Windows
  each call is the Win32 call made before: runz's output and logits identical to release 2026.11.12.19, all 13
  packaged programs and the bench / test tools build. Nothing runs on Linux yet; files, mapping, sockets and processes
  follow.

## 2026-11-12 (midnight): the H8 down GEMM reads the up GEMM's output; `runz --n 0` fixed

- **No `swiglu_quant` before an H8 down GEMM** (QUEUE 33.3). In a prompt chunk, the up GEMM already writes
  silu(gate) * up in f32, and `swiglu_quant` only copied those values for the H8 rotation (and wrote int8 rows that an
  H8 GEMM does not read). The down GEMM's rotation now reads the up GEMM's output itself, and `swiglu_quant` is left
  out (`ARC_H8_SWI=0`: as before). Bit-identical: 4B and 27B logits on and off (a 711-token prompt, padding rows in its
  last chunk), and a two-shard split (`ARC_TP_SIM`); the default laptop path identical to release 2026.11.12.18.
- Laptop, `ARC_H8=1 --bench-pp 2048`, 2 alternated rounds of 3 runs: 27B 120.7-123.0 -> 123.9-125.3 tok/s (+2-2.5 %;
  `swiglu_quant` 495-511 ms gone, the rotation 23-75 ms slower as it reads f32 values of its own), 4B 823-898 -> 911-958
  tok/s.
- **`runz --bench-pp N --n 0` crashed** after the prompt runs (an empty list of step times was read), and the crash lost
  the `pp` lines still buffered: the generation summary is now printed only when `--n` is above 0.

## 2026-11-12 (late): H8 prompt GEMMs on the same input rotate it once

- **The H8 prompt GEMMs** (on by default on discrete cards; `ARC_H8=1` elsewhere) rotated their input before each
  GEMM, even when the next GEMM reads the same input: q / k / v, gate / up, and DeltaNet's qkv / z. Those now rotate
  it once (QUEUE 33.3; `ARC_H8_REUSE=0` rotates before each as before). The same kernel on the same input, so the
  results are bit-identical (4B and 27B logits, a 711-token prompt, on and off; a split over two shards with
  `ARC_TP_SIM`; the default laptop path's logits identical to release 2026.11.12.17).
- Laptop, `ARC_H8=1 --bench-pp 2048`, alternated, 2 rounds of 3 runs: `h8_act` 800 -> 512 launches and 111 -> 83 ms
  on the 4B, 400 -> 256 and 463-501 -> 344-347 ms on the 27B; the prompt 4B 922-927 -> 934-937 tok/s, 27B 118.9-121.6
  -> 120.9-123.4 tok/s (about +1-2 %). On a B60 `h8_act` was 14 % of the GEMMs' time (2026-11-06): to be measured
  there.

## 2026-11-12 (night): Gemma's long-prompt attention, three shapes measured at 32K (none faster); the Linux plan

- **Gemma 4 12B, 32K-token prompt, one run each on the laptop** (the head-512 global attention, `attn_fa_gh`; the
  12B's 16 q heads per kv head spill 416 bytes per thread as it is): as it is 375 s (with `--prof`); two query rows
  per work-group (`ARC_FA_GR=2`) 866 s (7.5 KB spilled); 4 sub-groups of 8 dims (`ARC_FA_FS=4`) 1,018 s; and a new
  switch, **`ARC_FA_HL=n`** (n of the 16 q heads per work-group, the rest in other work-groups: no spill, each key row
  read 16 / n times) 418 s with 8, 564 s with 4. All bit-identical (12B logits, 2,311-token prompt, against the
  default and against release 2026.11.12.16). So the spill does not bound this kernel; the key reads do: the default
  stays, `ARC_FA_HL` is opt-in (for discrete cards to measure).
- **A Linux port: the plan** (issue #11), [docs/linux-port.md](docs/linux-port.md): what is Windows-only (measured by a
  search of the sources), the order (a platform layer with the Windows build bit-identical, then runz, serverz headless,
  packaging), and how to test it (WSL2 on the laptop; a 285H on Ubuntu for the speed). Nothing built yet.

## 2026-11-12 (evening): the fixes of DeepSeek's audit (docs/audit-2026-11-13.md)

A static audit by DeepSeek-V4.1-Flash through Pi (read-only tools, no server), every finding checked in the code:
- **MCP:** stopping an MCP server whose child left the job and still holds the pipe no longer hangs: a reader that
  does not end after 5 s of cancels is left behind, its client kept alive so it reads nothing freed.
- **Model files:** a GGUF may hold at most 1,000,000 tensors or keys, 64 M array elements and 1,024 split parts,
  refused before anything is allocated (a crafted 1-2 GB file could ask for 10+ GB of RAM).
- **PDFs:** object numbers, counts, page indexes and offsets range-checked before they are cast (undefined behaviour
  on a hostile file's 1e999).
- **The `--ctx auto` cache:** a damaged row is skipped and computed again ("invalid stoi argument" failed the load; a
  row cut inside its digits gave a 12-token context); `ctx-auto.txt` keeps its newest 1,000 rows past 512 KB.
- **Chats:** a `CHAT.JSON` in a chat's folder is the chat's own file (Windows' names are case-blind), not a user file.
- **The log and `/metrics`:** control bytes from a model file or a request (CR, ESC, NUL) show as `?`.
- `.gitignore`: video recordings under `tests/data/` too; `docs/logging.md`: three drifts corrected.
- Results unchanged: runz's output and logits identical to 2026.11.12.15; the PDF tests (300 damaged PDFs) pass.

## 2026-11-12 (evening): the sums in pieces on by default; the GEMV prefetch for MoE models on discrete cards

From a Pro B60 owner's runs (issue #4, package fffaf1a) and a Reddit user's two Arcs:
- **`ARC_TP_PIPE` is on by default** (a split model's prompt sums in pieces, both link directions at once): the 27B's
  4K prompt over 2 B60s 1,298.9 -> 1,368.8 tok/s (+5 %), over 4 B60s 1,629.6 -> 1,734.0 (+6 %); a 27B IQ4_XS 32K
  prompt over two Arcs (the second on PCIe x4) 903.9 -> 1,033.7 (+14 %). The same logits bit for bit (laptop,
  `ARC_TP_SIM`, 2 and 3 shards, against `ARC_TP_PIPE=0`). `ARC_TP_PIPE=0`: as before.
- **The single-token GEMV's prefetch (`ARC_GEMV_PF`) for MoE models too** on discrete cards: the 35B-A3B's generation
  79.93 -> 88.91 tok/s on a B60 (+11 %; dense models had it already). Bit-identical. An earlier run (2026-11-09, older
  code) had shown the opposite; asked for a rerun. `ARC_GEMV_PF=0`: off.
- **Measured and kept as they are:** `ARC_GEMV_GU=1` (gate and up in one launch) 22.79 against 23.72 on the 27B, so
  off; the wide XMX tiles (`ARC_GEMM_XW`) slower than the B60's default at every size (27B 4K: 992.9 default, 944.6
  at 32 x 64), so off on discrete cards; `ARC_BATCH` no gain on the 27B (35B: 85.1 at 16, not taken with the prefetch
  on, to measure together).
- **DeepSeek-V4.1-Flash ran on real hardware for the first time** (IQ2_XS, one B60, 1 TB of RAM): the expert file
  (161.8 GB) written in 129 s, "The capital of France is" -> " Paris." (top token 0.90), 3.6 tok/s with 9 % of the
  experts on the GPU, the rest from RAM.

## 2026-11-12 (evening): the GPU's fatal errors in words; the launcher's "Copy for an issue"; release names that sort

- **A GPU error that ends the process now says what it was** (an `E` line): "device lost: the GPU was reset, by a
  driver timeout or a fault; the process must restart", "out of memory: other programs, or this one, use the GPU
  memory", "not initialized: the Level Zero driver did not start", on `ZE_CHECK`'s line and on the kernel launch and
  wait errors. A device allocation that fails is an `E` line too (it was an `I`), with what to do, followed by the GPU
  budget, its use, the engine's allocations and the free RAM (the line `ZE_CHECK` already printed).
- **The launcher's log tab: "Copy for an issue".** The log since the start (its first lines name the build, system,
  GPU and switches) as a GitHub code block, with your user folder as `%USERPROFILE%` and your user name taken out of
  paths; 2,040 lines at most (the first 40 and the last 2,000).
- **Release names that sort (manoreken2, issue #4):** GitHub lists files by name, so `2026.11.12.13` came after
  `2026.11.12.9`. From tomorrow the counter has two digits (`2026.11.13.01`), and `scripts\make-package.ps1 -Release
  <number>` puts it in the package's name (`inferzero-2026.11.13.01-<commit>-win64.zip`); without it, as before.

## 2026-11-12: a split model's automatic context sized per GPU; -Gpus in zero-server, zero-pi and the launcher

- **`--ctx auto` with `--gpus N`** sizes the context for one GPU's share (a dry probe of shard 0 of N) instead of the
  whole model on one GPU: a model bigger than one card got only 4,096 tokens (e.g. a 27B over two B580s). Laptop:
  the 27B's weights per GPU 10.94 -> 5.47 GiB on 2 GPUs, the context's cost per 1K tokens halved; the 4B split over 2
  (`ARC_TP_SIM`) loaded at 262,144 tokens, 5.41 GiB on each GPU. A model that cannot be split is sized as on one GPU.
- **Start a split model from the scripts:** `zero-server -Gpus N`, `zero-pi --gpus N`, a "GPUs (one model split)"
  field in the launcher (kept on a restart like `-Slots`; the memory estimate per GPU). `serverz --list-models --gpus N`
  and `/props` "gpus": Pi's model list gives each model the context a split server loads it with.

## 2026-11-12: a split model's FFN sums beside the next chunk's work (opt-in, `ARC_TP_OVERLAP=1`, stage 1)

- **Why:** a tester's two Arcs (27B IQ4_XS, 32K prompt over 2 GPUs) spent ~31 s of a 31.7-s pass inside the 1,024
  sums, the GPU mostly idle while its thread waited; `ARC_TP_PIPE=1` gave only +14 %.
- **What:** in prompts that run layer by layer (just-in-time weights, longer than a chunk) each shard's sums go to a
  sum thread of its own, with its own command lists and kernels (`zero/mgpu/engine_tp.h`, `TpOverlap`). Chunk c's
  FFN sum is waited for only when the next layer reaches chunk c, so the GPU computes chunk c+1's attention while it
  crosses PCIe. Attention / DeltaNet sums are still waited for at once. One more partial buffer per GPU (80 MB for the
  27B at chunk 4096). Dense models only (not the 35B MoE); generation and chunk-by-chunk prompts unchanged.
- The sums run the same code as before (`tp_reduce_rs` / `_pipe` now take their buffers as arguments) in the same
  order on every shard: **the logits are the same bit for bit** with the switch on and off, and off equals the build
  before. Checked: see `docs/before-after.md` (2026-11-12) and `docs/multi-gpu.md`, "Sums beside the next chunk's
  work".
- **Failsafes:** a sum's error is thrown on the main thread; a pass that stops with sums queued fails the split model
  so the others leave their barriers; the hang tests (`ARC_TP_HANG`) end as before.
- Found on the way: on a GPU with one compute queue (the laptop's iGPU) a GPU-side wait of one list for an event of
  the other never ends; every wait between the two lists is a host wait first.
- **Off by default, speed unmeasured** (the laptop has one GPU). Stages 2 (the attention sums hidden too) and 3
  (chunk-major prompts) are written up, not built.

## 2026-11-12: the server's log, as llama.cpp's; GET /metrics, as vLLM's

[docs/logging.md](docs/logging.md) describes all of it.
- **Every line has a time and a level:** `2026-11-12 14:58:09.716 I ...` (E / W / I / D). Each line is written whole
  under one lock, so a line built in parts no longer gets another thread's text inside it. The old
  `fprintf(stderr)` calls in `src/`, `zero/` and `tools/` go through the log (all but the CLI tools' usage texts and
  the crash report) (`src/log.h`, force-included like the CPU check). runz and the tools print
  as before, byte for byte: runz's output and logits were checked against 2026.11.12.12's.
- **Options, llama.cpp's names:** `--log-verbosity N` / `-lv N` (0 errors ... 3 debug; `ARC_LOG_LEVEL`),
  `--log-timestamps`, `--log-colors`, `--log-file`, `--log-disable`.
- **At start:** the build, Windows, CPU and RAM, the GPU (name, driver, EUs, clock, memory, XMX), and every `ARC_*`
  switch set (secrets as "(set)").
- **At each model load:** the model's shape (arch, layers, heads, vocab, trained context, experts, file). With
  `--ctx auto`, how the GPU memory was shared out: kept free, snapshots, encoders and speech, weights, drafter, buffers,
  context per 1K tokens, what is left.
- **Per request:** an access line (method, path, status, bytes, time, client). Polls show at debug only. An error
  reply is a W / E line with its message (an overflow says the prompt and the context sizes). The completion line
  adds the time to the first token, the context used / served and the slot. At `-lv 3`, llama.cpp's
  `prompt eval time` / `eval time` / `total time` lines.
- **Loading:** the `progress:` lines (read by zero-server and the launcher, without the prefix) are one per 5 points or
  2 s: ~20 a load instead of ~100.
- **`GET /metrics`:** Prometheus' format with vLLM's names under `inferzero:`. Requests running / waiting, context use,
  tokens, prefix-cache hits, drafts accepted, finish reasons, HTTP requests, and histograms of the time to the first
  token, the end-to-end latency, the time per token, the prompt and generation times and sizes. Numbers only.
- **Old logs:** the newest 10 as `.log`, older ones each in a zip of the same name (about 10x smaller), and those older than 90
  days or past 200 MB removed.

## 2026-11-12: Pi's context for every model counts speech to text

- **Pi's model list now gives each model the context a switch to it would really get.** With Whisper or Qwen3-ASR
  loaded, the server keeps their GPU memory free when it sizes a model's context (`--ctx auto`), but Pi's list
  (`~/.pi/agent/models.json`, written by `zero-server` / `zero-pi` at each start) took the other models' contexts
  from `--list-models`, which did not: after `/model` inside Pi, Pi could count on more context than the server
  served, and compacted only after an overflow error.
- The server's `/props` reports `speech_reserve_bytes` (the GPU memory it keeps for speech to text now);
  `serverz --list-models --speech-reserve BYTES` keeps that much free in every model's context, as a load does;
  `zero-server`'s Pi sync passes the running server's value. The running model's context still comes from the
  server itself. Without speech to text nothing changes.

## 2026-11-12: `xw_peak`, the Arc 140T's XMX prompt GEMM in TOPS; layouts in the DPAS order to measure

The 140T's prompt GEMM (`gemm_xw8`, 32 x 16 tiles) runs the 27B's FFN at ~23 int8 TOPS, ~30 % of the 76.75 its
owner's `gpu_bench` measured; the whole 4K prompt ~16 TOPS. Toward ~80 % (issue #5), a bench first:
- **`bin\xw_peak.exe`** (`bench\xw_peak.cpp`, in the package): the GEMM alone at the 27B's FFN shapes (gate, up with
  the fused SwiGLU, down in 4 split-K pieces; `--shape NxK[:a]`, `--m`), the engine's launch geometry, TOPS and the
  share of the int8 peak it measures in the same run; each variant's output compared bit for bit with the default.
  It also prints what the f16 DPAS keeps (products, the sum inside one DPAS, the accumulator, chained against
  added in f32): which of them loses bits would say how to fix the chunked DeltaNet on XMX, which missed its
  start-up check on the 140T by about f16's precision (QUEUE row 4). About a minute; 8-lane GPUs only.
- **`gemm_xw8.cl -DXW_PACKA / -DXW_PACKW`** (not used by the engine): activations and weights in the order the DPAS
  operands take them, made on the CPU by the bench. ocloc `arl-h`, 256 registers, no spills in any: the main loop's
  instructions per DPAS 11.8 (default) -> 5.7 (both, 32 x 16) -> 4.1 (both, 32 x 32), its register moves 399 -> 0,
  its loads ~2.6 -> ~1.6 -> ~1.0 per DPAS. The 32 x 32 tile also loads 1/3 fewer bytes per DPAS than 32 x 16.
- Checked on the laptop (DPAS emulated): every variant's output identical to the default's at three shapes (split-K
  and SwiGLU included); the engine's logits with the kernel before and after the change bit-identical (4B, H8 + the
  wide tile, 853 tokens). The engine's path is the old code unchanged (`#if` around it). The speed needs a 140T.

## 2026-11-12: the fixes of the static audit (docs/audit-2026-11-12.md)

Every finding of the audit fixed (numbers as in the audit):
- **Kernel cache (#3):** one folder per driver (`kernels\drv-<version>\`); at start the old flat files, other
  drivers' folders unused for 7 days and temporary files older than an hour are removed, and the current folder is
  trimmed to `ARC_KCACHE_GB` (default 2 GB) by last use. Each process writes under a name of its own before the
  rename, so two compiling the same kernel cannot install a mixed file. The first start after the update compiles
  every kernel once (the laptop's 1.4 GB / 2,797 old files and its warm-up markers went).
- **Launcher (#12):** a Hugging Face file name with `..`, a drive letter or a UNC prefix is refused before anything
  is written (it could write outside the cache). **(#13)** the alias and the drafter path are quoted when the server
  starts. **(#11)** `make-package.ps1` keeps the newest 3 packages in `dist\` (`ARC_DIST_KEEP`).
- **Server:** a request body has a total deadline, 60 s plus 10 s per MiB (#14: a client trickling one byte every
  29 s held a connection); the MCP client keeps a reply only for a request that waits for it (#16); the terminal
  and MCP pipe readers are cancelled with `CancelSynchronousIo`, and a terminal read that still never ends is left
  behind instead of waited for (#15); a folder that cannot be read is remembered for 60 s (#10); the web UI shows a
  stored chat's image or audio only from an address `safeUrl` accepts (#17).
- **Engine and tools:** `runz --ppl-ctx` below 3 and a text shorter than one chunk are refused (#1: 0 divided by zero, 2 gave nan); the DXGI adapter loop stops on
  any error (#5); `ds_norm_rope` refuses rows it cannot hold (#6); `ARC_GVROWS` outside 1-8 and `ARC_MOE_TM_SMALL`
  outside 0 / 1 / 2 / 4 / 8 / 16 are ignored with a note (#8); the GEMV and MoE kernels laid out for 16 sub-groups
  return in a larger work-group instead of sharing windows (#7; logits bit-identical, four models); the bandwidth
  probe's allocation throws instead of ending runz.
- **Build:** the three f32 references `ref_qwen3a`, `gemma4v_ref` and `gemma4a_ref` are built `/fp:precise` like the
  others (#2); every program checks for AVX2 / FMA at start and says so instead of dying with an illegal
  instruction (#4, `src/cpu_check.h`, force-included).
- **Rules:** AGENTS.md says a kernel launched without a local size must not depend on it (#9).
- **Measured for the audit's "needs a GPU" #1:** split-K is not bit-identical to one pass (the pieces' sums in
  another order): 27B, KL against `ARC_GEMM_SPLIT=0` mean 0.00067, max 0.0017, the same greedy tokens.

## 2026-11-12: a split model's prompt sums in pieces (opt-in, `ARC_TP_PIPE=1`)

- **The f16 reduce-scatter of prompt chunks can run in pieces** (`zero/mgpu/engine_tp.h`, `tp_reduce_rs_pipe`),
  after manoreken2's HostCopy (PR #10: a block cut into pieces and copied by two command lists, 1.85x on two B60s).
  Each GPU's rows are cut into 2-8 pieces of ~2 MB. A copy list of its own (a copy engine) puts the pieces to host
  memory while the compute list already sums the pieces every GPU has put, so both directions of each GPU's link
  work at once, where the three steps of the sum ran one after another. Counters in host memory say which pieces
  have arrived; every wait keeps the barrier's time limit.
- The same results bit for bit (the 27B split over 2, 3 and 4 GPUs with `ARC_TP_SIM` on the laptop, chunks of 128
  and 512, 1 / 3 / default pieces, the f16-overflow failsafe forced: every logit the same as without it). The
  failsafes checked with it: a GPU that stops (`ARC_TP_HANG=1:20`) and a wait that fails (`1:20:wait`).
- **Off by default** until it is measured on B60s and B580s: the laptop runs every shard on one GPU and says
  nothing about the speed. Expected: the sums up to ~2x faster on 2 GPUs and ~1.7x on 4. `ARC_TP_PIPE_PIECES=N`
  sets the pieces per GPU block. Generation's per-token sums are unchanged.

## 2026-11-12: gpu_bench's memory read on discrete cards

- **`gpu_bench` and `membw_probe` read random data** instead of a buffer filled with one value. An Arc B580's run
  (Reddit, 2026-11-12) read 1,149 GB/s from 1 GiB, with 456 GB/s on paper and an 18 MB L2: the constant fill made
  the reads cheap, as the discrete cards' memory compression can. The laptop's iGPU reads the same as before (76.2 ->
  77.6 GB/s). Earlier read figures from discrete cards with these tools are not valid; the engine's own probe
  (`runz --prof-gen`, an unwritten buffer: a Pro B60 428 GB/s) was not affected.

## 2026-11-12: DeepSeek-V4.1's tokenizer

- **The `joyai-llm` pre-tokenizer** (DeepSeek-V4.1's GGUFs; llama.cpp's DeepSeek-V3 split, also taken for
  `deepseek-v3`): digits in groups of up to three, runs of CJK ideographs and kana apart, then DeepSeek-V3's word
  rule, applied in turn as llama.cpp does (`src/tokenizer.h`, `pretokenize_ds3`). Before, such models were split as
  Qwen's (single digits), so text prompts gave other tokens than llama.cpp's. Checked against llama.cpp b12292's
  `llama-tokenize` on the real model's vocabulary (its first shard): 16 texts, about 140,000 tokens (edge-case
  Unicode, random codepoint soup, the repository's docs and code), every token the same. Other models' tokens
  unchanged (Qwen3.6-35B and Gemma 4 26B, three texts each, before and after). DeepSeek-V4.1 runs can take text now
  instead of token ids.

## 2026-11-12: the 35B MoE's generation on discrete cards back to its speed

- **`ARC_GEMV_SMALL_WG` is off again by default** (discrete cards, `zero/xe2/`). A Pro B60's `decode_try` on package
  18acafc (issue #4, best of 2): the 35B-A3B generated 55.85 tok/s with it and 80.35 without (+44 %), the same tokens;
  the 27B 23.52 with it, 23.57 without. The single-token GEMV of the small matrices goes back to work-groups of 256,
  as before 2026-11-10. Bit-identical (only the work-group size changes). iGPUs unchanged (they always used 256).
  `ARC_GEMV_SMALL_WG=1` turns it on; `bench\decode_try.ps1` now has a `SMALL_WG=1` row instead of `SMALL_WG=0`.

## 2026-11-12: DeepSeek-V4.1's indexer scores faster, the same results

- The indexer's scores take a lane per compressed row: each row's index key is read once for all its heads (the
  real-widths test model's 1,200-token prompt: 318 -> 21 ms, 763 -> 986 tok/s), bit-identical. It matters more with
  long prompts, where the scores grow with rows x queries. `ARC_DS_IX1=1`: the previous kernel.
- The queries' RoPE reads and writes only the rotated 64 dims of each 512-wide head (65 -> 14 ms on that prompt),
  bit-identical.

## 2026-11-12: DeepSeek-V4.1 prompts nearly twice as fast on the test model

- **Prompt attention takes 16 heads at a time** (`ds_attn_gh`: a lane per key, the 16 heads' queries broadcast, as
  Gemma's head-512 prompt attention does), instead of one sub-group per head that read the keys again for each head
  and paid a reduction and an exp per key and head. The real-widths test model's 1,200-token prompt: attention 1,756
  -> 275 ms, the prompt 407 -> 765 tok/s on the laptop. KL against f32 as before (16-head test model 0.00308 against
  0.00304). `ARC_DS_ATTN_GH=0`: the old kernel. `tools/make_tiny_deepseek41.py --heads N` for such test models.

## 2026-11-12: DeepSeek-V4.1 generation 28 % faster on the test model

- The hyper-connection mixes of a token now run over 16 work-groups instead of one (`ds_mix` 99 -> 18 ms on the
  real-widths test model), and attention of 1-8 rows splits each head's keys over 8 sub-groups (`ds_attn` 90 -> 61
  ms); keys come in vector loads. Generation on the real-widths test model 32.2 -> 41.2 tok/s on the laptop. KL
  against the f32 reference unchanged within noise. `ARC_DS_MIX1=1` / `ARC_DS_ATTN1=1`: the single kernels.

## 2026-11-12: DeepSeek-V4.1's key caches in f16

- **The window ring, the compressed rows and the index keys of DeepSeek-V4.1 are kept in f16** (the chunk's own keys
  rounded the same way, so the results do not depend on how a prompt is cut): half the cache memory, ~325 MB less at
  128K of context on the real model. KL against the f32 reference unchanged within noise (tiny model 100 / 300 tokens:
  0.00295 / 0.00363 against 0.00323 / 0.00361 with f32). `ARC_DS_KV32=1` keeps f32. Saved DeepSeek conversations from
  before (f32) are processed again once.

## 2026-11-11: DeepSeek-V4.1 conversations saved to disk

- **The server's saved conversations now cover DeepSeek-V4.1** (a format of its own, version 3: the window rings,
  the compressed rows and index keys up to the saved position, the pending compressor rows, the Engram history, the
  snapshots' copies). `runz --snap-test` now also saves a conversation with a snapshot to a file, resets, loads it,
  continues, then restores the loaded snapshot and continues again: bit-identical against a fresh run on the tiny
  DeepSeek model and on Qwen3.5 4B.

## 2026-11-11: DeepSeek-V4.1 checked at the real model's widths

- `tools/make_tiny_deepseek41.py --real`: a 4.4 GB test model with the real widths (embedding 5120, 64 heads of 512,
  q LoRA 1280, 8 output groups of 1024, window 128, top-k 512, a 2048-block candidate pool, 32 indexer heads, 384
  experts with 6 active, Engram rows of 256) in 4 layers that cover every layer role, expert FFN 256, experts Q4_0.
  The engine loads and runs it (1,200-token prompt in two chunks: 7.5 s; generation 35 tok/s on the laptop).
- Against the f32 reference: up to 417 tokens every half-layer within 3 % of the reference (attention halves within
  2 % also at 600 tokens, where top-k 512 is active); over 1,200 positions token by token the median KL is 0.001 at
  every depth, the mean 0.049 from positions where an expert near-tie flips (the reference's 6th and 7th experts
  0.014 apart in the case traced): a random-weight router of 384 experts. The real model's check is against
  llama.cpp on the same prompts.

## 2026-11-11: DeepSeek-V4.1-Flash in the server's prompt cache

- **The server's snapshots now cover DeepSeek-V4.1** (`zero/engine_ds41.h` ds_save / ds_load): a follow-up
  request or a new conversation sharing the system prompt resumes from the cached prefix instead of processing it
  again. A snapshot copies what a step leaves besides the position-indexed compressed rows: every layer's window
  ring, the pending compressor rows, the Engram token history (~10 MB on the real model). Checked with the new
  `runz --snap-test` (a snapshot at an odd position, 100 other tokens fed, the snapshot restored, the next tokens'
  logits against a fresh run): bit-identical, device and host snapshots, in one chunk and in chunks of 128; the
  Qwen3.5 4B passes it too. Saved conversations (to disk) are still not for this model.

## 2026-11-11: the Arc 140T's prompts about 40 % faster (gemm_xw8 by default)

- **On the Arc 140T (its preset), the H8 prompt GEMMs now run through `gemm_xw8` with 32 x 16 tiles at 256
  registers.** Measured by its owner (issue #5), Qwen3.8 27B: 512 / 4K / 16K-token prompts 147.6 / 214.1 / 205.4 ->
  204.9 / 298.6 / 283.1 tok/s, the logits identical; the start-time check compares it with the old kernel bit for bit.
  `ARC_GEMM_XW=0` turns it off. Other GPUs: unchanged (it stays opt-in there, `ARC_GEMM_XW=1`).
- `bench\xw_sweep.ps1` on an 8-lane GPU: its "default" row is the old path, so the comparison stays the same.

## 2026-11-11: Gemma's long-prompt attention, an opt-in read pattern and a benchmark for it

- **`bench\gattn_bench.exe`** times Gemma 4's head-512 prompt attention (`attn_fa_gh`) at any depth on random data
  and compares variants bit for bit: the 12B's global layer, 32K deep, 4,096 queries: 4.07 s (1.0 TFLOPS of the
  laptop's 3.9).
- **`ARC_FA_QB=2`** (opt-in): the query of each head read two 16-dim chunks at a time in the scores (half the reads
  of the query). Bit-identical. 16 % faster in the benchmark, 4.5 % less attention time in the 12B's 8K prompt, but
  slower in the 12B's 32K prompt (attention 133 -> 162 s; in the reverse order 133.7 against 139.1 s, still 4 % slower), so it stays off.
- Tried and dropped: one lane per (query row, q head) so keys and values are read once for all lanes, without the
  per-multiply broadcast (bit-identical, but 23-60 % slower: more reads per key than before).

## 2026-11-11: DeepSeek-V4.1-Flash, a first engine path (tested on a small synthetic model)

- **`deepseek41` loads and runs** (`zero/engine_ds41.h`, `zero/kernels/ops_ds41.cl`), written from
  [docs/deepseek41-spec.md](docs/deepseek41-spec.md): the 4 hyper-connection streams with their Sinkhorn mixes, the
  Engram n-gram rows (read by rows from the GGUF, as Flash Next's PLE table), one 512-wide KV head over a sliding
  window plus up to top-k compressed rows picked by the indexer (ratio 2 and 1 compressors, the candidate block pool,
  attention sinks), the grouped output LoRA, and a MoE with 384 experts streamed from the SSD through the existing
  expert path: its router in f32, a sqrt(softplus) choice with the selection bias (`moe.cl -DROUTE_DS`), SwiGLU
  clamped at 10 (`-DSWI_CLAMP`).
- The real model is ~231 GB (ISTA-DASLab IQ1_M + its Engram table): it cannot run on the laptop. It was built and
  checked on `tools/make_tiny_deepseek41.py`, a 39 MB GGUF with the real keys, tensor names and layer roles and random
  weights: KL against the f32 reference (`tools/ref_forward`) 0.0032-0.0039 for 100 / 300-token prompts, in one
  chunk, in chunks of 128 and token by token. Not yet for this model: server prompt-cache snapshots, saved
  conversations, slots, drafters, multi-GPU; the caches are f32 and the kernels plain (speed comes after the real
  model's first run).
- Other models: their logits are bit-identical (Qwen3.6 35B, Gemma 4 26B, Qwen3.5 4B checked against the commit
  before); the MoE kernels' new switches are off for them.

## 2026-11-11: Gemma 4's attention keys in f16 (more context); a faster attention tried and dropped

- **Gemma 4's float attention keeps its K rows in f16** instead of f32, as Qwen3 dense does (`ARC_G4_KF16=0`: f32 as
  before). Half the key memory, so a larger automatic context: Gemma 4 26B 102,400 -> 131,072 tokens (with images
  110,592 -> 131,072; `-std` 131,072 -> 163,840); the 12B and E4B were at their caps already. The same speed (8K
  prompts within 1 %: the laptop's prompt attention is bound by its instructions, not by bytes). Accuracy as f32:
  20-token KL 12B 0.00000, E4B 0.00395 (0.00389 with f32), 26B 0.01013 (0.01014); after an 8,839-token prompt the
  12B's next 33 tokens KL 0.00007 against f32 K, every top token the same. Gemma conversations saved to disk with f32
  keys are processed again once (another layout).
- The automatic context is estimated again once (its cache's layout version is new, and the K row switches
  `ARC_G4_KF16` / `ARC_Q3_KF32` are part of its key: a context sized for f16 K is never taken for f32 K). The
  two-pass fallback attention (`ARC_FA_GH=0`) takes one row a work-group with f16 K (its multi-row form copies f32
  words), the same results. From a self audit of the change.
- **Tried and dropped:** Gemma's prompt attention with the softmax computed once instead of by every sub-group and a
  P.V without a check per key (ocloc: ~2 -> ~1.45 instructions per FMA where it matters). Bit-identical, but no
  faster: 12B 8K prompt 129.6 / 127.5 tok/s against 126.1 / 127.3 (twice as slow with the probabilities in a second
  local array: half the work-groups per core). The scores' loop over the keys, unchanged, is what bounds it.

## 2026-11-11: a GPU benchmark, in the launcher too

- **`bin\gpu_bench.exe`** (and the launcher's More > Benchmark this GPU) measures what the GPU really does: its
  matrix engines (XMX: int8, f16, bf16 DPAS, in the sub-group size its DPAS takes), its vector engines (FP32, FP16,
  int8 DP4a) and memory read bandwidth, each next to the GPU's figure on paper (EUs x maximum clock x the per-EU
  rate of its kind: the rates behind Intel's numbers, a multiply-add counted as two). The kind and generation come
  from the driver's IP version (Meteor Lake 12.70 / 12.71, Arrow Lake H 12.74, Battlemage 20.1, Lunar Lake 20.4,
  Panther Lake 30.x). It writes a result file and a page in `%USERPROFILE%\.inferzero\bench\gpu`: a comparison table
  of every result file in that folder (results from other PCs dropped there too), then a card per GPU in the layout
  of a spec database's page, with its notes.
- The laptop (Core Ultra 5 125H): FP32 3.85 of 3.94 TFLOPS on paper, FP16 7.69 of 7.88, int8 DP4a 12.98 of 15.77
  TOPS, memory 76.2 GB/s. To check the vendors' matrix figures (the 140T: Intel 77 int8 TOPS, a spec database 154;
  a Pro B60 ~197) on the cards themselves.
- `bench/peak_bench.cpp` removed: the laptop-only DP4a / FMA probe it was, with the 125H's EU count fixed in it.

## 2026-11-11: the 140T's H8 GEMM with coalesced loads and a wider tile, opt-in (QUEUE 33.4b)

- **`ARC_GEMM_XW=1` on a GPU with 8-lane XMX** (Arrow Lake H's Arc 140T): the H8 prompt GEMMs, the 140T preset's
  chunks of 512+ tokens included, run through `zero/kernels/gemm_xw8.cl` instead of `gemm_x8.cl`. A 16-column
  group's weights come in 4 block reads instead of 32 single-word ones, an activation row in one instead of two, and a
  sub-group holds 16 tokens x 16 columns (both 8-column halves). ocloc for the 140T (`arl-h`): 90 instructions and
  6.3 memory reads per DPAS instead of 110 and 13.3, no spills at the default register file. Larger shapes
  (`ARC_GEMM_XW_TM=32`, `_NC=2`) need `ARC_GEMM_XW_GRF=256`.
- Bit for bit gemm_x8's results: checked on the laptop with the 140T preset emulated (the 27B at four shapes) and
  at start against gemm.cl. Off until a 140T measures it; `bench\xw_sweep.ps1` finds the 8-lane shapes by itself.

## 2026-11-11: self audit of gemm_xw and the fused GEMV; the 140T's XMX described

- **gemm_xw's start-up check** built its gemm.cl reference with `ARC_GEMM_TM`'s tokens: with `ARC_GEMM_TM=32` and
  `ARC_GEMM_XW_TM=16` it always failed (the reference left rows unwritten) and turned gemm_xw off. It takes 16 now.
- **`ARC_GEMM_XW_TRY=1` on an 8-lane XMX GPU** (the 140T, Alchemist) ran the 16-lane DPAS instead of emulating it.
- **The split-K pieces of an H8 GEMM** come from one helper (`h8_splits`) for gemm.cl and gemm_xw: the pieces decide
  the result's bits, so two copies of the rule could drift apart; the one-token GEMV's kernel and rows likewise
  (`gemv1_rows`, matmul1 and the fused gate + up).
- **`bench\xw_sweep.ps1`** marks a shape "xw not used" when gemm_xw did not run (H8 off, its check failed) instead of
  "identical"; **`decode_try.ps1`** compares `GEMV_GU=1`'s logits with plain's bit for bit (32 steps), not only tokens.
- Checked: the default path's logits as before (4B, 48 steps), gemm_xw = gemm.cl (emulated), GEMV_GU = two launches.
- **The Arc 140T** in docs/xe2-tuning.md section 12: an Alchemist-kind XMX (16 engines of 1024 bits a core, ~77 int8
  TOPS), where the 27B's prompt reaches ~12; its GEMM's loads per DPAS read with ocloc, and QUEUE 33.4b.

## 2026-11-11: gate + up in one GEMV launch (opt-in), website_bench warmed up

- **`ARC_GEMV_GU=1`** (QUEUE 33.5): for one token, the FFN's gate and up GEMVs run as one launch of both matrices'
  rows, each row as in its own launch (bit-identical: every step's logits of the 4B, 9B and 27B). On the laptop no
  faster (27B about 3 % slower, 4B within noise: its iGPU already reads at ~84 % of its bandwidth), so it is off; a
  B60, at 57 % of its bandwidth with 837 launches a token, measures it with `decode_try.ps1`'s new `GEMV_GU=1` row.
- **`bench\website_bench.ps1`** makes one run per model that is not counted before the sizes: a model's first run
  compiles its kernels, which landed in the first size's generation (a B60: the 4B at 128 tokens 12.8 tok/s, 61 at
  512).

## 2026-11-11: the H8 prompt GEMMs with wider XMX tiles, opt-in (QUEUE 33.4)

- **`ARC_GEMM_XW=1`** (GPUs with 16-lane XMX; off until a B60 measures it): the H8 prompt GEMMs (per-column and
  per-token scales, the Qwen dense models' default on discrete cards) run through a new kernel,
  `zero/kernels/gemm_xw.cl`, whose sub-group holds 32 tokens x 64 columns instead of 16 or 32 x 16. A 64-value chunk
  of activations is then loaded once for 64 columns and a column group's weights once for 32 tokens: twice the
  matrix work per byte loaded. On a B60 the 27B's FFN GEMMs ran at ~33 of ~197 int8 TOPS.
- Bit for bit the results of `gemm.cl`: the same integer sums, split-K pieces and epilogue; checked at start bit for
  bit (a difference keeps gemm.cl and says so). `ARC_GEMM_XW_TM` / `_NC` / `_GRF` change the tile;
  `bench\xw_sweep.ps1` measures them against the default and compares the logits.
- Checked on the laptop with DPAS emulated (`ARC_GEMM_XW_TRY=1`): the start-up check passes at four tile shapes and
  the 4B's logits are identical to gemm.cl's H8; without the switch the default path and H8 are bit-identical to
  the build before. ocloc for the B60 (`bmg`): 32 x 64 without spills at 256 registers. For Panther Lake (`ptl-h`)
  the compiler sizes the register file itself when 256 is not asked for (192 at 32 x 64, 96 at the small shapes).

## 2026-11-11: packing a MoE model from the launcher, and deleting its original

- **More > Pack the selected MoE model** in the launcher (or `scripts\zero-server.ps1 pack -Model <name or path>`)
  makes the package `serverz --pack` makes (the GGUF's experts are kept once, not twice) and copies the model's own
  `mmproj` / `mtp-` files into it. It refuses a dense model, a package, a model the running server has loaded (or
  a server that does not say which it has), and a disk without room. The default model (`last-model.txt`) moves to
  the package. Run again on a packed model, it only copies the side files that are missing.
- Then it offers to **delete the original GGUF** (`zero-server remove-original`), with a warning: deleted for good,
  and the package runs in inferzero only. It deletes nothing unless the package's GGUF and expert file are both
  there and the package was made from that very file: a package now holds `package-source.txt`, its source's full path
  (two folders' models of one file name make one package folder). Every part of a split model goes, and in a
  Hugging Face cache the file the snapshot links to; a link whose file is missing stops it before anything goes.

### Self audit of the 2026-11-11 changes

- **`"cache_prompt": false` could make a later answer wrong:** the system prompt's saved state (the hybrid Qwen
  models: the DeltaNet states at its end) stayed after the request dropped the rest, and a later request that shared
  the start of the `cache_prompt: false` prompt got it restored under its own tokens. It is dropped now, and the
  conversation the server held is saved to the session cache first (it was lost).
- **Projector names** match at a `-` / `_` only: `mmproj-qwen3` is no longer Qwen3.5-4B's in a shared folder. The
  launcher's copy uses the same rules, independent of Windows' language.
- **`runz --kl-ref`** takes the prompt in one chunk only with a reference of every position: a last-position
  reference (and `--kl-decode`) checks the passes the server takes, as before 2026-11-11.

### Second self audit (pack and remove-original)

- **A running server picked the original after a pack:** its model list was from its start, so switching to the
  packed model loaded the original GGUF and wrote its whole expert file again next to the package (or failed once
  the original was gone). A model that has a package since, or whose file is gone, now makes it read the list again.
- **The package's record of its source is a file of its own** (`package-source.txt`): the folder's `source.txt` is
  the expert store's, rewritten whenever a same-named GGUF from another folder runs, so it could name a file the
  package was not made from. Packing again a package of another same-named file is refused too.
- **remove-original deletes nothing when any part is open** (a server, another program) and moves the default model
  first: before, it could stop halfway with the default model pointing at deleted files. The launcher's own model
  list being read, or a pack running, no longer counts as a running server; the launcher passes its port; it reads
  the list again only after the delete question; it does not start the model being packed. The free-space check
  leaves out an expert file the engine wrote before (it moves into the package).

## 2026-11-11: Gemma 4 12B's every-position KL check fixed (it was the check, not the model)

- **Why `tests\run_tests.ps1`'s Gemma 4 12B KL read 10.75:** with weights decoded just in time, a prompt of up to
  32 tokens (two-pass models; 40 otherwise) goes through the layers in passes of 8 tokens. After the 20-token test
  prompt the residual stream held the last pass's 4 rows only, and `runz --kl-ref` read rows 0..19 of it: every
  position compared garbage, while the model's own next token and text were right. E4B and 26B (weights resident:
  one chunk) passed.
- **Fix:** `Engine::keep_rows` (set by runz for `--kl-ref` and `--perplexity`) takes the prompt in one chunk, and
  `logits_at()` refuses a row the last pass did not leave instead of reading it. The server is unchanged.
- Checked: Gemma 4 12B KL 0.00000 (max 0.00001), top token 20 / 20; E4B 0.00389 and 26B 0.01014 as before; the
  4B's logits bit-identical; `--perplexity` runs (4B, 2 x 512: 12.11).

## 2026-11-11: the update notice, old packages explained, `cache_prompt`, projectors in a shared folder

- **The launcher said a newer package was out when it was this one** (issue #8): a check made by an older package
  was kept for 20 hours in the data folder every package shares. A check now holds for the package that made it
  (its folder and serverz.exe's time).
- **A package or GSQ conversion from before 2026-11-10 says why it cannot load:** the model list and the launcher
  show what to do (delete the package's folder, then load the original GGUF or `--pack` it again; convert a GSQ
  model again) instead of "NOT supported". The launcher's update bar no longer promises more than it keeps.
- **`"cache_prompt": false`** in a chat request evaluates the whole prompt (llama.cpp's option); it was ignored, and
  a repeated prompt came back from the cache (a user's report from a Panther Lake B390). Checked: 376 tokens
  evaluated again with it, 371 cached without.
- **A projector in a folder of several models** is taken only by the model it is named after (`mmproj-<name>...`);
  in a Hugging Face snapshot folder, or a folder with one model, as before. The first `mmproj*.gguf` of a folder
  went with every model in it (the same report: an unrelated one on a Qwen3.8 27B fine-tune). The log says when it
  found others; `--mmproj FILE` takes any.
- **Measured on a B60 (issue #4):** `ARC_WIDE_ROWS` +2.3 % on the 27B's generation (swiglu_quant -72 %),
  `ARC_GEMV_SMALL_WG` no gain (kept, bit-identical); docs/before-after.md.

## 2026-11-10 (evening): self audit after the transcription crash (docs/self-audit-2026-11-10b.md)

- **A live recording's draft is kept** when its accurate pass is stopped or a part of it fails: before, the partial
  text was written over the whole meeting's draft. Checked through the API: a 5-minute live recording's 70-segment
  draft survived a cancelled pass.
- **A live recording goes on after a failed write** (disk full): the WAV stream's error state is cleared, so the
  page's retry is written once there is room; before, every later piece failed.
- **A recording whose compression failed can be merged** (it was refused for good as "still being compressed").
- **Model files checked against their dimensions when loaded:** a tensor with more rows than the model's dimension
  overflowed a host buffer (`load_w_pad`); attention, DeltaNet and FFN matrices and the norms are now checked
  (`expect`, `load_vec`); `expert_used_count` 0, `full_attention_interval` out of range, tensors with too few
  dims and short scale vectors are refused with the tensor's name.
- **Launches:** a zero-size one runs nothing, one past 32 bits is refused. Staging buffers made again drop their
  "already decoded" mark; the crash log no longer reads a released kernel; Whisper's copies are checked.
- Checked: every model on the laptop loads and answers, the 4B's logits bit-identical (plain and MTP), Gemma 4
  E4B / 26B KL as before, Qwen3-ASR after a warm-up, Whisper.

## 2026-11-10: prompt attention on XMX by default on discrete cards; website_bench in a package

- **`ARC_ATTN_XMX` is on by default on discrete cards** (Xe2 / Xe3 with 16-lane DPAS; iGPUs unchanged, `=1`
  forces it, `=0` turns it off): a Pro B60's 64K-token prompts 4B +11.7 %, 9B +8.4 %, 27B +2.1 %, 35B-A3B +10.2 %,
  4K prompts +0.2-1 %, logits bit-identical (issue #4, package bb8952f; docs/before-after.md).
- **`bench\website_bench.ps1` ran nowhere but in a git folder:** it asked git for the commit, and in a package
  (no `.git`) Windows PowerShell stopped the whole benchmark at git's error (issue #4). The commit now comes from the
  package's `bin\PREBUILT.txt`, else from git when there is one.
- Found while testing: `tests\run_tests.ps1`'s Gemma 4 12B KL fails (10.75, every position) with the last commit's
  runz too, so not from today. Not its reference (made again: the same KL, and its last position equals the engine's):
  the every-position logits `--kl-ref` takes for the dense 12B are wrong; generation is not affected (QUEUE).

## 2026-11-10: the server's crash at the start of a meeting's transcription, found and fixed

- **What happened:** an 83-minute meeting (two tracks) recorded fine with its live text, then the server ended at
  the start of its accurate pass, every time it was started again (three times), with nothing in its log: an access
  violation in `ze_intel_gpu64.dll` (0xC0000005, offset 0x386975; the same as a crash on 2026-11-04).
- **Why:** the engine's argument cache (`zero/clrt.h`) skipped setting a kernel argument whose value was the same
  as at its last launch. Level Zero keeps the *allocation* an argument was set with, not only its address. The
  server warms Qwen3-ASR's encoder up on 1 s of silence at start, so a recording's first long part grows the
  encoder's buffers (freed, made again); a new buffer at the address of a freed one kept the kernel's argument on
  the freed allocation, and the next launch (`q3a_gather`) read it inside `zeCommandListAppendLaunchKernel`.
  Reproduced outside the server with `asr_testz --warm` (a 1-s clip, then 5 minutes): it crashed with the cache on
  and ran with `ARC_ARG_CACHE=0`.
- **Fix:** every free of Level Zero memory the engine makes goes through `zrt::mem_free`, which counts it; a cached
  argument counts as unchanged only within the same count, so after any free each kernel sets its arguments again
  once. The case that crashed runs (twice, and with its text: the same as without the warm-up); the 4B's logits are
  bit-identical, plain and with MTP; the meeting was transcribed to its end.
- **A crash now names itself in the log** (`src/crash_log.h`, the server): the exception and the address it read,
  the launch the thread was appending (kernel, sizes, arguments, any argument in no live buffer), and the stack
  with module!function (file:line with a .pdb next to the exe).
- `asr_testz --warm` encodes 1 s of silence first, as the server; `tests\run_tests.ps1` runs it on a generated
  300-s clip (section 7).

## 2026-11-10: small kernels and matrices of few rows in a generation step

- **Measured first:** a Pro B60's 27B GEMVs read 68 % of its bandwidth over all; the Q4_K output head 99 %, the
  codebook FFN 64-77 %, the 1,024-row K / V projections 45 % and DeltaNet's 96-row alpha / beta 14 % (issue #4's
  per-kernel profile against each matrix's bytes in the GGUF; docs/xe2-tuning.md section 10). `swiglu_quant` (15 us
  x 64 a token) and `rmsnorm_quant` (7 us x 129) ran one work-group per row.
- **`ARC_WIDE_ROWS`** (on, every GPU): a 1-8 token pass's `swiglu_quant` (Qwen and MoE paths), `gated_norm_quant`
  and `attn_gate_quant` run one sub-group per superblock (as Gemma's `swiglu_quant` did), `rmsnorm_quant` one
  work-group per 16 superblocks; the same arithmetic. `=0`: as before.
- **`ARC_GEMV_SMALL_WG`** (on, discrete cards, `zero/xe2/`): the single-token GEMV of a matrix with at most 512
  sub-groups in work-groups of 64 instead of 256; the same sums. `=0`: as before.
- `bench\decode_try.ps1` has rows for both off, their tokens checked against plain. Not measured yet: the laptop's
  check (bit-identical logits, speed) waits for a free GPU; the B60's speed for its owner.

## 2026-11-10: the GGUF keys are inferzero.*

- **The metadata keys inferzero writes into a GGUF** are `inferzero.package`, `inferzero.experts_file`,
  `inferzero.source`, `inferzero.virtual.names` / `.types` / `.dims` (a package, `zero/gguf_pack.h`) and
  `inferzero.expert_format` (a GSQ conversion, `tools/convert_gsq.py`). The old names `arcinferzero.*` (and
  `arcinfer.expert_format`) are no longer read: a package or a GSQ model made before this is refused when opened,
  with the way out in the message. A package: delete its folder under `%USERPROFILE%\.inferzero\zero\models`,
  then load the original GGUF (its first start writes the expert file again: 15 GB for the 35B, 52 GB for Flash
  Next) or make the package again with `serverz --pack <the original GGUF>`. A GSQ model: convert it again
  (`setup -Models 35b-gsq`). The server's model list shows such a file as unsupported and the log says why, once
  per file; setup no longer counts it as downloaded. The expert file format (`.experts`) is unchanged.
- **The desktop shortcut from before the rename** (`arcinferzero.lnk`, which opened the removed
  `arc-launcher.cmd`) counts as a shortcut only while its target exists, and a new `inferzero.lnk` replaces it.
- `tools/gguf_dump` prints why a file is refused (an old package, a truncated file) instead of aborting.
- Metadata names only: the logits are bit-identical (Qwen3.5 4B, `runz --dump-logits` before and after).

## 2026-11-10: the old command names are gone

- **Removed:** `arc-launcher.cmd`, `arc-server.cmd`, `scripts\arc-launcher.ps1`, `scripts\arc-server.ps1` and
  `scripts\pi-arc.ps1` (the aliases kept at the rename earlier today), zero-pi's `--arc-model`, and the `pi-arc` /
  `arc-server` shims `install-pi` wrote into Pi's bin folder (`install-pi` now deletes them). The commands are
  `zero-launcher`, `zero-server`, `zero-pi` and `zero-recorder` only; scripts and shortcuts that used the old names
  need the new ones.
- **Pi's command shims hold the folder's path:** after the repository folder is moved or renamed (this laptop's was,
  with the project), `zero-pi` and `zero-server` in a terminal fail with a missing-file error until `install-pi` (or
  `setup`) runs again; `install-pi` says so now.
- Also: three stale git worktrees from the old folder pruned; `.gitignore`'s comment names the data folder
  `%USERPROFILE%\.inferzero`.

## 2026-11-10: self audit of the website, logo and renames (docs/self-audit-2026-11-10.md)

- **The package's key filter** (`zero/gguf_pack.h`) is `arcinferzero.` again: the rename had made it `inferzero.`,
  so a re-pack doubled the package keys and a GSQ model's `expert_format` was copied into new packages.
- **Recordings of an older recorder** are recognised by the server (not imported while still written) and block a
  second `zero_recorder start`.
- **`zero/kernels/decode.cl`** is back to its text before the rename (a comment): the kernel cache stays valid.
- **Old names still found:** a model asked for as `arcinferzero/<name>`; `%TEMP%\arcinferzero-view-*` cleaned up; bench
  results recorded as `arcinferzero`; `bin\arc_recorder.exe` of an older build; Pi `settings.json` entries.
- **The alias scripts** return the script's exit code; the web UI's prompt explains the name as infer + zero.
- **Website and docs:** the grid animation, the speed-up claim (2.7-7.4x, as the bars show), numbers that disagreed,
  the code tabs' accessibility; `zero_recorder.exe` spelled right; the recording commands; links to the archived log;
  a note for packages from before the rename.
- Checked: serverz, runz and zero_recorder build; Qwen3.5 4B logits bit-identical to the package from before the rename
  (4ee6693, SHA-256); every changed script parses; the recorder and alias commands run.

## 2026-11-10: the commands start with zero-, as the logo

- `arc-launcher` -> **`zero-launcher`**, `arc-server` -> **`zero-server`**, `pi-arc` -> **`zero-pi`**, `arc-recorder` ->
  **`zero-recorder`** (`bin\zero_recorder.exe`); the files moved (`zero-launcher.cmd`, `zero-server.cmd`,
  `scripts\zero-launcher.ps1`, `scripts\zero-server.ps1`, `scripts\zero-pi.ps1`, `tools\zero_recorder.cpp`) and every
  living doc and help text with them (560 uses in 38 files; the planned `arc-transcribe` is `zero-transcribe`, zero-pi's
  `--arc-model` is `--zero-model`).
- **The old names still work:** `arc-launcher.cmd`, `arc-server.cmd`, `scripts\arc-server.ps1`, `scripts\pi-arc.ps1` and
  `scripts\arc-launcher.ps1` hand over to the new ones; `install-pi` puts `zero-pi`, `zero-server`, `pi-arc` and
  `arc-server` on PATH (shims installed before point at the old scripts, which hand over); `--arc-model` is still taken.
- **The recorder's stop event** became `inferzero-recorder-stop` with the project's rename; `bin\zero_recorder.exe
  stop` / `status` (what `zero-server record-stop` / `record-status` run) now also find a recording an older exe
  started (`arcinferzero-recorder-stop`).
- Checked: `zero-recorder` builds and reports "not recording"; every changed script parses; `zero-server help` and
  `status`, `zero-pi --help`, and the same through `arc-server.cmd`, `scripts\arc-server.ps1` and `scripts\pi-arc.ps1`;
  `zero-launcher -SelfTest`; `install-pi` on this laptop (the four shims; Pi's provider "inferzero" alone).

## 2026-11-10: arcinferzero is now inferzero

- **Why:** Intel's trademark guidelines ask third parties not to put an Intel mark, or any part of one, into a product
  name, and "Arc" is Intel's graphics brand. The new name keeps the rest; the 4 x 4 "Z" logo stays.
- **Renamed:** the code, the scripts, the web UI, the launcher, the website and the living docs (672 uses in 120
  files); packages are now `inferzero-<version>-win64.zip`; the README and the website say "formerly arcinferzero"
  and carry Intel's trademark notice.
- **Carried over by themselves:**
  - the data folder `%USERPROFILE%\.arcinferzero` moves once to `%USERPROFILE%\.inferzero` (the same drive: a rename,
    instant), and the paths in `last-model.txt`, `launcher.json`, `server.json` and `zero\ctx-auto.txt` follow it;
    if a file is held open (an older server running) the old folder is used as it is (`src/app_dir.h`,
    `scripts/app-dir.ps1`); the bench scripts and the launcher's Downloads dialog now ask `Get-AppDir` too;
  - Pi: the provider becomes "inferzero" (the "arcinferzero" and "arcinfer" entries are replaced), and `pi-arc`
    takes `arcinferzero/<id>` and `arcinfer/<id>` as before;
  - the web UI's settings in the browser (`arcinferzero-prefs` is read when `inferzero-prefs` is not there yet);
  - a desktop shortcut `arcinferzero.lnk` counts as the launcher's shortcut;
  - `arc-server` reads a package's `VERSION.txt` and the server log with either name.
- **Kept on purpose:** the GGUF keys packages already carry (`arcinferzero.package`, `.virtual.*`, `.experts_file`,
  `.expert_format`, `.source`: an older engine must still read a newer package); the `ARC_*` switches (settings
  people have in their scripts); the command names `arc-launcher`, `arc-server`, `pi-arc`, `arc-recorder` (renamed
  later the same day: the entry above); history
  (older CHANGELOG entries, `docs/archive`, dated reports, `docs/before-after.md`, `BUGS.md`).
- **Checked:** the ten package exes build; every changed PowerShell script parses; the data-folder move in a test
  profile for the scripts and an exe (the three state files rewritten, JSON's doubled backslashes and the lower-case
  context file included; a second start uses the new folder); `arc-launcher -SelfTest` in the real profile (it moved
  this laptop's data folder; the state files followed).

## 2026-11-10: the logo: a 4 x 4 grid drawing a Z, everywhere

- The "az" circle (two letters of a system font) becomes a drawn mark: 4 x 4 rounded tiles, ten lit, drawing a Z
  for zero; the tiles are the engine's int8 GEMM tiles and its mixture-of-experts slots, only some of them in memory.
  The launcher's accent colours stay (Windows' #0078D4, on a dark theme #59A7E3); the dim tiles are #C9D3DE /
  #2B3440. No font, so it looks the same on every system.
- Where: the launcher's splash, window and tray icon (`Draw-ArcMark` in scripts/arc-launcher.ps1, snapped to whole
  pixels at 16 / 32 px; the tray is accent when a model runs with a smaller green dot, grey when stopped), the web
  UI's `/icon.svg`, and the website's favicon, nav and footer (`site/icon.svg`, byte for byte the server's).
- Checked: `arc-launcher -SelfTest` in light and dark (the splash), the tray icon at 32 and 16 px on a light and a
  dark taskbar, the website in the browser; serverz rebuilt. The concepts compared (the grid, a slashed zero, a gauge,
  a drawn monogram) are in a local research page, not in the repository.

## 2026-11-10: one logo: the launcher's "az" mark on the website and the web UI

- The web UI's icon (`/icon.svg`: the browser tab, and the app icon when installed) was a green chat bubble, and the
  website had an arc of its own. Both are now the launcher's splash mark: a circle in Windows' default accent
  (#0078D4) with a bold white "az" in Segoe UI; on a dark browser theme the accent 35 % lighter (#59A7E3) with a black
  "az", as the launcher draws it on a dark window.
- `site/icon.svg` is the website's favicon and the same SVG the server sends; the website's wordmark is Segoe UI
  semibold, as in the splash. The app manifest lists the icon as `any` (a round mark, not a full-bleed one).
- Checked: the server's SVG is byte for byte `site/icon.svg`; both themes rendered in the browser next to
  `arc-launcher -SelfTest`'s splash picture.

## 2026-11-10: Qwen3.8 Flash Next MTP from its head file (QUEUE F34, GitHub #7)

- **MTP for Flash Next**: its MTP head as a file of its own, llama.cpp's form (`mtp-Qwen3.8-Flash-Next-Q4_0.gguf`,
  base Flash Next's head, in SC117's Swift-1.5 repository), found next to the model or given with `--mtp-model FILE`
  (runz; the server and the catalog list such a model with MTP). Its `blk.48.*` tensors join the model file's tensor
  list (`GgufFile::adopt`), so the existing loaders read the layer; the model's expert file is not touched (the
  layer is left out of it).
- The head as llama.cpp's `qwen4exp` `graph_mtp`: per row and residual stream `eh_proj([rmsnorm(e) * enorm |
  rmsnorm(h_s) * hnorm_s])` (new kernel `mtp_concat_hc`; the (row, stream) pairs as rows of one matmul), then a
  Flash Next layer (mixers, QSA attention with its own KV and QSA state, 512-expert MoE), its next h the layer's
  residual streams, logits through its own head mixer and the model's lm_head. The model's hidden rows for it are
  the residual streams before the head mixer (hc x E: `mtp_hw()`, also in slots, snapshots and sessions).
- Its 512 Q4_0 experts **pinned** in the expert cache: built in the expert file's record form into one device buffer
  (1.41 GiB, counted before the cache is sized; a dry load counts it too), the cache's table row fixed
  (`XCache::pin`, kept when the table is reset), no `ensure()` for that layer. Q4_0 experts with K 640 accepted.
- Laptop, code prompt, 128 tokens: plain 6.65 tok/s, MTP 1 / 2 / **3** / 5 drafts 7.02 / 7.57 / **7.84** / 7.06
  (96 % accepted at 3, 3.88 tokens a pass; more drafts read more experts from the SSD per pass): **3 by default**.
  Greedy text equal to plain on every run; the default path bit-identical to the build before (Flash Next, 35B,
  4B). serverz with DFlash2 and the head: both picked per step.

## 2026-11-10: website benchmark's KL for Gemma with BOS

- **`run-tests.cmd`** (bench\run_tests.ps1) for testers: double-click, leave it running, drag the zip it puts on the
  Desktop into the GitHub issue. It finds the models (Hugging Face cache, packages, and every folder the server
  reads), runs `f33_sweep` (Qwen models) and `website_bench`, keeps the PC awake, writes the user's folder in paths as
  %USERPROFILE%, and packs the logs. `run-tests.cmd quick` (about an hour), `run-tests.cmd quick 27B` (one model).
- **`website_bench.ps1` found no models outside the Hugging Face cache** (issue #4: models in `C:\hf`) and then stopped
  at its summary: it now also asks `serverz --list-models` (model-dirs.txt folders), takes DFlash2 drafters from
  there too, and ends with a plain message when nothing is found. Flash Next's split first part is found too.
- **`mgpu_probe`: host memory <-> GPU one way** (pinned / pageable, compute queue / copy engine, 40 / 256 MB) and every
  GPU at once: the link that MoE experts streamed from RAM come over. The 6.6 GB/s measured on the B60s before was
  GPU -> host -> GPU (two copies), not this. Laptop iGPU: pinned 36.8 GB/s host -> GPU, pageable 12.5, copy engine
  20.5.
- B60 results of 2026-11-10 (issue #4, `f33_sweep`, one GPU): `ARC_ATTN_XMX=1` bit-identical and faster where
  attention weighs most (35B 64K prompt 1,808 -> 1,993 tok/s, +10 %; 27B +2 %; attention kernel time -5 to -15 %);
  the chunked DeltaNet on XMX slower than `gdn_recur` (27B 4K: 900 vs 290 ms), so it stays off.

- `runz --dump-ids` (used by `bench\website_bench.ps1` for its KL against f32) now puts BOS first when the model
  asks for it (`tokenizer.ggml.add_bos_token`, the Gemma models); Qwen ids are unchanged. Without it the KL measured
  a prompt the model never sees: Gemma 4 E4B 0.01644 -> **0.00273**, 26B-A4B 0.09272 -> **0.00479** (480 tokens of
  USAGE.md, laptop). Only the measurement changed; the engine did not.
- runz warns when the model and its context need more GPU memory than the share less the driver's part
  ("does not fit: ... expect paging"); `--must-fit` stops there with exit code 3. `website_bench.ps1` passes it and
  marks such a size "does not fit" instead of measuring hours of paging: Qwen3.8 27B at 131K needs 17.13 GiB, over this
  laptop's 16.5 GiB share (back to its default after a driver update), and had measured 8.6 tok/s and a first token
  after 4.2 hours (at 32K: 14.09 GiB, 125 tok/s). The server's `--ctx auto` already stays inside the share.
- **Qwen3.8 Flash Next in setup's model list** (`setup -Models flash`, the launcher's first run; asked for in issue #4):
  the two IQ3_S parts from ISTA-DASLab (84.5 GB) and bartowski's f16 image projector, copied next to them. The disk
  check counts 141 GB (the first start writes the 56 GB expert file; `serverz --pack` then makes a package and the
  parts can go to a backup disk). `-Models all` leaves it out. A model kept as a package counts as present.
- **Launcher: a Downloads dialog** (Get models, and More > Download models; `scripts\launcher-downloads.ps1`): several
  models of the list at once, or a Hugging Face link (a repository, a folder or a file; the other parts of a split file
  and an image projector ticked with it, the projector copied next to a model in another folder); Pause / Continue
  (curl stopped, the `.part` files kept, resumed with `-C -`); a speed limit (`ARC_HF_RATE`, curl `--limit-rate`),
  changed while it runs; a cut connection tried again without end (5 s doubling to 60 s, back to 5 s once data flows;
  a token or a missing file shown at once instead); models kept as packages count as here. `hf-get.ps1` ends a
  stalled connection after 2 minutes under 1 byte/s (`--speed-time`), so it is tried again. Tested in Windows
  PowerShell 5.1: 5 MB/s measured 4.5-5.0, 1 MB/s 1.0; pause, continue and two cut connections on a 0.67 GB file; the
  first-run welcome dialog is unchanged.
- **Launcher splash screen**: shown as soon as the theme's colours are read, while the fonts and the window are made
  (before, a few seconds with nothing on the screen); the "az" mark in the Windows accent colour, a status line, the
  theme's light / dark colours and rounded corners. Nothing new to load: WinForms and System.Drawing as before, the
  mark drawn in code. It closes just before the window shows (closed in the window's Shown event it was the window's
  owner and took the window with it). `-SelfTest x.png` also saves it as `x-splash.png`.

## 2026-11-09: DeltaNet in chunks and prompt attention on XMX, opt-in (QUEUE F33.1 / F33.2)

For the Arc Pro B60 (issue #4) and the Arc 140T (issue #5); every switch is off by default, the default path is
bit-identical to before (laptop, `--dump-logits` of a ~1,500-token prompt and 24 tokens: Qwen3.5 4B, Qwen3.8 27B,
Qwen3.6 35B-A3B against the build before tonight's work, bb8fd33; `bench\f33_sweep.ps1`'s check, USAGE.md as the
prompt and 16 tokens, all seven models of the laptop: Gemma 4 E4B, Qwen3.5 4B and 9B, Gemma 4 26B-A4B, 27B, 35B,
Qwen3.8 Flash Next).

- **Chunked DeltaNet, `ARC_GDN_CHUNK`** (prompt passes of 32+ tokens; MTP / DFlash2 verification keeps `gdn_recur`):
  - The rule in chunks of 32 tokens: per chunk lg = the cumulative log decay, A = beta exp(lg_l - lg_m) k_l.k_m and
    P = exp(lg_i - lg_l) q_i.k_l; the tokens' deltas by one forward substitution, the outputs from P and the state at
    the chunk's start, the state moved once a chunk.
  - `=3`: on XMX. `gdn_chunk_x` (zero/kernels/gdn_x.cl, 16-lane DPAS: B580 / B60) holds the state as DPAS
    accumulators and does every product of the chunk as f16 DPAS (k.k and q.k, state x k, state x q, the deltas
    through (I + A)^-1, the outputs, the state update): 192 DPAS per sub-group and chunk, no spills (ocloc,
    Battlemage). `gdn_chunk_x8`: the same at 8 lanes for the 140T (264 DPAS, ocloc Arrow Lake H). Every f16
    operand is scaled by a power of two and split into high and low halves (~22 bits): the low halves stay out of
    f16's subnormal range, which a DPAS may flush (found by an independent review of the kernels). Checked once per
    device at start against `gdn_chunk`; a difference above 1e-4 keeps `gdn_recur`. Without XMX `=3` runs
    `gdn_recur`.
  - `=1` / `=2`: the vector form `gdn_chunk` (any GPU; 4 work-items per state row, 4x `gdn_recur`'s). **No gain on
    the laptop** (4K prompt, DeltaNet kernel time): 4B `gdn_recur` 241-272 ms, `=1` 3,506-3,662 ms, `=2` 2,367-2,409 ms;
    9B 235 / 3,507 / 2,358 ms (512 -> 353 / 388 tok/s); 27B 782-806 / 12,420 / 8,832 ms (prompt 136 -> 95 / 110
    tok/s); 35B 538-542 / 5,802 / 4,448 ms (782 -> 381 / 454 tok/s); Flash Next 435 / 7,492 / 5,090 ms (207 -> 156 /
    172 tok/s). Kept for a B60 to measure.
  - Its own kernel part, zero/kernels/ops_gdn.cl (the last ops part), so ops.cl and the other ops kernels' cache are
    as before.
- **Prompt attention on XMX, `ARC_ATTN_XMX=1`** (zero/kernels/attn_x.cl, put in front of the ops parts for
  `attn_q8`; 16-lane DPAS, Qwen): attn_q8's int8 q.k sums (8 DPAS per 16-key tile at 256 dims) and its u8 P x int8 V
  sums (16 DPAS) replace 2 x 256 DP4a per tile; everything else is attn_q8's in the same order. DPAS sums integers
  exactly: bit-identical, checked at start bit for bit (else attn_q8). Not for the 140T: its attention kernels run
  16-lane sub-groups, its DPAS takes 8 (QUEUE F33.2b).
- **Checks without XMX:**
  - The chunked math in numpy against the sequential rule (f32: 1e-7 relative, chunks of 16 and 64, decays to 1e-6).
  - On the CPU's OpenCL (oneAPI), against a double-precision reference, 100 tokens (three chunks and a short one),
    gated and fused gates: `gdn_chunk` output 5.5e-7, `gdn_chunk_x` / `gdn_chunk_x8` with DPAS emulated 6.8e-7
    (state 1.4e-7); `gdn_recur` 1.4e-7.
  - Laptop GPU with DPAS emulated (`ARC_GDN_XEMU=16` / `8`, `ARC_ATTN_XEMU=1`): both DeltaNet forms pass their start
    check (4.5e-7 of the result); the attention's start check and the logits of the 4B, 27B and 35B (~1,500-token
    prompt, 24 tokens) bit-identical to attn_q8, five shapes of the kernel test too (33 to 100 tokens, 3 and 4
    heads per sub-group).
  - The laptop GPU compiler got a byte of a shuffled word wrong in an unrolled loop (the emulation's first form):
    every attention output of the check apart, the CPU's OpenCL exact. The emulation now splits bytes before the
    shuffle (both agree). Only the emulation was affected; real DPAS does not shuffle.
- **KL against `gdn_recur`** (4 perplexity chunks of USAGE.md): 4B `=1` / `=2` 0.00084 (perplexity 13.2847 ->
  13.2987), `=3` emulated 0.00087 (16 lanes) / 0.00091 (8 lanes); 27B `=1` / `=2` 0.00063 (6.3926 -> 6.3941); 35B
  `=1` / `=2` 0.0052 (9.9937 -> 9.9532). Not bit-identical sums change later int8 roundings (and in the 35B the
  experts some tokens pick, as a prompt chunk size of 512 instead of 8,192 does: 0.0026-0.017): that is this size of
  KL, below 0.015. 9B `=1` / `=2` 0.00071 (10.9342 -> 10.9305), `=3` emulated (8 lanes) 0.00078; 27B and 35B `=3`
  emulated 0.00062 / 0.0053.
- **Flash Next** moves more: `=1` against `gdn_recur` 0.025 (perplexity 7.2929 -> 7.3449; precise mode 0.0125), `=3`
  emulated 0.026. Against the f32 reference (`ref_forward`, 480 tokens of USAGE.md) the chunked kernel is as close as
  `gdn_recur`: fast mode 0.0134 against 0.0117 (top token 456 / 480 against 451), precise 0.00879 against 0.00875
  (463 against 464): within its limits (~0.02 fast, 0.015 precise). The two runs sit ~0.012 from f32 each, on
  different sides, so they are 0.025 apart.
- `bench\f33_sweep.ps1` for the B60 / 140T owners: per model, prompt speed at 512 to 65,536 tokens for every mode,
  the kernels' own time, KL, and the attention's logits against attn_q8. `ref_forward.exe` is in the package now,
  for `bench\website_bench.ps1`'s KL against f32.

## 2026-11-09: MoE expert files kept with their model

- A MoE GGUF's expert file went loose into `%USERPROFILE%\.arcinferzero\zero\`, so a deleted GGUF left it there
  unnoticed (a Gemma 4 26B's 12.3 GB, found and deleted by hand). It is now written next to the GGUF when the GGUF is
  inside the data folder (`models\`, or a package in `zero\models\<name>\` as before), else into a folder of its
  own, `zero\models\<model name>\`, with a `source.txt` naming the GGUF (`zero/expert_store.h`, `place`).
- One written to `zero\` before moves to its place (a rename, with its `.lru` state) the next time its model is
  loaded: not written again. A package made from a GGUF still takes that file over as before.
- At each MoE load the log names an expert folder whose GGUF no longer exists (on a drive that is there: an unplugged
  drive's GGUF is not "gone") and any expert file still loose in `zero\`. Nothing is deleted automatically.
- Checked with a test in a temporary data folder (no model, no GPU): the outside and inside places, `source.txt`, an
  old file and its `.lru` moved, nothing moved while a package is made, and both log notes.
- Self-audit the same day: a package made from a GGUF lands in that GGUF's folder, and the original GGUF is usually
  deleted afterwards; the log then called the package's folder deletable. A folder holding `<folder>.gguf` (a
  package) is never named now. A split GGUF's folder is named without its `-00001-of-N`, as its package would be,
  and `source.txt` holds the GGUF's full path. Both cases added to the test.

## 2026-11-09: earlier server logs kept

- `arc-server` (and so the launcher) wrote `server.log` anew at each start, so the log of the server before a restart
  or a crash was lost. That log now moves to `%USERPROFILE%\.arcinferzero\logs\`, named by the local date and time
  of its last line, `server_2026-11-09_14-13-48.log` (readable, sorted by name = by time; two in the same second:
  `server_... (2).log`), and the newest 10 are kept. `server.log` stays the running server's. Checked over 14 starts
  in a test folder: 10 files, one from the day before removed as the oldest, a same-second pair kept apart; a log
  still held open is left as it is without an error.

## 2026-11-09: discrete cards, MoE generation defaults from a B60's run

- `ARC_XALL` is on by default on discrete cards: when every expert of a MoE model is in the GPU's cache, a token's
  layers no longer read their experts' ids on the host. A Pro B60's 35B: 78.77 -> 88.05 tok/s, host waits 43 -> 3 a
  token, the same tokens. `ARC_XALL=0` turns it off.
- The single-token GEMV prefetch (`ARC_GEMV_PF`) is off for MoE models on discrete cards (the same B60's 35B 78.77
  with it, 85.15 without); dense models keep it (27B 23.21 with it, 22.45 without).
- iGPUs are unchanged: both switches stay off there.
- `bench\decode_try.ps1`: for MoE models the rows now try the old settings (`XALL=0`, `XALL=0 GEMV_PF=1`, `mtp XALL=0`)
  against the new defaults; dense models keep `GEMV_PF=0`.
- `docs/xe2-tuning.md`: the B60's 2026-11-09 generation and prompt numbers.

## 2026-11-09: web UI, queue and steer messages while an answer runs

- Send (or Enter) while the model answers no longer does nothing: the message goes into a queue above the message
  box and is sent when the answer ends, the queue in order; the draft in the box is kept. A queue belongs to its chat
  and goes on when the chat is opened again.
- Each queued message has **Steer**, **Edit** (back into the box) and **×**. Steer puts it into the answer under
  way: while tools run it is added to the conversation before the model's next round (the tools are not stopped);
  while the model writes, the answer stops where it is (kept) and the message is sent at once.
- Checked in the browser with the server's replies simulated in the page (no request reached the model, no chat
  saved): two queued messages sent in order after the answer, the draft kept; Steer while writing stopped the answer
  after 11 of 30 words and sent the message within 0.6 s; Steer while a 2.5 s web search ran left it running and the
  next request held assistant (call), tool, user (the steered message).

## 2026-11-09: web UI, the menu button on your own messages visible

- The ▾ button of a message (Edit, Copy, Delete) was drawn in muted grey on the user's green bubble, nearly
  invisible, so deleting one's own message looked impossible. It now takes the bubble's text colour. Checked: the
  user's menu holds Edit, Copy and "Delete (and its files)", an answer's Copy and Delete.
- Near the bottom of the chat the menu opened under the message box (its place assumed a 140 px menu; yours has three
  rows), so its Delete could not be clicked. It is now measured and opens above the button when there is no room
  below. Checked at 1280x800: the last message's menu at y 518-636 above its button (647), every row clickable (it
  was placed down to y 769, under the box at 742); the first message's still opens downwards.

## 2026-11-09: Qwen's reasoning: the line-break exemption undone

- The change below ("no stray ctlsc") is undone: every token the reasoning used is penalized again, line breaks
  included. With line breaks free of the penalty the 35B circled twice in one chat within hours, writing short lines
  ("(Done).", "(End).", "<br>", indented) without closing its reasoning (18,038 characters once, 82 `<br>` lines;
  the second stopped by hand). The presence penalty is there against such loops, and these lines are mostly line
  breaks and indents. A stray "ctlsc" now and then is the smaller harm. `Sampler::set_presence_penalty` keeps its
  optional exempt list (unused, tested in `test_sampler`).

## 2026-11-09: web UI, the thinking box rendered as Markdown

- The thinking box showed the reasoning as plain text, so a model's `**bold**`, tables and `<br>` appeared as
  written (the 35B uses them often). It is now rendered with the answer's Markdown renderer (HTML escaped as there;
  `<br>` the only tag kept), while streaming, when the box is opened and in a reopened chat. Checked: bold, a table
  with `<br>`, a list, and a `<script>` line shown as text.

## 2026-11-09: web UI, several files attached at once

- Choosing several files with the attach button (or dropping them on the chat) attached only the first. The page
  cleared the file input right away, which also empties the browser's live list of the chosen files while the first
  upload was still running. The list is now copied first (checked: 3 files chosen, the live list 0 after clearing,
  the copy 3). Every model was affected, OCR ones included; the transcript upload already copied its list.

## 2026-11-09: Qwen's reasoning, no stray "ctlsc"

- The 35B sometimes ended its reasoning with a stray "ctlsc" (as in "Let's investigate.ctlsc") and then copied it
  in later turns of the same chat. Cause: the presence penalty of 1.5 the server uses while a Qwen model thinks
  also hit "\n" and "\n\n", which every reasoning uses, so the line break before `</think>` lost to the model's
  4.4 % "ctl" (raised to ~17 %). Tokens of spaces and line breaks alone are now free of that penalty; words are
  still penalized. A request's own presence_penalty and Gemma are unchanged. DFlash2 / MTP were not involved:
  their drafts never change which token is drawn. A chat that already holds "ctlsc" may keep copying it: start a
  new one.

## 2026-11-09: web UI, the time in an answer's bubble

- The time under a model's answer sat on the bubble's bottom edge (0 px): the floated time fell below the answer's
  blocks with the message bubble's -4 px offset. It is now its own right-aligned line with 8 px below it, as in the
  user's bubble (checked: user, short and long answers all 8 px).

## 2026-11-09: self-audit: web UI headers, OCR Stop and saves, model file checks, CET

- **Web UI page:** `X-Frame-Options: DENY` and a CSP (`frame-ancestors 'none'; object-src 'none'; base-uri 'none'`):
  no other site can frame it and trick a click. **Every reply:** `X-Content-Type-Options: nosniff`.
- **Web UI:** a chat file's link only to this server's files, `data:` or `blob:`; a stream's error event is shown
  (it was skipped: the answer just stopped); a summary answer with no text, a damaged saved tool call and a minutes
  answer that is not an object give a clear error instead of a TypeError.
- **OCR:** Stop ends the page being read (it ran to the end of the page); a `.md` that cannot be written stops the
  reading with "could not write ... (is the disk full?)"; saving into the chat's folder is refused (403) from a page
  another local program serves, as a folder of one's own was; a file name with bad UTF-8 is cleaned (it threw); a
  project's image up to 64 MB and sent with its own type; a PDF page at most 4000 px on its long side.
- **Model files:** the embedding tables' width is checked against `n_embd` (and Gemma's per-layer table against
  `n_ple x n_layer`), every matrix is checked to be 2-D with int-sized dimensions and bytes, a GGUF's array counts
  against their element width, tensors without data only for a package's experts, and the package's expert file
  name must be a plain name. A damaged or hostile GGUF now stops with an error at load instead of reading past a
  tensor.
- **Fatal errors:** `CL_CHECK`, a failed device allocation and a failed kernel module end the process with
  `_Exit(1)` like `ZE_CHECK` (no destructors run under the server's threads); a module build log is no longer left
  behind on a soft failure.
- **Build:** linked with `/CETCOMPAT` (the CPU's shadow stack may be used).
- Checked: `runz --dump-logits` bit-identical before / after for Qwen3.5 4B, Gemma 4 E4B and the Qwen3.6 35B-A3B
  package; serverz headers read back; Xiaomi-OCR-0: two images into one `both.md`, the same request with a foreign
  Origin 403, a bad UTF-8 name answered; `node --check` of the web UI's script. Stop during a page was not tried
  by hand. Findings left as they are, and why: BUGS.md "Audit 2026-11-09".

## 2026-11-09: the session cache's size holds for all models together

- **`--session-cache-gb` (default 8 GB in arc-server, pi-arc and the launcher) is one limit for every model now.**
  Before, each model file (and each layout of it: `-std`, `-nomtp`, `-q2`) had its own 8 GB folder, so trying many
  models filled the disk: 49 GB of saved conversations on the laptop, 20 folders, most of models no longer there.
  When the server starts and after each save, other models' saved conversations go first, oldest first, until all
  of them fit; then this model's least recently used ones, as before (`prune_other_sessions`, `tools/server_cache.h`).
  Only finished `.arcs` files are removed (a `.tmp` being written is left). The size is set in the launcher's
  "Session cache (GB on disk)" field ("Set as default" keeps it), `arc-server start -SessionCacheGb N`, a preset, or
  `serverz --session-cache-gb N`; 0 turns it off.
- Checked: a sessions folder with another model's five 3 MB files and a 10 MB limit: the server removed the two
  oldest at start (9 MB left), left the `.tmp` and its own folder alone. Its own folder is found by file identity
  (a model renamed only in case), and only files that start with the conversation header are removed.

## 2026-11-09: launcher and CLI, stages 0 and 1 of docs/launcher-research.md

- **Launcher, look:** follows Windows' light / dark mode and accent colour; Segoe UI Variable; on Windows 11 a dark
  or light title bar, rounded corners and Mica behind the title bar (`DwmSetWindowAttribute`); sharp at 125-200 %
  scaling (DPI-aware, sizes scaled). Main actions in one row (Start, Stop, Get models), the rest under **More**; the
  running server, the selected model and the **Log** (live) as tabs; a GPU memory bar.
- **Launcher, behaviour:** the server state is read by a background runspace, so a busy or hung server no longer
  freezes the window; errors and notices go to a bar under the title ("Copy details", "Show log") instead of boxes, and
  an error in a WinForms event too (GitHub #5: boxes repeated under the ticking timer); a failed start says what its
  exit code means.
- **Launcher, new:** whether the options fit the GPU memory, under the context field; **presets** (save, pick, delete;
  the same files `arc-server start -Preset` reads); a **tray icon** (on / off, Start the default model, Stop, web UI;
  minimized, the window waits there); a **doctor report** with Copy; an **update check** once a day against the latest
  GitHub release (a link only; off in More).
- **No terminal window next to the launcher:** `arc-launcher.cmd` runs PowerShell in `conhost --headless`; with Windows
  Terminal as the default terminal, `-WindowStyle Hidden` still opened one (GitHub #5).
- **`arc-server`, new commands:** `list`, `logs [-Follow]`, `doctor`, `version [-Check]`, `pull <keys>`, `presets`,
  `save-preset -Preset <name>`; `-Json` on `status`, `list`, `doctor`, `version`, `presets`.
- **`arc-server start`, new options:** `-Preset <name>` (the command line wins), `-DryRun` (checks everything, prints
  the `serverz.exe` command and a GPU memory estimate, starts nothing), `-Drafter auto|mtp|dflash|off|<folder>` (one
  name for `-NoMtp`, `-DFlash` and the `-nodf` / `-df` ids). A start whose estimate does not fit prints a warning.
- **Exit codes:** 1 a wrong option, 2 the server stopped while loading, 3 not ready within 10 minutes, 4 the port is
  used by another program (checked before the server starts).
- **`arc-server help`** is a short page now; `help options`, `help writing`, `help errors`, `help all` show the rest.
- **`arc-server.cmd` in the package root**, so the command works without Pi; setup offers to put the folder on PATH
  when Pi is not installed.
- Checked: both scripts parse on PowerShell 5.1 and 7; every new command run on both (the dry runs, presets, `-Drafter`,
  exit 4 with a port held by another program); the launcher's `-SelfTest` (opens, lists 26 models, loads a preset,
  saves a picture, closes) on 5.1 and 7, light and dark. No server was started for these checks (other sessions'
  servers were running).

## 2026-11-08: the launcher's start command (issue #6); self-audit of ARC_BATCH / ARC_XALL

- **Launcher, "Copy command" (GitHub issue #6):** copies the command that starts the server the way Start would,
  with the selected model and the options as set, for a script or shortcut of one's own:
  `powershell -NoProfile -ExecutionPolicy Bypass -File "...\scripts\arc-server.ps1" start -Model <id> ... -Port N`
  (also shown as `arc-server ...` when Pi's commands are installed).
- **`arc-server` prints the server's own command line** (`bin\serverz.exe -m ... --ctx ...`) when it starts one and
  in `arc-server status`; it is kept in `server.json`, and the launcher's "Running server" panel shows it.
- **Self-audit fix, ARC_BATCH on a split model:** a recorded list that had to wait for an earlier one (64 lists
  behind) waited without the multi-GPU time limit and abort; it now waits through the guarded wait. One GPU and
  ARC_BATCH off: unchanged. Checked: 4B tokens the same plain / ARC_BATCH=32 / 2 GPUs simulated / both.
- Audit notes, no change needed: ARC_XALL refuses while an expert prefetch thread exists (its copies finish before
  it ends), counts the MTP layer (n_layer + 1 layers), and drops to the usual path after a shrink(); the recorded
  lists and their events live as long as the process (no growth on model switches); the server's engine lock keeps
  Whisper and the chat model from recording at the same time.

## 2026-11-08: batch OCR: many images and PDFs into Markdown at once

- **Several images and PDFs in one message** to an OCR model (Xiaomi-OCR-0) are read one after another, in their
  order, each under `# File K / F: name`, a PDF's pages under `## Page N / M` (`ocr_batch`, `tools/server_ocr.h`).
  Before, two images went into one prompt together, and of a PDF with other files only the PDF was read.
- **Saving (`ocr_save` gets `per_file`):** one merged `.md` with every file under its own `# name` heading, or one
  `.md` per file named after it. The web UI asks with a dialog when 2+ such files are sent: one Markdown per file,
  one merged Markdown, or just show it, and a folder.
- The single PDF's rules hold: no existing file is written over (`name (2).md`), 50 pages a request over all the
  files (an image is one), and "lanjut" goes on from the last file and page with text. The bar shows the file and
  page under way.
- **OCR from a project's folder, no upload:** file parts `project:<id>/<path>` are read in place (`ocr_file`), with
  read_file's rule: only the chat's own project (`project_id`, which the web UI now sends in OCR mode), from this
  computer and the server's own page. A project's chat saves its `.md` into the project's `ocr` folder by default
  (made when needed; the `name (2).md` rule holds), with no copy in the chat's files: every chat of the project lists
  it. A folder named inside the project is treated the same. The web UI's new folder button next to the paperclip
  (an OCR model in a project's chat) lists the project's PDFs and images, subfolders too, to tick several or all.
  A chat model later in the same chat sees such a part as a pointer to `read_file`.
- **Self audit fixes** (single PDF too, where the cause was older):
  - A heading is a whole line `# File K / F: ` / `## Page N / M`. Before, the model's own "## File menu" or
    "## Page 9" in a page's text moved the place "lanjut" starts from, so pages were read twice or skipped.
  - A page or file that failed is read again by the next "lanjut". Before, its error counted as its text; the web
    UI's kept note now names the page before it.
  - Per-file saving makes the first `.md` before the stream begins, so a folder that cannot be written is an HTTP
    error. A later file's error stops the answer with a note.
  - "lanjut" after Stop appends only to a `.md` named after the file in progress, never into an earlier file's.
  - A refused project file stops only its own turn, not every later turn of the chat.
  - The project's `ocr` folder is made only when the `.md` goes there.
- `ocr_pdf`'s file handling is now shared helpers (names, folder check, a new `.md`, appending, one image read);
  its output is the same. A save error (bad folder, every name taken) now comes before the stream begins, as a
  proper HTTP error.

## 2026-11-08: generation switches for the B60: recorded command lists (`ARC_BATCH`), MoE tokens without host waits (`ARC_XALL`)

Both are off by default and bit-identical; they wait for a Pro B60's numbers (issue #4).

- **`ARC_BATCH=n`:** kernels are recorded into regular in-order command lists of up to n kernels. Each list is
  appended to the immediate list when full (`zeCommandListImmediateAppendCommandListsExp`, Level Zero 1.9), instead
  of every kernel being appended there on its own.
  - On a B60, `replay_probe` measured 2.56 us a kernel appended one by one against 0.66 replayed. The 27B's
    generation step has ~3 ms between kernels (46.6 ms against 43.6 of kernel time).
  - The lists start at 4 kernels after every host wait and double up to n.
  - Copies, fills, barriers and waits first append the list being recorded (`zrt::imm()`, `zrt::sync()`).
  - Kernels with an event (profiling, `ARC_ZE_WATCH`) and `ARC_ZE_SYNC_EACH` go to the immediate list as before.
  - A driver without the call says so and runs as before.
- **`ARC_XALL=1`:** a MoE model whose experts are all in the GPU's expert cache (the 35B-A3B on a 24 GB card) no
  longer reads each layer's expert ids on the host. That read is ~40 host waits a 35B token, and each one leaves the
  GPU idle.
  - The check runs once a token (`XCache::all_in`) and sets every expert's address, as `ensure()` does for its hits.
  - It is cleared when a launch that ran out of memory takes slots away.
  - Not used for multi-GPU or prompt chunks.
- **`ARC_LAUNCH_STATS=1`** also prints the host waits per token.
- **`bench\replay_probe.exe`** also times lists appended to the immediate list and lists recorded again each step
  in chunks of 8 / 32 / 128 / all.
- **`bench\decode_try.ps1`:**
  - new rows `BATCH=16/32/128`, `mtp BATCH=32`, and for MoE models `XALL=1` and `XALL=1 BATCH=32`; these rows say
    whether the tokens match plain;
  - launches and host waits per token;
  - the prefetch row is now `GEMV_PF=0`, since the prefetch is the default on discrete cards.
- **Laptop (no gain expected there):**
  - Logits bit-identical with `ARC_BATCH` 8 / 32 / 256 on the 4B and the 35B, with MTP, and over a 2,161-token
    prompt.
  - Speed: the 4B the same; the 35B -5 %. Its cache holds 83-87 % of the experts, so a token waits 43 times and the
    lists stay short.
- **Tried and removed: a dense prompt GEMM straight from the raw rows (DP4a)**, meant for short prompts.
  - On the laptop `decode_w8` takes 1.1 s of a 4.6 s 512-token 27B prompt.
  - The fused kernel was bit-identical but not faster (512 tokens 106-110 against 110 tok/s): the GEMM and the
    decoding both run on the vector units.
  - An XMX version could still pay on B-series cards; see `docs/before-after.md`.

## 2026-11-08: B60 defaults: H8 + 4 column groups for Qwen dense models, GEMV prefetch; arc-server -DFlash

- **H8 + `ARC_GEMM_CG=4` by default** on discrete cards with 16-lane XMX, for the Qwen dense models. A Pro B60
  (issue #4, package d980970) measured the 27B at 512 tokens 615 -> 677 tok/s, 4K 766 -> 1,024 and 16K 706 -> 937.
  The fused H8 decode made H8 win at short prompts too. KL against the plain path 0.00154 (max 0.064).
  - Not for MoE models (the 35B: H8 -1 %), Gemma, Flash Next or two-pass models.
  - Off again if XMX does not pass its start-time check, which now runs the H8 kernel with 4 column groups.
  - `ARC_H8=0` turns it off.
- **`ARC_GEMV_PF=1` by default on discrete cards** (bit-identical). The B60's 27B generated 22.33 -> 23.08 tok/s, as
  in an earlier run (22.39 -> 23.11). iGPUs keep it off. `ARC_GEMV_PF=0` turns it off.
- **`scripts\arc-server.ps1 -DFlash <folder>|off`:** a DFlash2 drafter of your own, e.g. for a fine-tuned variant the
  model list does not pair with one (asked for on issue #5).
- **`bench\b60_sweep.ps1` / `gemm_sweep.ps1`:** their "no H8" rows set `ARC_H8=0`; the KL base is the plain path.
- **Also measured on the B60:** `ARC_GEMM_WPF=2` +3.6 % without H8 (794 tok/s) but slower with it; f16 XMX 400-550,
  so f16 is out on the B60 too. Two cards with H8: 1,292 tok/s at 4K, four cards 1,624. MTP 47.7 tok/s.
- The laptop's iGPU path is unchanged: logits bit-identical (4B, ~6,000-token prompt in 1,024-token chunks).
- **The GPU memory left to the driver is 0.5 GiB instead of 1.2**, on iGPUs and discrete cards alike (the user's
  choice). On four B60s that frees 2.8 GiB for context and the expert cache. The auto context and the expert cache use
  the share less 0.5 GiB. It is a setting, not fixed: `arc-server -DriverReserveGb N`, `serverz` / `runz
  --driver-reserve-gb N`, or `ARC_DRIVER_RESERVE_GB=N` (0 to 8 GiB). The laptop: the 35B's expert cache 83 % of its
  experts.
- **Self audit of these changes:**
  - The `--ctx auto` estimate now counts H8's buffers (~0.35 GB on the 27B).
  - The H8 kernels are checked apart from the rest of XMX, and per kind of check, not once per process. A failing H8
    kernel turns H8 off, not XMX.
  - The 140T check covers split-K.
  - The 140T preset keeps to Qwen dense models and stays off with `ARC_H8_SB=1`.
  - The per-256 H8 path tracks its staged form.
  - `-DFlash` takes an absolute path and refuses `-NoMtp`.
  - The B60 sweep's FUSE rows and its CG labels for H8 (4 column groups now by default).

## 2026-11-08: web UI: bold that holds italic

- `**not *clean* yet**` was shown with its asterisks: the bold pattern allowed no `*` inside, so italic inside bold, as
  models often write it, broke the whole pair. Bold may now hold single asterisks, and the italic inside it is
  formatted too. `***both***` stays `<em><strong>`. Checked with the web UI's own functions on the answer where it
  showed (22 bold pairs, none left raw) and on edge cases (two bolds on a line, an unclosed `**`).

## 2026-11-08: the Arc 140T preset (XMX + H8 for long prompt chunks), on by default

- **Measured on a 140T** (issue #5, package 65e9e81, 27B, two runs): XMX + H8 + 32 tokens per sub-group + 2 column
  groups + 128 registers, against DP4a: 512 tokens 154-159 / 154 tok/s, 2K 221 / 180-181, 8K 230-232 / 184-186.
  The fused H8 decode is part of the gain: without it 123 / 201 / 226. KL against DP4a 0.00154.
- **On iGPUs with 8-lane DPAS and weights decoded just in time, it is now the default, per GEMM.** Chunks of 512+
  tokens take XMX + H8 with those tiles; shorter chunks and resident weights keep the DP4a default (16 tokens, 256
  registers). TM 32 needs a multiple of 32 tokens, and DP4a spills with 128 registers.
- `gemm_kernel` takes its tokens per sub-group and register file per call. The staging buffer remembers whether it
  holds the W8 or the rotated H8 form. The start-time check runs the preset's kernels against DP4a bit for bit.
  `ARC_140T=0` turns it off.
- **Laptop 125H, the preset forced with DP4a standing in for DPAS** (`ARC_140T=1 ARC_XMX_TRY=2`, 27B):
  - a short prompt is bit-identical to the default;
  - 2,048 tokens in 1,024-token chunks are bit-identical to H8;
  - 2,148 tokens (H8, H8, then a W8 chunk) run with the same greedy text as the default.

## 2026-11-08: generation switches on every GPU (for the 140T)

- `ARC_GEMV_GRF`, `ARC_GEMV_PF` and `ARC_GEMV_MT_GRF=128` now act on any GPU when set. They had needed the xe2
  switches, which the 140T no longer has since it keeps DP4a for prompts, so its generation runs ignored them.
  Unset, nothing changes.
- `bench\decode_try.ps1`: a row for MTP with 128 registers (iGPUs default to 256).
- Docs: the 140T's generation is within ~15 % of its bandwidth bound; MTP / DFlash2 drafting is the lever.

## 2026-11-08: generation switches for the B60, and a generation profile in decode_try

- `ARC_GV_IQ3S=1`: single-token IQ3_S weights (the 27B) through `gemv_v`, loaded straight into registers, instead of
  `gemv`'s local window with two barriers a step. Bit-identical (27B logits). The 125H's iGPU is 6 % slower with it
  (27B step 210 -> 223 ms), so it is off until a B60 measures it.
- `bench\decode_try.ps1`: rows for it (with 2 and 4 rows per sub-group) and, for MoE models, the experts' GEMV rows per
  sub-group (`ARC_MOE_GR_GU` / `ARC_MOE_GR_DN`). It ends with a `--prof-gen` table per model.
- Docs: where a B60's generation time goes, and command-list replay as the largest lever left (35B ~10 %, from
  `replay_probe`: 2.56 against 0.66 us a kernel).

## 2026-11-08: H8's weight rotation in the decode pass; a weight prefetch switch for the B60

- H8 with weights decoded just in time (the 27B on a B60): `decode.cl h8r_decode` decodes the raw rows and rotates
  them in one pass. Before, `decode_w8` wrote a W8 copy and `h8r_weights` read it back. It runs from a weight's second
  decode on, once its column scales are known. Q2_K / Q4_K / Q5_K and the per-32-float types keep the two passes.
  `ARC_H8_FUSE=0` turns it off.
- Laptop 125H, 27B, 2K prompt in 1K chunks, H8: logits bit-identical; decode + rotation 4,569 -> 1,566 ms of GPU time;
  101.5 -> 114.3 tok/s. A B60 spent 561 ms there per 4K prompt.
- `ARC_GEMM_WPF=n`: the XMX prompt GEMM prefetches its weights n 64-value chunks ahead. Off by default, a B60 to
  measure; bit-identical.
- `bench\b60_sweep.ps1`: rows WPF and FUSE, and a phase E over several GPUs (`-Gpus 2,4`).
- Self audit (`docs/self-audit-2026-11-08.md`):
  - XMX is off again only on iGPUs with 8-lane DPAS. Discrete Alchemist cards had been caught too; they now go
    through the model-sized speed check.
  - The sweeps' FUSE rows and f16 labels fixed.
  - `ARC_MOE_CG` x `ARC_MOE_WY` bounded.

## 2026-11-08: f16 XMX at 8 lanes for the Arc 140T (opt-in)

- `ARC_X16=1` now also works on iGPUs with 8-lane DPAS (`gemm_x16.cl -DX16_SG8`). The weights and activations are f16,
  so one DPAS sums the whole row with no per-32 rescaling between DPAS, where the 140T's int8 XMX kernel lost to DP4a.
  Strata chose f16 XMX for the A770, the same 8-lane XMX generation.
- ocloc for Arrow Lake H: real DPAS, no spills at 8 / 16 / 32 tokens and 128 / 256 registers.
- On the 125H with DPAS emulated: the start-time check passes (largest error 2.9e-4). KL against DP4a: 4B 0.00087, 27B
  0.00069.
- `bench\igpu_grf_sweep.ps1` measures it with its own settings, and its KL.
- Off by default; the 140T has to measure it.

## 2026-11-08: a 140T sweep with 128 registers and H8; its last results recorded

- The 140T's `igpu_sweep` (issue #5): DP4a still ahead, 172-185 tok/s on the 27B's prompt. The best XMX variant was 128
  registers at 162-168 tok/s; `ARC_X8_W` did not help. GEMV switches stayed within noise.
- `bench\igpu_grf_sweep.ps1` sweeps the tiles of XMX with 128 registers and of DP4a, each without and with H8. Then the
  H8 forms, and KL / perplexity against DP4a.
- `ARC_GEMM_GRF=128` and `ARC_GEMM_CG` also act on the DP4a GEMM of iGPUs when set. Bit-identical on the 125H (4B,
  27B, H8). Defaults unchanged.
- ocloc: DP4a with 128 registers spills on Arrow Lake H as on the 125H, so the sweep measures it once only.

## 2026-11-07: variants of the 8-lane XMX GEMM for the 140T, and a sweep script

- `ARC_X8_W=1`: `gemm_x8.cl` with one 8-lane sub-group for both halves of a 16-column group, so 8 tokens' activations
  are loaded once for 16 columns. ocloc (`arl-h`): 28 % fewer memory messages and 17 % fewer instructions per DPAS, no
  spills.
- `ARC_GEMM_GRF=128` now also applies on 8-lane GPUs.
- `bench\igpu_sweep.ps1`: prompt GEMM settings with XMX forced; prompts of real use (512 / 2048 / 8192 tokens, chunk
  sizes); the GEMV switches for generation.
- `bench\b60_sweep.ps1` for the B60: every tile, register and 2D-load setting changed alone from the default (764 tok/s)
  and from H8 (918), the winners put together; then prompts of real use (512 / 4K / 16K, 16K in 4K chunks), KL and
  perplexity against the default, and the 35B's expert GEMM shapes.
- Bit-identical on the 125H with DP4a standing in for DPAS (4B, 27B, H8). Defaults unchanged: XMX stays off on 8-lane
  GPUs until a 140T measures a variant faster than DP4a.

## 2026-11-07: a model-sized XMX speed check on iGPUs; Panther Lake's XMX code read

- iGPUs with 16-lane XMX (Lunar Lake, Panther Lake such as the B390) now time a 512 x 17,408 x 5,120 GEMM at start, XMX
  against DP4a, and keep XMX only if it is at least 3 % faster. DP4a is timed with the iGPU's own settings. If XMX loses,
  the discrete cards' settings are dropped too, so the GPU runs as one without XMX.
- Discrete cards skip the check (`ARC_XMX_SPEED=1` runs it). `ARC_XMX_FORCE=1` skips it anywhere.
- ocloc for Panther Lake (`ptl-h`): real DPAS in all 19 builds, 2D loads, no spills. The 140T's `gemm_x8` (`arl-h`, the
  options of its log): no spills either, but about 2.5x the vector work per multiply.
- Tested on the 125H only with DP4a standing in for XMX: DP4a kept, the same text. No iGPU with XMX has run it yet.

## 2026-11-07: XMX off again on 8-lane GPUs (Arc 140T)

- A 140T ran package 911b6fa (issue #5): `gemm_x8.cl` passed the start-time check, and the 27B's logits were
  bit-identical to DP4a on real DPAS. But its 2K prompt fell from 178-182 to 113-120 tok/s (GEMMs 1.6x slower).
- The start-time speed check had picked XMX: its 64 x 64 x 1024 GEMM stays in the caches. It is removed.
- 8-lane GPUs run as with `ARC_NO_XMX=1` again; `ARC_XMX_FORCE=1` still takes XMX after the bit-for-bit check. Other GPUs
  are unchanged (the 125H: no XMX; `ARC_XMX_TRY=2` still runs).

## 2026-11-07: MoE expert GEMM work-group shapes (for the B60)

- A B60's profile of the 35B-A3B (16K prompt, issue #4) showed the expert GEMMs at about 21 TOPS, against about 50 for
  the dense GEMMs: each work-group held one sub-group, 16 columns x 16 slots, so nothing was shared through L1.
- `ARC_MOE_CG` / `ARC_MOE_WY`: several sub-groups per work-group, along a tile's columns (activations shared) and along
  the tiles, mostly one expert's (weights shared). Default 1 x 1, as before.
- Logits bit-identical at every shape (35B on the laptop, five shapes). Laptop speed within noise (745-766 tok/s), so
  the default stays until a B60 measures it: `bench\gemm_sweep.ps1` has the rows MOECG and MOEWY.

## 2026-11-07: an XMX kernel for GPUs with 8-lane sub-groups (Arc 140T)

- A 140T's log (issue #5) showed the 16-lane XMX GEMM differing from DP4a in every layout. The extension requires an 8-lane
  sub-group on such GPUs, so that form was undefined there, and the earlier note that its DPAS looked emulated was wrong.
- `zero/kernels/gemm_x8.cl`: the W8 and H8 layouts at 8 lanes, with real `dpas` for Arrow Lake H in ocloc and no spills.
  Native types and MoE tiles stay on DP4a on these GPUs. It is checked against DP4a at start, then taken only if at least
  5 % faster.
- Not run on real DPAS yet. On the 125H with DP4a in its place (`ARC_XMX_TRY=2`): logits bit-identical (4B, 27B, H8).
  GPUs without XMX, as the 125H, are unchanged.

## 2026-11-07: self audit of remote access, speech, the web UI and H8

Full list: docs/self-audit-2026-11-07.md. The ones that matter most:

- **Security:** with `--host`, a peer on the network could set the remote password through `/v1/remote/config`: it is from
  this computer only now. A sign-in being checked can no longer outlive a new password or "sign everyone out". Bodies
  through a tunnel with access off are refused before they are read. `/sessions` is fenced, and plain http moves to https.
- **Crash:** the first load of an offered speech model could end the server when GPU memory was short. It is an error now.
- **Lost work:** the transcript model, accuracy, kept format and glossary were not being saved (a stray comment in
  `trSavePrefs`). A pass again could replace a good transcript with an empty one, or an open edit could overwrite the new
  text.
- **Audio without speech models:** a chat with audio and `/v1/audio/*` reach a model that hears again.
- The tool loop guard looks at results too: a cycle stops, and polling a job whose output changes goes on.
- The default path's logits are bit-identical (4B, 27B). H8's KL is unchanged. Tests: 53 passed, 5 skipped.

## 2026-11-07: a Pro B60 measures the new H8 (no code changed)

- 27B, 4K prompt, one card: default with 2D loads 764 tok/s, `ARC_H8=1` 918 tok/s (+20 %). KL against the default path
  0.00144. Wider tiles (750) and 256 registers (783) do not help H8.
- Over 2 and 4 cards: default 1,097 and 1,501 tok/s, H8 1,570 over 4. The sums limit it there.
- 2D loads gave bit-identical logits on a short prompt. H8 stays opt-in until the 35B and a model with resident weights
  have run on it.

## 2026-11-07: iGPUs with XMX use it, with the discrete cards' defaults

- Lunar Lake, Panther Lake and Arrow Lake H now go through the XMX check and take the defaults found on the B60: XMX
  GEMMs, 2D block loads with a prefetch where the driver has them, and 128 registers on GPUs with 64-byte registers.
  None of this has been measured on such an iGPU yet.
- Arrow Lake H (Arc 140T) was refused XMX until now because it has 8-lane sub-groups. XMX is taken there only if a
  start-time speed check finds it at least 5 % faster than DP4a; the log gives both times. `ARC_XMX_FORCE=1` takes it
  anyway. It keeps 256 registers (128 spills). (Its kernel: see the entry above.)
- A GPU without XMX, as the 125H, is unchanged: logits bit-identical (4B, 27B).

## 2026-11-06: H8 with one scale per column and token, the rotation in registers

- `ARC_H8=1` (opt-in) now works as urakozz's int8 prefill does: one scale per column, computed once per weight, and one
  per token, so a GEMM's dot products over K are one integer sum; the rotation in registers instead of a barrier per
  stage through local memory; random signs before it; on XMX the sums stay in the DPAS accumulators. `ARC_H8_SB=1`
  keeps the per-256 form, `ARC_H8_NOD=1` leaves out the signs.
- Laptop iGPU, 27B 4K prompt: H8 111.9 -> 122.5 tok/s, rotations 6.64 -> 4.3 s, and KL against the default path 4B
  0.00403 -> 0.00243 (largest row 2.41 -> 0.072), 27B 0.00100 -> 0.00148. **The default path stays faster on the iGPU
  (140 tok/s) and stays the default:** DP4a's GEMMs gained only 1-4 % from the scales.
- Not run on an XMX GPU yet; a B60 has to say what it gains (docs/xe2-tuning.md section 8 has the commands and the
  comparison with urakozz's code).
- The start-time XMX check tests an H8 kernel only when H8 is asked for, so one cannot take XMX away from other runs.
- The default path's logits are bit-identical (4B, 27B).
- Against f32, 480 tokens: 27B default KL 0.00054, H8 0.00141, urakozz's method simulated on our model 0.00080; 4B
  0.00100, 0.00274, 0.00150. `runz --kl-ref` also prints the logits' relative L2, worst cosine and top-5 overlap;
  `ref_forward REF_HAD_D=1` simulates the signs. The signs show no clear gain.
- H8 on Qwen models now rotates the activations from f32 instead of their int8 form, without the signs (`ARC_H8_INT=1`:
  from int8; `ARC_H8_D=1`: with signs). KL against f32: 27B 0.00141 -> 0.00113, 4B 0.00274 -> 0.00195. The iGPU's 27B
  H8 prompt goes 122 -> 116-117 tok/s, +285 MB. The default path's logits stay bit-identical (4B, 27B); its memory and
  kernels do not change.

## 2026-11-06: 2D block loads by default on discrete cards; the B60 and 140T results

- **Prompts on discrete cards:** the XMX GEMM now loads its activations with 2D block loads and a prefetch. A Pro B60's
  27B prompt goes 608 -> 763 tok/s (+25 %, the owner's runs with the switches on). `ARC_GEMM_2D=0` and
  `ARC_GEMM_2D_PF=0` turn them off. If they do not build or differ from DP4a at start, XMX goes on without them. iGPUs
  are unchanged (laptop: 4B logits bit-identical). Logits on a B60 and runs over several GPUs are still to be checked.
- `bench\gemm_sweep.ps1`: the X2D row is pf (default), on, off.
- **Measured and recorded, nothing changed** (docs/xe2-tuning.md section 8, docs/multi-gpu.md):
  - H8 stays off: 570 tok/s on one B60 against 608, because the rotations cost more than the GEMMs gain. KL 0.0012
    against the default path.
  - H8 over 2 and 4 B60s: 863 and 1,232 tok/s (27B).
  - `ARC_TP_ATT_QSPLIT=1`, 35B over 4 B60s: 3,441 -> 4,290 tok/s. It stays opt-in.
  - The 140T has not run XMX or H8 yet: the engine's XMX GEMM needs a GPU without 8-lane sub-groups, and its H8 runs
    came from a package older than H8.

## 2026-11-06: the speech models load from Settings whatever the server was started with

- Settings > GPU memory said "not on this server" for Whisper when the server had started without `--whisper`, with
  the model in the Hugging Face cache, and a server started with a chat model alone offered no speech models at all.
  Now Whisper and Qwen3-ASR found in the cache can be loaded and unloaded there, and by Transcribe again, whatever the
  start said. They are not loaded at the start, so they take no GPU memory until asked. Only an explicit
  `--whisper off` or `--asr off` keeps one away.
- The Transcripts tab is therefore there on every server, with its saved transcripts.

## 2026-11-05 (night): remote access with a password; motion in the web UI; expert cache policies measured

- **Remote access** (docs/remote-access.md): the web UI from a phone or anywhere through a Cloudflare Tunnel, with a
  password.
  - Settings > Remote access (on this computer only): on / off, the tunnel's host names (`*.trycloudflare.com` for
    a quick tunnel), the password, sign everyone out. `scripts\remote-tunnel.ps1` starts the tunnel.
  - The password is kept as PBKDF2-HMAC-SHA256 (200,000 rounds, random salt) in `remote.json`. A session cookie is
    `HttpOnly`, `Secure` and `SameSite=Strict`, valid for 30 days.
  - Wrong passwords: one second's wait each. 8 within 15 minutes lock that visitor out; 40 from everyone lock every
    sign-in. Parallel guesses count before they finish.
  - Before signing in, a remote request may only fetch the sign-in page or send a small sign-in. Anything else is
    refused before its body is read.
  - A remote page may only send its own origin. A request through a tunnel or proxy for any other host is refused,
    so a tunnel that rewrites the Host header cannot pass as local.
  - Unless "Full access remotely" is on, a remote session is not "this computer". Every local-only check refuses
    it: the terminal, projects and their folders, memory, MCP servers (their tokens too), slots, loading models by
    hand, and the OCR save into a folder.
  - Local use is unchanged: requests that name localhost go through the same checks as before.
- **Files the model wrote (HTML, SVG, XML) open sandboxed at their own address** (`/files/<id>`): a page with
  scripts cannot act as the web UI. The web UI already showed them in a sandboxed frame.
- A self audit (an independent review of this change) found the gaps fixed above before they shipped:
  - `local_peer` was true for every tunnel request;
  - `mcp.json` was readable remotely;
  - bodies were read before the password check;
  - parallel guesses passed the lockout;
  - the app manifest needed the cookie;
  - sessions outlived turning access off;
  - forwarding headers of other proxies were not checked.
- **The web UI installs as an app** (`/manifest.webmanifest`, `/icon.svg`): "Add to home screen" on a phone.
- **Motion** (docs/webui-plan.md):
  - new messages slide in from their side;
  - three dots wait in the answer until the first token;
  - the prompt bar shows a sheen until the server reports progress;
  - menus and dropdowns pop in, buttons press, and on a phone a chat slides in.
  - Opening a chat draws its messages at once. Everything uses transform and opacity, and nothing animates with
    reduced motion on.
- **Security fix:** a page of another local server (`http://localhost:<another port>`) could reach `/v1/mcp`, which
  starts MCP servers (programs). `/v1/mcp` is now only for the server's own page or a client without an Origin, as
  the tools and files already were.
- **Measured, not changed: the expert cache by use instead of recency** (QUEUE). Every expert route was recorded
  (`ARC_XTRACE=file`, zero/xcache.h) and replayed with the cache's own slots (`bench/xtrace_sim.py`).
  - Flash Next, 19 % cached: a protected part for experts used again saves at most 2.8 % of the generation misses;
    choosing the least-used victim costs 0.5-19 % more.
  - Qwen3.6 35B, 98.4 % hits: -1.1 to +1.1 %.
  - The bound with the future known (Belady) is -52 % / -76 %: what would help is predicting the next layer's
    experts, not counting the past. The LRU cache stays.
- **Measured, not changed: Flash Next's prompt gate / up** (QUEUE 21.15). The 4K prompt runs at 186-194 tok/s, and
  gate / up take 6.2-6.3 s of its 20.6 s of GPU time. Two variants were slower, so both were left out:
  - through moe_rg, as the down projection: 8.6 s, 170-174 tok/s;
  - the tile's activations shared through local memory: bit-identical, but 9.6 s, 166-170 tok/s.
- **Tests:** `test_hc` and `test_fa` (hyper-connection and float attention kernels) failed in `run_tests.ps1` since
  `ops.cl` was split into parts on 2026-11-03. They built from `ops.cl` alone, so their kernels were not found.
  They now build from every part in order and pass. The whole suite: 53 passed, 5 skipped (4B, E4B, 27B, 35B, Flash
  Next, DFlash2, server, error paths).

## 2026-11-05 (evening): measured, not changed: Vulkan, fewer kernels, GEMVs inside the model

- **Vulkan instead of Level Zero: not worth a port** (`bench/vk_probe`). Bandwidth 74.1 against 72-75 GB/s; a
  launch 1.07 against 1.50 µs. llama.cpp's Vulkan backend on the 4B: prompt 254 against 855 tok/s, generation
  15.0 against 16.9.
- **Fewer kernels per token: not worth it.** 426-852 empty kernels added to every 4B token cost nothing. Fusing would
  save only the small kernels' own work (~1.7 ms a token, 3 %). Gate and up as one stacked GEMV were 1-2 ms a token
  slower.
- **The GEMVs are not slower inside the model.** Every GEMV of the 4B back to back, with no other kernel, took
  43.7 ms, and ~45 ms in the model (`--prof-gen`, measured right after). The earlier 48-60 GB/s profile was taken
  while other programs used the memory: one CPU thread copying memory slows the GEMVs from 67.5 to 55 GB/s, while CPU
  arithmetic changes nothing. The size of the data read and the weights' alignment change nothing either. This
  corrects the 2026-11-04 note below that put the lower profile numbers on its events.
- Details: `docs/lessons-intel-arc-igpu.md` section 5, QUEUE 31.13.

## 2026-11-05 (afternoon): transcribe again with another model; summaries kept per model; minutes not cut short

- **↻ Transcribe again** (a transcript's header): a finished, failed or stopped transcript is transcribed again with
  the speech model, language, accuracy and glossary chosen in the list's options (`POST /v1/transcripts/<id>/redo`).
  - The text there now stays until the new pass is done.
  - If the new pass is stopped or fails, the record and the text are put back as they were.
  - A two-track recording (Me / Others) kept as M4A or MP3 is read one channel per track. Before, only its WAV could
    be read, and a second pass failed with "not a WAV file".
- **Summarize again keeps the summary before it.** The button reads "↻ Summarize again" once there is a summary.
  - Each summary is saved with the chat model that wrote it.
  - A list above the summary shows up to 10 earlier ones (model and time); picking one shows it and puts it in the
    minutes, and the shown one is kept in its place (`summaries.json`, `PUT {summary_use: k}`).
- **Transcribe again loads the speech model it needs.**
  - Whisper or Qwen3-ASR are offered also while they are unloaded, marked "(unloaded)". The page loads the chosen
    one first, for Transcribe again and for new uploads; before, a new upload waited in the queue forever.
  - If the GPU memory is short, the page unloads the chat model first. The next chat loads it again. It does not
    unload it while this page is writing an answer.
  - Gemma 4 E4B transcribes as the chat model, so Whisper and Qwen3-ASR are unloaded to give it room, after a
    question.
  - Gemma 4 E4B is offered while Whisper and Qwen3-ASR are unloaded too (`/v1/server/memory` names it). Before, it
    was missing from the list just when it was the only speech model there.
- **Failsafes of Transcribe again:**
  - stopped while it waits in the queue, the transcript is put back as it was (before, it was marked stopped);
  - if the server stops twice during it, the record is put back with a note, not marked failed;
  - a pass again that failed is noted apart from the record's error (`redo_error`), shown after its details. The
    first build wrote it as the record's error, and the page then said "could not open the transcript".
- **No length limit on summaries and minutes.** Each answer goes on until the model ends it or the context is full.
  - The notes of each part of a long meeting had 1200 tokens. On a 2-hour meeting, part 2 reached that limit and its
    notes stopped mid-way, so the rest of that part was missing from the minutes.
  - The part summaries (700 tokens, at most 15 points), the merges (900, at most 20) and the form (6000) had limits too.

## 2026-11-05: a recording survives a muted or switched microphone; a tab's sound without the speakers

- **A live recording no longer stops when the microphone goes away.** Muting it, Bluetooth earphones connected or
  disconnected, or another default input all ended the microphone's track, and the page took that as Stop: a
  meeting was cut into three recordings. Now the page opens the microphone again (the new default) and the same
  recording goes on.
  - The gap is silence on that track, so the timeline stays right.
  - If the device is still switching, the page tries again after 1, 2, 3 ... s.
  - If the microphone cannot be opened, the recording continues with silence on its track.
  - Stopping a share from the browser's bar still ends the recording.
- New source **"A tab's sound, silent here"**: a tab's sound (a video) is recorded and transcribed without being
  played on the speakers (`suppressLocalAudioPlayback`, Chrome / Edge). It works for browser tabs only; a desktop app
  such as Zoom is captured through "Entire screen" with "Share system audio".
- Every dropdown takes the page's colours. The minutes form's dropdown was the browser's white, and open lists were
  light in the dark theme (`color-scheme` now follows the theme).
- Every dropdown has one look:
  - rounded corners, an arrow, a ring on hover and on keyboard focus;
  - in Chrome / Edge with `appearance: base-select` (checked in Chrome 152), the open list is a rounded card with
    rounded rows, and the arrow turns while it is open;
  - a choice longer than its room wraps instead of running under the arrow;
  - the minutes form's dropdown looks like the buttons beside it.
- Self audit of these changes:
  - a microphone retry still pending after Stop could join the next recording and replace its tab sound; each
    retry now belongs to its own recording;
  - "silent here" with a window or the screen picked says to pick a tab, and the dialog opens on the Tab list;
  - the arrow sits at the right edge of a wide dropdown;
  - buttons, dropdowns and Run / Don't run share an 8 px corner;
  - neutral buttons hover in the accent colour, and only the delete buttons turn red;
  - disabled buttons look disabled.
- Talk and voice messages keep their microphone as a live recording does. When its track ends (Bluetooth
  earphones switched, connected or gone), the new default input is opened and feeds the same capture: Talk keeps
  listening, and a voice message keeps recording. Retries come after 1, 2, 3 ... s, only for the session they
  started in. Checked in the page with stand-in microphones: replaced twice, and nothing after a stop.
- **A shared sound that drops no longer stops a live recording.** Chrome ends the shared sound's track (a tab's, or
  the system's with the screen) when the sound output changes, for example Bluetooth earphones on or off, while the
  share itself goes on. The recording now goes on with that track silent (the microphone keeps recording in a Me /
  Others recording), and a "↻ Share sound again" button takes it up again. The browser's dialog needs a click, so
  this cannot be done for you. Stop sharing in the browser's bar still ends the recording. Checked in the page with
  stand-in shares: kept on, shared again, and the old share's late end ignored.
- **Transcripts stays open with speech to text unloaded.** Unloading Whisper and Qwen3-ASR to make room for
  Qwen3.6 35B locked the whole menu, saved transcripts, summaries and minutes forms included, though none of them
  needs a speech model. The menu is now open while the server has its transcript library, and says that new
  transcriptions need speech to text loaded again. `@` on a transcript in a chat works then too. Checked with the 35B
  loaded and both speech models unloaded: the menu opens, a transcript shows Summarize and the forms.
- **No limit on tool rounds in the web UI.** An answer stopped after 8 rounds of tool calls (the model said it had
  run out), whereas Pi has no such limit. Now it goes on until the model answers. Stop ends an answer at any time,
  and a full context is compacted as before. Only a loop ends it: the very same calls (names and arguments) a third
  time in a row are not run, and the model is asked to answer from what it has.

## 2026-11-04 (night): MCP servers for the model, a self audit of every kernel, the web UI and speech

- **MCP servers** (docs/mcp.md): the web UI's model can use the tools of other programs on this computer, Blender
  through blender-mcp for example. The server is the MCP client:
  - stdio servers come from `%USERPROFILE%\.arcinferzero\mcp.json` in the usual `mcpServers` form;
  - each server is switched on per project, or for the chats outside projects;
  - a server starts at its first use and runs in a job object;
  - its tools reach the model as `<server>__<tool>`;
  - a call that may change something asks Run / Always for this tool / Don't run;
  - returned images go into the chat's files.

  Settings has a section for the servers, their switches and the file. The endpoints are `/v1/mcp*`, from this
  computer only. Checked against a test server: 22 API checks, covering tools over two pages, a missing program,
  the instructions, text, image and error results, an unknown tool, a timeout, parallel calls, a crash and its
  restart, stop, and a config change. In the browser with Qwen3.5 4B: a read-only tool ran without asking, a tool
  that changes something asked for Run, and its image was saved and described by the model.
- **Self audit** (docs/self-audit-2026-11-04.md). Six static reviews covered the 15 kernel files against their
  launches, the web UI against the server routes, and the audio and speech code. Each finding was checked before
  it was fixed.
  - The kernel findings were all latent. They are now errors or harmless, with logits bit-identical to the commit
    before (4B, 9B, 27B, E4B, 12B, 26B).
  - Speech:
    - a job that ends the process (a lost GPU) is no longer restarted at every start: after two attempts it is
      marked failed;
    - live audio keeps its timeline when a piece is lost;
    - utterances can no longer be queued after the recording's finish;
    - a deleted merge stops;
    - merged audio follows an MP4's start delay and gaps, as the pieces' transcripts did;
    - the engine is rebound to slot 0 after a failed part;
    - a lost wakeup after loading a speech model is fixed.
  - Web UI:
    - unloading the chat model no longer leaves the page thinking it is loaded, and the page keeps that model
      instead of the catalog's first (`/v1/models` lists the unloaded model first);
    - a U+0000 HTML injection is closed;
    - transcript preferences are no longer reset;
    - Summarize and Minutes cannot run twice;
    - an upload whose last answer was lost is no longer reported as failed;
    - Talk is refused during a recording;
    - plus smaller fixes.
- QUEUE 31.11 (1), measured and not taken: the driver's wait for the GPU does not spin a CPU core.
  - Default: Qwen3.8 27B, prompt 2048 + 32 tokens, used 16-25 s of user CPU time in ~70 s, most of it loading.
  - Polling with 200-µs sleeps: no faster (prompt 132-137 tok/s either way, generation step 204-206 against 208-212
    ms).

## 2026-11-04 (afternoon): unload models, a model that does not fit refused

- **A chat model that does not fit is refused instead of loaded**: `--ctx auto` raised "no context fits" to 4,096
  tokens and loaded anyway. After a restart Windows gave the GPU 16.55 GiB (20.42 before), Gemma 4 12B needed 8.1 GiB
  next to Whisper and Qwen3-ASR's 7.7, ran out of GPU memory at layer 40 and ended the server. Now the switch fails
  with what it needs and what is free, and the server stays up (`ctx-auto.txt` entries made again: m24).
- **Unload / load** (Settings, GPU memory; `GET /v1/server/memory`, `POST /v1/server/unload {"which": "model" |
  "whisper" | "asr"}`, `POST /v1/server/load`): the chat model, Whisper or Qwen3-ASR freed, the speech models loaded
  again as at start (Whisper 16 s, Qwen3-ASR 10 s), each only while nothing of it runs (no recording; a running
  transcription makes it say busy). The next chat model's `--ctx auto` counts the speech memory as it is then:
  Gemma 4 12B with both speech models unloaded loaded with 106,496 tokens of context.
- **Minutes in your own Word forms**: any `.docx` in `%USERPROFILE%\.arcinferzero\templates` (a company's minutes form
  with its letterhead and signature block) is offered next to "Minutes DOCX" in a transcript. The loaded chat model
  reads the transcript (a long one through notes of its parts) and answers the form's content as JSON held to the
  form's shape (`response_format` json_schema); the server fills the form's tables (tools/doc_template.h: the first
  table's "Label : value" rows by label, every other numbered table rebuilt row by row from its first data row; the
  signature block, header, footer and styles kept) and saves it as a file. Today's 2-hour meeting in the user's
  event form with Qwen3.6 35B: agenda, date and time, 11 participants, 6 agenda items, the Q&A with units, in ~2 min
  of generation; Word opens it. New endpoints: `POST /v1/tools/minutes_templates`, `POST /v1/tools/minutes_docx`.
- A data row that repeats a table's header ("No | Acara", "No. | Acara", "Nomor | Catatan") is left out by the
  server too: the 35B began each table with it and the minutes showed the header twice.
- A form is written whole: a numbered table the model left empty gets one row of "-", a field "-", so a form's own
  sample text (a template made from an earlier meeting's minutes) never stays in new minutes. Keep the templates
  clean anyway: fields empty, one empty row per numbered table, the signature block as it should appear.
- Summarize and the forms load the chat model first when none is loaded, so the transcript is cut to its context
  (it was cut to 8K characters: a 2-hour meeting in 12 parts on a 100K model).
- create_file: a model asked for a 2-hour meeting's minutes wrote six files (one per part, and versions). The tool's
  description now says one call per file, a long document whole in one call, never one file per part; its reply says
  the file is done instead of inviting "a new version".
- A `<br>` in an answer (models write it in table cells) is a line break, not the tag's text; any other HTML stays
  shown as text, and a `<br>` inside a code span as written.
- **Select** in the Transcripts list: a circle on each row; the picked ones are numbered in the order Merge joins
  them (the order they were made), Merge N then joins them, Cancel drops the picks (Ctrl+click and a long press start
  it too).
- **Transcripts and Meetings are one menu** (they were the same list and tools; a recording is not always a meeting):
  one list of files, recordings and meetings; **Record** takes the microphone, a tab's or the screen's sound, or both
  as two tracks (Me / Others, named "Meeting ..."); **upload files**; Summarize and the DOCX / PDF minutes for every
  finished transcript, not only meetings.
- After a bug hunt of the above: loading Whisper / Qwen3-ASR back checks the free GPU memory first (next to Gemma 4 12B
  it is refused: "unload the chat model first"; it would run out of memory); a split model (`--gpus`) keeps the old
  4,096 floor; the speech engines are read under their lock (a request during an unload could read a freed one) and a
  job whose engine was unloaded takes the other one; with both unloaded `/props` has no speech to text, queued
  transcripts wait instead of failing one by one, recordings and transcriptions are refused with the reason; the chat
  model cannot be unloaded where a request could not load it again (`--no-swap`); a model that does not fit at start
  leaves the server running without it (models switchable); the image variant is not the default when it fits no
  context. Web UI: the page is always asked again (no-cache), a stray 22-px SVG that let the page scroll and cut the
  header off is gone, and "GPU memory settings" scrolls the Settings list only.
- A failed answer in a chat is a box with a title and the server's message as text (it was the raw JSON); a model
  that does not fit offers "Unload speech to text" and "GPU memory settings".

## 2026-11-04: transcripts through E4B keep a timeline, warm-up after an update

- A transcript or meeting transcribed by Gemma 4 E4B was one segment per 2 minutes (the model gives no times), so a
  meeting whose live draft showed its utterances with times finished as one block. Its parts are now cut at the
  first pause after 10 s, at most 30 s (the quietest 100 ms near it in speech without pauses), a segment each: an
  89-s meeting 1 -> 5 segments, a 163-s voice message 2 -> 6. Words against Qwen3-ASR's text: 0.129 -> 0.133 on
  the voice message (one word of 241), 0.436 -> 0.487 on the meeting (four words; casual speech the reference also
  got wrong). `/v1/audio/transcriptions` keeps its 2-minute parts.
- **Merge**: recordings picked in the Transcripts or Meetings list (Ctrl+click or a long press, then Merge;
  `POST /v1/transcripts/merge {"ids": [...]}`) become one transcript: their audio one after another as one mono
  48-kHz M4A (a two-track meeting's Me / Others mixed; each segment keeps its speaker), their segments on that
  timeline, each piece's start a mark, `pieces` naming them with their clock times; the pieces stay. Today's meeting,
  four pieces after two server crashes (8 + 1 + 67 + 45 min), merged in 175 s into 2:01:03, 33 segments, 118 MB.
- Merge after a self audit: a merge the server stopped in (or a crash) is "failed" at the next start with its partial
  audio removed (it stayed "merging" for good); a failed merge removes its WAV (gigabytes, a full disk); the server
  waits for merges at its stop; pieces still being compressed are refused for a minute; the 4-GB WAV limit is checked
  as it writes; a merging transcript cannot be edited; a read error inside a piece's audio fails the merge instead of
  cutting the piece short (every later piece's times shifted). Checked: refused requests, a 2-piece merge, a server
  killed mid-merge.
- Web UI after the same audit: a failing part of a meeting summary stops the others and its error stays shown; picks
  for Merge stay in their tab; a voice message keeps the name of what wrote it at recording time; a long press on a
  row no longer selects text.
- The Transcripts tab records mono only (the Mono / Stereo choice is gone; the text was made from the mono mix
  anyway).
- Debug switches `ARC_ZE_SYNC_EACH`, `ARC_LAUNCH_STATS`, `ARC_ZE_KPROPS`: `=0` now means off (any value turned
  them on).
- `ARC_ZE_WATCH=ms` (diagnostics, after docs/device-lost-recording.md §5): every launch signals an event of its own
  and is waited for on it, timed; a launch over `ms` prints as "slow kernel" with its sizes and arguments (device
  addresses), one still running after 1.5 s is printed while it runs (Windows resets the GPU at 2 s), and an error
  prints the last 64 launches before the process ends. For finding the kernels behind the GPU timeouts (TDR,
  LiveKernelEvent 141) Windows logged dozens of times on 2026-11-01..04; slow (4B generation 1.2 tok/s with it on).
- A long meeting's summary asks for its parts' summaries (and each merging round's groups) at once, as many as the
  server has slots (`--slots`, or Slots in the web UI's Settings), so they run together instead of one after
  another; with one slot nothing changes. Not measured yet.
- The web UI's Speech setting picks what writes voice messages and Talk as text: Qwen3-ASR, Whisper, or Gemma 4 E4B
  (while the chat is with E4B). A model no longer hears the clip itself when the server has speech to text (Gemma 4
  12B's own hearing was less exact).
- The first start after an update warms Gemma 4's drafter (MTP) with the generation kernels, and the audio encoder
  with a second of silence: a first meeting after this morning's update took 116 s for 89 s of audio (E4B
  generating at 3.7 tok/s and its audio encoder 21.6 s while their kernels compiled), the next one 3.4 s.

## 2026-11-04: codebook GEMVs, DeltaNet with fewer launches

- One generated token of a DeltaNet layer launches two kernels fewer: `gdn_conv` stores the conv history itself (`conv_state_update` was a launch of its own) and `gdn_recur` turns the raw alpha / beta into decay and beta (`gdn_gates` was one). The same expressions, so the logits are bit-identical on every Qwen3.5 / 3.6 / 3.8 model, with MTP and on split models. Verification passes keep the separate kernels (their replay needs the gated values and the history before the pass). `ARC_GDN_FUSE=0` turns it off. On the laptop's iGPU the speed is unchanged (4B 48.8 -> 48.7 ms a token, 27B within noise): there the GPU's work, not the gaps between launches, sets the time; it is for GPUs where the host's launches show, as between a split model's sums.
- Generation through the codebook types (IQ1_M / IQ1_S / IQ2_* / IQ3_XXS) reads 4 rows per sub-group where it read 2 (`ARC_GV4=0` for 2): Qwen3.8 27B, whose FFN is mostly these types, 181.8 -> 179.2 ms a token (1.4 %), bit-identical; the other models have no such matrices.
- Measured, not changed (QUEUE 31.13): the GEMVs on their own (`bench/gemv_bench`, weights streaming) read 68-76 GB/s on the 4B's Q5_K, Q4_K and Q6_K matrices; the lower numbers of `--prof-gen` come from its events. Q6_K FFN down projections through `gemv` instead of `gemv_v` were 4 % faster alone but no faster in the models (4B 49.0 -> 48.8, 27B slower), so the choice stays.

## 2026-11-03: kernel audit, files split, lighter launches

- **Kernel self audit (every `.cl` file, read against the code that launches it; each finding checked):**
  - Qwen3 dense with 5 q heads per kv head (Qwen3-14B): the split decode attention (`attn_fa_g -DFA_SPLIT`) took 4 q
    heads per sub-group and read the first one's kv head for all four, so heads 5-7 of each group attended with the
    wrong K / V at every generated token. Now the largest divisor of the ratio (`Engine::gq_div`); Gemma 4 (2, 4, 8)
    and the Qwen3 models with 2 or 4 are unchanged.
  - MoE routing: a router row of NaN logits picked no expert, and the id NE then wrote past the counts and read past
    the experts (device lost). The id is clamped to the last expert (`moe_route`, `moe_route1`); finite logits are
    unchanged.
  - `ARC_H8=1` with `ARC_NPU`: the NPU shares copy a weight over the same buffers, so the in-place H8 conversion ran
    twice on shared rows and the NPU read H8 rows as W8. H8 is now off with the NPU, with a line saying so.
  - Guards for shapes no shipped model has (each one silently wrong before): Qwen3.5 attention with a q / kv ratio
    that leaves key tile words unloaded (5, 7, 10 ...) and a RoPE width other than 64; Flash Next with hc_n not
    dividing 8 or inject logits past the padded low rank; QSA's local block kept within 32 KB; Whisper text contexts
    over 448; a DFlash2 window below 1. Gemma 4 vision K / V buffers got the 64-byte slack the other encoders had
    (head 72 read 16 bytes past). Multi-GPU f16 sums keep 1/1024 below the f16 limit (parts that each rounded up could
    add to inf).
- **Files split** (code moved, not changed: logits bit-identical on every model):
  - `zero/kernels/ops.cl` (3,516 lines) into `ops.cl`, `ops_attn.cl`, `ops_attn_g.cl`, `ops_vit.cl`,
    `ops_flashnext.cl`, `ops_dflash.cl`; `decode.cl` (2,047) into `decode.cl` and `decode_moe.cl`. A kernel builds from
    its part and the ones before it (`KernelParts`, zero/clrt.h), the same text it had in the one file, so a norm's
    or a GEMV's build no longer compiles every attention, encoder, Flash Next and MoE kernel: shorter first-run
    kernel builds. The cache of compiled kernels is rebuilt once.
  - `zero/engine.h` (4,262 lines -> 1,378) into `engine_setup.h`, `engine_load.h`, `engine_load_gemma.h`,
    `engine_alloc.h`, `engine_stream.h`, `engine_slots.h`, `engine_sessions.h`; `tools/server.cpp` (3,308 -> 2,185)
    gives its file tools to `server_file_tools.h`. Each part is included where it was cut, so the compiled class is
    the same text.
- **Lighter kernel launches:** an argument given the same value as at the kernel's last launch (as passed to Level
  Zero: a buffer as its device pointer) is not set again; most arguments of a decode step repeat. `ARC_ARG_CACHE=0`
  sets every argument as before.
- **Web UI tool and system prompts tightened**: 3,013 -> 2,520 prompt tokens with every tool on (-16 %); on Gemma 4
  E4B the same tool chosen with the same arguments in 8 of 8 test requests (web, page view, memory, Word / Excel
  file, terminal, file list, no tool).

## 2026-11-03: launcher without serverz.exe

- **2D block loads for the XMX prompt GEMM (QUEUE 31.8; built, not yet run, off by default: `ARC_GEMM_2D=1`).** On
  Arc B-series cards a step's activations, 8 token rows x 64 bytes, come in one 2D block load
  (`cl_intel_subgroup_2d_block_io`) shaped as the DPAS A operand, instead of 16 one-row reads; `ARC_GEMM_2D_PF=1`
  prefetches the next 64 bytes. Checked against DP4a at start with the other XMX layouts; when it fails, XMX stays
  on without it. GPUs without the extension (the Core Ultra 125H's Xe-LPG) are unchanged: logits bit-identical.
  `bench/gemm_sweep.ps1` row X2D.
- **H8 prompt GEMMs (QUEUE 31.12; built, not yet run, off by default: `ARC_H8=1`).** Weights and activations rotated
  by a Walsh-Hadamard transform in blocks of up to 1024 values along K, both int8 with one scale per 256 values, so
  the GEMM is pure int8 dot products with one multiply-add per 256 values instead of a rescale every 16 or 32
  (zero/engine_h8.h, zero/kernels/h8.cl, gemm.cl QTYPE 110). One kernel for the iGPU (DP4a), Arc B-series (int8 DPAS,
  checked against DP4a at start like every XMX layout) and split models (each shard rotates within its own K).
  Accuracy simulated first on the CPU (`tools/ref_forward` `REF_HAD`): Qwen3.8 27B KL 0.00057 against f32 (plain
  per-row int8 0.038), Gemma 4 E4B 0.00198. Two-pass models (Gemma 4 26B / 12B) and expert GEMMs keep their path.
  `bench/gemm_sweep.ps1` has an H8 row for B-series owners.
- **Second self audit of the web tools and multi-GPU** (a static review, each finding checked in the code; built, not
  yet run: the GPU was busy):
  - web tools: audio from a URL decoded with a 30-minute cap (~115 MB of samples instead of up to 700 MB) and 60 MB
    at most; each kind's size decided from the headers before the body (a refused type or a too-large Content-Length
    reads nothing); a body cut at its size or the deadline says so; a read of 0 bytes ends the loop; any web port
    (browsers' restricted service ports refused) instead of 80 / 443 only, on every redirect too; one URL parser.
  - web_view: Edge must run inside its job (else not at all), a timed-out run's read is cancelled so a request
    cannot hang; the job holds at most 3 GB and 32 processes; the proxy serves only Edge's processes of a render
    running now (no other program can tunnel through it), counts connections atomically and stops a tunnel at
    256 MB; the URL is percent-encoded on Edge's command line; the DOM read from its first 4 MB; profile folders
    swept at every render; WIC's buffer lock checked.
  - multi-GPU: a pass with no barrier completing for `ARC_TP_TIMEOUT` + 30 s is given up on (a shard stuck inside
    a driver call waited forever), with a log line 30 s before; after a fault left a split model unfreed, no new split
    model in that process; a job is never published over one still running, the stuck list read under its lock;
    `ARC_TP_TIMEOUT` 5-3600 s; odd row lengths, a bad layer number and a vocabulary too small for the GPU count
    refused at load.
  - the re-audit of these fixes: the size check from Content-Length applies only to files (whole or nothing; a page,
    a search page or a GitHub API reply is read up to its limit as before, which the first fix had broken); a file too
    large is 413 and one cut by the deadline 504 (both had come back as 415); a page or text that is not cut no longer
    falls into the file path (a slip of the first fix, found in the re-check); internationalized hosts (münchen.de) are
    checked and opened as punycode; the proxy looks a connection up again before refusing it and logs refusals; a page
    stopped by Edge's 3 GB limit says so; a 304 is not reported as a redirect; `WSAStartup` once in `check_host` too.
    tests/test_web_tools.cpp checks the kept links, the web ports, the URL parser, IDN hosts and web_view's
    command-line URL.
  - the third round: a body that stalls or drops mid-way marks itself not whole, so a cut file is refused (504)
    instead of saved as whole; web_fetch connects to an internationalized host by the punycode name it checked; the
    proxy's refusal log never goes silent (every 100th after 20); Edge's memory note kept next to a timeout's.
- **web_view** (new tool, tools/web_view.h): a page as a browser shows it. Microsoft Edge, headless, takes a
  screenshot of one screen (shown to a model that sees images) and gives the page's text after its scripts ran, so
  pages built by scripts are read too (a GitHub issue page: its comments). Edge's requests go through a proxy in the
  server that reaches public addresses on ports 80 / 443 only; a new empty profile each run, removed after.
- **Audio from a URL**: web_fetch saves MP3, M4A, WAV, FLAC, Ogg Opus... (up to 60 MB, 30 minutes transcribed) and returns its speech as
  text from Qwen3-ASR or Whisper (whisper.cpp's jfk.wav: 5.7 s, word for word).
- Self audit of the web tools: Edge inherited every inheritable handle of the server (other clients' sockets): now
  only its output pipe; a page or text is read from its first 4 MB inside the request (100 MB was for any URL), and at
  most 4 fetches or views run at once (64 at once could hold 6 GB); the proxy's listener cannot spin a core on errors;
  COM set up and taken down in pairs; Edge's profile folders removed with retries and old ones swept; the Latin-1 to
  UTF-8 conversion only for text (it could change a PDF's bytes).
- Every exe is built with Control Flow Guard (`/guard:cf`; build.ps1), as the server now parses files from the
  internet. Logits bit-identical; Gemma 4 E4B generation 14.06 / 15.00 without against 13.46 / 13.56 / 15.28 tok/s
  with, run-to-run noise (docs/before-after.md).
- **web_fetch reads what a person reads** (tools/web_tools.h, tools/web_github.h; docs/server.md):
  - links to files and to other issues / pull requests keep their URLs (`[text](url)`), so the model can open them;
    the other links show their text, and `"links": true` gives every URL to explore a site (the URLs of every link made
    pages 30-75 % longer; this way 0-22 %);
  - a GitHub issue or pull request comes from GitHub's API with every comment, the files attached in the thread listed
    at the top (its page, built by scripts, often had no comments);
  - PDF, Word, Excel and PowerPoint files (up to 20 MB) are read and saved in the chat's files; images are saved and
    shown to a model that sees images; text attachments (`.txt`, `.log`) read as before; archives and programs refused.
  - Checked: Gemma 4 E4B asked about the event_probe log attached to issue #4 opened the issue, picked that attachment
    from the list and answered from it; it read a screenshot fetched from issue #5.
- `bin\membw_probe.exe` (new, in the package): memory read bandwidth of the CPU (1..all threads), the GPU, and both
  at once. On the 125H the iGPU reads 73-74.5 GB/s, the CPU at most 72.5 (all 18 threads), both together ~84: decoding on
  the CPU (xenolith's choice on Raptor Lake) would not be faster here (docs/speed-progress.md).
- Discrete GPUs: the 2-8 token GEMV of MTP / DFlash2 verification now uses 128 registers per thread by default (a
  Pro B60: 27B MTP 43.3 -> 47.1 tok/s). `ARC_GEMV_MT_GRF=256` goes back; `decode_try.ps1`'s row is now "mtp
  MT_GRF=256". iGPUs unchanged.
- When `bin\serverz.exe` is missing (a git clone has no exes), the launcher says so in the model panel: download the
  ready package from the releases page, or run `setup.cmd` to build it. It also offers to run `setup.cmd` once per
  window. Before, only the line under the empty list said it, which was easy to miss (issue #5).
- `ARC_GEMV_MT_GRF=128` tried on the 125H iGPU: 27B with MTP 11.78 -> 9.57 tok/s, so the iGPUs keep 256 registers
  (docs/xe2-tuning.md). Nothing changed in the engine.

## 2026-11-02 (night): Gemma 4 31B, memory and projects in the web UI, B60 switches

**Gemma 4 31B QAT** (`g31` in setup and the catalog; Google's Q4_0, Unsloth's drafter)
- KL against the f32 reference on a 536-token prompt: 0.00000 on the prompt path and the decode path (max 7e-5),
  the same top token at every position.
- A tight-memory mode for a dense model whose weights fill more than 85 % of an iGPU's share
  (`Engine::plan_tight`): part of the weights in host memory (bit-identical logits), prompt chunks of 1024,
  sliding-window rings with a margin of one chunk, `--ctx auto` from 1024 tokens. Only the 31B here; the other
  models plan as before. The host share is bounded by the RAM free at the start and is not cached; the server's load
  places the same amount its context probe did.
- On this laptop: generation 3.6 tok/s, 7.8 with the drafter; context 2048 (1024 with vision).

**Web UI memory and projects**
- `MEMORY.md` in the data folder: the model's `remember` / `forget` tools and Settings > Memory. A chat keeps the
  notes it started with; chats from before memory keep their prompt.
- Projects: a folder of yours, with its `AGENTS.md` (or `CLAUDE.md`) as instructions, a memory of its own, its
  files for the file tools (`project:<id>/<path>`) and the terminal opening there.
- Self audit: a chat reads only its own project's files; memory and projects only on this computer; `forget`
  removes at most 3 notes; the user folder, AppData and key folders cannot be projects; project files have a
  document cache of their own; a removed project's chats show under "No project"; a project file's digest notes
  get their own name.
- `AGENTS.md` for coding agents (Claude Code reads it through `CLAUDE.md`).

**Discrete GPUs and multi-GPU** (off by default; no change on the iGPU)
- `ARC_GEMV_PF=1`: the single-token GEMV issues all its rows' reads first (bit-identical; a row in
  `bench/decode_try.ps1`).
- `ARC_TP_ATT_QSPLIT=1`: the 35B's attention split by q-head over 4 GPUs (KL to one GPU 0.011 vs 0.008 whole).

## 2026-11-02 (late night): self audit of the day's B60 work

Three independent reviews (f16 XMX GEMM, multi-GPU changes, event probe and scripts). Each finding was checked in
the code before it was fixed.

**`event_probe`**
- Events were reused every 2 sums. A GPU that ran ahead could pass its wait on a peer's signal from 2 sums earlier
  and read stale data. The probe now uses a ring of 3 events and checks the data of 200 sums. QUEUE F31.3's design
  is corrected to match.
- A wait that never ends now releases every GPU and stops the probe instead of hanging it.
- Every call's result is checked.
- Output is unbuffered, so nothing is lost when it goes to a file.
- On real GPUs, one thread per GPU appends the work, as the engine does.

**f16 XMX GEMM**
- Weights are decoded times 16, not 256. With 256, a weight of 256 or more became inf and its column NaN.
- The f16 activations' buffer is reserved with the others, so the memory plan counts it (143 MB on the 27B).
- Chunks of 41-127 tokens whose size is not a multiple of 32 now use 16 or 8 tokens per sub-group instead of
  falling back to DP4a. A 48-token prompt: KL against DP4a 0.00019.
- The start-up check covers more cases and goes through the engine's own launch.
- The check's verdict is kept per device. A GPU wait that gives up during the check (this applies to
  `xmx_check` too) is not taken as a verdict.
- `act_half` is built without the GEMM.
- Checked: CG 1 / 2 / 4 bit-identical; 27B KL against DP4a 0.00069, unchanged by the new scale.

**Multi-GPU**
- A faulted split model now also sets every GPU's abort, so a given-up thread stops polling its wait.
- The wait-limit fields are atomic.
- The weight upload's staging buffer comes from the throwing allocation, so a shard short of host memory falls back
  to one GPU.
- Freeing a half-built shard keeps the first error.

**Also**
- A fused SwiGLU is never cut into split-K pieces (gemm.cl too; no model reached it).
- `gemm_sweep.ps1` / `decode_try.ps1`:
  - environment variables are removed even after Ctrl+C;
  - stderr no longer stops a sweep;
  - X16 rows are skipped where the f16 GEMM does not run;
  - X16 is never combined with DP4a only.

**Regression:** one GPU bit-identical; split 27B as before; split 35B as before with `ARC_TP_RG_GU=0`, and equal to
b3da0bf by default; f16 fallback equal to f32; the server's wait fault, OOM fallback and E4B tests pass.

## 2026-11-02 (night, 2): B60 queue F31.1 and F31.2

- **`ARC_X16_CG=2 / 4`**: the f16 XMX GEMM covers 2 or 4 column groups per sub-group, so each activation read feeds
  that many DPAS. With 1 group, every 16 columns re-read the activations from L2, which likely caps it far below
  XMX's rate. Logits are bit-identical for CG 1 / 2 / 4 (27B, DPAS emulated). `gemm_sweep.ps1` has an X16CG row.
- **`bin\event_probe.exe`** (in the package): can GPUs wait on each other's events through host memory, without
  peer access, and what does a sum cost that way against today's host wait and CPU barrier? It also checks the
  failsafe (the host releases a GPU waiting on a dead peer) and the driver's memory-wait extension. Laptop
  (`--sim 2`, one iGPU, a test of the tool): parts intact, 22.4 -> 16.7 us per sum, failsafe works.

## 2026-11-02 (late): B60 checklist items

- **Split MoE prompts: gate / up through moe_rg** as the down projection already was. Before, gate / up ran the
  per-tile kernel. Laptop (simulated 2-GPU split, 35B, 1K prompt): 498.7 -> 511.9 tok/s; KL to one GPU 0.0113 ->
  0.0066. `ARC_TP_RG_GU=0`: as before.
- `ARC_GEMV_MT_GRF=128` (the 2-8 token GEMV's register file, discrete GPUs) and `ARC_TP_RS_MIN=N` (the f16
  reduce-scatter's row threshold), for measuring on B60s.
- The XMX checks run once per process, not once per GPU of a split model.
- `tp_half` no longer writes a GPU's own rows (only range-checks them). Bit-identical.
- One GPU: logits bit-identical; split 27B and split 35B with gate / up per tile: bit-identical to before.

## 2026-11-02 (night): an f16 XMX prompt GEMM for discrete GPUs (`ARC_X16=1`)

- **`kernels/gemm_x16.cl`**: the prompt GEMM for W8 weights (the 27B's IQ3_S) in f16 on the XMX engines. Weights are
  decoded to f16 once per 16 columns with their scales applied, activations are made f16 once per GEMM, and one
  DPAS accumulator per 8 tokens runs over the whole row. The int8 XMX kernel rescaled every 32-value block on the
  vector unit, so it was only ~10 % faster than DP4a on a B60.
- Off until a B60 measures it: `ARC_X16=1` on a discrete GPU with XMX, checked against DP4a at start.
  `bench/gemm_sweep.ps1` has rows for it (X16, X16TM, X16GRF). Checked on the laptop with DPAS emulated
  (`ARC_X16_TRY=1`): KL against DP4a 0.00069 on the 27B. iGPUs: unchanged, logits bit-identical.

## 2026-11-02 (night): fixes from a second audit (DeepSeek, static)

- **A split shard out of memory ended the server** instead of loading the model on one GPU: its allocations now
  throw while it is made (`ARC_TP_OOM=r` tests it).
- **Whisper / Qwen3-ASR on GPU 0 shared the split model's wait limit**: a long wait of theirs could mark the split
  model faulty. The limit now applies only to the split model's own work.
- **A faulted split model left its wait limit on the GPUs** for the next model loaded; it is now taken back.
- `verify_batch` reads this engine's own logits rows (a split shard has a part of the vocabulary; not reachable yet,
  a split model has no MTP); the RAM tier's size is read under its lock. One GPU: logits bit-identical.

## 2026-11-02 (evening): multi-GPU self audit and failsafes

- **f16 sums out of range**: a prompt chunk's partial sum beyond f16 (65504 / GPUs) on any GPU makes every GPU sum
  that projection in f32 instead of turning it into inf / NaN (`ARC_TP_F16_LIMIT=0`, tests: bit-identical to f32).
- **A GPU wait that gave up was taken as "in step"** (from a review): with 2 GPUs a hung GPU 1 could leave the
  barrier count off by one and later sums silently wrong. Now any barrier or wait give-up breaks the split model
  for good (`/health` 503); a clean failure needs every GPU to have stopped alike, the same sums and no barrier in
  progress.
- **Unloading after a GPU fault** could end the process (a wait throwing in a destructor) or free memory a GPU or
  thread still used: the waits go back to no limit first, and after a fault nothing is freed until a restart.
- Shards with another weight layout / chunk / context than GPU 0 fail at load (runz too); `serverz --gpus` loads a
  model that cannot be split, or a split that fails, on one GPU; the three new sum kernels are released; a broken
  model's mirror no longer writes to the other shards first. One GPU: logits bit-identical.

## 2026-11-02: `serverz --gpus`, with a failsafe for a GPU that hangs

- **`serverz --gpus N`**: the server runs a split model (Qwen3.5 / 3.6 / 3.8 dense, the 35B-A3B) with `--slots` and
  batched steps. Shard 0 is the server's engine; its state-changing calls are mirrored on the other shards
  (`zero/mgpu/tp_mirror.h`) and their logits joined; a request's cancel is decided once for all. Text chat only for
  now (no MTP / DFlash2, images, prompt or session caches on disk). Over HTTP on the laptop (simulated split): the
  same greedy answers as one GPU, two chats at once the same as each alone, cancel and the next request fine.
- **Failsafe**: a shard waiting at a sum longer than `ARC_TP_TIMEOUT` s (default 120), or a split model's GPU wait that
  long, stops every shard with an error naming the GPUs that did not come; a thread stuck in the driver is given up
  on. The request fails, the server stays up, `/health` answers 503 until it is restarted. One GPU: no limit as
  before, logits bit-identical.

## 2026-11-02: batched decoding over a split model

- **Slots and batched decoding with `--gpus`**: a split model keeps its conversation slots (each GPU its part of
  every slot's state) and `TpEngine::forward_batch` runs several slots' generation steps in one pass, the sums
  carrying all their rows at once. `runz --gpus N --bench-batch M` checks the batched logits against each slot alone
  (bit-identical for the 27B over 2 and the 35B over 4, simulated on the laptop) and times 1..M slots. Not yet in
  `serverz`. One GPU: logits bit-identical.

## 2026-11-02: multi-GPU prompt sums in f16, reduce-scatter

- **A split model's prompt chunks sum through a reduce-scatter in f16** (`zero/mgpu`, `tp.cl`): each GPU sums one
  block of rows over all parts and shares that sum, instead of every GPU reading every part in f32. About 4.4x fewer
  bytes through host memory per sum on 4 GPUs, 4x on 2: on 4x B60 the split 35B's prompt spent 65-80 % of its time
  in the sums and ran slower than one GPU. Not yet measured there. Generation keeps the one-step f32 sum;
  `ARC_TP_SUM=f32` for the old sum everywhere. KL against one GPU (laptop, split simulated): 27B 0.0016-0.0018 mean
  (f32 sums 0.0009-0.0012), the same greedy text; one GPU unchanged (bit-identical).

## 2026-11-02: 128 registers for the prompt GEMM on discrete GPUs

- **Arc B-series (one card or several): the dense prompt GEMM now uses 128 registers per thread**, from the B60
  owner's sweep (issue #4: Qwen3.8 27B 450 -> 617 tok/s). `ARC_GEMM_GRF=256` for the old build; the MoE expert
  GEMMs keep 256 until measured (`ARC_MOE_GRF=128`).
- The `zero/xe2` switches act on discrete GPUs only: iGPUs with XMX (Lunar Lake, Arrow Lake) keep the iGPU kernels,
  where 128 registers made the 35B's prompt about 3x slower on the 125H. iGPUs: the same kernel builds as before,
  logits bit-identical.
- `bench\gemm_sweep.ps1`: GRF 128 is the default it sweeps from; new `MOEGRF` row for MoE models.

## 2026-11-02: launcher "Failed" pop-ups (issue #5)

- **Every start and stop showed "Failed:" over and over in Windows PowerShell 5.1**: there `Start-Process
  -PassThru` gives a `$null` ExitCode once the process ends, unless its handle was taken while it ran, so a start
  that went well counted as failed. The launcher now takes the handle (start / stop and recording), and clears
  the finished process before showing a message box: the timer goes on ticking while one is open, which showed
  the same box again every 0.5 s. The box now also gives the exit code. PowerShell 7 was not affected.

## 2026-11-01 (night): switches for Arc B-series tuning

- **Switches to measure on GPUs with XMX** (B580 / B60), in `zero/xe2/engine_xe2.h`: `ARC_GEMM_GRF=128` (the prompt
  GEMMs without 256 registers per thread), `ARC_GEMM_CG=2|4` (column groups per work-group), `ARC_GEMV_GRF=256`.
  Ignored on the iGPUs, where every kernel is built as before (checked: the same kernel builds, bit-identical logits,
  same speed); with them on, the XMX check passes and the logits are bit-identical.
- `serverz --act256` (as in `runz`): one activation scale per 256 values; less accurate (35B: KL 0.009-0.026 against
  the default), a switch to measure, not a default.
- `bench\gemm_sweep.ps1` sweeps `GRF`, `CG` and `ACT` too; new `bench\decode_try.ps1` (generation plain, MTP,
  DFlash2, GEMV switches); what to run on a B-series card: [docs/xe2-tuning.md](docs/xe2-tuning.md).

## 2026-11-01 (evening): XMX check at 32 tokens per sub-group

- **`ARC_GEMM_TM=32` kept DP4a on XMX GPUs**: the XMX check's 64 test tokens gave its split-K case an empty grid
  at 32 tokens per sub-group, so the check failed and `bench/gemm_sweep.ps1`'s TM 32 rows measured DP4a. It now
  tests 4 tiles of whatever TM is set (64 tokens at the default 16, unchanged). Found in a DeepSeek review.
- `ARC_GEMM_TM` takes only 16 or 32 (others warn and use 16; `gemm.cl` stops at build time on any other TM).

## 2026-11-01 (later): multi-GPU MoE, experts in RAM

- **Qwen3.6 35B-A3B over several GPUs** (`runz --gpus N`, and its fine-tunes such as Ornith 1.5 35B): the routed
  experts divided over the GPUs, each streaming and caching only its own; the attention layers stay whole on every
  GPU over 3-4 GPUs (the 35B has 2 kv-heads). Checked on one GPU with `ARC_TP_SIM` over 2, 3 and 4 shards: KL to
  one GPU no larger than one GPU's own between two chunk sizes. Details: [docs/multi-gpu.md](docs/multi-gpu.md).
- **Experts in RAM** on GPUs with memory of their own (B580 / B60): each expert read from the SSD stays in pinned
  RAM (up to the free RAM less 16 GiB, `serverz --expert-ram-gb N`, 0: off) and is copied from there by the GPU
  the next time. Never on an iGPU. Logits bit-identical.

## 2026-11-01: Ornith 1.5 35B-A3B checked

- **Ornith 1.5 35B-A3B** (a Qwen 3.5 MoE fine-tune for coding agents, MIT) runs as it is: the official Q4_K_M GGUF
  with its MTP block and `mmproj`, listed with MTP and images; generation 15.9 tok/s plain, ~23-24 with drafts (its
  MTP block 71 % accepted, Qwen3.6's DFlash2 83 %). Its Q4_K_M experts are 28 % larger than Qwen3.6's UD-IQ4_XS, so
  less of them fit the cache. Details: [docs/models.md](docs/models.md).

## 2026-10-31 (late): fixes from a static review ([BUGS.md](BUGS.md))

- **OCR save to a folder of your own** only from the server's own page or a client without an Origin: a page
  another local program serves could call the chat API and write a `.md` into any folder (checked: 403).
- An OCR save never replaces a file after 999 taken names (409); `chat.json.tmp` and `CHAT.JSON` are reserved names
  in a chat's folder; chat file references found in pretty-printed chat JSON too (they could be deleted as unused).
- Numbers from requests and prompts read without overflow (`read_file` offset / pages, OCR page numbers, transcript
  export options); `read_file` never starts inside a UTF-8 character; error replies say `Connection: close`;
  `ARC_QWEN_PS` checked; the web UI skips a non-JSON stream event; smaller hardening from `/analyze`.

## 2026-10-31 (last): multi-GPU measured on 4 Arc Pro B60

- **First numbers from real GPUs** (issue #4): Qwen3.8 27B generation 1.63x on 2 B60s, 1.97x on 3, 2.44x on 4
  (45.7 -> 18.7 ms a token, ~53 tok/s); pp512 1.28-1.38x. Table and what limits it: [docs/multi-gpu.md](docs/multi-gpu.md).
- **Fixed**: `serverz --npu on` found no NPU when started without Whisper / Qwen3-ASR (Gemma 4 12B: "no NPU device
  in Level Zero"): Level Zero was initialized for every driver type only in the speech service's start, after its
  early return, and `--ctx auto` started the GPU runtime (GPUs only) before the engine could. The server now does it
  first, right after reading its options.
- **Fixed**: 3 and 4 GPUs sometimes ended while loading in the driver's "Abort was called at 252 line in file:":
  the shards after the first were made at the same time on one Level Zero context; now one after another
  (`ARC_TP_PARALLEL_LOAD=1` for the old way).

## 2026-10-31 (late night): self audit fixes, the README as a landing page

- **Gemma ring snapshots, after an audit** (docs/self-audit-2026-10-31.md):
  - after a restore no rewind reaches below the restored window (it could read an abandoned conversation's keys);
  - the rings' copy is taken only when the rings are about to overwrite it, not at every snapshot: a turn that never
    goes ~4K positions past its start costs nothing (a copy: E4B 30 MB, 12B 480 MB of RAM, 12B 233 ms);
  - `runz --ring-snap-test` sizes its run to the model's rings and also checks the rewind below; E4B and 12B pass.
- **OCR**: a page stopped while being read is read again by the next "lanjut" (its heading had counted it as read).
- **digest_file**: a combining step that does not shrink the notes keeps the ones before it.
- **Documentation**: the README is a landing page (what it is, why, how to try it, results, models, an index of the
  docs); its long sections moved word for word to [docs/install.md](docs/install.md),
  [docs/server.md](docs/server.md), [docs/models.md](docs/models.md) and [docs/architecture.md](docs/architecture.md);
  [docs/features.md](docs/features.md) tours the best features.

## 2026-10-31 (night): Gemma 4 conversations keep their cache after long answers and tool results

- **The same prompt every turn**: Gemma 4's template keeps an assistant tool call's reasoning while the turn's tool
  rounds go on and drops it once the user writes again, so the next turn's prompt differed right before the tool
  results. The server now renders a tool call without its reasoning every time (`gemma_history`). E4B in the web UI,
  "Ringkas README.md" (a 20K-token read) then "Model apa saja yang didukung?": the follow-up's prompt 22,092 tokens,
  **21,846 from the cache, 0.8 s** (it was 2,234 from the cache, 38.4 s).
- **Ring snapshots**: Gemma's sliding-window layers keep a ring of ~4K positions beyond their window, so the cache
  could go back only that far (`rewind`). The engine's snapshots (end of the prompt, start of the model's turn) now
  also keep a host copy of the rings' last window (13.7 ms to take, 6.4 ms to restore on E4B); when a request
  differs further back than the rings reach, the latest snapshot before it is restored instead of processing the
  whole conversation again. `runz --ring-snap-test` (a Gemma model, a prompt of 7,600+ tokens): restored after 6,000
  more tokens, the next 600 tokens' logits bit-identical to a fresh run.
- `digest_file`'s outline note says "do not answer from it": E4B, asked about all of a 159 KB file, had answered from
  the outline alone; now it called `digest_file` (4 parts, 306 s) and answered from the notes.
- `ARC_DUMP_PROMPT=file`: every rendered prompt appended to the file (where two turns' prompts differ).

## 2026-10-31 (late night): digest_file, tool-call fixes from an audit, a shorter prompt

- **`digest_file {file_id, focus}`** (Files on): a document longer than one read, for all of it (a summary, a review,
  its weak points). The server reads it part by part with the loaded model, each part a request of its own (no
  compaction, no tool rounds used), and returns notes of every part with their pages.
  - Parts end before a section's heading (a function's start in source files), else a page's heading, else a
    paragraph's end; a paragraph longer than a part is cut at its lines; a part that starts inside a section is
    told which one it goes on with.
  - The notes are saved as `<name> (notes).md` in the chat's files: a later call reads them at once (whatever its
    focus), an unfinished one goes on from its last part (Stop keeps the notes so far).
  - Notes longer than one read are combined again until they fit, and never come back longer than one read.
  - Checked with Qwen3.5 4B at 16K context: README.md (57 KB) 3 parts in 177 s; `tools/server_ocr.h` (30 KB) 3 parts in
    85-92 s; the same call again 0 s.
- **Fixed** (an audit of the tool-call path):
  - Qwen: a `<tool_call>` written in the reasoning was taken as an open call, so after `</think>` every token failed
    the constraint (a scan of the whole vocabulary each) and the real call went unchecked.
  - Qwen: string parameters with an `enum` (list_files' `scope` / `kind`, create_file's `page`, terminal_panel's
    `action`) took any text; now one of their values.
  - `tool_choice: "none"` keeps the tools in the prompt and only stops the call's open marker, so the web UI's last
    answer after 8 tool rounds no longer processes the whole chat again (the prompt is the same 2,680 tokens; three
    tries answered in words).
  - A chat answer that no longer fits in the middle of its tool rounds shortens its own earlier rounds' results
    before it gives up (a compaction keeps them).
  - `list_files` 20 files by default and `search_chats` without a query: at most one read of the context.
  - `read_file`: the outline for any text one read cannot hold (it was only over 24,000 characters), and its parts
    point to `digest_file`.
- **Shorter tool texts:** what a system text and a tool's description both said is said once (the sheet and slide
  layout, the terminal's session rules, digest_file's use), and two texts that disagreed on who allows a command now
  agree. The prompt with every tool on: 2,870 -> 2,680 tokens (Files alone 2,132 -> 1,995).

## 2026-10-31 (night): reading an OCR .md with a chat model

- **`read_file` reads a document whole** when the model's context takes it in one read (`files::read_budget`) and
  no `max_chars` is asked: without arguments, and from an `offset` too (the rest at once); a longer one still gets
  its outline, and parts. Qwen3.5 4B (262K context) read the 56-page OCR `.md` (98 KB, 25K tokens) a page-section at
  a time, then with `offset: 0` in 12,000-character parts, and ran out of its 8 tool rounds at 80K of 98K. Now, in
  the web UI, "Baca seluruh file ... lalu berikan kekurangannya": one call, the whole file (23K-token prompt,
  670 t/s), the answer from all of it.
- **`pages: "A-B"` in a Markdown file with `## Page N / M` headings** (an OCR model's `.md`): those pages' sections
  in one call ("26-30" said "no text" before). The web UI's `read_file` description says so.
- **Qwen tool calls with a number argument** (`section: 12`): every token after its `</parameter` was refused by the
  call's constraint, each one a scan of the whole vocabulary, so tool calls ran at ~13.5 tok/s instead of ~18.
  Fixed in `tools/tool_constraint.h` (a closed value is checked as one that has ended); details in
  [docs/before-after.md](docs/before-after.md). `ARC_DEBUG_TOOLFIX=1` logs each refused first choice.

## 2026-10-31 (late evening): saved OCR text stays in the file

- **Saved to a .md**, the OCR answer shows only the bar of the pages done ("reading page N / M…": the server now
  sends each page's heading before it reads the page), not the text.
- The chat keeps a note instead of the pages: the last page's heading (the next "lanjut" goes on after it) and
  "the text of pages A-B is saved in the file NAME in this chat's files (read_file it)". A chat model switched to in
  the same chat reads the file from there.
- A folder of the user's gets a copy in the chat's files under the same name (a chat model's `read_file` reads no
  folder of this computer). Checked in the web UI on a 2-page PDF: the bar only; the external `.md` and its copy the
  same 1,881 bytes; `list_files` / `read_file doc_sample.md` of that chat find it.
- Not saved: every page shows, as before.

## 2026-10-31 (evening): launcher errors, PDFs of a new chat, NPU messages

- **Launcher** (issue #5): the server monitor and the model details popped up "Index (zero based) must be greater
  than or equal to zero..." every 2 s. In `$L.Add("... {0} {1}" -f a, b)` the commas split the method's arguments,
  so `-f` got one value; each such line now has its own parentheses (`scripts/arc-launcher.ps1`).
- **PDFs for OCR models**: the first PDF of a new chat could fail with "An item cannot be found at the specified
  path". The chat's folder is renamed when the chat is first saved, which came while the PDF was being opened. The
  PDF is now read into memory once and rendered from there (`tools/pdf_render.h`).
- **Self audit** of the OCR mode (docs/self-audit-2026-10-31.md, evening):
  - a page the model could not read stops the pages (`[OCR: stopped at page P: why]`), so the web UI no longer
    sends "lanjut" into the same error to the PDF's end;
  - PDF answers carry the pages' `timings` (the statistics line said "stopped");
  - an `.md` that cannot be written is refused before any page; `CON` / `NUL` / `COM1` as its name become
    `ocr CON.md`.
- **NPU**: a driver's newer graph extension (1.9 and up) is taken too, and "no NPU" now says what is missing: no NPU
  device in Level Zero (the Intel NPU driver), or a driver too old for graph extension 1.8.

## 2026-10-31 (late night): long PDFs for OCR models

- **Every page, by itself**: a PDF is read 50 pages a turn. An answer with pages left ends with
  `[OCR: N more page(s); "lanjut" reads pages A-B]`, and the web UI sends "lanjut" itself until the last page
  (Stop ends it). The words may also name pages (`halaman 51-100`, `page 7`). A "lanjut" goes on with the same
  PDF and task after the highest `## Page N` of the answers so far.
- **Saved to a file**: when a PDF is sent to an OCR model, the web UI offers to save its text to
  `<the PDF's name>.md` in the chat's files (`ocr_save`, `chat_id`).
  - The question comes in a modal of the page, asked again with every PDF sent, not the browser's confirm box:
    - the file's name (`<the PDF's name>.md`) and folder (empty: the chat's files; or a folder of this computer,
      taken from requests of this computer only);
    - Save to .md / Just show it (Esc: just show it).
  - An existing file is never written over: `name (2).md` then. The "lanjut" turns go on with the file the server
    named at the end of its answer.
  - Each page is appended once read, and the file shows in the Files panel.
  - Saved or not, the answer shows every page under a bar of the pages done, and the chat keeps them all. A chat
    with an OCR model is not compacted (the server reads only its last turn), so no page is lost to a summary.
  - Checked on a 450-page service manual: pages 1-2, then 449-450 appended to the same file; in the web UI the bar
    counted on and "lanjut" went by itself.

## 2026-10-31 (night): faster BF16 models (the OCR model)

- **Float matrices stacked as quantized ones are** (`zero/engine.h`):
  - DeltaNet `qkv` + `z` + alpha / beta: one GEMV, since a BF16 GGUF's float matrices become the same Q8_0 rows
    first; alpha / beta join the stack when their rows come in 16s, not only 64s;
  - FFN `gate` + `up`;
  - gated attention `q` + `k` + `v`.

  Xiaomi-OCR-0 (Qwen3.5 0.8B) runs 9 GEMV launches fewer per layer:
  - generation 69.3 -> 73.2 tok/s, prompts +2 %;
  - logits bit-identical;
  - quantized models take none of these paths.

  The output head as Q4_0 was 86 tok/s, but KL 0.048 against f32 (the limit is 0.015), so it stays Q8_0. The image
  encoder (~1.44 s for a 2,000-token page) is already near the iGPU's f32 rate; its attention in f16 is left as the
  next step. Switches for A/B tests: `ARC_NO_QZ_FLT`, `ARC_NO_GU_FLT`, `ARC_NO_QKV_FLT`, `ARC_NO_QZAB16`.

## 2026-10-31 (evening): OCR mode

- **OCR models get their own task prompts** (`tools/server_ocr.h`). Xiaomi-OCR-0 (Qwen3.5 0.8B with an image
  encoder, converted to GGUF with llama.cpp's `convert_hf_to_gguf.py`, model + mmproj) knows a few English task
  prompts only:
  - from the web UI (a long system prompt, tools, thinking, sampling) it answered in Chinese;
  - asked in Indonesian it wrote a letter's title only.

  A model whose name says "OCR" is now asked the way it was trained:
  - its last turn with the image and the task its words name (text by default; `dokumen` / `document`, `tabel` /
    `table`, `rumus` / `formula`, `json` + fields);
  - no system prompt, tools or thinking, greedy;
  - fields named in Indonesian become the model's English keys (`tanggal` -> date, `nama toko` -> seller_name).

  Checked with a chat client's long system prompt, a tool, thinking and temperature 0.8:
  - an Indonesian letter: word for word (similarity 1.000), also with no text at all; as a document 0.995;
  - the README's receipt: `json: tanggal, total, nama toko, alamat toko, pajak` all right;
  - the repo's examples: 0.975 / 0.988 to its outputs.

  The web UI's message box shows the tasks.
- **OCR mode from the web UI** (the same evening): the web UI sends its images as saved files (`"type": "file"`),
  which became image parts only after the OCR rewrite looked for one, so its requests went through unchanged (and
  still came back in Chinese). The rewrite now runs after the files are expanded and skips their `[file: ...]`
  lines. A follow-up without an image ("now as a table") takes the conversation's latest image. Checked with
  files from a Downloads folder through the web UI's way (file part, system prompt, thinking, temperature 0.8), all
  in Indonesian words: no Chinese in any answer.
- **PDFs for OCR models**: a PDF in the last turn is rendered page by page by Windows (`tools/pdf_render.h`,
  Windows.Data.Pdf: every kind of page, as a viewer draws it, ~0.2 s a page at 150 dpi). Each page is read as an image
  and streamed under `## Page N / M` (at most 50 pages a request). Before, a PDF went in as its text layer, and
  the follow-up rule took the conversation's last image instead (the user's report came back as an earlier picture);
  a turn that brings a file now never looks back for an image. A 5-page report: first page after 12 s, all in 50 s.
- **OCR tables as Markdown**: the model writes tables in OTSL (`<fcel>cell<fcel>cell<nl>...`). The server turns
  them into Markdown tables in non-streamed answers, and the web UI does so while an answer streams. A report page's
  similarity to its PDF text layer went from 0.61 to 0.87.

## 2026-10-31 (afternoon): multi-GPU audit, uneven splits

- **Self audit** (docs/self-audit-2026-10-31.md):
  - a constructor error ended `runz --gpus` without a message (worker threads still running);
  - the cut tensors' host copies held the model a second time in RAM;
  - a failed shard left the others out of step;
  - the 27B over 4 GPUs could not load prompts (24 alpha / beta rows; padded to 16 now).
- **Any number of GPUs** the model's heads allow: counts that do not divide are split as evenly as their units allow
  (27B over 3 GPUs: 6 / 6 / 4 k-heads, 2 / 1 / 1 kv-heads). 27B over 2 / 3 / 4 shards: greedy text identical to one
  GPU.
- **The multi-GPU code moved apart:** `zero/mgpu/` and `zero/kernels/tp.cl`; the one-GPU code only calls hooks,
  and its logits stay bit-identical.

## 2026-10-31: one model over several GPUs (runz --gpus N)

- **Tensor parallel for dense Qwen3.5 / 3.6 / 3.8** (docs/multi-gpu.md):
  - `runz --gpus N` splits the DeltaNet and attention heads, the FFN columns and the output head's vocabulary over N
    GPUs, so each GPU reads 1/N of the weights per token.
  - The output projections' partial sums meet in pinned host memory (B60s on Windows have no peer access).
  - Checked on one GPU with `ARC_TP_SIM=1`: greedy text identical, KL against an f32 reference the same as one GPU
    (4B: 0.00113 / 0.00096 / 0.00109 for 1 / 2 / 4 shards).
  - Speed needs a machine with several GPUs; the server does not use it yet.
- **The runtime keeps its state per GPU** (`zero/clrt.h` `zrt::Dev`, `zrt::D()`):
  - one Level Zero context holds every GPU of the driver;
  - a thread works on the GPU it chose, GPU 0 by default;
  - one GPU is unchanged, with logits of the 4B and 27B bit-identical to before.

## 2026-10-31: speech to text without a chat model

- **`serverz -m none`** (`arc-server start -Model none`, the launcher's **Speech only** button): only the speech
  service loads (Qwen3-ASR alone, ~3.2 GB of GPU memory; `--whisper auto` adds Whisper), for transcripts through
  the web UI's Transcripts tab or `POST /v1/audio/transcriptions`. `/health`, `/props` and `/v1/models` answer with no
  model loaded (the model started last listed first, none `loaded`), and the first chat request naming a model loads
  it, context auto next to the speech models. A 2:43 voice message: 25 s.
- **The first transcript after a start no longer waits for the encoder**: Qwen3-ASR's audio encoder loaded its
  weights and kernels on its first clip (60 s of audio: encoder 4.1 s instead of 0.5); now when the server starts.
- **Web UI recordings: the MP3 bitrate is a choice** (Transcripts tab, "MP3 96 / 128 / 160 / 192 / 256 /
  320 kbps", or WAV) instead of a fixed 192. Windows' MP3 encoder takes at most 128 kbps for one track (a
  microphone): higher choices give 128 there; a meeting's two tracks get the bitrate chosen (checked: 96, 192, 320).
- **Recordings kept as M4A (AAC)**: "M4A 96 / 128 / 160 / 192 kbps" next to WAV and MP3 (`POST /v1/live`
  `keep: "m4a"`), with Windows' AAC encoder through Media Foundation as the MP3 (on the CPU: audio encoding is
  sequential and light, and the GPU has no audio unit). AAC takes 192 kbps for one track too (MP3 stops at 128);
  60 s at 96 / 128 / 192 kbps: 0.73 / 0.97 / 1.45 MB, mono and stereo, played back by the page (audio/mp4).
- **Mono or stereo recordings**: the Transcripts tab's new Mono / Stereo choice (default Mono). Stereo keeps left
  and right of the one source as it plays them (a tab's music or video; a stereo microphone, then without the
  browser's echo and noise suppression; `POST /v1/live` `channels: 2` with one track), so the MP3 gets the bitrate
  chosen; the transcript is made from its mono mix (60 s: the same 164 tokens as the mono recording). A source that
  reports one channel stays mono; the Meetings tab keeps one mono track per source.

## 2026-10-30 (night): multi-GPU groundwork

- **bin\mgpu_probe.exe** (zero/mgpu_probe.cpp, in the package): lists the GPUs and measures what moving data between
  them costs: peer access, a 20 KB transfer (a tensor-parallel sync of token generation, ~128 per 27B token) and a
  40 MB one (a prompt chunk), GPU to GPU directly and through host memory, on the compute queue and the copy engine.
  Its numbers decide how a multi-GPU arcinferzero splits the work. On this laptop (one GPU, to itself): 20 KB 5 us
  on the compute queue, 70 us on the copy engine; through host memory 11 us.
- **bench\gemm_sweep.ps1**: sweeps the prompt GEMM's settings (XMX on / off, tokens per sub-group, row groups,
  work-group height, split-K, chunk size) one at a time from the defaults with `runz --bench-pp`, then the best of
  each together, and prints which path ran (XMX, or DP4a when the XMX check failed). For B-series owners to report
  what their GPU prefers: the XMX kernel was written without an XMX GPU at hand (on a B60 the 27B's prompt runs at
  ~12 % of the card's int8 peak). Reading it for this: each DPAS sums only 32 values before the per-token unpack and
  the per-16 / per-32 scales, and the per-16 layouts fill half of every DPAS tile with zeros, so the large gain on
  XMX needs a kernel that accumulates longer runs, beyond what these settings reach.
- **runz compares its GB/s with the GPU it runs on**: it measures the GPU's read bandwidth (Engine::read_gbs, a
  256 MB block-read stream) instead of assuming this laptop's ~75 GB/s (measured here: 70 GB/s).

## 2026-10-30 (evening): faster tool-call turns on the 27B

- **Short prompts of just-in-time models in passes of 8 rows**: a chunk of a model whose weights are decoded just in
  time (the 27B) decoded every matrix first, ~2 s however few tokens it held, so every tool-call turn waited ~2 s
  before its first token. Up to 40 new tokens now go through the generation's multi-row GEMVs (~0.35 s per 8):
  a tool turn's prompt 2.3 -> 1.3-1.4 s. DFlash2's capture writes each pass at its own rows (it overwrote the first
  rows and the drafter fell back to MTP).
- **Gemma 4 12B the same up to 32 tokens** (its two-pass passes cost more; past ~32 the chunk is faster again): 16 /
  22 / 27 new tokens 1.1 / 1.2 / 1.2 -> 0.6 / 0.7 / 0.9 s, replies byte-identical. Qwen 9B (resident weights) and
  the 4B, E4B and MoE models are unaffected.

## 2026-10-30 (afternoon): a progress bar while the model loads

- **arc-server and the launcher show how far the load is**: the server writes `progress: P % <step>` lines to its
  log while it loads (src/load_progress.h): Qwen3-ASR's and the model's weights (the bytes uploaded, without the
  experts streamed from the SSD), Whisper's layers, the expert file written on a first start, the expert cache filled
  from the SSD, and the kernels' warm-up (no percentage: a busy bar). `arc-server start` draws them as a PowerShell
  progress bar with the seconds so far; the launcher has a bar under its status line and the step in the message
  line (a switch reads the log only once the new start has rewritten it, not the stopped server's lines). The
  command-line tools (runz, transcribez, asr_testz) print none.
- Self audit of today's changes, fixed: the web UI no longer follows Gemma 4 E4B when the speech service loads it for
  a transcript (the chat keeps its model); `--asr` without Whisper that cannot load Qwen3-ASR stops the server with
  the reason instead of advertising speech to text with no model behind it; the expert file's per-layer line is
  back for the command-line tools; the weights bar leaves out the tensors kept on the host (E4B's stalled at ~60 %).

## 2026-10-30 (later): Qwen3-ASR in 3.15 GiB

- **Qwen3-ASR's encoder on two int8 passes** (its blocks and projector; the convolutions stay f16): 0.037 % from
  the f32 encoder instead of 0.059 % (the int8 GEMM reproduces the mmproj's Q8_0 blocks exactly, f16 rounded them),
  34 % faster, 0.26 GiB less; the f32 text model's KL over a transcript max 0.000002, the same text. `ARC_ASR_Q2=0`:
  f16. On the way: in two-pass engines the encoder's quantization wrote its residual at the engine's offsets (the
  ops program carries them), so the encoder now uses the engine's layout there.
- **One Qwen3-ASR slot by default** (`--asr-slots N`, new `arc-server -AsrSlots N`, 1-4, kept like `-Asr`): each
  slot past the first ~0.6 GiB, faster only for clips over 4 minutes (a 14-minute file 129 s with one, 80 s with 4).
  Qwen3-ASR alone in the server: 5.37 -> 3.15 GiB; the 35B keeps 55 % of its experts cached (40 % before, 10 % with
  Whisper too).
- `asr_testz --kl-files P` and `ref_forward` `REF_EMB` for the Qwen models: the encoder's effect on the text model's
  KL.
- **The web UI shows the model the server has loaded**: a server started on another model than the one used last
  (arc-server, pi-arc, Pi) still showed the earlier one, because opening a chat took the model saved with it (the
  page reopens the last chat), and the next message then switched the server back to that model. The page now
  follows the loaded model unless a model is picked in its menu (that pick stays until the server loads another
  one), and asks the server again whenever the page is shown again.
- **Whisper large-v3 in 3.94 GiB instead of 4.39** (NPU mode 3.28): the decoder layers' cross keys / values share one
  f32 buffer before they are packed to f16 (it was one per layer); logits and transcripts byte-identical.

## 2026-10-30: Qwen3-ASR without Whisper; the terminal asks once

- **Qwen3-ASR alone**: `--asr auto` (or a GGUF) without `--whisper` (`arc-server start -Asr auto`) starts the speech
  service without Whisper: 5.37 GiB of GPU memory with its 4 parts at a time instead of 9.76 with Whisper, so the 35B
  kept 40 % of its experts cached instead of 10 %. Transcription requests, the Transcripts and Meetings tabs and chat
  audio go to Qwen3-ASR (a 2:43 voice note in 29 s, SRT included). What needs Whisper says so: translations and a
  request naming Whisper get 400; a live recording keeps recording and gets its text from the pass after Stop (none
  while recording; the web UI notes it). The web UI hides and disables "Model: Whisper" then (`/props` "whisper":
  {"whisper": false}). arc-server keeps `-Asr` on a restart with or without `-Whisper`.
- **The web UI's terminal asks once per session**: Settings > Terminal is now **Ask once per terminal session** (the
  default: the first command of a session waits for Run), **Ask before every command** or **Model never**. The old
  default asked before every command, and "Run without asking" in fact asked once: both saved choices become the
  first. The calls of one answer that run at once wait for the first one's Run instead of each asking (three
  `run_command` calls in one answer: one Run, all three ran; the next answer's command ran without asking).

## 2026-10-30: Qwen's reasoning with a presence penalty

- **Qwen models reason with presence_penalty 1.5** (Qwen's recommendation for thinking) when the request sets none:
  the Qwen3.6 35B, asked in the web UI to turn a sprite sheet into a GIF, circled its plan ("Let's run.", "Wait, ...")
  for 7,000 tokens without closing `</think>`, so its `run_command` call never came (17 minutes with Whisper and
  Qwen3-ASR loaded beside it). `</think>` was never held back: at those points it was the most likely token (~50 %),
  and temperature 1.0 drew another. The penalty applies to the reasoning only; the answer and its tool calls keep
  the request's value (0 by default), so code is not penalized for reusing its tokens. A request's own
  `presence_penalty` applies to both, as before. Gemma 4 unchanged.

## 2026-10-29 (evening): a ready-to-use package for people who never open a terminal

- **Downloads in a Windows PowerShell started from PowerShell 7** (GitHub issue #3: "The term 'Get-FileHash' is not
  recognized" at the SHA256 check): the hash is computed with .NET, and setup.cmd, arc-launcher.cmd and the pi-arc /
  arc-server commands clear the inherited PSModulePath, so Windows PowerShell loads its own modules.
- **setup (the installer from the sources)**: a model the server already finds outside the Hugging Face cache (a
  package in `.arcinferzero\zero\models`, a `model-dirs.txt` folder) counts as there (the packaged 35B was offered for
  download again); `setup.cmd` ends with setup's exit code (it said 0 after a failure) and, double-clicked, keeps its
  window open at the end; the last lines point to `arc-launcher.cmd`.
- **Precise mode as an option**: `arc-server start -Precise` / `-NoPrecise`, `pi-arc --precise`, the launcher's
  "Accuracy" box (Flash Next only) and the server's `--precise` (the same as `ARC_FN_PRECISE=1`, which works as
  before); kept on a restart and a model switch like `-Npu`, reported by `arc-server status` and in the server log.
  `arc-server restart` without `-Model` now restarts the running model (it kept the server without restarting it).
- **Flash Next starts with images through arc-server, the launcher and pi-arc**: the model list did not look for a
  projector next to a `qwen4exp` model, so it listed Flash Next without images and arc-server passed --no-mmproj.
  Now it lists `qwen3.8-flash-next-iq4_xs` (images, the default) and `...-std` (none).
- **The launcher sets everything up**: unpack the zip, double-click `arc-launcher.cmd`. The first time, a window
  offers a model in plain words (the one that suits the laptop's RAM ticked: 4B on 16 GB, the 35B on 32 GB), speech
  to text as an add-on and a desktop shortcut; "Download and start" downloads it with a progress bar, starts it and
  opens the web UI. "Get models" adds more later; "Open the web UI when started" opens it after every start. The
  package's files are unblocked once (a zip from the internet marks them), and a missing Intel driver is named with
  Intel's download page (scripts\launcher-firstrun.ps1).
- **No compiler, no Python, no Visual C++ Redistributable**: `scripts\make-package.ps1` makes
  `dist\arcinferzero-<version>-win64.zip` (6 MB: every tracked file, the built exes, `bin\PREBUILT.txt`); setup
  keeps the shipped exes (`setup -Rebuild` builds them). The exes are built with `/MT` (the C++ runtime inside) and
  classic `/openmp`, whose `vcomp140.dll` may be shipped (build.ps1 copies it into bin); `/openmp:llvm`'s
  libomp140 may not. Logits bit-identical to the old build, speeds the same (4B: prompt 840-894 against 857-876 tok/s,
  generation 18.6-18.7 against 18.0-18.4).
- **Model downloads without Python**: `scripts\hf-get.ps1` (curl.exe, part of Windows: resumed after a cut, each
  file's SHA256 checked, the Hugging Face cache layout the server reads; HF_TOKEN for gated repositories) and
  `scripts\get-models.ps1` over one catalog for setup and the launcher (`scripts\model-catalog.ps1`). The 4B, 9B
  and 27B come with their image projectors now; `35b` is Unsloth's UD-IQ4_XS (MTP, images, no conversion) and
  `35b-gsq` the converted 2-bit file (the one step that still needs Python).


## 2026-10-29: the data folder is %USERPROFILE%\.arcinferzero; Qwen3-ASR keeps the language; Flash Next images

- The data folder (kernel cache, model packages and expert files, caches, logs, settings, chats, transcripts,
  recordings) is now `%USERPROFILE%\.arcinferzero`, a dot folder like the other tools' folders in the user's profile
  (`.cache`, `.pi`, `.vscode`). The first exe or script that finds only the old `%USERPROFILE%\arcinferzero` (or an
  older `%LOCALAPPDATA%` one) moves it: a rename on the same drive, instant, nothing is copied. The paths in
  `last-model.txt`, `launcher.json`, `server.json` and `zero\ctx-auto.txt` follow it, in any case (the context
  cache's keys are lower case; until now that file was not rewritten, so an earlier move left its entries behind and
  the contexts were computed again). `ARC_DATA_DIR` still overrides the place.
- The help's file lists line up again after the longer folder name.
- **Option mistakes say what is expected** (serverz, arc-server, pi-arc): a value missing (`--whisper` at the end, or
  followed by the next option), a file where a folder belongs or the other way round, a path with a stray quote (a
  quoted path ending in a backslash), a .gguf that is not one, an unknown option (with the closest one: `--wisper`:
  did you mean `--whisper`?; `-whisper` to serverz: it takes two dashes), each with the kind of value it takes.
  arc-server no longer ignores an option it does not have (`-Whispr on` started the server without speech), checks
  `-Whisper` / `-Asr` paths and drops a path's last backslash; pi-arc passes `--whisper`, `--asr`,
  `--whisper-device` and `--whisper-max-minutes` to arc-server (they went to Pi). The helps of both have WRITING
  OPTIONS (dashes, on / off, paths, what each path is) and WHEN IT FAILS (each message and what to do).
- **`--whisper on`** was taken for a folder named "on" and stopped the server with "cannot open model.bin". on / yes / true / 1
  now mean auto (off / no / false / 0: off), for `--whisper` and `--asr` in serverz and in arc-server, and a folder
  without a model.bin is named in the message. arc-server's help says that options take one dash or two (`-Whisper`,
  `--whisper`).
- **arc-server start without -Model and no console** (a script, a scheduled task, a tool with its input
  redirected, `-NonInteractive`): the model menu asked again on every empty answer, forever, at full CPU. Now a
  running server is kept, else the model used last starts (as Enter and pi-arc do), else it stops with a message.
- **Web UI on a phone**: tapping a chat in the list did not open it: the row's delete button appeared on hover, and a
  touch screen (iOS) takes the first tap that shows something as the hover alone. Touch screens now show the delete
  button (and a message's menu button) all the time; with a mouse they appear on hover as before.
- **Web UI lighter on the browser**: the transcript player's word highlight runs a frame loop only while the media
  plays (it ran 60 times a second all the time, so the page never went idle); a streaming answer is rendered again
  only when its text changed, at a gap that grows with the render's cost (8x its time, at least 120 ms, so a very
  long answer no longer takes the browser's time quadratically); the transcript list's rows are rewritten only
  when what they show changed.
- **Image prompts in one pass** (Qwen images on models without an MTP block or DFlash2 drafter, so Flash Next):
  the text and the image rows go through the layers together instead of a pass per piece, so the streamed experts
  are read once: prompt time -8 to -24 % (290-798 tokens with a picture), the same answers. `ARC_MEDIA_PASSES=1`:
  the old way. Flash Next's image encoder stays f16 (int8 two passes would be 26-28 % faster at KL up to 0.0098).
- **Flash Next reads images right**: an image's rows went into the model as single rows of its width, while Flash
  Next keeps every row as 4 hyper-connection streams, so four image rows shared one row and the PLE n-gram block read
  the previous text's rows. Each row now goes in as its 4 streams with the n-gram rows of `<|image_pad|>` (the model's
  `ple.image_token_id`). A test picture's "PROFIL RISIKO 2026", read as "PFILIKO 2013" before, is now read right.
  Its image projector: `mmproj-Qwen3.8-Flash-Next-f16.gguf` from bartowski/Qwen3.8-Flash-Next-GGUF, next to the
  model (or its package).
- **Qwen3-ASR keeps to the transcript's language**: the request's language (a code, or the first of a list such as
  `id,en`) is now forced (`language Indonesian<asr_text>`), and the Web UI's default is Indonesian. Detection named
  most parts of a 26-min Indonesian meeting Malay, and one part drifted into Malay words; forced Indonesian kept every
  English term the detected text had. A glossary (the Transcripts tab's field) spells the jargon as given and gives
  the punctuation back (docs/qwen3-asr.md).
- **Qwen3-ASR BF16** (`Qwen3-ASR-1.7B-bf16.gguf`) loads: the stacked Q / K / V and gate / up matrices refused BF16 /
  F16 / F32 matrices; they are now requantized to Q8_0 row by row, as a single matrix already was. The transcript is
  word for word the one of the Q8_0 file (the engine holds both as Q8_0), the load takes 17.6 s instead of 5.4 s, so
  the BF16 file brings nothing over Q8_0.

## 2026-10-28: the launcher, setup and the docs brought up to date

- **arc-launcher**: every arc-server option now has its field: speech to text (Speech, Qwen3-ASR, Whisper's encoder,
  the longest clip), NPU, conversations at once, images / audio, no MTP, no streaming, earlier turns' reasoning;
  Record / Stop recording buttons for meetings (when `bin\arc_recorder.exe` is there).
- **setup**: `-Models whisper asr` downloads the speech models (Whisper large-v3, Qwen3-ASR 1.7B with its projector).
- **Transcripts tab**: Accurate (beam 5) / Fast (greedy) shows only for Whisper: Qwen3-ASR and Gemma 4 E4B always
  decode greedily.
- **Docs**: what was written two or three times (setup, Pi, the launcher, troubleshooting, the web UI, NPU, speech)
  now lives in one place with links from the others; out-of-date statements corrected (chats are kept by the server,
  the speech models, files copied between chats); the finished Gemma two-pass plan moved to docs/archive/.

## 2026-10-28: an external audit checked and fixed

A further static audit (23 findings) was checked finding by finding in the code; all held and are fixed:

- **Model and file input checked**: a Whisper model whose tokenizer or alignment heads point outside the model is
  refused at load (it could write past a buffer); a PDF's stream length past the file no longer overflows; JSON
  with a lone surrogate gives U+FFFD, raw control characters (other than tab / newline) are refused; numbers from a
  request are clamped before they are converted (slots, top_k, max_tokens, offsets, live recording rates).
- **Files**: uploads are written to a temporary file and moved into place (a crash left a cut file under the full
  file's id); a file edited to the same size gets a new id (its time is kept); a file reused for an upload is
  compared byte for byte first; an extension longer than 16 characters no longer escapes the name limit.
- **Terminal**: a command never inherits the server's other handles; its output reader cannot end the server.
- **API**: `seed: 0` is a seed (it was taken as none); `top_k: 0` means no top-k; the last streamed token's
  logprobs are sent; web_fetch keeps one deadline over its redirects; `http://[::1]:port` pages are recognised;
  a chunked body with a broken chunk end gets 400; saved conversations with positions outside the context are
  not loaded.

## 2026-10-28: an audit of every source and document

Six reviews (server, file tools, speech, web UI, the GPU engine's host code, scripts and docs); what they found is
fixed (docs/self-audit-2026-10-28.md, "All sources"):

- **Other local pages are kept out**: a page of another server on this computer (another port: a dev server, a
  folder served by Python) could call the terminal tool, the slots, chats, files and transcripts. It now reaches
  only the model endpoints (chat completions, the audio ones, /tokenize); the web UI, Pi and command-line clients
  work as before. Text inside a fetched page, a file or your message that looks like the model's control tokens is
  read as text, no longer as a turn of the conversation.
- **PDFs that hung the server**: a broken font map looped forever, and some files took hours to parse; reading a
  PDF now has a work limit and keeps what it read with a note. Very long paragraphs, a huge word in a written PDF
  and long CSV fields no longer slow to a crawl.
- **Qwen3-ASR keeps the end of long recordings** (parts of 4 minutes, with room for their text; 6-minute parts lost
  their last minute), lets voice messages in between a queued file's parts, falls back to Whisper when it cannot
  load, and keeps the text before a part that failed. Two-track meetings come out in the order they were said.
- **Web UI**: a double click on Talk or the microphone no longer opens it twice (an echo and a microphone that
  stayed on); Compact, Stop, background chats in Talk and a chat's model after switching fixed.
- **A file attached in another chat is copied there**, not linked: a command run in one chat no longer changes the
  other chat's copy.
- `arc-server stop -Port N` stops only that port's server; `setup` downloads E4B's projector (images, audio); the
  launcher offers 1-7 drafts and chunks up to 8192.

## 2026-10-28: Gemma 4 E4B audio encoder checked against f32

- **E4B's GPU conformer matches an f32 reference** (new `tools/gemma4a_ref.cpp`, written from llama.cpp's
  `models/gemma4a.cpp` and `mtmd_audio_preprocessor_gemma4a`: log-mel in double, the two convolutions, 12 conformer
  blocks with every clamp, f32 activations, the mmproj's BF16 weights as f32). Rows: cosine 1.000000, relative error
  max 0.004 % (the 49.8-s test clip, 1,246 tokens; rec1 part_00 / part_03, 722 / 612 tokens). Rounding the weights
  to f16 as the GPU stores them changes nothing measurable. The model's KL, 48 forced reply tokens of the test clip:
  arcinferzero against the full f32 path (`ref_forward` REF_EMB on the reference rows) mean 0.00018, max 0.0054,
  top-1 48 / 48; the encoder alone (the GPU model on GPU rows vs on f32 rows) 0.00019 / 0.0047 (test clip), 0.00058
  / 0.0025 (part_00), 0.00068 / 0.0038 (part_03). Nothing to fix; E4B's noisy-clip mistakes are the model's.
- `vision_testz`: `ARC_AUDIO_EMB=file` runs the model on audio rows from a file (as `ARC_VIS_EMB` for images).

## 2026-10-28: Qwen3-ASR's attention with f16 K

- **f16 K rows in Qwen3's float attention** (Gemma 4 keeps f32): attention -10 %, generation +4 %, the two
  recordings' text 2-3 % sooner, the KV cache a third smaller (server -0.32 GiB); KL < 0.00001, the transcripts
  identical, Gemma 4 E4B / 12B and Qwen3.5 4B bit-identical. `ARC_Q3_KF32=1`: f32 K.

## 2026-10-28: Qwen3-ASR GEMV / GEMM; the speech code in its own files

- **Qwen3-ASR's q / k / v and gate / up each one matrix** (their Q8_0 rows stacked as they are): one token at 2048
  of context 23.8 -> 25.1 tok/s, the 2:34 recording's text 2 % sooner; logits bit-identical. Measured and left: rows
  per sub-group (within 1 %), the prompt GEMMs (11 % of a transcription, ~half the DP4a peak, shared by every
  model); the GEMVs run at 58-70 GB/s of ~75. The larger remaining parts of a token are the float attention (21 %:
  f32 K rows) and the 152K-word output head (12 %).
- **Speech code out of tools/server.cpp**: tools/server_speech.h (starting Whisper / Qwen3-ASR, Gemma 4 E4B for
  transcripts, `/v1/audio/*`), tools/server_whisper_asr.h (Qwen3-ASR / E4B inside the speech service); the Qwen3-ASR
  encoder's kernels in zero/kernels/asr.cl instead of ops.cl. Behaviour unchanged (every route, the queue, the
  encoder's error against f32, the transcript checked).

## 2026-10-28: Qwen3-ASR the default speech model; Gemma 4 E4B and Whisper large-v3 the other choices

- **Transcript models: Qwen3-ASR (default), Gemma 4 E4B, Whisper large-v3** (the user's choice after comparing two
  recordings, [docs/qwen3-asr.md](docs/qwen3-asr.md)): `--whisper auto` loads large-v3 (turbo only without it) and
  Qwen3-ASR with it (`--asr auto` is the default); the Transcripts tab picks the model per file, changeable any
  time; `/v1/audio/transcriptions` without a model (or OpenAI's `whisper-1`) and the chat's voice messages / audio
  go to Qwen3-ASR; `gemma4-e4b` transcribes through the chat model in 2-minute parts; Gemma 4 12B is refused for
  transcription (it garbled a clear recording and summarized a noisy one). Live text while speaking stays Whisper's.
- **Fixed**: a transcription request naming an audio chat model other than the loaded one sent its clip to the
  speech service instead of that model (now `"speech": "model"` in its chat request).

- **Qwen3-ASR 1.7B** ([docs/qwen3-asr.md](docs/qwen3-asr.md)): `serverz --whisper auto --asr auto`, then
  `/v1/audio/transcriptions` with `model=qwen3-asr` or "Model: Qwen3-ASR" in the Transcripts tab (a live
  recording's pass after Stop; its text while spoken stays Whisper's). On the user's 2:34 recording of noisy mixed
  Indonesian and English: every utterance, English kept, punctuation, 3 invented words over the last 40 s of
  background sound (Whisper turbo lost two sentences and wrote Japanese there, large-v3 Thai, Gemma 4 E4B a
  paragraph); 23-25 s for the 2:34. No timestamps (one segment per part of up to 6 minutes); the model calls this
  Indonesian "Malay" (reported as `id` when the request's languages have `id`).
- **Qwen3 dense on the engine** (GGUF arch `qwen3` / `qwen3vl`): float attention and two int8 passes, KL < 0.00001
  against the f32 reference; the int8 attention of the Qwen3.5 path gave 0.236 on it. `tools/ref_forward` takes
  these models; `ARC_Q3_Q2=0`: one pass (KL 0.0064 / 0.0093, prompt +50 %).
- **The qwen3a audio encoder on the GPU** (zero/audio_qwen3a.h; f32 CPU reference tools/ref_qwen3a.cpp: 0.08 %);
  zero/asr_testz.cpp runs and checks it from the command line.
- **1.37 GiB less GPU memory for Qwen3-ASR**: its weights' GEMM form decoded just in time per 512-token chunk
  instead of kept (3.72 GiB with the encoder; the server with Whisper 5.31 instead of 6.68). The same transcript and
  KL, the 2:34 recording 23.2 s either way. The server logs each Qwen3-ASR transcription's times.

## 2026-10-28: a live recording's language preferred, not forced; recordings kept as WAV; a cap for audio to a chat model

- **English in an Indonesian live recording stays English** (src/whisper.h `prefer_language`, tools/server_whisper.h
  `live_lock`): the language a recording locked to was forced on every utterance, and Whisper forced to one language
  translates the others: "How come you translate my words to Indonesian?" came out in Indonesian. It is preferred
  now: an utterance detected in another language with less than 0.7 is decoded in it (noise, short words), one
  clearly in another keeps its own. Checked with the user's 2:34 recording through the live API: its English
  utterances in English in the draft.
- **Recordings kept as WAV** (Transcripts tab: "Keep recordings as WAV / MP3", WAV by default; `mp3_kbps: 0`):
  lossless for a later pass by another model (~5.5 MB a minute a track; MP3 192 kbps ~1.4 MB). Checked: the
  recording's ext stays wav, its header 153.0 s, ranges read back whole.
- **Audio to a chat model** (`/v1/audio/transcriptions` with a Gemma model): at most 8 tokens a second of audio
  (+64) and repeat_penalty 1.1 unless given: greedy, Gemma 4 E4B fell into a loop on a 29-s clip and wrote 16,687
  tokens in 489 s. Compared on that recording (mixed Indonesian and English, a noisy microphone): Whisper turbo 11
  s, large-v3 40 s (no better: repeats, Thai over the closing silence), Gemma 4 E4B 26 s (the most readable, English
  kept, but a whole made-up paragraph over a nearly silent piece), Gemma 4 12B: its audio does not reach the model
  ("the audio is not provided"), to be looked into.

## 2026-10-28: the model's tools audited; create_document is create_file

- **create_file** replaces create_document (it writes any text file now, not only documents): the tool, its texts,
  `/v1/tools/create_file` and the minutes' DOCX / PDF. The old name is gone, from the server too (one tool, one
  name); older chats still show their create_document cards and files.
- **search_chats** replaces list_chats and search_chats (the same job, two tools in every prompt): no query lists
  the other chats newest first, a query finds their exchanges (BM25); `limit` for both. `/v1/tools/list_chats` is
  gone. The six file tools are five.
- **Talk mode**: every tool runs in the background there, but only the web tools' calls were explained, and only
  with Web on: with Files alone a model did not know its read_file results come later, and could say a file was
  written before create_file ran. The text now covers every tool whenever tools are on; a result's label names its
  file ("[Results of #3: read_file laporan.pdf]", before "read_file " alone).
- **The 8-round limit**: calls in the 8th round were dropped without a word, the answer often empty. Now they get
  a "not run" result and their card says so, and the model answers once more without tools from what it has.
- **read_file of a file the model wrote** returns its source within the model's read budget, in parts by offset
  (a 60 KB page overflowed a small context). **web_fetch** keeps to the same budget (12,000 characters a fetch
  whatever the context before).
- **search_file** with nothing to search (an image or audio file, a chat without documents) says so, instead of
  "no passage has these words: try other words", which sent the model off retrying.
- create_file: an empty filename takes the title (it made `file.docx`); md with a title and content that starts
  with its own `# ` heading no longer has the heading twice.
- Earlier exchanges' long call arguments (a file's whole content, a long script) are shortened in what is sent, as
  long results already were; search_chats results too. The card labels: "PDF" / ".html" formats, search_file
  "searching…", read_file without file_id.

## 2026-10-28: a pipe inside a table cell

- **`\|` and a `|` inside a code span stay in their table cell** (webui/index.html `mdCells`, tools/doc_writer.h
  `cells`): "`` `ls | sort` ``" or "a \| b" split the cell in two and moved the row's later cells one column right,
  in the chat and in the docx / pdf / rtf / xlsx files create_document writes. A row with an odd number of backticks
  has no code spans (one would take the rest of the row). Checked: the chat's table in the browser; a docx and an
  xlsx read back cell by cell.

## 2026-10-28: create_document writes any text file

- **The model can save HTML, SVG, code and other text files** (tools/server.cpp `create_document`, webui/index.html
  `FILE_TOOLS`): before, the format had to be docx, pdf, doc, rtf, md or xlsx, so a page the model wrote stayed a code
  block in its answer (Open / Download only) and never became a saved file. Now every other text extension (html,
  svg, css, js, py, json, csv, txt, xml...) is saved exactly as written; no format: the file name's extension.
  Binary formats (images, audio, video, pptx, zip, exe...) are refused with a note. The tool's text says a code
  block is not a file. Source and markup files are served as text/plain with nosniff (never run as this origin);
  an .html / .svg card opens sandboxed (`openPage`).

## 2026-10-28: the prompt cache kept with the terminal; terminal_panel; HTML pages open sandboxed

- **Opening or closing the terminal no longer recomputes the whole chat** (webui/index.html `TERM_SYSTEM`): the
  terminal's system text changed with its state (open / closed, the shell note added once it opened), and the
  system text comes before the tools in the prompt, so each change made the server compute the tools and the whole
  conversation again (the log: "cached 0", 20 s for 2,400 tokens on the 27B instead of 2.7 s). It is one text now;
  the shell's version goes in the session's first result. Checked: the system text the same open and closed.
- **terminal_panel** (open / hide / exit) for the model: asked only to open the terminal, a model made up a command
  to open it with (Get-Location; hostname; python --version) that waited for Run. Checked with the 27B: "open
  terminal" and "buka terminal dong" -> open, "Python versi berapa" -> run_command, "venv vs conda" -> no tool. The
  power button is called Exit now (it was Close, and the model took "tutup" for exit).
- The system text names the server (a model's thinking read "arcinferzero" in its paths as "Arc Infra Zero").
- **HTML and SVG open in a new tab, sandboxed** (`openPage`): a code block's Open (next to Copy and Download) and a
  saved .html / .svg chip. An iframe sandboxed without allow-same-origin and a CSP of https only: checked from inside
  the page, its origin is null, fetch to /props, POST to run_command (also no-cors), an <img> of 127.0.0.1, the web
  UI's localStorage and window are all blocked; its scripts run, an https image loads. The server still sends .html
  as text/plain (a blob: URL or text/html would have run as the web UI's origin, with its API).

## 2026-10-28: the model opens the chat's terminal itself; shorter tool texts

- **The model opens a closed terminal itself** (webui/index.html `runTermTool`, `termOffered`): before, `run_command`
  was there only while the user had opened the chat's terminal, so a model that had to make a `.venv` could not.
  Now it is offered in every chat unless Settings > Terminal is "Model never"; a call in a closed terminal waits for
  Run / Don't run, and the terminal opens once allowed. The first command of every terminal session asks, with
  "Run without asking" too (like a coding agent's approval), so nothing runs on the computer unasked.
- **The tools' texts are shorter, each rule and parameter kept**: what the system text and a tool's description both
  said is said once, the wording is tighter. Measured with the server's tokenizer: all tools 2,320 -> 1,884 tokens
  (-19 %), Files alone with the time 1,732 -> 1,421. The rest is mostly the parameters' JSON (create_document: 20).

## 2026-10-28: Whisper over silence, a live recording's language, replay after the MP3, the Transcripts tab

- **No more text over silence and noise** (src/whisper.h `drop_unvoiced`): a live recording in Indonesian came out
  "Продолжение следует...", "Спасибо.", "Obrigada.", "MBC 뉴스", "Thank you." in its quiet stretches. large-v3-turbo
  writes these with a no-speech probability near 0 (the f32 reference too), so the no-speech check never skipped
  them. Segments with almost no voice under them are left out now, Whisper's stock credit lines and thanks in
  another language when short; a spoken "terima kasih" stays. 12 s of silence: no segment (before: one, Russian with
  auto, "Terima kasih." with id). docs/whisper-native-status.md; tests/test_audio.cpp.
- **A live recording with language auto keeps one language** (tools/server_whisper.h `live_lock`): each utterance
  was detected alone, and quiet ones came out Russian, Spanish, Portuguese; once two clear utterances are in one
  language, the rest is decoded in it ("id,en" for mixed speech is left as it is).
- **Replay stopped mid-way after a recording ended** (~0:55 of 2 minutes): once the recording was kept as MP3, the
  WAV was deleted and the same media URL gave the player, reading the WAV in ranges, the MP3's bytes. The page asks
  for the file by its type (`/media?f=wav`), the server keeps the WAV 30 minutes for it, and the player takes the MP3
  when it is not playing, at the same time.
- Stop in the Live panel now hides the header's Stop at once (it stayed while the last audio was sent, the server
  still saying "live"); the glossary box has 3 lines and grows with its text.

## 2026-10-28: `arc-server restart` keeps -Npu and -Whisper

- `arc-server restart` (e.g. `restart -Model 27b -Whisper auto`) kept the slots but turned the NPU and Whisper off
  unless they were given again: the 27B lost `--npu on` when Whisper was added. Both are now kept like the slots;
  `-Npu off` / `-Whisper off` turn them off, a start after `stop` still begins with both off (scripts/arc-server.ps1).
- A kept `-WhisperDevice npu` read as "npu|think" (the state's options end in "|think N"), so a switch to another
  model moved Whisper's encoder back to the iGPU; the option is now read up to the "|".

## 2026-10-28: Transcripts and Meetings in the web UI

- **A self audit of all of it, and its fixes** (docs/self-audit-2026-10-28.md):
  among them a 32-byte MP4 that hung the server, an open in the Transcripts tab that was slow or failed (Delete
  then deleted another transcript), recordings past 4 GB that lost their end, a recorder killed or a power cut
  (the recording is now picked up at the server's next start), a headset plugged in mid-meeting (now followed),
  a live recording whose page closed (stopped and transcribed when the page has sent nothing for 5 minutes), subtitle cues of
  two speakers that ended before they started. Recordings up to 6 hours are transcribed (180 minutes before). A
  3-hour unattended recording passed, and showed that recordings' WAV header overwrote their first samples (fixed).
- **A chat reads your other chats and recordings**: type `@` to attach an earlier chat or a recording / meeting
  next to the files, or just ask ("what budget did we settle on for Bali?"): the model searches your other chats
  (list_chats / search_chats) and reads the part it needs. Files now tell the model an image's size and a
  recording's length.
- **Meetings recorded apart from the browser**: `arc-server record -App Zoom.exe` records your microphone and
  the Zoom (or Teams, or any app's) sound as two tracks until `arc-server record-stop`; it goes on if the server
  stops, and the Meetings tab transcribes it as Me / Others when it ends.
- **Every model hears**: an audio file or voice note attached in a chat is written by Whisper, so Qwen and the
  others answer about it (a 10-minute voice note: 39 s the first time, 4 s for each next question). Settings >
  Voice > **Speech** chooses, for Gemma 4 E4B / 12B, whether Whisper writes it or the model hears it itself.
- **Transcripts tab**: drop audio or video files; each is transcribed in the queue with its progress; the
  transcript plays with its media, the word being said lit; click a word to play from it, double-click a line to
  correct it, find and replace a term everywhere, and export SRT, VTT, YouTube, ASS, TTML, text, Markdown, CSV,
  Remotion or JSON.
- **Live**: record the microphone or a tab's / the screen's sound and read the text about 2 s behind; when you
  stop, the whole recording is transcribed again accurately and kept as MP3 (192 kbps).
- **Meetings tab**: "Start a meeting" records your microphone as "Me" and the meeting (a browser tab, or the
  screen's sound) as "Others"; afterwards "Summarize" (the loaded chat model: summary, decisions, tasks, open
  questions with their minute, in the meeting's language) and the minutes as DOCX or PDF.

## 2026-10-28: Whisper takes video files and keeps transcripts

- **Video and voice notes**: the audio of MP4, MOV, MKV, WebM, AVI and WMV files, and Ogg Opus voice notes
  (WhatsApp, Telegram), is transcribed; the times follow the video even when its audio starts late. An MKV with
  16-kHz AAC used to come out as garbage. Long files are read as they go instead of into memory first.
- **Saved transcripts** on the server (`/v1/transcripts`, for the Transcript tab): upload files of any size, a queue
  with progress, cancel, delete, export in every format, edits saved; a server restart continues the queue.
- **Subtitles**: no line longer than 42 characters any more (the second line could be), and two lines evened out;
  WebVTT word timing as the standard asks. Checked in Remotion's caption library and Chrome's subtitle parser.

## 2026-10-28: Whisper for mixed languages and jargon

- **Indonesian and English in one recording**: `--lang id,en` (server `language=id,en`) checks each utterance's
  language and transcribes it in that language, instead of translating the other language's sentences or leaving
  them out. On a test file alternating Indonesian speech and English sentences: 18.5 % / 6.9 % of the words wrong
  (was 45.7 % / 43.5 %; faster-whisper 61.6 % / 19.1 %). About 3x slower than one language (14 minutes in 72 s).
- **A glossary**: `--terms "Qwen, GGUF, Level Zero, ..."` (server `terms`) before every window: the 14-minute
  channel recording's hard terms 14 -> 21 of 24. Tip: the terms inside a casual sentence keep the casual spelling.
- **The previous text is no longer the next window's prompt by default** (`--previous-text` turns it on): with it
  Whisper dropped whole windows (one became "Terima kasih.") and carried errors over.

## 2026-10-28: Whisper faster

- **A 14-minute recording transcribed in 25 s** (greedy, 33.8x realtime; 30.8 s before) and 32 s with beam 5
  (38.2 s): 2.27x OpenVINO's speed on the same iGPU. The decoder reads its weights through the engine's GEMVs, small
  kernels are fused, the cross-attention computes only the rows a pass has, and the word timestamps' host work runs
  on every core. Same accuracy (decoder logits KL 0.0004 against the f32 reference). Whisper takes 0.16 GiB more
  GPU memory (1.55 GiB).

## 2026-10-27: failure paths cleaned up (a static audit by DeepSeek, each finding checked)

- **A model or Whisper that fails to load frees what it already put on the GPU**: a constructor that throws runs
  no destructor, so a failed model switch in the server kept the half-loaded engine's buffers, kernels and programs
  until the process ended (Engine, WhisperEngine); Whisper's `model.bin` is closed when it is malformed.
- **No more process abort from a background save**: the threads writing saved conversations and system-prompt
  states are joined by their owner however it lets go (a `std::thread` still joinable when destroyed ends the process).
- **Request bodies: 1 GiB for all connections at once** (64 connections x 256 MiB could ask for 16 GiB); past it,
  503. The per-request limit stays 256 MiB.
- **Gemma 4 image and audio projectors of another model size are refused** at load, as the Qwen one already was
  (their embeddings would have been read at the wrong width).
- GGUF tensors must hold whole blocks (also the virtual expert tensors, whose unknown type divided by zero); the
  base64 table is built thread-safely; WIC and Media Foundation objects, the terminal tool's pipe / job / process
  and a compiled program are released when an allocation throws; the expert cache's saved state is written whole
  or not at all, never from a throwing destructor; NPU memory import and `OpenCL.dll` cleaned up on failure.
- Checked and not changed: the encoders' GPU buffers when a projector is refused (they load lazily: nothing is
  allocated yet); tokenizing before the context check (memory now bounded by the body budget).
- **Web UI** (DeepSeek's second audit, each finding tried in the browser):
  - Talk while every slot was busy could delete an older message and everything after it (the next utterance
    "joined" a message that was not its own); background web results could overwrite an older message's bubble.
    Both now act only on a message they really posted, and the results wait until a slot is free.
  - Stopping Talk clears its utterance state; something said just before stopping lands in the message box instead
    of being sent.
  - Opening chat A and then chat B quickly could show A (its slower load won); a file upload going on while you
    switch chats stays in the chat it was dropped on.
  - The microphone is released when the browser cannot build its audio graph; closing the terminal ends the command
    still running; deleting an answer not in the history no longer throws; voice conversion errors are shown; pages
    in browsers without speech output keep working.
- **DeepSeek's re-audit of the fixes above** (each finding checked):
  - A model or Whisper failing before the GPU runtime existed (no GPU / driver) gives its error again instead of
    crashing: the cleanup added above waited on a command list that was not there yet.
  - The DFlash2 drafter's file is closed when the model is unloaded, and a damaged one is refused (header length,
    tensor ranges, shapes, codebook sizes checked against the file and the model).
  - Saved-conversation and system-prompt files are closed when loading or saving throws; Whisper decodes an upload
    in its turn (four waiting 3-hour uploads were decoded at once, ~2 GB each); requests with both
    `Transfer-Encoding` and `Content-Length`, or two different lengths, get 400 (request smuggling behind a proxy).
  - Web UI: Compact by hand no longer writes its summary into another chat opened meanwhile (on screen and on
    disk); deleting a file that is attached to the message being written takes it off the message; Talk stopped
    and started again does not send what was said before; a voice message is not sent into a chat opened while it
    was written down, and its file stays in its own chat.
  - Smaller: `OpenCL.dll` let go on every failed start; a model folder that hangs is checked by one thread, not
    one more per model list; the terminal's allocations before its handles; JSON mode's limit on leading
    whitespace; the sampler refuses empty logits.
  - Not changed: tokenizing before the context check (as above).
- **Third round** (each finding checked):
  - A prompt longer than the context can possibly hold (more than context x the longest token's bytes) is refused
    before tokenizing: 30 MB in 0.8 s, with the usual context-overflow error ("at least N tokens"); every prompt that
    might fit is tokenized and checked exactly as before.
  - `/v1/completions` with `n` > 1 returns n completions (the API adapters skipped the n > 1 handling: one came back).
  - Closing the terminal stops the model's running command too (the answer goes on, told the terminal was closed).
  - Web UI: Talk stopped and started again leaves no clip of the old session attached (without Whisper); a file the
    server could not delete stays attached and the failure is shown.
  - Smaller: a half-written cache file is removed when saving throws; DFlash2 dimensions past INT_MAX refused; COM
    released when a server thread that decoded an image ends.
  - Not changed: background web results are not capped (they pile up only while you never pause; capping them
    would drop results the model was promised).
- **Fourth round** (each finding checked): a model's terminal command that is waiting for Run is cancelled when
  the terminal is closed (Run no longer starts it in a reset session), and one whose answer was stopped just
  before it began does not run or ask; a half-written `chat.json.tmp` is removed when saving fails; a model folder
  check that cannot get a thread skips the folder instead of failing the model list. Not changed: re-checking a
  folder once its hanging check has answered (its state may have changed: a drive plugged back in).

## 2026-10-27: web UI audit: every function tried, security and robustness fixes

- **Tried with Qwen3.5 4B (vision, Whisper) in the web UI**, each passing:
  - a voice message through Whisper; DOCX, XLSX (two sheets) and PPTX attachments (speaker notes, the hidden slide);
  - a scanned PDF transcribed by the vision encoder; an XLSX the model wrote, opened in Excel (2 sheets, numbers as numbers);
  - the terminal panel (folder and environment carried, PowerShell variables not, as designed), the model's commands
    with Run / Don't run, and no terminal tool for the model while the panel is closed;
  - web search, the Files panel (@, delete unused keeping an unsent attachment), deleting a message with its file, edit;
  - two chats answering at once after switching to 2 slots in Settings, reopening a running chat, Stop all, Compact.
- **Fixed**:
  - **DNS rebinding through a name that starts like an address** (security): "127.0.0.1.evil.com" passed the Host check
    (it only looked at the "127." start), so a page of that site could run terminal commands and read their output.
    The Host must now be an address literal (or localhost); a server opened to the network (--host) also refuses
    names (open it by its IP). src/http.h; tests/test_http.cpp.
  - **Voice messages and Talk never used Whisper**: the page read /props "speech_to_text", the server sends "whisper".
  - **Run / Don't run took the keyboard focus**: a key typed for the message box could approve a command. A chat whose
    answer waits for that choice in the background now says "needs your OK" in the chat list.
  - **Stop and Esc end a running command** (it ran on to its timeout, up to 10 minutes): the server polls the client and
    ends the command with every process in its job; Esc in the terminal panel stops the command typed there. A command
    whose process left its job no longer blocks the reply (the pipe read ends 2 s after the command).
  - **Command output within the model's context** (the share a read_file part gets), its start and end kept.
  - The terminal session's paths (cwd, VIRTUAL_ENV, PATH) follow the chat folder when it is renamed to its title.
  - A file put into a chat folder (e.g. a download from the terminal) is hashed in parts, not read whole into RAM;
    files over 512 MB are not read as documents.
  - A chat saved in the middle of a tool round gets placeholder results for its unfinished calls (a valid history after
    a closed page); opening a chat waits for a save on its way.
  - XLSX: elapsed times ([h]:mm) as 30:00, not a 1900 date; phone and account numbers (+62..., 12+ digits) stay text;
    slash dates month first in a US table. PPTX: mc:AlternateContent read once. Office files: at most 512 MB unpacked.
  - The terminal tool's description says which programs can outlive a command (COM objects, WMI, scheduled tasks).
  - **Answers render Markdown fully** (webui/index.html `mdBlocks`): tables (column alignment, a header row, scrolling
    sideways in the bubble on a phone; before, they showed as "| a | b |" text), numbered and nested lists, task boxes,
    quotes, rules, links (http(s) and mailto only, in a new tab) and bare URLs, strike; empty lines make one gap.
    Text is escaped first, quotes too (a URL could otherwise leave its href); every prefix of a streamed answer
    renders, a 2,000-row table in 7 ms.
- Not changed, by design: a program on this computer can call run_command without the page's Run (it has the user's
  rights already); the approval guards against the model. Not tried live: a request from another computer to a
  server opened with --host (a firewall prompt), Talk with a real microphone.

## 2026-10-27: scanned PDFs read by the vision model

- **Scanned PDF pages (images without a text layer) are read in the web UI**: `read_file` with `pages` on such a page
  has the loaded vision model (Qwen 3.5 4B / 9B, Gemma 4 E4B / 12B / 26B) transcribe it from its image, up to 4
  pages a call; the text is kept per file and page (the chat folder's `.pages\<id>.json`), so later reads and
  `search_file` find it without the vision model. A model without an image encoder names the scanned pages and says
  a vision model is needed. The server transcribes (a chat request of its own) instead of the web UI attaching the
  page images to the model's next turn: the text can be kept and searched, a page costs its text in the
  conversation rather than ~1,000 image tokens on every later turn, and no chat template has to take images in
  tool results. The cost: with one slot the chat's prompt is processed again at its next turn.
- `tools/pdf_images.h`: the image a page draws (its content stream and form XObjects), out of the PDF on the CPU
  without a library: DCTDecode as the JPEG it is, FlateDecode or unfiltered samples (gray, RGB, CMYK, ICCBased,
  Indexed, image masks, 1-16 bits, PNG predictors) as a BMP, CCITTFaxDecode (Group 4, Group 3) behind a minimal
  TIFF header, all decoded by WIC; JBIG2 and JPEG 2000 a note. Sizes capped as uploads (16384 pixels a side, 64
  megapixels), raw samples only as far as they decoded. `tools/doc_reader.h`: PNG predictors for more than one byte
  a pixel, decode parameters per filter.
- Checks: `tests/test_pdf_scan.cpp` (model-free; in `tests/run_tests.ps1`, CI and setup): PDFs built from known
  images give the same pixels back (raw, CCITT G4 made by WIC) or the same JPEG bytes; scanned pages found, a page
  with text is not one; transcriptions merged and searched; huge sizes, broken streams, a form that draws itself and
  300 randomly damaged PDFs. `test_pdf_scan FILE.pdf` lists any PDF's scanned pages and their images.

## 2026-10-27: batched decoding for Flash Next

- **Flash Next generates for several conversations in one pass** (`--slots 2` or more, without a drafter): the
  PLE n-gram state and history and the QSA blocks run per conversation, everything else once. Bit-identical to each
  conversation alone. With the experts cached 2 slots give 1.43x the tokens per second together; in real chats on
  this laptop only ~1.05x, as each conversation brings its own expert reads from the SSD.
- Batched steps and verification passes of MoE models run their experts on smaller tiles: the 35B 2 / 4 slots
  1.33x -> 1.44x / 1.99x -> 2.19x together, bit-identical.
- `runz --bench-batch N --prof-gen`: the GPU time per kernel of a step alone and batched.
- Prompt MoE on the 35B 2 % faster (an emulated integer multiply in the expert kernel done in float, same results);
  found by reading the kernels' GPU assembly with ocloc (docs/kernel-audit-2026-10-27.md).
- Prompt attention of the 27B 9 % faster (two independent sums per head in the score loop, same results).
- The small kernels that quantize activations between the matrix products read their inputs as vectors: about 1 ms
  less per generated token on the 35B (~2 %), the 27B's prompt normalization a third faster; same results.

## 2026-10-27: Qwen vision encoder faster on the dense models

- **Images 22-30 % faster to encode on Qwen3.5 4B / 9B and Qwen3.8 27B**: the vision encoder's matrix products run on
  the engine's int8 GEMM in two passes, as the Gemma 4 encoder's (27B: a 768-token image 1.76 -> 1.27 s, a photo
  571 -> 450 ms); the model's KL against the f16 encoder 0.0004-0.0024 on 5 test images. The 35B and Flash Next keep
  the f16 encoder (their models are too sensitive to it); `ARC_VIT_Q2=0 / 1` turns it off / on for any model.
- Measured, not changed: the NPU next to the GPU does not pay off on the 35B (-3 to -8 %) or Flash Next (noise),
  whatever share it gets; `--npu on` keeps them off.

## 2026-10-27: the chat's terminal in a right panel, a venv that stays

- **Terminal panel** (webui/index.html, the Terminal button above the chat): the chat's PowerShell on the right of the
  chat, as an editor's terminal pane. You type commands there; the model's `run_command` calls show in the same log
  (marked AI). It belongs to the chat (chat.json `terminal`: open or not, the log, folder, venv) and is there again
  when the chat is opened. **The model gets `run_command` only while the chat's terminal is open**; Close ends it
  (the model loses it, the next one starts fresh in the chat's folder), × only hides the panel. Settings > Terminal is
  now the model's access to an open terminal: Ask each time (the default), Run without asking, Model never.
- **The model knows which shell it writes for**: when the terminal opens, it is asked once for PowerShell's version
  and edition and Windows' name, version and build (kept with the chat), and the system prompt says it ("Windows
  PowerShell 5.1.26100 (Desktop) on Microsoft Windows 11 Home Single Language 25H2 (build 26200), AMD64") with the
  syntax that differs: PowerShell, not bash or cmd; curl.exe / where.exe; and for 5.1 no `&&` / `||`, no ternary or
  `??`, Set-Content's encoding.
- **Folder and environment kept between commands** (tools/server_terminal.h, `run_command` with `session: true`):
  each command still runs in its own PowerShell, and the variables that differ from a fresh one plus the folder are
  written to the chat folder's `.terminal.json` after it and set again before the next. `cd`, `$env:X`, an activated
  `.venv` (and its `deactivate`) stay; exit codes are as before. `reset: true` ends the session. Checked:
  test_chat_store (folder, variable set and removed, exit codes, PATH not doubled, a venv and deactivate) and a real
  `python -m venv`: the next command's `python` and `pip` were the venv's, after `deactivate` the system Python.
- **Thinking card follows the newest text** while it streams (unless scrolled up to read), and opens at its end.

## 2026-10-27: web UI sends earlier tool results shortened

- **Long tool results of earlier exchanges go to the server as a short stub** (webui/index.html `trimTools`): a
  `read_file` part, a `web_fetch` page, `search_file` passages or `run_command` output (1,500+ characters) from
  before the previous user message becomes one line with what was read and how to read it again (file id and name,
  section and title, pages, size, next_offset; the query and the passages' sections; the URL), e.g. `[read earlier:
  read_file doc1 (kontrak.docx), section 3 "PASAL 3" pages 3, 12,000 characters of 20,000; shortened here: read it
  again with read_file if needed]`. Whole: the last exchange, every round of the answer running now, and the last 2
  results before them; `list_files` and `create_document` results always. Only in the request: the saved chat and
  the screen keep every result. The cut moves only when the user sends a message, so the server's prompt cache holds
  through the rounds of an answer. A few reads of a long document no longer fill the context, and compaction comes
  later (the context estimate counts the shortened form). Compaction's summary request gets the same shortened
  messages, so it reads less and starts as the last request did (the server's cache still fits). Checked in the browser with a scripted fake server: three
  12,000-character reads, a search and a page over four questions; the fourth request had the two oldest reads as
  stubs, the requests within one answer kept the same start, the summary request matched the last request's start,
  and the saved history stayed whole.

## 2026-10-27: Excel spreadsheets in the web UI (read and written), PowerPoint read

- **Uploaded .xlsx / .xlsm are read** (tools/xlsx.h, no library; `read_file` / `search_file` and attachments): each
  sheet is a heading "Sheet: <name>" and each row a line "A | B | C"; with a header row every line repeats the
  column names ("Nama: Budi | Nilai: 80"), so a passage `search_file` finds still says what each value is. Shared and
  inline strings (rich text joined), numbers, booleans, errors, formula cells as their cached value, dates and
  times from the cell's number format (built-in and custom) as yyyy-mm-dd / hh:mm, percentages, a currency prefix
  ("Rp 1234000"), merged cells (the value in the first cell), hidden sheets marked; empty rows and columns left out,
  at most 200,000 cells (a note says so). A binary .xls gets a note (save it as XLSX). **CSV / TSV** now read the same
  way (delimiter , ; or tab, quotes, line breaks inside quotes) instead of as plain text.
- **`create_document` format "xlsx"**: from the same Markdown, each table becomes a sheet named by the heading before
  it (31 characters, no `[ ] : * ? / \`, a repeated name gets "(2)"); the text around the tables goes to a first
  "Notes" sheet (with a line where each table was), and the workbook opens on the first table. Numbers become real
  numbers with a number format ("1.234,5", "1,234.5", "12.500", "Rp 1.234.000", "$12.50", "15 %", "12,5%"; Indonesian
  or US style per table: "1.500" / "1,500" alone cannot tell, so each table goes by its numbers that do, "1.234,5" or
  "2,5" against "1,234.5" or "3.14", else the whole document's, else Indonesian; `number_format: "id" | "us"` fixes
  it), dates
  real dates ("2026-10-27", "27/10/2026", with a time); a leading zero ("0812...") or a text starting with "=" stays
  text. The header row is bold on a fill, frozen, with an autofilter; column widths from the content. Checked with
  Excel itself (COM): the file opens without repair (a copy with one element out of order does not open), values,
  number formats, frozen pane and filter as written, and Excel's own re-save reads back the same.
- **Uploaded PowerPoint .pptx / .pptm / .ppsx / .potx are read** (tools/pptx.h): each slide is a heading "Slide N:
  <title>" on page N (the outline lists the slides, `read_file` pages are slides), then its text in the slide's
  order: bullet points with their level, tables as rows ("Column: value" with a header row), SmartArt text, a
  picture's alt text as "[image: ...]", the speaker notes as "Notes: ..."; footer, date and slide number left out,
  hidden slides marked. A binary .ppt gets a note (save it as PPTX). Not read: text inside charts and embedded objects.
- Checks: `tests/test_office.cpp` (a workbook made in Excel, `tests/data/xlsx_sample.xlsx`, and a deck made in
  PowerPoint, `tests/data/pptx_sample.pptx`, metadata scrubbed: dates, percent, currency, formulas, an error, rich
  text, merged cells; titles, bullet levels, a table, notes, alt text, a hidden slide; Markdown -> xlsx -> text; the
  element order; hostile input: huge references, bad shared string indexes, 200,000-deep nesting, 20,000 merges over
  the whole sheet, 260,000 cells, 100,000 slide ids, a part whose notes point at itself; CSV), in
  `tests/run_tests.ps1`, CI and setup. It runs in 0.5 s.

## 2026-10-26: Whisper native (speech to text on the iGPU or the NPU)

- **`arc-transcribe` / `bin\transcribez.exe`**: Whisper large-v3-turbo (faster-whisper's `model.bin` from the
  Hugging Face cache) on Level Zero: files, folders or wildcards to json, fw / captions (Remotion), srt, vtt, sbv,
  srv3, ass, ttml, txt, md, csv, tsv in `%USERPROFILE%\arcinferzero\transcripts`; language auto per window, fixed or
  a list (`id,en`); greedy or beam 5; word timestamps; a progress bar or `--progress` JSON lines. `--device npu`: the
  encoder on the NPU, the decoder on the iGPU. 14-minute file: iGPU greedy 31.8 s (26.7x realtime, 1.8x OpenVINO's
  iGPU), beam 5 39.3 s; NPU mode 81 s (OpenVINO NPU 144 s); decoder logits KL 0.00039 (iGPU) / 0.00052 (NPU) against
  the f32 reference (`tools\ref_whisper`).
- **Server**: `serverz --whisper auto [--whisper-device npu]` (`arc-server -Whisper auto`) answers
  `/v1/audio/transcriptions` and `/translations` as OpenAI's API (json, text, srt, vtt, verbose_json with words,
  `stream=true`) plus the formats above; its GPU memory comes off `--ctx auto`'s context; it takes turns with the
  model on the GPU; limits: 180 minutes, one at a time and 4 waiting. Details: [docs/whisper-native.md](docs/whisper-native.md).
- **MP3 / M4A / video audio 13x faster to decode**: the resampler to 16 kHz computes its filter weights once per
  phase (a 14-minute 48-kHz MP3: 21.4 s -> 1.6 s; bit-identical samples). Checked: MP3 and M4A through
  `arc-transcribe` and `/v1/audio/transcriptions`.

## 2026-10-27: PDF outlines from bookmarks, fewer false headings

- **PDF bookmarks are the outline** when a PDF has them (tools/doc_reader.h: /Outlines, /Dest or a GoTo action,
  explicit or named destinations, PDFDocEncoding or UTF-16 titles): each marks the first paragraph on its page that
  starts with its title (the rest split off as body text); a title not in the page's text (PowerPoint's "Slide 5")
  marks the page's top heading by font instead, or is put in when the page has none; font headings before the first
  bookmark's page stay (bookmarks for only a part of the document).
- **Fewer false headings by font**: list items (a bullet, "a.", "1)") and lines ending in ':' ',' ';', table of
  contents lines (dot leaders), numbers ("39,84%") and sentences are never headings; a bold line counts only when it
  stands apart (a gap before it larger than a line's), not as a table's cell (the next cell a little up the page) and
  not when the same heading comes three times or more ("Done", a table's labels); a larger font's line of 16 words or
  more is a sentence. Numbered lines in capitals ("II. PIHAK KEDUA") and short numbered titles in a larger font
  ("1. Peserta") stay headings. On real documents: a 215-page memo 817 -> 108 headings (its sections and risks
  R1-R10, not its list items), a 70-page slide deck 523 -> 0 level-4+ headings (one slide title per slide; 138 by font
  alone), a 18-page agreement keeps all its PASAL headings.
- **Heading levels with many font sizes**: with more than five sizes above the body text's, the levels go to the five
  on the most pages in short lines, the others take the level of the nearest of them below. A 142-page requirements
  document with a cover and slides attached: its chapters, sections and subsections (I, II.1, II.3.1, IV.1.1.1) at
  four levels instead of all at one (the cover's and the slides' sizes had taken the five levels).
- **Long documents within a small context**: a long section without headings over several pages (a five-page risk
  table, a chapter without subheadings) shows in the outline with its pages as parts one level below it (about 3,600+
  characters each), so the model reads it a piece at a time: a 215-page memo's largest section 34,900 -> 7,300
  characters. One `read_file` gives at most a fifth of the loaded model's context (3 characters a token; 4,900
  characters at 8K, the old 12,000 from 20K up) whatever `max_chars` asks, cut at a paragraph's end, the rest by
  `next_offset` with a note; `search_file`'s passages together stay within it too (the best one always). A document
  in a message goes in whole only when it takes at most a quarter of the context (else its outline, as above
  24,000 characters before). The context is read without waiting during a model switch (then the old limits).
- Checks: `tests/test_doc_read.cpp` with a Word-made bookmarked fixture (`tests/data/doc_bookmarks.pdf`: headings
  1-3, bold list items; also read with its bookmarks taken out), a hand-made PDF (a named destination, a GoTo action,
  a UTF-16 title, a title not in the text), PDF text strings and a long section's page parts;
  `tests/test_chat_store.cpp`: a document in a message whole or as its outline by the context, the read budget.

## 2026-10-27: web UI answers in the background, conversations at once from Settings

- **An answer goes on when another chat is opened** (webui/index.html: one job per chat): its chat in the list shows
  "answering…", opening it again shows the answer as it runs (with Stop), and it is saved when it ends; the question
  is saved at once and each tool round too, so a closed page keeps them. At most as many answers run as the server
  has slots: with every slot answering, New chat is off and a message in another chat stays in its box (no queue)
  until one ends or is stopped. Failsafes: a bar over the chat list
  ("2 chats answering, 1 waiting", **Stop all**), a watchdog that stops an answer after 10 minutes without a byte from
  the server, deleting or clearing a chat stops its answer (and nothing is saved again), closing the page ends every
  request (the server cancels a request whose connection closes).
- **Conversations at once** in Settings (`POST /v1/server/slots {"slots": N}`, this computer only): the loaded model
  again with N slots (5-40 s, after the running requests), the context shared out (`--ctx auto`); later model switches
  keep N, a server restart goes back to its `--slots`. A chat that no longer fits (more slots, a smaller model, a chat
  opened again) is compacted before its next answer, from an estimate of its size against the target model's
  context; when the server still says it does not fit, every tool round compacts and tries once more. The session
  cache skips (keeps) saved conversations longer than the context instead of failing to load and dropping them.
- Fixes from a review: Talk background results stay in their chat (the old chat's abort is checked, not the new
  one's); Stop while a command waits for Run / Don't run stops the answer instead of saving a refusal; the overflow
  retry compacts everything before the question (it kept the last exchange); two commands at once no longer wait for
  each other (only the command's own handles go to it: PROC_THREAD_ATTRIBUTE_HANDLE_LIST); "delete unused" leaves the
  files of a chat whose first answer runs, of the message box and of running answers; editing a message keeps images
  and audio sent inside it; Talk results never take the message box's attachments; an upload that ends after another
  chat was opened stays out of that chat; a streamed answer's redraw no longer waits for an animation frame (one that
  never came, in a hidden tab, stopped every later redraw).
- Small screens: the gear in the chat header too; a written document's card also under the final answer.

## 2026-10-26: long DOCX / PDF without a long prompt

- **Uploaded DOCX and PDF are read** (tools/doc_reader.h, tools/inflate.h: DEFLATE, ZIP, a PDF object and content
  stream reader, no libraries): paragraphs with their page and heading level. DOCX: headings from the styles
  ("heading N", outline levels), tables as `a | b | c` rows, lists, pages from Word's page breaks or, when it marked
  few, from `docProps/app.xml` spread over the text. PDF: fonts with ToUnicode maps, simple encodings and their
  Differences, glyph widths for the spaces, form XObjects, object streams; lines grouped into paragraphs by their
  spacing, headings by font size and by short bold lines, table rows by column gaps, running headers and footers
  (the same line on most pages) left out; a scan without a text layer, an encrypted PDF or a binary .doc gets a note
  instead. A 227-page PDF reads in 0.25 s, the same document as DOCX in 0.04 s.
- **A long document is not put into the prompt whole**: above ~6K tokens the model gets its outline (the top heading
  levels, at most ~60-120 sections, with pages and sizes) and its beginning, and reads what the question needs:
  `read_file` with `section` (a chapter with its subsections) or `pages`, or the new `search_file` (BM25 over chunks
  of ~2,400 characters within sections; Indonesian particles -nya / -lah / -kah / -pun folded, words of 5+ letters
  also find longer words; one document or every document of the chat), each passage with its section and pages.
  Tried with Qwen3.5 4B on a 227-page PDF (~219K tokens whole): the question about a fact on page 151 took a 6,357-token
  prompt (8.1 s) and one search, and the answer named the page; a chapter summary came from its pages 8-10.
- Checks: `tests/test_doc_read.cpp` (inflate, Word-made DOCX and PDF fixtures in `tests/data/`, doc_writer output,
  search, damaged and hostile files: delimiter runs, self-referencing forms, a 512 MB decode budget per PDF), in
  `tests/run_tests.ps1`, CI and setup. `test_doc_read FILE [QUERY]` shows any document's outline or search hits.

## 2026-10-26: web UI documents, one folder per chat, terminal tool

- **One folder per chat** (tools/server_files.h): `chats\<date> <first words> (<id>)\` with `chat.json` and the
  chat's files under their own names (a name taken: `name (2).ext`), `.files.json` (id, name, time) and `.src\` (the
  sources of the documents the model wrote). The folder gets its name at the chat's first save. A file attached from
  another chat (@) becomes a hard link, so its bytes are kept once; deleting a chat deletes its folder (a file another
  chat holds stays there); files put into a folder or deleted by hand show up in the Files panel at its next listing.
  Uploads take a `chat_id` (form field or query). The first version's `uploads\` and `chats\<id>.json` are still read.
- **Documents the model writes** (tools/doc_writer.h, `POST /v1/tools/create_document`): DOCX (every part in the
  schema's element order, checked in Word: no repair prompt), PDF 1.4 (standard fonts, real text, justified lines,
  tables that continue on the next page with their header row, page numbers), DOC (RTF, which Word opens as a
  document), RTF and Markdown, from Markdown with inline HTML for underline, colors and sizes. Options: page size and
  orientation, margins, font, size, line spacing, heading / table / text colors, header, footer, page numbers. The
  file shows as a card in the answer. `POST /v1/tools/list_files` and `/v1/tools/read_file` (a text file in parts; a
  document the model wrote: its source, so a correction makes a new version; a file's name works as well as its id).
  The **Files** setting (on) gives the model the three tools.
- **Terminal tool** (tools/server_terminal.h, `POST /v1/tools/run_command`): one PowerShell command in the chat's
  folder, with arcinferzero's folder on PATH, 2 minutes by default (up to 10), ended with every process it started
  (a job object), output up to 64 KB (the start and the end of a longer one). The **Terminal** setting: off (default)
  / ask each time (Run / Don't run in the command's card) / run without asking. Refused to every connection that does
  not come from this computer (`--host` too), to foreign pages (Origin check), and with `--no-terminal`.
- Per-message **Edit** and **Delete** (with the message's files), and **Data** in the settings (clear the chats, or
  the chats and files). Saved chats show their tool cards again (documents as file cards, commands with their output).
- Checks: `tests/test_chat_store.cpp` (28 checks: folders, renames, hard links, files changed by hand, deleting; the
  terminal's output, exit code, timeout and long output), in `tests/run_tests.ps1`; `tools/doc_test.cpp` writes a
  sample in every format. Tried with Qwen3.5 4B in the web UI: notes to DOCX, a corrected version (Arial, dark red
  headings; Word opens it), a command allowed and one refused (the model asked instead of trying again), @ a file into
  a new chat, the old chat deleted.

## 2026-10-25: the NPU next to the GPU (`--npu on`); Flint support removed; the model list without duplicates

- **Web UI rebuilt, laid out like WhatsApp Web**: the chats kept in the browser (searchable), the model picker in the
  chat header (the server switches on the next message), message bubbles with times, the settings behind a gear.
  No max-tokens field any more and nothing sent for it: an answer may use the rest of the context (before, the UI's
  4096 counted the reasoning too, so a long thinking pass could end the reply before its answer). No temperature
  sent unless set: the server samples as the model file recommends. Thinking offered as the model has it (Qwen3.8
  off / low / medium / high, others on / off), from `/props` (new: `thinking`, `reasoning_effort`, `sampling`,
  `think_budget`). Voice messages (record, then send) next to Talk, both ready for Whisper speech to text
  (`speech_to_text` in `/props`). Streamed answers are drawn once per frame instead of per token.
- **Chats and files on disk** (tools/server_files.h): the web UI's chats in `%USERPROFILE%\arcinferzero\chats\`, attached
  files uploaded once into `uploads\<yyyy-mm>\` (content-addressed: the same file kept once) and sent as references
  (`{"type": "file", "file": {"file_id"}}` parts; the server reads documents as text with their name, type, size and
  upload time, images and audio from disk). `@` in the message attaches an earlier file again; a Files panel lists
  every file with the chats using it and deletes files (one, selected, or all unused). New endpoints: `/v1/files`
  (POST upload, GET list, DELETE `/v1/files/<id>`, POST `/v1/files/delete_unused`), `/files/<id>`, `/v1/chats`
  (GET, PUT, DELETE per chat). Every user message carries the time it was sent, so the model knows the day and time.

- **The data folder moved to `%USERPROFILE%\arcinferzero`** (`C:\Users\<name>\arcinferzero`, visible in Explorer;
  `ARC_DATA_DIR` overrides it): kernel cache, NPU graphs, model packages and expert files, sessions, prefix cache,
  logs, settings, and later transcripts and recordings. The exes and scripts move `%LOCALAPPDATA%\arcinferzero` there
  once on the first start (a rename on the same drive: instant) and fix the paths in `last-model.txt`,
  `launcher.json` and `ctx-auto.txt`. Results unchanged (27B with the NPU, 35B, 4B bit-identical to the previous
  build).

- **Cleanup** (no change in results: 4B, E4B, Gemma 4 12B, 35B, 27B with and without MTP / the NPU bit-identical
  before and after; Qwen3.5 4B and Gemma 4 26B images checked): unused code removed (`zero/kernels/gemm8.cl`,
  `weight_ext` / `gemm_ext`, the `attn_decode` kernel built at load and never used, the next-layer expert guess and
  its always-zero counters, the `scale_rows` / `vit_ln_quant` / `vit_quant` kernels, a few unused helpers); the
  JSON / tool-call constraint checkers share one copy of their type / literal helpers; outdated help texts and docs
  fixed (Gemma 4 26B prompt chunks 2048, no Windows RAM reserve since 10-22, Gemma 4 26B's default of 3 drafts, the
  model folders, the NPU), QUEUE's open list tidied.

- **The NPU next to the GPU for prompts** (`--npu on|off|force` for serverz / runz, `arc-server -Npu on`, `pi-arc
  --npu on`; off by default; docs/npu-experiment.md; built on the branch npu-experiment, merged): the Meteor Lake NPU
  computes part of the Qwen3.8 27B's FFN and DeltaNet projections in each prompt chunk through Level Zero's graph
  extension alone (no OpenVINO): 2K prompt +15 %, 11K +15 %, 34K +13 % (time to first token 292 -> 258 s), KL vs
  f32 0.0028-0.0031; off bit-identical to before, MTP / DFlash2 still bit-identical to plain decoding. The setting
  is the server's: switching models keeps it, a restart turns it off. Models with resident W8 (4B, 9B, 35B) gain
  nothing and stay off. Every model can use it (`--npu force`): Gemma 4 (its FFN in f16; the 12B gains +10 % and is
  on with `--npu on`), the Q8_0-form matrices (Gemma QAT Q4_0, the 35B's attn_qkv), Flash Next's attn_qkv. RAM: 0.37 GiB (the NPU's weight rows decoded per layer by the GPU into the shared block; a
  resident copy took 6.1 GiB, same speed).
- **Flint is no longer supported** (the user's decision): the `flint` architecture in the engine and the f32
  reference, its pre-tokenizer (and `src/unicode_ps.h`), the GLM-style tool calls, the server's Flint template and
  thinking handling, the catalog entry and the `flint` shortcut, and the kernel options only it used (attention
  sinks, f32 V rows, attention prep without int8 copies, sigmoid routing with an ungated shared FFN) are gone. Kept,
  as Flash Next's precise mode uses them: BF16 router rows, two passes for the experts, `moe_gelu_mul -DSILU`. The
  other models' logits are bit-identical before and after.
- **The model list shows each model once**: a copy with the same file name and size in another folder (e.g. a
  backup on an external drive) and the GGUF an arcinferzero package was made from are no longer listed as extra
  `-2` entries; the package, then the Hugging Face cache, then the Windows drive keep the entry (35B and Gemma 4 26B
  on D: next to their C: packages: 39 -> 33 entries).
- A plain folder of `.gguf` files inside the Hugging Face cache or a model folder (a copied model, not the
  `models--org--repo\snapshots\...` layout) is listed too, one level deep (e.g. a copied Qwen3.8-27B-AP-IQ4_XS).

## 2026-10-24: renamed to arcinferzero; Flint runs; Gemma sliding attention, /W4 and CI

- **Self-audit fixes** (all source files read for bugs, each finding checked against the code):
  - Server: the Anthropic / Ollama / Responses adapter no longer outlives its request (a cancel or an engine error
    called into freed memory); their streams end once (the finish chunk and the usage chunk both ended them: Ollama
    sent its tool calls twice); a Responses stream with two tool calls closes each with its own data; a stop string
    split over tokens is held back in streams (it reached the client before the cut); a request naming a model no
    catalog has ("gpt-4o", "claude-...") no longer takes the switch lock and rescans every time (a minute's memory);
    `-m model.gguf` without a folder loads.
  - Kernels: `gemv_f32` (Gemma 4 26B's router) and `vit_gemm_f16` (every vision / audio encoder) had lost their
    16-lane sub-group requirement to a kernel inserted before them (right today only because the compiler chose 16).
  - Engine: the vision / audio encoders free their smaller buffers when they grow (a server getting larger and
    larger images kept every set); a Gemma image chunk that throws no longer leaves its non-causal span set;
    DFlash2's head rows fit `logits_` with `--mtp-drafts` below its block; `selector_rank` is bounded (a stack
    array); the MTP layer is prefetched only when the MTP block runs; a second engine reuses the OpenCL compiler
    context; a damaged kernel-cache file is rebuilt from source instead of ending every start, and a short write is
    never installed; the expert cache's shrink also unlinks slots a failed prefetch freed.
  - Tools: GGUF files that fail to parse close their handle and mapping; very thin images stay within the image
    token budget; `runz` checks option values and `--bench-batch` prompts, reads `--kl-ref` files over 2 GB;
    `hf-download.py` keeps `HF_TOKEN` on huggingface.co (urllib copied it to the CDN); `setup.ps1` downloads the 4B
    before the 35B conversion that reads it; `arc-server -NoStream` restarts a running server that streams.
- **Flint** (unsloth/flint-GGUF UD-Q4_K_XL, arch `flint`) runs: Gemma 4's float attention with attention sinks,
  1/sqrt(d), no RoPE on the global layers and a 513-token window on the sliding ones; two leading dense SwiGLU
  layers, then 256 experts (sigmoid routing, streamed from the SSD) and a shared FFN without a gate. Server: its
  template through minja, its stop tokens, `</think>`, tool calls in its GLM form (`<tool_call>NAME<arg_key>..`),
  model id / shortcut `flint`. 4K prompt 248 tok/s, generation 13-17 tok/s.
- **Flint's accuracy, KL 0.029 -> 0.0043** (prompt; token by token 0.067 -> 0.0027) against the f32 reference: two
  int8 passes as Gemma (the experts too), the **V rows in f32** (f16 V alone, simulated in the f32 reference,
  moved it as far as the whole engine and flipped the same near-tie expert choice) and the **router on the file's
  BF16 rows with f32 inputs** (its sigmoid scores have near-ties 0.005 apart). `ARC_FL_Q1=1`: one pass.
- **From reading the GPU assembly (ocloc)**: Flint's attn_prep_g 8,855 -> 481 instructions (no unused int8 copies;
  the in-kernel RoPE angles only where no table is passed, the Gemma drafter); the two-pass expert down kernels no
  longer spill (256 registers, as gate / up); Flint's router GEMV 4 rows x 4 tokens per sub-group (744 -> 229 ms per
  4K prompt). **Gemma 4 26B** gets the same router tiling on its f32 rows: prompt +4.6 %, logits bit-identical.
- **Flash Next precise mode** (`ARC_FN_PRECISE=1`, opt-in): BF16 mixer and router rows, two passes for every
  matmul input: KL 0.0143 / 0.0147 -> 0.0117 / 0.0096 at about half the prompt speed. On a 3,584-token prompt with
  QSA the default measures 0.0186 / 0.0177, precise 0.0127 / 0.0132; the user keeps the fast default (a float
  attention is not needed).
- **KL on the tail of long prompts**: `REF_ALL_FROM=N` (ref_forward) writes the reference from position N on,
  `ARC_KL_FROM=N` (runz) compares from there.
- **Tests**: 27B long-prompt MTP / DFlash2 logits equal plain decoding bit for bit; Flint's tokenizer against
  llama-tokenize; Flash Next auto-detect takes the packaged single file first (its expert file exists; the GGUF parts
  would write a 56 GB one).
- `tools/server.cpp` split into `server_cache.h`, `server_chat.h`, `server_catalog.h` and `server_load.h` (3,337 ->
  1,316 lines, the same code); setup rebuilds serverz when `tools\*.h` change.
- (The 2026-10-24 entries IQ1_S weights, the Qwen verification rows bit-identical again and the Gemma ring slot per
  16-key block are listed under 2026-10-23 below.)
- **Gemma 4 E2B / E4B images fixed** (wrong since the Level Zero engine): the vision attention kernel was built for
  256-dim heads (an earlier default in the kernel file won), so it read the wrong rows; now 3.1 s instead of 51.5 s
  for a 1,064-token image, and the model sees what is in it.
- **Gemma 4 26B images 12.7 -> 7.4 s**: the encoder's matrix products on the engine's int8 GEMM with two passes, and
  its attention with 8-wide block reads and tiles of 8 keys; the model's KL against an exact encoder stays ~0.001-0.003
  (4 test images). An int8 attention was measured too: accurate enough, but no faster, not taken. Tools: `tools/gemma4v_ref.cpp` (the f32 encoder reference in
  C++, 3x faster than the numpy one) and `vision_testz` `ARC_VIS_LM` (the model's KL on an image).
- Qwen3.5 9B / Qwen3.8 27B images: the vision attention in tiles of 8 keys, 3-7 % faster (KL <= 0.0006). `vision_testz`
  now also runs Qwen3-VL mmprojs (timing, `ARC_VIS_LM`).
- **Setup / build** find Visual Studio in more places (scripts/vcvars.ps1: vswhere anywhere, every drive's usual
  folders, a Developer prompt's own compiler).

- **The project is now called arcinferzero** (the repository `rizagp/arcinferzero`; GitHub forwards the old
  `rizagp/arcinfer` address). Entries below this one keep the old name: there "arcinfer" is the project, and before
  2026-10-20 also its first, OpenCL engine. What moved with the name:
  - the data folder is `%LOCALAPPDATA%\arcinferzero` (kernel cache, model packages and expert files, context /
    prefix / session caches, logs, `model-dirs.txt`): an existing `%LOCALAPPDATA%\arcinfer` is renamed once by the
    first exe or script that runs (`src/app_dir.h`, `scripts/app-dir.ps1`), on the same drive, so nothing is copied;
  - Pi's provider is `arcinferzero` (models `arcinferzero/<id>`): the next `pi-arc`, `arc-server` or `setup` replaces
    the old `arcinfer` entry in `~/.pi/agent/models.json`; `pi-arc --model arcinfer/<id>` is still understood;
  - models converted with `tools/convert_gsq.py` carry `arcinferzero.expert_format`; files with the old
    `arcinfer.expert_format` load as before. The web UI keeps its settings (the old storage key is read once).
- **Gemma 4 26B / 12B long prompts**: the sliding layers' attention scores two blocks of 16 keys per step, each
  query load used for both: 11K-token prompt, sliding attention 26B 8.6 -> 7.1 s, 12B 16.5-20.1 -> 13.4-14.0 s,
  bit-identical. E4B and the global layers keep one block (slower with two).
- **Flint groundwork**: the float attention kernels take 128-dim heads and
  attention sinks, attn_prep_g has V without a norm and layers without RoPE, the MoE routing has sigmoid weights
  with an ungated shared FFN. New GPU tests against double / CPU computations: `tests/test_fa.cpp` (every
  attention form, heads 128 / 256, global / sliding ring, with and without sinks) and `tests/test_route.cpp`.
- **Build**: `/W4 /permissive-` (our sources warning-free; third_party as external headers) and a GitHub Actions
  workflow that builds every exe and runs the model-free unit tests on Windows.
- Measured and dropped: Flash Next gate / up on 32-slot tiles (bit-identical, 6.7 -> 8.4 s); larger key blocks or
  other shapes for the head-512 attention (all slower).

## 2026-10-23: Xe2 / Xe3 GPUs (no 8-lane sub-groups); Unsloth Studio statistics and context

- Every exe embeds a UTF-8 code page manifest: model and data paths with any characters work.
- **Gemma 4 26B long prompts ~2 % faster**: the attention's ring slot computed once per 16-key block (global
  layers 16.4 -> 15.2 s at 13,921 tokens), bit-identical.
- **IQ1_S weights** (CPU and GPU decoders; IQ1_M's codebook with one scale per 32 values): agentionai's
  Qwen3.8-27B-AP quants (two IQ1_S `ffn_up` matrices in the IQ3_S file) load. Against ISTA's IQ3_S on this laptop:
  AP-IQ3_S same prompt speed (116 tok/s), generation 4.9 -> 5.05 tok/s, with MTP 11.6-11.9 -> 12.4-12.6 (drafts
  accepted 76 -> 81 %); AP-IQ4_XS generation 4.44, with MTP 12.25 (83.5 % accepted), 18.3 GiB GPU with MTP + DFlash2;
  auto context 40,960 for all three. KL of AP-IQ3_S against its own f32 reference 0.00071 (64 tokens, top token 64 / 64).
- **Qwen 27B MTP / DFlash2 output equals plain decoding again** (verification rows' attention one by one, as single
  tokens: they had been summed differently once the context needed more than one key split); MTP -3 %.
- **Gemma 4 E4B prompts +12-33 %**: its global layers' prompt attention reads an int8 copy of K / V again
  (`attn_ps`), the sliding layers and generation stay float: 13,921 tokens 30.2 -> 22.6 s, 4K 648 -> 724 tok/s;
  KL 0.00075 -> 0.00088 (the limit is 0.015); +0.34 GiB GPU. `ARC_G4_ATT8=0` turns it off.
- **Fixes from DeepSeek's second audit** (static reading, confirmed in the code):
  - a model switch whose model and then the previous model both fail to load no longer leaves the server to crash on
    the next request: it answers 503 until a request names a model that loads;
  - a stop string is cut where it matched at the end of the text, not at an earlier copy of it;
  - a server bound to this machine (the default) answers only requests whose Host names it (localhost, 127.x,
    [::1]): a web page under a name that resolves to 127.0.0.1 can no longer drive it (DNS rebinding);
  - web fetch connects directly, never through a system proxy (through one the checked address was the proxy's);
  - GGUF files: every bound checked without overflow, counts and alignment validated, tensor ranges checked before
    use; invalid head counts / layer counts refused before they divide anything;
  - images larger than 16384 pixels a side or 64 megapixels refused before decoding; `/tokenize` answers 400 to
    invalid JSON; `--ctx`, `--chunk` and the cache sizes clamped (no nan / inf).
- **Flash Next prompt**: the per-tile expert gate / up kernel reads the activation scales token by token (a small
  transpose first): gate / up 8.1 -> 7.0 s per 4K prompt, bit-identical.
- **XMX safer** (from DeepSeek's audit): a GPU module or kernel that cannot be created during the start-up check no
  longer ends the process (it keeps DP4a), and the check now launches the kernels as the engine does (split-K pieces,
  accumulation, the fused SwiGLU, several work-group shapes) and covers the mixture-of-experts tile kernel, which now
  takes XMX too when the check passes (it had been left on DP4a). The check itself is tested here with a right and a
  wrong DP4a variant in the XMX slot (`ARC_XMX_TRY=2` / `=3`); the XMX tile kernels compile for Lunar Lake,
  Battlemage and Panther Lake.
- **Web fetch hardened** (same audit): the address WinHTTP actually connected to is checked before anything is read
  (a host that resolves to a public address first and a local one at connection, DNS rebinding, gets nothing back);
  more reserved ranges refused (192.0.0.0/24, documentation and benchmark ranges, 240/4, IPv4 inside IPv6 judged
  as IPv4: compatible, NAT64, 6to4; Teredo, site-local, 2001:db8::/32, 100::/64); a read never grows the body past its
  limit.
- **Qwen3.8 Flash Next with the DFlash2 drafter** (`--dflash DIR`, incoai/Qwen3.8-Flash-Next-DFlash2): the drafter's
  features are the means of the 4 residual streams, a verification pass of up to 8 rows computes every row as a single
  token is (the HC mixers' single-token kernels for several rows), and rejected drafts roll back the PLE history and
  token ids and the QSA tails. Greedy logits bit-identical to plain decoding (1,097-token prompt, 128 tokens;
  3,291-token prompt with QSA, 96 tokens); 79 % of the drafts accepted, 3.66 tokens per pass, but 5.8-6.6 tok/s
  against 6.4-6.8 plain: decoding here waits on the SSD, and a pass reads the experts of all its rows. Not picked
  automatically.

- **GPUs without 8-lane sub-groups** (Xe2 / Xe3: Lunar Lake, Battlemage, Panther Lake's Arc B390; GitHub issue #1):
  the engine reads the sub-group sizes the GPU supports and builds the two 8-lane kernels (`gdn_recur`, Qwen
  DeltaNet layers; `vit_attn_rows`, Qwen image encoder) with 16 lanes there. Same arithmetic in the same order:
  forced on this Xe-LPG laptop (`ARC_NO_SG8=1`) the 35B's logits are bit-identical and Qwen3.5 4B / 9B describe
  images as before (the 16-lane DeltaNet kernel spills here, Xe-LPG's registers are half as wide; not on Xe2 / Xe3).
- **Gemma 4 E4B: matmuls in one int8 pass** (the float attention stays): KL against f32 0.00075 / 0.00093
  (prompt / generation, 480 tokens), within the user's new limit of 0.015; 13,921-token prompt 45.4 -> 30.1 s,
  generation with the drafter 29.3 -> 32.2 tok/s. 26B and 12B keep both passes (their one-pass KL was 0.098 /
  0.033). Saved E4B states of the two-pass mode are not loaded (state flag 16). `ARC_G4_Q2MM=1`: two passes.
- **XMX for prompt GEMMs** (Xe2 / Xe3 GPUs with XMX, e.g. Arc B390; not this laptop's Xe-LPG): the prompt GEMMs'
  int8 dot products on DPAS, picked automatically and only after a start-up check that the XMX kernels give the DP4a
  kernels' results bit for bit (else DP4a stays). `ARC_NO_XMX=1` turns it off. Untested on XMX hardware so far
  (compiled for Lunar Lake / Battlemage / Panther Lake with ocloc); waiting for a report from issue #1.
- **Unsloth Studio** (arcinfer as a llama.cpp endpoint): the last streamed chunk carries `usage` and `timings`
  together, as llama-server's does (Studio keeps the timings only from a chunk with usage, so it showed no token
  statistics), and `/v1/models` gives the served window as llama-server's `meta.n_ctx` (plus `max_model_len`), which
  Studio reads for the context size.


- **Talk** (web UI, Gemma 4 E4B / 12B): an always-open microphone with speech detection, each utterance sent as
  audio to the model, the answer read aloud sentence by sentence while it streams (browser speech synthesis, voice
  and speed settings), speaking interrupts the AI, a pause inside a sentence joins the two parts. Tested with
  synthesized speech fed to the detector (E4B: speech starts 2.3 s after the user stops talking).
- **Web search while talking** (Talk + Web): the tools run in the background; the AI says it is looking it up
  and talks about the results when they arrive (a message of their own), as the user designed it (F22.4 / F24.5).

## 2026-10-22 (night): web search in the web UI

- **Web search and page reading in the web UI** (the **Web** checkbox): the model gets `web_search` and
  `web_fetch`; the server runs them (`tools/web_tools.h`, WinHTTP: Brave Search's result page, Bing as the
  fallback; HTML to plain text with headings and lists) at `POST /v1/tools/web_search` / `web_fetch`, also while a
  model loads or generates. Each call shows as a card; the answer continues with the results (8 tool turns at
  most); Stop keeps the conversation valid. Only public http(s) addresses are fetched (every host and redirect is
  resolved first; loopback, private, link-local, multicast and reserved addresses are refused, and the connection's
  address is checked again). `--no-web-tools` turns the
  endpoints off. An independent implementation (no code from the AGPL `pi-unsloth-webtools`).

## 2026-10-22: Gemma 4 runs only in its accurate mode; faster sampled tool calls

- **Gemma 4's two-pass mode is its only mode** (the user's decision: "buat apa mode tidak akurat"; Qwen models are
  unchanged). KL against an f32 reference, 480-token code prompt: 26B 0.0042, 12B 0.0002, E4B 0.00003 (one-pass:
  0.0985 / 0.033 / 0.0010). The day's speed work for it: one-token and verification GEMVs and the experts read each
  weight once for both passes, head-512 prompt attention reads each key row once per row, flash-decoding for
  generation, 2048-token chunks with streamed experts. 26B with its drafter ~25 tok/s (one-pass 26.8), after a 14K
  prompt 13.0 (14.9), 4K prompt ~220 tok/s (526); E4B with its drafter 38.1 (42.5). `--q2`, `-Q2` and `ARC_Q2` are
  gone; saved one-pass Gemma conversations are not loaded.

- **The two-pass mode's saved state** (first as the option `--q2`). It is part of the server's context key, its saved conversations and system-prompt cache get their own folders
  (`-q2`), and the saved state carries a flag and the float K / V rows: a conversation saved, a second one held,
  the first one loaded again gives the same greedy answer as a fresh server (26B, 1,511 tokens); a plain server
  refuses the file. Saved states of the other modes are unchanged.
- **GPU memory = the GPU share Windows reports - 1.2 GB** (the user's rule): the share is the user's own setting, so
  the 14 GB Windows RAM reserve and `--windows-reserve-gb` are gone (with a 22 GB share: 17.7 -> 19.2 GiB for
  arcinfer). Gemma 4 26B prompts of the head-512 layers keep `attn_fa_gh` only up to 8 q heads per kv head (the 12B's
  16 spilled and were slower: 4K prompt global attention 4.4-4.9 -> 3.5 s; DeepSeek's audit).
- **`--v4` removed** (the 4-bit V copy for Qwen generation at very long contexts; the user does not use it): the
  option, `-V4`, `ARC_V4`, `ARC_V4_SINK` / `ARC_V4_RECENT`, its kernel paths and buffers (-179 lines). Default
  output unchanged bit for bit (Qwen 35B / 27B with and without drafting, Gemma 26B).
- **Gemma 4 one-pass code removed** (-227 lines): its int8 attention paths, `gemv_glu` and its buffers, the fused
  GELU epilogues, `ARC_Q2_FUSE` / `ARC_Q2_ATT` / `ARC_Q2_KF32`; output bit-identical.
- **Server hardening (audit by DeepSeek, reviewed)**: a request with an `Origin` from another site gets 403 (a web
  page elsewhere could otherwise drive this no-authentication server; local pages and the server's own web UI, also
  opened from another device with `--host`, keep working; command-line clients send no Origin), CORS only for those,
  at most 64 connections, request heads up to 1 MB, an unknown `--host` is an error instead of every interface (IPv6
  literals and names accepted), and a bad option value (`--port abc`) is a message instead of a crash.
- **Sampled tool calls 4x faster**: constrained sampling with top_k > 1 sorted and tested the whole 248K vocabulary
  whenever the 512 most likely tokens held too few allowed ones (after every JSON key); it now stops when the rest
  could not be drawn (e^-20 below the best allowed token). 35B, as Pi samples: 4.8 -> 20.4 tok/s; a 1,240-token
  `write` call 24.3 (greedy 24.4). The same tokens as before in the seeded tests.

## 2026-10-20: the OpenCL engine removed; robustness fixes

- **Two-pass mode for Gemma 4 (`ARC_Q2=1`, opt-in, slow for now)**: every activation in two int8 passes (the value,
  then its rounding residual), the experts too, and the attention in float (f32 query and K, f16 V). KL against the
  f32 reference, 480-token code prompt: 26B 0.0985 -> 0.0045 (decode path 0.104 -> 0.0045), 12B 0.033 -> 0.0003,
  E4B 0.0010 -> 0.00003; drafters stay bit-identical, slots too. Speed work comes next (QUEUE F21.11).
  `tools/ref_forward` now has Gemma 4's per-layer embeddings (E2B / E4B).
- **Gemma 4 26B-A4B with its drafter** (Unsloth's `mtp-gemma-4-26B-A4B-it.gguf`, next to the model): 3 drafts per
  step; server answers of 400 tokens, code 18.8 -> 27.0 tok/s, Indonesian 19.4 -> 21.7.
- **Gemma 4 drafts verified bit for bit**: a verification pass ran several rows per attention launch with the keys
  split by the last row's position, so its logits were not those of decoding each token alone (26B: greedy text
  changed after 7 tokens; 12B / E4B logits up to 1.3 / 0.85 off). One row per launch now: bit-identical over 128
  tokens on the 26B, 12B and E4B, as fast or faster (12B 21.3 -> 23.2 tok/s with the drafter).
- Gemma 4 26B prompts: the experts' gate / up through `moe_rg` with a GELU epilogue (they were per tile because of
  the 704 -> 768 padding): 4K 451 -> 506 tok/s, 8K 405 -> 457; KL unchanged (0.0985 -> 0.0993).
- `tools/ref_forward` reads the 26B's package (its experts from the expert file) and simulates two int8 passes
  (`REF_Q2`, `REF_ATTQ2`): with the matrix inputs and the attention's q / k / v in two passes the 26B's KL would fall
  from 0.0985 to 0.0053 (QUEUE F21: the plan). `runz --dump-logits` also with `--mtp`; `vision_testz` with
  `ARC_PROF=1` prints the encoder's kernels.

- **The OpenCL engine is removed** (`bin\server.exe`, `bin\run.exe`, `src/engine.h`, `kernels/`, the OpenCL copies of
  the vision / audio encoders, `tools/run.cpp`, `bitcheck`, `test_gemm`, `vision_test`, `bench/gemm_bench`, the old
  `stable/` server). arcinferzero (`bin\serverz.exe`, `bin\runz.exe`), which matched its output bit for bit on every
  model, is the only engine: `arc-server -OpenCL` / `pi-arc --opencl` stop with a message, `-Zero` / `--zero` are
  still accepted, the launcher has no Engine field, `setup` builds serverz / runz / clinfo, `tests\run_tests.ps1`
  has no `-Engine opencl|both`. The micro-benchmarks read `zero\kernels`. `runz` now warns about a running serverz /
  runz (it looked for the old names).
- Slots and batches refuse misuse instead of corrupting state: a slot out of range or given twice, a batch over
  drafts not yet settled. A batch pass that throws binds the slot it started from again and changes no slot's
  position or pending drafts (they change once the pass is done); the server fails every request of that pass
  (requests waiting for it to lead their step no longer wait for good) and starts their conversations over. The
  estimated memory per slot equals the allocation exactly (`ARC_SLOT_BYTES=1` prints both; `runz --slots-check`
  and `tests\run_tests.ps1` fail when it counts less).
- MTP block + DFlash2: a step drafted by DFlash2 now also writes its token's row of the MTP block's KV cache; the
  MTP drafts after DFlash2 steps read stale memory there (output was always right; drafts depended on what the
  slot held before). Speed unchanged (35B / 27B, prose and code).
- A failed generation step (alone or batched) fails its request(s) and starts the slot's conversation over; a
  streamed request ends with an error event (`data: {"error": ...}`) instead of a cut stream, and a 500's message is
  escaped as JSON. Tested by `tests/fail_inject.ps1` (in `run_tests`): `ARC_FAIL_PASS=N` makes the engine's N-th
  verification or batched pass throw after its layers.
- QUASAR-QAT's Q4_0 files load (ids from their file names): the Qwen3.5 4B is about as accurate as Unsloth's
  UD-Q4_K_XL (perplexity 14.93 vs 14.83), its Gemma 4 E4B clearly worse than Google's QAT file (48.0 vs 30.4;
  llama.cpp the same). `serverz --list-models` marks mixture-of-experts files (`moe`), which `arc-server` uses for
  its engine line (Gemma 4 26B was described as dense).
- The expert cache's prefetch thread passes its errors to the caller instead of ending the process; the expert
  file is written to a `.tmp` and renamed only when complete.
- JSON nested more than 256 deep is a 400 (it overflowed the connection thread's stack); the build uses `/utf-8`.
- Saved conversations remember the MTP draft count and the slot layout (an older save is not reused).

## 2026-10-19: two conversations at once, Gemma 4 26B-A4B, Flash Next accuracy knob

- **Gemma 4 26B-A4B** (Google QAT Q4_0; arcinferzero, `pi-arc g26`): the dense FFN and 128 routed experts side by
  side, experts streamed from the SSD like the 35B's (gate / up from one tensor; GELU; the router in f32 on its own
  input norm; a scale per expert). Prompt 450 tok/s (4K), generation ~20 tok/s; KL 0.07 against the CPU reference
  (`tools/ref_forward` now runs its experts), the size of int8 activations' own effect. Images too: its vision
  encoder's 72-dim heads (`g4v_qkv_hd`) and std_bias / std_scale; the embeddings match `tools/gemma4v_ref.py` (cosine
  >= 0.999997; `bin\vision_testz.exe` writes them for arcinferzero). `--ctx auto` keeps no memory for an encoder it
  cannot run.
- The server prints why a model cannot load instead of stopping silently.

- **Conversation slots** (arcinferzero; `serverz --slots N`, `arc-server -Slots N`, `pi-arc --slots N`): the server
  holds N conversations, each with its own KV caches, DeltaNet / conv states, QSA keys, MTP / DFlash2 state and
  snapshots (`Engine::add_slots` / `bind`: the state buffers are re-pointed to the slot's memory, no copies).
  Requests in different slots take turns step by step by GPU time used, so two Pi sessions answer at the same time;
  a short question during another slot's 46K-token prompt: 49 -> 9.6 s. Logits bit-identical alone and interleaved
  (`runz --slots-check`: 27B, 35B with MTP and DFlash2, Flash Next with QSA); one slot unchanged.
- **Batched decoding for Gemma 4** too (26B-A4B: two answers at once 19.8 -> 26.2 tok/s); `--ctx auto` with slots on
  streamed MoE models aims at 128K in all and keeps at least 32K per slot (26B 2 x 32,768 instead of 2 x 8,192).
- **Batched decoding over slots** (Qwen dense and MoE): requests generating at the same time run their next tokens
  and drafts in one engine pass (`Engine::forward_batch` / `verify_batch`; every slot drafts in its own state, each
  settles its own drafts). Two answers at once: 27B 7.3 -> 10.5 tok/s together (prose), 11.6 -> 15.4 (code); 35B
  20.2 -> 24.7, 29.0 -> 30.7. Logits bit-identical to each slot alone; `runz --bench-batch N` times it (27B, no
  drafts: 2 / 3 / 4 slots 1.78 / 2.43 / 3.04x).

- `ARC_HC_F32` (arcinferzero, Flash Next prompts; `4` the default since 20 Oct, `0` the old single int8 pass): the hyper-connection mixers of prompt chunks with more
  precision. `4` takes the up projection's input as two int8 passes (values, then their rounding residual): KL
  against the f32 reference 0.0647 -> 0.0483, top token 324 -> 331 of 363, prompts 5-7 % slower. `1` f32 for both
  projections (KL 0.0369, about 22 % slower), `2` / `3` the down / up projection only. The default (int8) is unchanged.
- `tools/ref_forward` reads arcinferzero packages; `runz --kl-ref` reports top-token agreement.

## 2026-10-18: SSD expert prefetch on the copy engine, 35B chunks of 8192, drafter per model, long contexts

- **Deep expert prefetch** (arcinferzero, prompts of >= 2048 tokens): while the GPU runs a layer, a thread reads every
  missing expert of the next layer from the SSD and the GPU's copy engine copies them into their cache slots; the
  kernels wait on an event. Flash Next prompts 4096 tokens 107 -> 166 tok/s, 9000 116.9 -> 144.5, 32K 131.7 -> 162.6;
  35B 32K at ctx 131K 488.6 -> 557. Logits bit-identical; 56 regression runs (small caches, simulated GPU
  out-of-memory, 40 prompts in a row) identical. `ARC_XCOPY=0` turns it off.
- **Qwen3.6 35B prompts in 8192-token chunks** (streamed MoE): 32K prompt 533 -> 604 tok/s, 128K 306 -> 324,
  generation about the same (ctx 262K; expert cache 81 -> 77 %).
- **Drafter per model** in Pi's `/model` and `pi-arc`: `<id>-nodf` drafts with the MTP block only, `<id>-df` with
  DFlash2 only (the MTP block is not loaded: 27B context 69,632 -> 94,208). 35B: Python code fastest with `-df`
  (49.9 tok/s), Indonesian with `-nodf` (21.9). `--draft auto|mtp|dflash` for the server and `runz`.
- `--v4`: a 4-bit copy of V for generation, opt-in (35B at 250K depth 10.6 -> 11.5 tok/s, KV +25 %); the
  automatic context gives the 35B and Flash Next 131,072 tokens; Flash Next's QSA scores 8 rows per work-group
  (125K prompt 78.6 -> 123.4 tok/s).
- Fixes: a server switching models now keeps one Level Zero context (the new model used the old context's copy
  list, events and staging buffer); the expert prefetch thread's read errors no longer end the process; GPU
  out-of-memory during a prefetch is handled; numeric `ARC_*` knobs that divide are kept >= 1; `/fp:precise` for the
  accuracy tools.
- `scripts/hf-download.py` keeps trying until a download is complete (a router that drops out is waited out), with
  no speed limit by default and several files per command.
- Docs: [docs/before-after.md](docs/before-after.md) (every change with its numbers before and after, accuracy and
  consistency), a new README overview.

## 2026-10-16 (night): DFlash2 block drafter for Qwen3.8 27B

- `--dflash DIR` (runz, serverz) loads a z-lab DFlash2 drafter (z-lab/Qwen3.8-27B-DFlash2): 5 layers fed by the
  target's hidden states after layers 5 / 19 / 33 / 47 / 61 (fc + norm into a context K/V ring of 4096 positions,
  attention window 2048), a block of 8 rows [token, mask x 7] per draft with two-tap grouped dynamic convolutions
  around attention and MLP, the target's output head and the candidate selector (top 16, greedy bilinear path).
  Weights BF16 -> Q8_0 on the GPU (~1.9 GB), selector codebooks read from the mapped file.
- With the model's MTP block also loaded (the default with `--dflash`), each step picks the drafter: DFlash2 while
  its recent accepted drafts per pass stay >= 2 (`ARC_DFL_MIN`), else the MTP block (3 drafts), with a DFlash2 step
  every 8th to re-check. `ARC_DFL_P` (default 0.3) drops a DFlash2 path's tail below that probability.
- 27B, greedy, answers identical to plain decoding: Python code 4.85 -> 11.5 tok/s (MTP alone ~8.9; 89 % of drafts
  taken, 6.9 tokens per main-model pass); Indonesian prose 4.67 -> 5.6 (MTP alone 5.8-6.3; DFlash2 takes ~11 % of
  its drafts there: its training data is mostly English and code).

## 2026-10-16 (evening): Qwen3.8 Flash Next on arcinferzero

- New architecture `qwen4exp` (Qwen3.8 Flash Next, ISTA-DASLab GSQ-RCO IQ3_S, split GGUF of 2 parts) in arcinferzero:
  4 hyper-connection streams with their mixers before every block, Gated DeltaNet with a sigmoid output gate, gated
  attention, 512-expert MoE (10 active) streamed from the SSD with 640-wide experts (down rows padded to 768), the
  PLE n-gram block (rows read from the 26.8 GiB table in part 2 per token, its conv history kept with the recurrent
  state) and QSA sparse attention (indexer, pooled block keys, top 512 blocks per query; dense below ~2K tokens).
- Accuracy against the CPU reference (`tools/ref_forward`, which now covers QSA and dumps per-block residuals):
  short chat prompt KL 0.0014 ("Jakarta" 0.92 vs 0.93). On a 337-token raw text the KL is 0.15, the same as the
  reference with int8 activations (0.13): this model is more sensitive to int8 activation quantization than the
  others (every matrix group adds 0.02-0.10).
- Speed (first numbers, 32 GB laptop, expert cache 7.9 GiB = 12 %): generation ~4 tok/s, short prompts 3-6 tok/s
  while the cache is cold. The expert file (56 GB) is written once on the first start (~9.5 minutes).
- Expert cache: with more than two record sizes (Flash Next has 5 type mixes) one pool of the largest record serves
  every layer; separate pools each needed a whole layer and ran out of GPU memory.
- `src/quants.h`: Q8_0 rows whose length is not a multiple of 256 dequantize fully (the CPU reference read only the
  first 512 of 640 values of Flash Next's last shared expert).
- 27B and 35B logits unchanged (bit-identical).

## 2026-10-16 (afternoon): extra model folders, kernel timings on Level Zero

- Models can live on another drive: `%LOCALAPPDATA%\arcinfer\model-dirs.txt` (one folder per line) or
  `ARC_MODEL_DIRS` (`;`-separated) adds folders in the Hugging Face cache layout or with `.gguf` files directly in
  them. The Hugging Face cache is searched first, so a model kept in both places still loads from there.
- `runz --prof-gen` on arcinferzero now reports per-kernel times (Level Zero kernel timestamps; before: 0 ms).
- `zero/page_probe.cpp`: GPU page size and random-read speed by span (2 MiB pages, ~4 MiB GPU cache, random 256 B
  reads slow down above ~1 GiB; the Level Zero virtual-memory API is refused on this Windows driver).

## 2026-10-16: arcinferzero is the default engine; MTP up to 7 drafts

- `arc-server`, `pi-arc` and the launcher start arcinferzero (Level Zero) by default; `--opencl` / `-OpenCL` (launcher:
  Engine "arcinfer") runs the OpenCL server as before. `--zero` is still accepted. Tested the night before: identical
  logits to OpenCL for every model (35B with the same prompt chunking), images and audio identical (E4B, 12B, 9B),
  every API format, Pi, tool calls, MTP identical to plain decoding.
- MTP drafts up to 7 on arcinferzero (`--mtp-drafts 1..7`): a verification pass writes its DeltaNet states to a second
  buffer and keeps its inputs; rejected drafts are undone by replaying the recurrence over the accepted tokens (one
  state copy instead of one per draft, no 150 MB copies per step). Default drafts: 35B 1 -> 3 (23.9 -> 26.1 tok/s),
  4B 2 -> 3 (22.9 -> 24.8); 9B 2, 27B 3 (5 and 7 are not faster yet: each draft runs the MTP block with the full
  output head).
- Multi-token GEMVs (verification of 3..7 drafts) load each token's activations where they are used instead of
  holding all of them in registers: 4-7 drafts no longer spill (27B, 5 drafts: 3.8 -> 9.2 tok/s).
- Qwen3.6 35B decode attention: attn_dsplit (4 q heads per sub-group, dims split over 4), 1.16 -> 0.94 ms per layer at
  64K; verification rows stay bit-identical to single-token decoding (fixed key splits, per-row P bound).
- Gemma kernels without register spills: attention prep with the large register file and its norm / RoPE work in
  pieces of 8 (8.8 KB -> 0 per thread), the post-attention / post-FFN norm sized to the model (3.7 KB -> 64 B);
  identical logits. E4B prompt 2K 733 -> 827 tok/s, 16K 571 -> 625; 12B 271 -> 289, 218 -> 228.
- Fixes: constrained JSON / tool calls no longer allow a comma after the last allowed key (a `json_schema` answer
  dead-ended as `{"name": "x", "age": 34,`); `/api/tags` lists the loaded model with `--no-swap`.
- `arc-server stop` waits for the process to end plus 5 s: a server started while Windows was still releasing the
  previous one's memory sometimes ran at half speed for its whole life. The server and runz opt out of Windows power
  throttling (EcoQoS) as a precaution (no effect measured).
- Diagnostics: `ARC_ZE_KPROPS=1` prints each kernel's spill and its build options; `zero/grf_probe.cpp` (also
  `grf_probe FILE.cl KERNEL "OPTIONS"`: one kernel's spill); `runz --bench-pp` prints the time of each generation step
  (min / median / max) and takes `--prof-gen`; `ARC_CONV_GRF=256` (experiment: no spill, not faster).
- `ARC_DRAFT_VOCAB=N` (off): the drafts' output head over the first N token ids only; faster for English code, slower
  for Indonesian (its tokens lie beyond 65536).

## 2026-10-15 (night): Level Zero tests, Qwen tool-call constraint, saving the held conversation

- Constrained tool calls also for Qwen's XML format (`<tool_call><function=..><parameter=..>`): the function and
  parameter names are the tool's own, every required parameter is there before `</function>`, non-string values are
  JSON of their schema type, string values cannot contain `</parameter>` / `<parameter=` (the parser would cut them
  there). Qwen: only after the reasoning (it never calls tools inside it). Gemma stress test (E4B, temperature 1.3, a
  long edit forced through tool_choice): 2/10 well-formed calls without the constraint, 9/10 with it.
- `POST /sessions/save`: the held conversation goes to the session cache and the call returns once it is on disk;
  `arc-server stop` / `restart` call it first, and the server does it before a model switch, so continuing that
  conversation afterwards loads it instead of processing it again.
- `ARC_LOG_TOOLCALLS=file`: every answer that writes a tool call is appended as a JSON line (raw text, parsed calls,
  tokens the constraint corrected), to see what a model wrote when a call fails.
- Fix: `logprobs` sent tokens that are part of a UTF-8 character as raw bytes, making the JSON invalid; the text is
  now valid UTF-8 (U+FFFD for a partial character), `bytes` keeps the exact bytes.
- Web UI compaction: the summary request asks for a list that copies every specific item exactly (temperature 0);
  with the first wording Qwen 4B dropped one of seven code words, now none.

## 2026-10-15: web UI compaction

- `--preserve-thinking` (server; `pi-arc --preserve-thinking`, `arc-server -PreserveThinking`): Qwen3.6's template
  option, earlier turns keep their reasoning in the prompt (requests can set `chat_template_kwargs.preserve_thinking`).
  The prompt then only grows, so a new message reuses the whole cache; the context fills sooner. Only for models
  whose template has the option (Qwen3.6 35B); Qwen3.8 27B always keeps the reasoning.

- Qwen: a third recurrent-state snapshot at the start of the assistant turn that follows the user's last message.
  Qwen's chat template drops the reasoning of earlier turns once the user writes again, so every new message in a
  Pi session diverged right after the previous user message and was processed again from the end of the system
  prompt; now only the last turn is processed again. In `server-new.exe` until tested on the GPU.

- Constrained tool calls (Gemma 4; `tools/tool_constraint.h`): while the model writes a tool call, only tokens that
  keep the call well formed and fitting the tool's parameter schema are drawn: required arguments cannot be left
  out (Pi's edit without `path`), keys are the tool's own (no code as a key, no `edits` inside an edit), strings use
  `<|"|>`, enum values and integers are respected, and the call ends with `<tool_call|>`. A call that is already well
  formed is sampled exactly as before. The request log shows "tool call: N of M tokens corrected" when the
  constraint changed something; `ARC_TOOL_CONSTRAINT=0` turns it off. In `server-new.exe` until tested on the GPU.

- The disk session cache is on by default (16 GB; `pi-arc --session-cache 0` turns it off): a Pi conversation that
  another session or the web UI pushed out of the GPU is loaded from disk in seconds instead of processed again.

- The web UI shows the context the conversation uses ("context 12.3K / 40.9K") and compacts it: when the next
  request would nearly fill the context, or the server answers `exceed_context_size_error`, the earlier messages are
  summarized by the model (thinking off, the cached prefix reused) and the conversation goes on with that summary
  plus the latest exchange. A history too long to summarize at once is summarized in parts. The Compact button does
  it on demand; Stop cancels it and keeps the conversation. The old messages stay on screen, dimmed, with the
  summary under "Earlier messages compacted".

## 2026-10-15: the full API (OpenAI, Anthropic, Ollama formats), structured output, one server source

- One server source: `zero/serverz.cpp` is `tools/server.cpp` built with `ARC_ZERO` (Level Zero engine, its own
  data folder `%LOCALAPPDATA%\arcinfer\zero`); `setup.ps1` builds both, and `runz`.
- Sampling: `frequency_penalty`, `repeat_penalty` / `repeat_last_n`, `logit_bias`; `logprobs` / `top_logprobs`;
  `n` > 1 (without streaming).
- Structured output: `response_format` `json_object` / `json_schema` constrains sampling with a JSON-schema prefix
  checker (`tools/json_constraint.h`), so the answer always parses and fits; `tool_choice` `required` / a named
  function starts the answer with a tool call.
- Other formats on the same handler (`tools/api_compat.h`): OpenAI `/v1/completions` (raw prompt) and
  `/v1/responses`; Anthropic `/v1/messages` (+ `count_tokens`); Ollama `/api/chat`, `/api/generate`, `/api/tags`,
  `/api/show`, `/api/ps`, `/api/version`; `/v1/audio/transcriptions` and `/translations` (Gemma 4 audio).
- `tests/test_api_compat.cpp`: 44 CPU-only checks of the JSON checker and the request translation (all pass).
- Launcher: the API address for each format with a Copy button. `pi-arc --zero` / `--no-stream`, `arc-server -Zero`
  / `-NoStream`; startup messages name the engine.
- Not yet: `/v1/embeddings` (needs an embedding model architecture the engine does not have).

## 2026-10-14: arcinferzero: Qwen3.6 35B with its experts streamed from the SSD (Level Zero)

- New engine in `zero/` (arcinfer itself is unchanged): `zero/clrt.h` runs arcinfer's OpenCL kernels under Level
  Zero (native binaries from the shared kernel cache, USM buffers). Same results bit for bit, faster: 35B TTFT 6.45 ->
  4.07 s, generation 16.9 -> 18.3 tok/s; load 4.3 -> 2.3 s (4B).
- Routed experts live in `%LOCALAPPDATA%\arcinfer\zero\<model>-<id>.experts` (GPU block form, one 4 KB-aligned record
  per expert; made once from the GGUF, 15.1 GB / 144 s for 35B; remade when the GGUF changes; may be deleted) and are
  read with unbuffered I/O (no Windows file cache) into an LRU cache on the GPU. The expert kernels take each
  expert's address from a table (`-DXCACHE`); prompts use the back of the LRU list as their transient area and read
  the next layer's missing experts on a thread while the GPU works; the cache's contents are saved (`.lru`) and loaded
  again at start (10.6 GB in ~6 s).
- Cache size: automatic so the process stays within RAM - 14 GiB (Windows) and the GPU share - 1.4 GiB;
  `--expert-cache-gb`, `--expert-cache-pct`, `--windows-reserve-gb`, `--no-stream`. When the GPU runs out of memory
  anyway (the share is shared with every program), the cache gives back 1 GiB and the launch is retried.
- Qwen3.6 35B at 262K context, 4096-token chunks: MTP 22.2 tok/s (cache 79 %, hit 97.6 %), without MTP 19.3 tok/s,
  time to first token 3.3-3.5 s for 2,216 tokens (arcinfer: 4.1-6.5 s at 40K context); output identical to arcinfer.
  With images and MTP together: 262K context, cache ~70 %.
- `bin\runz.exe` / `bin\serverz.exe` (build `zero\runz.cpp` / `zero\serverz.cpp`); `arc-server -Zero`,
  `pi-arc --zero` and the launcher's Engine field choose it; without them the usual arcinfer runs.

## 2026-10-14: Qwen3.6 35B images; projectors that leave too little context are opt-in

- Qwen3.6 35B-A3B reads images with Unsloth's `mmproj-BF16.gguf` (Qwen3-VL, 27 layers, 72-dim heads): text in a
  1920x1080 screenshot quoted exactly; 1,008 image tokens encoded in ~3 s (~60 s the first time: kernel compile).
- `--ctx auto` now counts the image encoder of MoE models too (it reserved nothing). For 35B the encoder (~1.1-1.3 GB)
  leaves only 4,096 tokens at 18.3 GiB, so a projector that cuts a model's auto context below 16K (to less than
  half of it) is no longer part of the default variant: `35b` stays MTP without images (40,960 tokens), images go
  through `35b-vision`. Other models keep their variants.

## 2026-10-14: launcher window, GPU memory by part, monitoring

- `arc-launcher.cmd` / `scripts/arc-launcher.ps1`: a window with every model and variant (MTP, images, automatic
  context, GPU memory at that context, compatibility), how the automatic context was counted for the selected one,
  a field for every server option, and the running server's state, GPU memory by part, requests and speeds. Start /
  Stop go through `arc-server`; the server keeps running when the window closes. Every field starts at the selected
  model's best value (`--list-models` "defaults": chunk, MTP drafts, image tokens, ...); changed fields are marked
  and are the only ones passed on; "Set as default" saves them and the default model, "Reset options" clears them.
- `arc-server.ps1`: every server option (`-MtpDrafts`, `-ThinkBudget`, `-ImageMinTokens` / `-ImageMaxTokens`,
  `-BudgetGb`, `-AttnF32`, `-NoPrefixCache`, `-PrefixCacheGb`, `-NoSwap`, `-ListenHost`, `-SessionDir`); a running
  server is restarted when one given differs (a plain `start`, as pi-arc does, keeps it).
- The engine counts its GPU memory by part (weights, MTP block / drafter, working buffers, context); `server.exe
  --list-models` gives each variant's counts as `--ctx auto` used them ("memory": device, kept free, snapshots,
  encoders, weights, mtp, buffers, context, context per 1K tokens), `/props` the loaded model's ("memory") and
  monitoring ("stats": requests, busy, filled context, the last request's timings). `/props` reports audio when the
  model has an audio encoder (it said false). The ctx-auto cache key carries layout version m9.

## 2026-10-13: Qwen3.6 35B: prompt experts through the int8 GEMM, MTP, more context; Qwen prompt attention

- MoE prompt chunks of >= 1152 tokens run the experts through the int8 GEMM (kernels/gemm.cl `MOE_TILES`): per group
  of 64 experts the raw rows are decoded to W8, then one GEMM over that group's slot tiles. Qwen3.6 35B UD-Q3_K_M,
  2048-token prompt: 476 -> 642 tok/s; per-layer outputs equal to the per-tile kernels (differences ~1e-7).
- `decode_w8_sl`: whole rows (up to 16 superblocks) are copied to local memory with coalesced loads, then decoded
  from there (types without mins). IQ3_XXS 1038 -> 518 ms, IQ4_XS 433 -> 303 ms per 2048-token chunk; Q8_0 / Q4_0
  use their vector block decoder (`decode_w8_vb`): Q8_0 211 -> 72 ms.
- MoE buffers share memory with buffers only used in the attention / DeltaNet half of a layer (expert activations
  over z_, expert outputs over conv_out_ / o_ / qrope_): 167 MB. Automatic context: `35b` 86K -> 102K (94K before
  the GEMM staging), `35b-s` 217K.
- Single tokens: DeltaNet qkv + z (+ alpha / beta when Q8_0) stacked into one GEMV (views, nothing stored twice).
  35B 19.2 -> 19.8 tok/s, 27B 4.75 -> 4.8-5.0, output identical; 9B unchanged (qkv / z types differ in most layers).
- MTP for Qwen3.6 35B (Unsloth's MTP repository, `35b-mtp`): MoE in the MTP block, Q3_K on the GPU (decode32), expert
  types with mins (Q4_K / Q2_K / Q5_K) in the expert kernels (activation sums computed in the kernel). Output
  bit-identical to plain decoding; 19.7 -> 22.3 tok/s (83 % of drafts accepted), prompts -3.4 %, automatic context
  40,960 (`35b-mtp-std`: 98,304). MoE models default to 1 draft per step (1 and 2 equally fast; one DeltaNet state
  copy less). Verification rows of models whose single token takes `attn_decode` (35B: 2 kv heads) now take it as
  well, row by row: `attn_split` split the keys differently and MTP output drifted after ~40 tokens of a 2K prompt.
- Qwen prompt attention (`attn_q8`): 4 q heads per sub-group where the GQA ratio allows. 21,590-token prompt:
  35B 9.4 -> 8.4 s, 9B 8.2 -> 7.0 s of attention, same output; 27B (6 q heads per kv head) unchanged.
- Server: Qwen3-VL projectors also load for MoE models (a projector whose output width does not match the model is
  refused); no 35B mmproj was available to test with.
- tools/bitcheck: residual stream per layer and logits of single-token vs 2-token passes and of the MTP flow
  (draft, verify, reject / accept); tools/ref_forward reads standard GGUF experts.
- Tried and left off: Q8_0 GEMV through `gemv_v` (slower), down + combine fused for single tokens (212 vs 211 ms),
  `attn_ps` for Qwen shapes (slower), Q4_0 requantization of the MTP block's experts (replaced by native Q3_K / Q4_K).
- The ctx-auto cache key carries layout version m8.

## 2026-10-13: Qwen3.6 35B: both Unsloth files selectable, router + shared expert in one GEMV

- `35b` = UD-Q3_K_M (IQ4_XS down experts; automatic context 94,208), `35b-s` = UD-Q3_K_S (204,800); both in Pi's
  /model (several files of one repository are named with their file), and the server switches between them.
- MoE layers keep the router rows (padded to 384) and the shared expert's gate and up rows in one Q8_0 buffer:
  a single token runs one GEMV over the 1,408 rows instead of three small ones (each paid a launch's fixed cost);
  the three matrices are sub-buffers of it for prompts, so nothing is stored twice. Qwen3.6 35B UD-Q3_K_M
  generation 17.4 / 17.8 / 18.5 -> 18.2 / 18.8 / 18.8 tok/s (alternating runs), output identical.
- Tried and left off: split-K GEMV for matrices with few rows (`ARC_SPLITK`): 512-row matrices 10-15 % faster,
  larger ones slower, no gain overall; per-type GEMV rows / kernels tuned on single repeated matrices: no gain in
  the model (Qwen3.8 27B). Qwen3.8 27B with its MTP block measured 4.5 -> 7.9 tok/s on code (2 drafts, 91 %
  accepted).
- The ctx-auto cache key carries layout version m4.

## 2026-10-13: memory: Q8_0 stored once, encoders reserved by --ctx auto; `35b` alias for Unsloth's GGUF

- Q8_0 matrices keep their repacked GEMM copy only with resident weights of a dense model; MoE models and
  just-in-time models decode the raw rows into the same form per prompt chunk (decode_w8, PER32). Qwen3.6-35B-A3B
  UD-Q3_K_S: GPU 16.47 -> 15.04 GiB, load 62 -> 28 s, automatic context 61,440 -> 208,896 tokens (a 24K-token prompt
  at 356 tok/s); prompts ~4 % slower (decoding per chunk), results identical.
- `--ctx auto` keeps the GPU encoders' memory free (weights + buffers for the largest image / a 30-second clip; they
  upload on first use and were not counted): Qwen3.5 9B 217,088 -> 143,360 tokens (~1.4 GB for the vision encoder at
  2048 image tokens), Gemma 4 E4B unchanged at 131,072 with 1.6 GB kept. The ctx-auto cache key carries an engine
  memory-layout version ("m2"), so stale contexts are computed again.
- `qwen3.6-35b-a3b` / `35b` (`pi-arc 35b`) also name Unsloth's `Qwen3.6-35B-A3B-UD-Q3_K_S.gguf`.
- Qwen3.5 9B vision attention (72-dim heads): q read again per key tile instead of held in registers (spills
  295 -> 164); speed unchanged at ~1.9 TFLOPS (64-dim heads: 2.8). The compiler keeps 128 registers per thread for
  this kernel despite -cl-intel-256-GRF-per-thread; going further needs another kernel layout.

## 2026-10-12: GEMV bandwidth, Qwen3.5 9B images, Qwen3.6-35B-A3B from standard GGUFs

- GPU raw forms regrouped for aligned vector loads: Q8_0 (the 8 scales first; GEMV 37 -> 77 GB/s), IQ2_XXS /
  IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S (the f16 scale moved behind the quants; IQ2_XXS 27 -> 60, IQ3_XXS 41 -> 62,
  IQ3_S 61 -> 66 GB/s); `gemv_v` vector decoders for the codebook types where faster. Token generation, share of
  the measured ~75 GB/s: Qwen3.5 4B 77 -> 79 %, 9B 83 -> 86 %, Gemma 4 E4B 76 -> 81 %, 12B 80 -> 82 %, Qwen3.8 27B
  65 -> 72 % (4.3 -> 4.7 tok/s). All regression tests pass.
- Qwen3.5 9B images (head size 72, width 1152): `vit_attn_rows` / `vit_qkv_rope` take the head size as a build
  option (72-dim heads read K / V with interleaved sub-group block reads: 6.0 -> 1.6 s of attention for a 1222-token
  image), a LayerNorm for any width; 0.38 % from the f32 reference (tools/vision_test.cpp, generalized).
- Qwen3.6-35B-A3B from standard GGUFs (Unsloth UD-Q3_K_S: IQ3_XXS / IQ3_S / IQ4_XS experts): `moe_gateup_q` /
  `moe_down_q` (kernels/decode.cl) on the experts' raw rows, each decoded weight serving a tile of 16 slots;
  generation 17-18 tok/s, 2048-token prompts 473 tok/s (150 with 8-slot tiles re-decoding per tile), automatic
  context 61K at 17 GiB. Prefill and token-by-token agree (top-1 0.9718 / 0.9712).

## 2026-10-12: Google's Gemma 4 QAT models (Q4_0), E4B images and audio on the GPU

- **Q4_0** weights (Google's QAT GGUFs): the GPU raw form regroups each 8-block superblock (8 scales, then the
  16-byte nibble groups, same 144 bytes) so every block is one aligned vector load. Token generation reads them in
  `gemv_v` / `gemv_mt_v` / the fused `gemv_glu` at 66-73 GB/s (bit-identical to `gemv`, ~1e-7 from the CPU
  reference); prompts decode them losslessly into the GEMM's native Q8_0 form (per-32 f32 scales; there is no such
  W8 variant: code 104 was read as per-16 + mins). Google Gemma 4 12B: 9.1 tok/s (85 % of the measured bandwidth),
  17-20 with the drafter; KL 0.0206 vs its f32 reference, greedy = llama.cpp's. E4B: 20.9 tok/s, 26.8 with the
  drafter, prompts 788 tok/s.
- Google's GGUFs omit `attn_k_norm` on shared-KV layers (they compute no K): optional there.
- Google's chat template (2026-07-09) concatenates adjacent string literals (`"a " "b"`), which minja cannot parse:
  the server crashed at start. Adjacent literals inside `{{ }}` / `{% %}` are now joined first; renders identical
  to jinja2 (tests/test_chat_jinja.py, new tests/data/gemma4_google_template.jinja).
- **E2B / E4B images** (mmproj gemma4v, src/vision_gemma4v.h): a 16-layer ViT on the GPU (f16 weights, f32
  activations, clamped linears, RMS sandwich norms, NEOX 2-D rotary, unscaled attention, 3x3 pooling), 0.11-0.17 %
  from an f32 numpy reference (tools/gemma4v_ref.py); a 1120-token image in 3.2 s. The FFN uses gelu_tanh as
  Google's config says (llama.cpp falls back to gelu_quick for these files). In the language model, image rows take
  the per-layer embedding of token 0 and stay causal (llama.cpp: E2B / E4B are always causal).
- **E2B / E4B audio** (mmproj gemma4a, src/audio_gemma4a.h): log-mel and the subsampling convolutions on the CPU, the
  12-layer conformer on the GPU; 7.2 s of speech encoded in 226 ms and transcribed word for word.
- `--image-min-tokens` / `--image-max-tokens` for Gemma 4 12B (default 1120..2048: small screenshots were read
  wrongly at ~140 tokens); Gemma models with an image projector keep one image per prompt step (`min_chunk`).
- Model list: projectors named `*-mmproj.gguf` (Google) are no longer listed as models; Qwen3.6-35B-A3B GGUFs
  without GSQ experts are listed as unsupported (they failed to load). setup.ps1 / docs: g12 and e4b now download
  Google's QAT files plus Unsloth's drafters (copied next to the model).
- `run.exe` prints the weight bytes the GEMVs read per token and the resulting GB/s.

## 2026-10-12: Gemma 4 12B reads small text in images (1120..2048 tokens per image)

- Gemma 4 12B images now become 1120 tokens by default (`--image-min-tokens` default 1120, was 70 as in
  llama.cpp): its encoder turns each 48x48-pixel patch into one token with a single linear layer, so a ~550-pixel
  screenshot became ~140 tokens with ~10-pixel text per patch, and the model misread words ("BORENK", "AEROWR") and
  invented numbers in a spec table. Scaled up to 1089 tokens it read both test screenshots exactly (five 5x3
  tables, part numbers, prices). Large images may now use up to 2048 tokens (`--image-max-tokens` default 2048,
  was 1120, the most Gemma 4 was trained with); at 2025 tokens it still read the logo and the 4x-upscaled table
  screenshot exactly. Cost: 1120 tokens ~0.7 s encoder (CPU) + ~7 s prefill; 2025 tokens ~1.5 s + ~11 s.
- `--image-min-tokens` / `--image-max-tokens` apply to Gemma 4 12B too (before: Qwen only); scaling up no longer
  rounds past the maximum.
- Image placeholder ids in the prefix cache include the token count, so a saved conversation whose image was
  encoded with other limits is not matched against the new image tokens.

## 2026-10-12: Qwen3.5 9B keeps its decoded weights (short prompts ~4x sooner)

- `--ctx auto` now also chooses whether the decoded int8 (W8) weights stay on the GPU: they do when that still
  leaves at least min(128K, trained) tokens of context; otherwise they are decoded just in time, with the largest
  context, as before. Before, a fixed 9 GB budget decided, so the 9B (raw + W8 12.4 GB) decoded every matrix again
  on every request. Qwen3.5 9B profile (11,240-token prompt, 27.4 s): GEMMs 76 % at ~7.7 TOPS (the kernel's
  practical ceiling; its Q6_K per-16 matrices cost only ~4 % more than the per-32 ones, not worth requantizing),
  decoding 1.4 s. Resident: 83-token prompt 1.05 -> 0.27 s to the first token, 11K prompt 27.4 -> 26.6 s; context
  262,144 -> 217,088 at 18.3 GiB (17.0 GiB used). The other models keep their mode and context (4B, E4B resident;
  12B, 27B just in time: resident would not fit). An explicit `--budget` keeps the old rule.
- The `ctx-auto.txt` cache key now includes this choice (before, a `--budget` run reused a context computed for
  just-in-time weights: the 9B with resident weights then took 17.8 GiB).

## 2026-10-12: image input for Qwen3.5 4B (in Pi, the web UI and the API)

- Qwen3.5 4B reads images: its `mmproj-F16.gguf` (Qwen3-VL projector) is found next to the model and runs on the GPU
  (`src/vision_qwen3vl.h`). `--list-models` and Pi's model list mark it `text, image`; Pi's `read` tool, the web
  UI's Media button and OpenAI `image_url` parts work. Checked in real Pi: it read the text of a test image exactly
  ("HELLO 42") and described a photo correctly (ship, terminal and crane names), 2,320-token prompt with two images
  in 2.2 s.
- Encoder precision: with the language model's int8 GEMMs the encoder drifted 10-23 % from an f32 CPU reference on
  photos. A CPU emulation of each rounding (`vision_test`, `VT_EMU`) showed every layer's int8 activations
  contributing, and int8 attention a further 7-9 %. The encoder now runs f16 weights with f32 activations
  (`vit_gemm_f16`, 2.4 TFLOPS) and f32 attention: 0.2-0.5 % error, worst token cosine 0.999. A typical image
  (~250 tokens) encodes in ~0.4 s, a full-size 1080p screenshot (1008 tokens) in ~2.3 s; the first image after a
  start also uploads the encoder (~3 s).
- Language model: M-RoPE (Qwen3-VL interleaved sections 11/11/10): image tokens get (t, h, w) = start + (0, row,
  column) of the merged-token grid, and the text after an image runs max(grid w, h) positions on instead of the
  token count (`Engine::rope_shift`, per-row positions in `attn_prep`). Text-only output is bitwise unchanged
  (regression tests: same KL to 5 decimals).
- `/v1/models` also carries llama-server's own list: a `models` array whose `capabilities` hold `multimodal` for
  the models with an image encoder (next to OpenAI's `data` with `capabilities`, `input_modalities`, `type: vlm`),
  for clients that treat arcinfer as a llama.cpp server.
- Encoder attention: one query row per lane (SIMD8) with q and the accumulators in registers, K / V rows broadcast,
  4-key tiles, native_exp (`vit_attn_rows`): 1008-token image 1118 -> 588 ms (2.7 TFLOPS), 4096 tokens 36.1 ->
  15.3 s, 2048 ~11 -> 5.0 s; error against the f32 reference unchanged (0.2-0.6 %).
- Images are scaled to 256-2048 tokens (32x32 pixels each; `--image-min-tokens`, `--image-max-tokens`): at 1024 a
  6750-pixel logo was misread ("Logan" for "Logam", llama.cpp at 1024 too), at 2048 read right (~8 s to encode). Not yet:
  Qwen3.5 9B (its encoder is 1152 wide with 72-dim heads), Gemma 4 E4B, the 12B's encoder on the GPU.

## 2026-10-12: just-in-time weight decoding 2.7x faster (27B prompts); GEMM measured against its ceilings

- `decode_w8_sb` (the W8 conversion of models whose weights do not stay decoded, e.g. Qwen3.8 27B) loads each
  superblock into registers with vector loads and decodes it from there, and is now used for every type. The
  per-32-block kernel used for IQ3_S / IQ3_XXS / IQ4_XS read global memory byte by byte. 27B, 11,240-token prompt:
  those three types 5.8 -> 2.2 s (all decoding ~6.3 -> ~2.6 s, ~3.5 % of the time to first token). The prompt
  logits are bitwise identical (6,615-token prompt; `ARC_W8_SB=0` selects the old kernel). It also shortens model
  loads that decode at load time (E4B).
- `run --dump-logits` writes the prompt's logits too; `ARC_PROF_W8TYPE=1` profiles decoding per weight type.
- GEMM, measured in isolation (new `bench/gemm_w8_bench`, 27B shapes, bitwise comparison with the original):
  W8 with per-32 scales 8.3-8.6 TOPS, per-16 7.1; no-load ablation 9.7, no loads and no scale epilogue 11.0, against
  13.2 TOPS for signed DP4a with a broadcast operand (`peak_ss_bc_a`; the 7.1 TOPS `peak_dp4a_bcast` is the slower
  signed x unsigned form). Tried and not kept: a software-pipelined chunk loop (identical results, 7.0 TOPS: it
  loses the unrolled superblock), 32 tokens per sub-group (2.6 TOPS), 24-bit multiplies in the scale step (no
  change). In the model the 27B's GEMMs run at ~6.8 TOPS over a 100 s prompt: sustained heat, the per-16-scale
  types (about a fifth of its weights) and the kernels in between, not the kernel's shapes (all 8.3-8.4 alone).
  Experiment switches: `ARC_GEMM_DEFS`, `ARC_GEMM_TM`, `ARC_GEMM_WGY`, `ARC_GEMM_RG`.

## 2026-10-12: web UI follows the reply; Copy / Download on code blocks

- Every code block in a reply has **Copy** and **Download** buttons, above and below it (no scrolling to either end
  of a long block), and shows its language. Download names the file
  from the block's language (```` ```html ```` -> `code.html`, `py`, `js`, `ps1`, ...; `code.txt` when none). The
  copied / saved text is the code exactly as written (HTML special characters included).

- The chat view stays at the newest output whenever the thread grows (streamed text, the thinking box appearing or
  opened, images loading, tool calls, the stats line). Before, it only scrolled when a new piece of text arrived,
  and it pulled the view back down on every piece even when you had scrolled up to read. Now scrolling up (wheel,
  touch, scrollbar, PageUp / Home / arrow keys outside text boxes) pauses following at once and shows a "↓ Latest"
  button; scrolling back to the bottom, the button, or sending a message resumes it. Checked in the browser against
  the server: 0 px behind while following (also with the thinking box opened mid-stream), the view stays put while
  paused.

## 2026-10-12: Gemma 4 E4B generation 17 -> 21 tok/s (31 with the drafter), faster long prompts

Profile of one generated token (E4B IQ4_XS, short context): 56.5 ms of GPU time; the matrices (2.74 GB read per
token) would take ~37 ms at the 75 GB/s the iGPU reads. The IQ4_XS GEMVs ran at 44-65 GB/s, and ablations showed
why: without its memory loads the kernel still took 75-90 % of its time, i.e. decoding the 4-bit codebook indices
(byte-wise reads from local memory, a table lookup and shift / or packing per weight) cost as much as reading them.

- **`gemv_v` / `gemv_mt_v`** (`kernels/decode.cl`): each lane vector-loads its block's header and indices straight
  from global memory (no local-memory window, no barriers), and IQ4_XS's codebook comes from four local tables
  pre-shifted into byte 0..3, so a packed int8x4 word is four lookups ORed together (~8 -> ~4 instructions per
  weight). Used for IQ4_XS and for Q6_K output heads; Q4_K / Q5_K stay on `gemv` (their decoding is cheap, and there
  `gemv` was as fast or faster). gemv_bench, weights streaming from memory: E4B FFN gate 65 -> 70 GB/s, down 57 ->
  69, attention q 44 -> 67, output 48 -> 69; Q6_K output heads (E4B, Qwen3.5 4B) 66-67 -> 72 GB/s. Results are
  bit-identical to `gemv` / `gemv_mt` (new `GB_CMP` mode of `bench/gemv_bench`: every matrix compared bitwise).
- **`gemv_glu`**: Gemma's FFN for 1-8 tokens in one launch: gate and up rows of each 32-value block in the same
  work-group, then `gelu_tanh(gate) * up` quantized per 32 values in the epilogue (the same formulas as
  `swiglu_quant`), into a second activation buffer that the down projection reads. Replaces gemm_gate + gemm_up +
  swiglu_quant (two launches per layer fewer).
- `attn_prep_g` / `add_norm_quant`: vector loads for each lane's run of values (same arithmetic; ~5 % faster).
- Result, E4B on `bench/prompts/gemma-code160` (128 tokens, same session): 17.2 -> 21.2 tok/s without drafting,
  ~23 -> 30.7 tok/s with the assistant drafter (`run --mtp`; the drafter itself is unchanged); through the server
  with sampling (temperature 0.7) 30.9 tok/s. GPU time per token 56.5 -> 46.6 ms. The logits of 64 generation steps
  are bitwise identical to before (new `run --dump-logits F`; `ARC_NO_GEMV_V=1` / `ARC_NO_GLU=1` select the old
  path for such comparisons).
- **Long prompts**: Gemma 4 E4B's global layers (head size 512, 4 q heads per kv head) now use `attn_ps` (head dims
  split over sub-groups) like the 12B's, with 8 tokens per work-group: attn_q8 spilled 2 KB of registers there.
  `bench/attn_bench` at 13,921 tokens: 870 -> 619 ms per layer (same error against the CPU reference). E4B, 13,921-
  token prompt: time to first token 23.5 -> 21.8 s (591 -> 638 t/s). Not bit-identical (partial scores summed in
  another order): perplexity over 6 chunks of 4,096 tokens 88.44 -> 88.34.
- Prompt speed in context: E4B does ~27 % more arithmetic per token than Qwen3.5 4B (wider FFN, head-512 global
  attention); pp2048 789 t/s (Qwen 4B: 880).
- `tests/run_tests.ps1` (Qwen3.5 4B, Gemma 4 12B): all checks pass (Gemma KL 0.0188, greedy = llama.cpp, drafter
  and MTP output identical).

## 2026-10-12: pi-arc opens the model it started

- Fixed: pi-arc started Swift but Pi opened Gemma 4 E4B, shown with Swift's context (86,016). `last-model.txt`
  holds the path as `server.exe --list-models` wrote it, with mixed separators (`C:\Users\me/.cache/huggingface/
  hub\models--...`); arc-server compared it to the resolved path as text, found no model and started the file under
  a name made from the file name (`swift-...-iq3_s-mtp`). That name was not in `/v1/models`, whose first entry was
  then `gemma4-e4b`; pi-arc and arc-server took the first entry as the running model, so Pi was given
  `gemma4-e4b`, its list got the running context, and Pi's first request made the server switch to the E4B.
  - server: the scanned paths have one kind of separator; model files are compared as files
    (`std::filesystem::equivalent`), so the loaded model is listed first and marked `loaded` however its path was
    spelled, and a request for its catalog id does not load it again. A model file outside the scanned folders is
    listed first under the alias it was started with.
  - arc-server / pi-arc: model paths compared as full paths; the running model is the entry marked `loaded`.
- web_search on networks where DuckDuckGo is blocked (Indonesia): pi-unsloth-webtools searches DuckDuckGo and
  Startpage by default, and from here DuckDuckGo times out while Startpage returns nothing, so every search ended
  in "No results found" (or was cancelled while waiting). Yandex answers (5 results in ~1 s), so Pi's
  `settings.json` now has `"webSearch": {"engines": ["yandex", "startpage", "duckduckgo"]}` (Pi's `/search-config`
  changes it). Not an arcinfer change: the model's `web_search` calls were well formed.

## 2026-10-11 (late night): image input published per model

- `/v1/models` says per model whether it takes images, in the fields clients read: `capabilities` (llama-server /
  Ollama style: `completion`, `tools`, `vision`, `multimodal`), `architecture.input_modalities` (OpenRouter style),
  `type` (`vlm` / `llm`, LM Studio style), `modalities`; `/props` adds llama-server's `modalities` block. Before,
  only `/props` had a `vision` flag, for the loaded model only, so a client such as Unsloth Studio could not see
  that `gemma4-12b` reads images. Only models whose image encoder arcinfer runs are marked: Gemma 4 12B
  (`gemma4uv`). Qwen3.5 4B / 9B (`qwen3vl_merger`) and Gemma 4 E4B (`gemma4v`) ship an mmproj whose encoder is
  not implemented yet, so they stay text-only.

## 2026-10-11 (late night): Pi's edit streams; Pi's model list follows the disk

- Tool calls: array and object arguments now stream while they are generated, as the model's JSON text kept as
  written (`chat::json_prefix` cuts it where strict JSON can continue: never inside an escape, a number, a literal
  or a UTF-8 character). Before, only string arguments streamed, and Pi's `edit` (its whole change is the `edits`
  array) arrived in one piece at the end. Measured on Swift 1.5 Qwen3.8 27B: a 282-character edit arrived in 95
  pieces over 14 s instead of at once. If the model's JSON is not strict (e.g. a raw line break inside a string),
  the part already sent is closed and the parameter is repeated with the whole-response value; JSON readers take the
  last one, so Pi sees exactly what it saw before. `tools/tool_stream_test`: new cases (streams before the array is
  closed, non-strict JSON, numbers and literals), 20,000 random responses; regression tests pass.
- All of Pi's tools checked (pi-coding-agent 1.1 with pi-unsloth-webtools, pi-browser-control and MCP `codemode`:
  17 tools, their parameter types captured from Pi's own request): `tools/tool_stream_test` now calls each of them
  as Qwen writes it and checks the final types against the schema and that every long string / array value
  streams before its parameter closes (bash command, edit edits, write content, browser_type text, codemode
  code, ...); numbers and booleans arrive when their parameter closes. Boolean parameters also accept Python-style
  `True` / `False` (before: passed on as the string "True", which Pi's validation rejects).
- Live, Swift 1.5 Qwen3.8 27B with Pi's real tool list (one streamed request per tool, nothing executed): all 17
  tools give arguments that match their schema; long values stream (write: 282 pieces over 54 s, edit: 99 over
  19 s, codemode: 34, bash: 22). Before the boolean fix 5 of them failed: this model writes `True` for
  browser_navigate, browser_screenshot, browser_click, browser_type and browser_console_logs.
- `arc-server start` / `restart` bring Pi's model list up to date (they rescan the Hugging Face cache), also when
  the server is already running, so a deleted model leaves Pi's `/model` without running `pi-arc` (which no longer
  writes the list itself). Checked with a model deleted from a temporary cache.

## 2026-10-11 (late night): help for every option, README brought up to date

- Every command documents all of its options: `pi-arc --help` (aliases, `-c` / `-r`, setup, what happens to the
  caches when you continue a session, all server options and files), `arc-server help`, `bin\server.exe --help`,
  `bin\run.exe --help` (all 24 options; before: a one-line usage) and `setup.cmd --help` (new: steps, models,
  what it does not install). `--help` exits with 0 everywhere.
- `pi-arc --model <text>` (or `--model arcinfer/<id>`, as in Pi) chooses the model like `-Model`; before, it went
  to Pi next to the `--model` pi-arc sets.
- README: "Using Pi" rewritten (model choice and `/model`, contexts per GPU memory setting, `--ctx auto`, a table of
  what the server processes for each way of starting or continuing a session, Esc), the Pi commands, the server's
  model switching and `--list-models`, the prompt cache under "How it works", troubleshooting (compiler search,
  Windows slow after raising the GPU memory, the first session in a project), layout.

## 2026-10-11 (late night): system prompts kept per project folder

- Pi's saved system prompts are grouped by project folder, read from the `<cwd>` section of Pi's system prompt
  (`<folder>-<folder hash>-<prompt hash>-<tokens>.arcp`; the folder's spelling normalized, so `C:/X` and `c:\x` are
  one folder). Each folder keeps its 2 latest versions (an `AGENTS.md` edit or a Pi update replaces the oldest;
  two variants used in turn both stay), and other folders do not replace them; only the size limit below removes a
  folder's files, least recently used first. Before: the 8 used last per model,
  across all folders. Prompts without a folder (other clients): the 8 used last, as before. All models' files
  together stay under `--prefix-cache-gb` (default 8; least recently used first). Files saved before this are
  renamed into their folder's group the first time they are used (no reprocessing).
- Checked on an isolated 4B server: per-folder names, 2 versions per folder, other folders untouched, load after a
  restart (0.07 s), a pre-change file adopted, the size cap; and in real Pi (file named after its folder).
- Not changed: the session cache (whole conversations, `--session-cache`) stays separate and off by default. Only a
  request that starts with exactly the saved tokens can reuse a saved state, so a folder's cache can cover the
  system prompt its sessions share but not one session's conversation for another.

## 2026-10-11 (late night): one setup command

- `setup.cmd` (`setup.ps1`): checks the requirements (Windows, Intel Arc GPU, Visual Studio C++ compiler, Python,
  Pi; says where to get what is missing), builds `server.exe` / `run.exe` / `clinfo.exe` when missing or older than
  the sources (compiler output in `%LOCALAPPDATA%\arcinfer\build.log`, shown on failure), checks the GPU with
  `clinfo` (OpenCL 3.0, DP4a, sub-groups, GPU memory), downloads the chosen models (`-Models 27b,4b`, `all`, `none`,
  or asked; Gemma with drafter and projector; the 35B downloaded and converted, with the 4B for its tokenizer),
  installs `huggingface_hub` / `numpy` with pip after asking, connects Pi, and runs the regression tests with
  `-Tests`. Safe to run again: done steps are skipped.
- `build.ps1` finds the compiler with `vswhere` (any Visual Studio / Build Tools with `vcvars64.bat`;
  `$env:ARC_VCVARS` overrides) instead of a fixed path, and builds with `_CRT_SECURE_NO_WARNINGS` (no
  getenv / fopen deprecation notes).
- `install-pi.ps1` writes Pi's model list from `server.exe --list-models` (every model found, each with its
  context) instead of a fixed list with fixed contexts; `scripts\pi-models.ps1` holds that code for it and `pi-arc`.

## 2026-10-11 (night): Esc in Pi stops the server right away

- A client that closes its connection (Pi's Esc) cancels the request: the server polls the socket (no write needed)
  after every layer of prompt processing and before every generated token. Before, a prompt kept being processed to
  its end (a 7K-token prompt on the 27B kept the GPU ~73 s after Esc; 50K tokens: ~10 minutes) and the next request
  waited for it.
- Prompt processing stops within one layer (`Engine::cancel`): the block it was in (layer-major segment or chunk) is
  undone (DeltaNet states from a host copy taken at the block start, 27B: 157 MB, ~20 ms per block; Gemma moves back
  in its sliding-window rings, else resets), complete blocks stay cached.
- Pi drops an aborted (or failed) reply and sends the conversation without it, so the next request diverges inside
  the `<|im_start|>assistant` header. The resume snapshot was right after that header: the whole conversation after
  the system prompt was processed again (2.5K-token conversation: 21.5 s). It is now taken at the header's start:
  5.0 s.
- Checked in real Pi on the 27B (a scratch project with Pi's normal tools and extensions plus a local MCP server):
  parallel tool calls, edits, files with backslashes, quotes, braces and Unicode, bash with long output and failing
  commands, MCP through codemode, git, follow-ups: 27 replies, 21 tool calls, all arguments valid, no errors from the
  server or Pi; files on disk exactly as asked. The regression tests' server runs with `--no-prefix-cache`.
- `pi-arc --ctx auto` / `arc-server -Ctx auto` work (pi-arc failed on a non-number): auto, and a server running a
  fixed context is restarted with auto; without the option a running server keeps its context. Bad values give a
  message instead of a PowerShell error.
- Help: `server.exe --help` (exit code 0) lists every option with its default; `pi-arc --help` has a WORKING IN PI
  section (Esc, new sessions, project folders); `arc-server help` says where saved system prompts are.
- `docs/review.html` and `docs/review_notes.md` (now docs/archive/) updated: stages 12–14, contexts at 17.4 GiB, Pi on the 27B with
  today's measurements and the use-case run, two new limits.

## 2026-10-11 (evening): system prompt kept on disk

- The state after a client's system prompt is saved to `%LOCALAPPDATA%\arcinfer\prefix\<model>` the first time it is
  processed and loaded when a request starts with it while the server holds something else (server start, model
  switch, a conversation with another system prompt). Pi's system prompt contains the project folder and its
  `AGENTS.md`, so each project has its own file; the 8 used last per model are kept. `--no-prefix-cache` turns it
  off. Measured in Pi, Qwen3.8 27B, 4146-token system prompt: first new session after a server restart 40.2 s ->
  5.3 s to the first token (file loaded in 0.53 s); the file (290 MB) is written in the background.
- The state is saved without the duplicate host snapshot at the same position (157 MB less per file; the snapshot
  is taken again after loading). Works for Gemma 4 too (its system turn is recognised the same way).
- `--ctx auto` measures with a dry load that only adds up the GPU allocations (seconds, nothing uploaded), so
  `--list-models` reports every model's context and Pi's `/model` list gets each model's own context.

## 2026-10-11 (later): one command for Pi, models switched from inside Pi

- `pi-arc` starts Pi right away on the model the server runs, else the one used last (`--pick`: choose from the
  list of every supported GGUF model in the Hugging Face cache, by full name). It registers all those models in
  Pi's `models.json`, so Pi's `/model` lists them.
- The server switches models itself: a request whose `model` is another model of `server.exe --list-models` (its
  id, `org/repo` name or file) unloads the current one and loads that one (`--no-swap` turns this off).
  `/v1/models` lists them all, the loaded one first. E4B <-> 4B: 4-9 s; GPU memory is the same after any number of
  switches (the engine now releases its buffers, kernels and context: the `--ctx auto` measuring load used to stay
  in memory).
- `--ctx auto` for every model by default (the context follows the GPU memory setting); `--list-models` reports each
  model's id, shortcut, image support (a `gemma4uv` projector: Gemma 4 12B, not the E4B's) and computed context.
- Disk session cache off by default (`pi-arc --session-cache 16`); `pi-arc --help` documents every option.

## 2026-10-11: session cache (switching between conversations), 27B back to 4096-token chunks

### Saved conversations

The server held one conversation: a new Pi session, or going back to an older one, processed the whole prompt again
(Qwen also its ~4.2K-token system prompt: the DeltaNet state can only be resumed where a snapshot was taken). Pi on
the 27B, two sessions taking turns: every first request had 0 tokens cached, 36 s for a new session, 55–64 s to go
back to a 6.6–7.5K-token one.

- `Engine::save_state` / `load_state`: the KV cache (int8 blocks and scales of every layer with its own cache; for
  sliding-window layers only the window before the resume point), the MTP block's cache and hidden state, the DeltaNet
  states and snapshots; copied in batches of up to 256 MB.
- Server: when a request does not continue the held conversation's last prompt, that conversation is saved to
  `%LOCALAPPDATA%\arcinfer\sessions\<model>-<size>\` first; a saved conversation that shares at least 512 more
  reusable tokens with the request is loaded. `--session-cache-gb` (default 0 = off; least recently used first),
  `--session-dir`; `GET /sessions` (size), `DELETE /sessions` (clear); `arc-server status` / `arc-server clear-cache`.
- Qwen: a recurrent-state snapshot at the end of the system prompt, kept in host memory (no GPU memory), so a new
  conversation with the same system prompt processes only its own messages.
- Answers are identical with and without other conversations in between (greedy; Qwen3.5 4B, Gemma 4 E4B, Gemma 4
  12B). Saved size for ~4.5K tokens: 4B 225 MB, E4B 49 MB, 12B 215 MB; saving 0.1–0.9 s, loading 0.05–0.3 s.

### Qwen3.8 27B: 86,016 tokens with 4096-token chunks

Default changed from 118,784 tokens with 1024-token chunks to 86,016 with 4096 (the largest that fits with the same
~1 GiB left for other apps). `arc-server start -Model 27b -Ctx 118784 -Chunk 1024` (or `pi-arc --ctx 118784
--chunk 1024`) gives the larger context; Pi's configured window stays 86,016.

## 2026-10-10 (evening): prompts at 90 % of the context

GPU memory is now 18.34 GiB (`CL_DEVICE_GLOBAL_MEM_SIZE`, was 16.5); the desktop and other apps hold 0.7–1.0 GiB of
it, and the driver ~0.22 GiB on top of the server's own buffers (Windows counter `GPU Process Memory(*)\Shared Usage`).

### Measured: a prompt of 90 % of the default context (arcinfer's own sources, `arc-server` settings)

| model | context | prompt | prompt processing | time to first token | generation after it |
|---|---|---|---|---|---|
| Qwen3.8 27B (before) | 90,112 | 81,100 | 82.5 t/s | 16.4 min | 4.92 t/s (MTP) |
| Qwen3.8 27B (2048-token chunks) | 106,496 | 95,846 | 78.6 t/s | 20.3 min | 4.55 t/s (MTP) |
| Qwen3.8 27B (now, 1024-token chunks) | 118,784 | 106,906 | 72.8 t/s | 24.5 min | 4.59 t/s (MTP) |
| Gemma 4 E4B | 131,072 | 117,965 | 165.5 t/s | 11.9 min | 11.6 t/s (drafter) |
| Qwen3.5 4B | 262,144 | 235,929 | 144 t/s | 27.3 min | 9.0 t/s (MTP) |

(35B-A3B, 9B and Gemma 4 12B were not run to the end.) At that depth attention takes a large share: ~38 % of the
27B's time at 81K, about half of the 4B's at 236K, most of the Gemma 4 12B's.

### Prompt attention (`bench/attn_bench`, new `ATT_P0`: only the queries from a position on, i.e. one chunk at depth)

- `attn_q8` at depth: 1.9–2.2 T MAC/s for the Qwen shapes (head 256), 0.87–0.91 for Gemma's head-512 global layers,
  which spill 1984 bytes of registers. Ablations on the 27B shape (compile-time `ABL_NOQK` / `ABL_NOSM` / `ABL_NOPV`):
  tile loads + barriers 19 %, QK 24 %, P·V with P packing 33 %, online softmax 19 %.
- **Gemma 4 12B global layers now use `attn_ps`** (new, `kernels/ops.cl`): the head dims are split over sub-groups
  (4 groups of 128 dims, 4 q heads each, 2 tokens per work-group) and partial scores are summed through SLM, so no
  registers spill: 1.25–1.4x faster at every prompt length (0.96 → 1.34 T MAC/s at 4K, 0.87 → 1.10 at 102K). KL vs
  the f32 reference on 160 code positions 0.0174 (0.0180 before); greedy output and drafter tests unchanged.
  `ARC_NO_PS=1` switches back. With one scale per row (Qwen) `attn_ps` is bit-identical to `attn_q8`
  (`ATT_CMP=1` checks it), but it is slower there (1.0–1.6 T MAC/s): splitting the dims doubles the per-row softmax
  and P packing work, which already take ~40 %. E4B's global layers (4 q heads) gain nothing either.
- Tried and removed: 32-key tiles (one max reduction and barrier pair per 32 keys): the extra live state spills
  (2–3 KB) and is 14–20 % slower for Qwen; 4 heads per sub-group: +19 % on the 4B / 35B shapes but spills.
  `attn_q8` is at the register limit of the Qwen shapes.

### Qwen3.8 27B: context 90,112 → 118,784 tokens

- The Qwen KV scale buffers were sized for one scale per 64 dims while Qwen uses one per row: 1,632 bytes per token
  less (36,992 → 35,360; −0.14 GiB at 90K).
- `arc-server` runs the 27B with 1024-token prompt chunks (`--chunk 1024`), which frees 1.1 GiB of activation
  buffers (GPU buffers at 106,496 tokens: 4096-token chunks ~17.5, 2048 16.78, 1024 16.41 GiB). Same outputs. Time
  to first token, 11,240 tokens: 90.1 s (4096), 91.9 s (2048), 94.0 s (1024); 33,937 tokens: 321.6 s (2048),
  322.2 s (1024); at 96K–107K the per-16K-segment times equal those of 4096-token chunks.
- At 118,784 the server holds 17.29 GiB; with 0.88 GiB of other apps that is 18.17 of 18.34 GiB. The previous
  90,112 with 4096-token chunks needs 17.62 GiB: over the budget once other apps hold more than ~0.7 GiB (paging).
- Not done: switching the chunk size during a session (2048 while the context is small). The KV cache is
  allocated for the whole context at load, so the memory for larger chunk buffers is never free; it would need a
  growing KV cache, for ~2 s on an 11K-token first prompt and nothing on long prompts or ordinary agent turns.

### Streaming

- A character whose UTF-8 bytes are split over two tokens (e.g. emoji) was streamed in two invalid pieces (the
  client shows two replacement characters). `emit` now holds back an incomplete character at the end of the text
  (Qwen and Gemma parsers); `tools/tool_stream_test` checks every delta byte by byte.

## 2026-10-10 (later): streamed tool calls

- **Tool calls are streamed while they are generated** (`tools/server.cpp`, `chat::partial_tool_calls`,
  `gemma4::partial_tool_calls`). Before, the server held a tool call back until the end of the response and sent it
  in one piece, so Pi showed nothing while the model wrote a file through its `write` tool (one call: 2,368 tokens,
  6 minutes without output). Now the first delta of a call carries its id and name, and the following ones the
  arguments JSON piece by piece, as OpenAI and llama-server send them. String parameters grow token by token;
  Qwen's non-string parameters (e.g. the `edits` array of Pi's `edit` tool) appear once the parameter is closed,
  because the model's JSON may still turn out invalid (it then becomes a string, as before). Gemma 4 delimits its
  strings, so all of its arguments stream. The pieces of a call only grow and end as the arguments of the
  whole-response parser, so non-streaming responses are unchanged.
- `tools/tool_stream_test.cpp` checks this for every prefix of hand-written and 3,000 random responses (both
  formats); part of `tests/run_tests.ps1`. The web UI and `tests/test_tools.py` join the pieces by index.
- Gemma 4 tool-call parser: an array closed by a stray `}` (`call:x{a:[1}`) made it loop forever while allocating
  memory (the server hung). A stray `}` now ends the array.
- `run.exe` warns when other GPU processes (`server.exe`, `run.exe`, `llama-*`) are running. This explains the
  "`run.exe --mtp` on E4B at 3–5 tok/s" below: alone, `run.exe --mtp` gives 23 tok/s on E4B (17 without drafting;
  20.5 after a 13.9K-token prompt), but with a 27B server processing a prompt at the same time it gave 8 tok/s
  (5–6 without drafting), and with an idle 27B server loaded the E4B load took 13–43 s instead of 2 s. The iGPU
  and its 16.5 GiB are shared between processes.

## 2026-10-10: token generation for Gemma 4 (12B and E4B)

Goal: faster generation on Gemma 4 12B and E4B, at short and long context, without changing the model's output.
The Gemma 4 assistant drafter (speculative decoding) was deliberately left as it was.

### Where the time went (Gemma 4 12B, before)

Profile of a single-token step after a short prompt: 134 ms per token (7.45 tok/s), 139–144 ms of GPU time when
profiled. The 7.4 GB of weights would take ~100 ms at the 72–77 GB/s the iGPU reads from memory, so ~30 % was
overhead:

| part | ms / token | note |
|---|---|---|
| FFN GEMVs (gate, up, down) | 78 | 61–71 GB/s, close to the limit |
| lm_head (262,144 × 3840, Q6_K, 826 MB) | 21–28 | **30 GB/s**, half the speed of the same kernel in isolation |
| attention GEMVs (q, k, v, o) | 24 | 50–60 GB/s |
| small kernels (attn_prep, swiglu, norms, attention, combine) | ~11 | attn_prep alone 55 µs per layer |

At 13.9K context the step grew to 157 ms: decode attention of the 8 global layers (head size 512, 16 q heads on
one kv head) ran at only ~20 GB/s, 0.7 ms per layer.

### Changes

1. **Output head allocated first** (`src/engine.h`, `load_weights`). The GPU read the 826 MB head buffer at about
   half speed when it was allocated after gigabytes of other weights (the same kernel on the same data took
   13–14 ms in isolation, 21–28 ms in the engine; loading the head first gives 14 ms in the engine). Applies to
   every architecture.
2. **Gemma's final logit soft-capping on the GPU.** `c·tanh(x/c)` is now the epilogue of the head's GEMV
   (`-DSOFTCAP`, `kernels/decode.cl`), instead of a host loop over 262K logits per token whose OpenMP worker
   threads kept spinning between tokens.
3. **RoPE table once per pass** (`rope_table` kernel). `attn_prep_g` used to compute `pow`, `sin` and `cos` for
   every element in all 48 layers; the angles are now computed once per pass for the sliding and the global layer
   type and read from a table (bit-identical values). attn_prep: 3.4 → 1.9 ms per token.
4. **Fused residual + norm + quantization** (`add_norm_quant`). For single-token and verification passes, the
   post-attention / post-FFN sandwich norm with the residual add and the next block's input norm and int8
   quantization run as one kernel (2 launches fewer per layer). Output is identical to the two separate kernels.
5. **Quantization of the SwiGLU and attention outputs** for single-token passes: one sub-group per 256-value block
   of the valid rows instead of one work-group per padded row (swiglu_quant 2.4 → 1.2 ms per token).
6. **Decode attention**:
   - New kernel `attn_dsplit` (`kernels/ops.cl`): a work-group covers one key split of one kv head; its sub-groups
     split the head dimensions (64 dims each) and the q heads, so every byte of the K/V cache is read once for all
     heads and up to 4 query rows. Partial scores of the dim groups are summed through local memory; softmax and
     P·V use the same int8 math as before. Used for Gemma 4 (not for the drafter's own layers, not for Qwen).
   - K/V tiles in local memory are single-buffered (`attn_q8`, `attn_split`): twice the work-groups per Xe-core.
     Decode with head size 512: 1.5× faster; prompt processing unchanged or up to 4 % faster.
   - New `attn_combine`: the lanes of a sub-group take the per-split maxima 16 at a time, no data-dependent
     branch; 3–5× faster (26 → 5 µs for 33 splits).

   Attention + combine per layer, `bench/attn_dec_bench` at 14K context:

   | layer type | rows | before | after |
   |---|---|---|---|
   | 12B global (16 q / 1 kv, head 512) | 1 | 742 µs | 310 µs |
   | 12B global, 36K context | 1 | 1,818 µs | 803 µs |
   | 12B sliding (window 1024) | 1 | 125–220 µs | 73 µs |
   | E4B global (8 q / 2 kv, head 512) | 1 | 1,008 µs | 420 µs |
   | E4B sliding (window 512) | 1 | 56 µs | 17 µs |
   | 12B global, verification | 2 | 779 µs | 520 µs |
   | E4B global, verification | 2 | 944 µs | 465 µs |
   | 12B sliding, verification | 2 | 142 µs | 78 µs |

   At 14K context this saves about 6–8 ms per token on the 12B (8 global + 40 sliding layers, plus attn_prep) and
   ~5.5 ms on the E4B (7 global + 35 sliding layers); at 36K the 12B's global layers alone save 8 ms.
7. **Server sampler** (`tools/server.cpp`): the top-k candidates are taken from the tokens within 40 temperatures
   of the best logit (one pass) instead of building and partially sorting a pair for each of the 262K vocabulary
   entries per token; when fewer than k candidates are left after penalties it falls back to the full vocabulary,
   so the sampled distribution is exactly the same.

### Results

- Gemma 4 12B, short context, 128 tokens, same session A/B: 7.79–7.83 → 8.09–8.16 tok/s without drafting
  (a later run on a warm laptop: 7.66); with the assistant drafter 8.74 tok/s (50 % of drafts accepted,
  1.94 tokens per main-model pass). GPU time per token 139–144 → 123 ms (profiled).
- Gemma 4 E4B, short context: 16.9 tok/s without drafting; with the drafter through the server 17.4–18.1 tok/s on a
  story prompt (40 % of drafts accepted; code accepts more).
- Long-context end-to-end generation after these changes was not measured; the per-layer attention savings above
  were. Before them, the 12B generated 6.4 tok/s after a 13.9K-token prompt (llama.cpp Vulkan 5.5).
- At short context the end-to-end gain is within the laptop's run-to-run variation (±4 %); the gains are at long
  context, where attention grows.

### Accuracy

- `tests/run_tests.ps1` with Qwen3.5-4B and Gemma 4 12B: all 10 checks pass (Gemma KL vs the f32 reference
  0.02927, greedy equal to llama.cpp, drafter output identical, Qwen MTP output identical).
- New `run --kl-decode`: feeds a prompt token by token through the generation path and reports the KL against the
  f32 reference at every position. Gemma 4 12B, 160 code positions: 0.013 (0.020 before the changes); 120 raw-text
  positions: 0.029 (0.034 before). A 20-position test swung between 0.009 and 0.043 between builds whose
  intermediate values differ by at most 1e-6: one near-tied position flips. Use the longer references.

### Tried and not kept

- IQ4_XS (E4B) weight decoding: a 256-entry byte-pair lookup table was slower (local-memory bank conflicts,
  66 → 55 GB/s), and register shuffles with a variable lane index halved the speed. The 16-byte table stays.
- `attn_dsplit` for Qwen: up to 2× faster for 2-row verification, but the verification rows are then no longer
  bit-identical to single-token decoding, which the MTP regression test requires. Qwen keeps the previous kernels.
- Drafter changes (native Q4_0 GEMV for the drafter's weights, which halves their bytes; draft argmax on the GPU
  instead of reading 1 MB of logits per draft): written and tested, then reverted on request. The drafter runs as
  before.

### New tools

- `bench/gemv_bench.cpp`: the single-token GEMV on real tensors of a GGUF (all layers in turn, so weights stream
  from memory), checked against the CPU dequantization; variants via build defines or another kernel file.
- `bench/attn_dec_bench.cpp`: decode attention (attn_dsplit / attn_split / attn_decode + combine) for 1–4 query
  rows at any head shape and window, checked against a CPU reference.
- `run --kl-decode` (with `--kl-ref` of every position) and `ARC_PROF_LAYERS=n` (profile per layer group).

### Known issues

- ~~`bin\run.exe --mtp` on Gemma 4 E4B generates at only 3–5 tok/s~~: not reproducible on an otherwise idle GPU
  (23 tok/s); the slow runs shared the GPU with other processes (see the entry above).
- E4B still has headroom: its q/o projections (IQ4_XS, K = 2560) run at 38–49 GB/s, the per-layer-embedding
  projections (256 rows, or K = 256) at 11–22 GB/s, and its small kernels take ~7 ms per token.
- Gemma 4 audio input (12B): the code compiles and is loaded when the mmproj has the audio projector; not yet tested.

## Earlier in this round (before 2026-10-10)

- Qwen3.6-35B-A3B, ISTA-DASLab GSQ 2-bit: `tools/convert_gsq.py` (safetensors → GGUF with the exact 2-bit expert
  codes) and `kernels/moe.cl` (routing, counting sort, DP4a GEMMs on the 2-bit codes). 11K-token prompt in ~14 s,
  ~15 tok/s generation, KL vs f32 0.0008.
- `gelu_tanh` attribute fix in `kernels/ops.cl`.
- Gemma 4 at long context: attention for head size 512 and windows, the assistant drafters, a persistent kernel
  binary cache (`%LOCALAPPDATA%\arcinfer\kernels`), lm_head tuning.
- Gemma 4 accuracy: one int8 scale per 64 dims for q/K/V (was one per row): KL on code 0.23 → 0.018.
- Segmented layer-major prefill (16K tokens) and the largest contexts that fit 16.5 GiB for every model
  (`scripts/arc-server.ps1`, `scripts/install-pi.ps1`).
- Review page `docs/review.html` (EN/ID) with `bench/review_data.py`; prompt benchmark `bench/prompt_bench.py`.

