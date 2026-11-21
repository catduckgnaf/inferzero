# Contributing to inferzero

inferzero is far from what Intel's GPUs can do. On every card we have measured, prompt processing and generation
leave a large share of the hardware unused, and the cards and setups we have never seen are the biggest unknowns.
Help of two kinds moves it most: **measurements from your hardware** and **code**.

## Where the speed is still missing

Measured, not guessed (details in [docs/xe2-tuning.md](docs/xe2-tuning.md) and [docs/before-after.md](docs/before-after.md)):

| Hardware | What is measured | Where it could be |
|---|---|---|
| Arc 140T (Arrow Lake-H iGPU) | the prompt GEMM at ~33 % of the XMX int8 peak | ~85 % with its operands in L1 (`bin\xw_peak --probe`) |
| Arc Pro B70 | the prompt GEMMs at ~62 % of the int8 peak; generation at ~62 % of the memory bandwidth | int8 GEMMs at 80 %+ are known on this card; int4 GEMVs reach 89-97 % of bandwidth |
| Arc B580 (12 GB) | generation at 61-70 % of bandwidth on the 4B / 9B; Flash Next limited by PCIe copies of its experts | more of the bandwidth; a smaller expert cache waste; fewer copies |
| Several GPUs | on Windows the sums go through host memory; a slow link (PCIe x4) halves a split model's prompt speed | GPU-to-GPU sums where the driver allows them (done on Linux); a layer split for slow links |
| Arc A770 / A750 / A580 (Alchemist) | first runs only (issue #35) | almost everything is untuned |
| Core Ultra iGPUs (Meteor Lake, Lunar Lake, Panther Lake) | tuned on one 125H laptop; the B390 measured by one user | every other laptop |

The open work, in order, is in [QUEUE.md](QUEUE.md) ("Open now" at the top) and
[docs/arc-plan-2026-11-17.md](docs/arc-plan-2026-11-17.md).

## 1. Measure on your hardware (no code needed)

This is the most useful thing most people can do: we can only tune for a card someone has measured.

1. Download the latest release ([Windows zip or Linux tar.gz](https://github.com/rizagp/inferzero/releases/latest))
   and a model with the launcher (or `setup.cmd`).
2. Run the benchmark script. Windows: `run-tests.cmd quick` (about an hour; `run-tests.cmd` alone runs overnight). It
   writes one zip to the Desktop with the paths' user names taken out. Linux: `bench/website_bench.sh`.
   A single model by hand: `bin\runz.exe <model.gguf> --bench-pp 4096 --n 128 --prof`.
3. Open an issue with the **Hardware results** template and attach the zip or paste the output: the GPU, the driver
   version, the OS, the PCIe slot, the release.

Results for discrete cards go in [issue #4](https://github.com/rizagp/inferzero/issues/4) or a new issue of their
own. When a switch is asked for on an issue, a run with it and one without, alternated, is worth more than a single
number.

## 2. Code

The engine is C++20 on Level Zero, with kernels in OpenCL C. Start with [AGENTS.md](AGENTS.md): the rules every
change follows, and a map of the source.

**Build.** Windows: Visual Studio Build Tools (C++), then `.\build.ps1 zero\runz.cpp zero\serverz.cpp`. Linux: g++ 13
or newer, then `./build.sh` ([docs/install.md](docs/install.md)). Rebuild both runz and serverz after any change to
the engine or the kernels.

**The rules a change is held to:**
- **Accuracy.** Every model's next-token KL against the f32 reference (`tools/ref_forward.cpp`, `runz --kl-ref`) stays
  at or below 0.015. A change meant to give the same results is checked bit for bit: `runz --dump-logits` before and
  after, the files identical.
- **Discrete-card code** goes in `zero/xe2/` and kernel files of its own; multi-GPU code in `zero/mgpu/`. The iGPU
  path must stay bit-identical.
- **Every speed or accuracy change** gets a row in [docs/before-after.md](docs/before-after.md) (measured the same way
  before and after) and a CHANGELOG entry. A new switch starts opt-in until a card measures it faster.
- **Memory:** extra VRAM or RAM only opt-in or from the room left free at load, and the load log says what it took.
- **Nothing a user runs may need Python.**

**Good places to start:**
- the issues labelled [good first issue](https://github.com/rizagp/inferzero/labels/good%20first%20issue) and
  [help wanted](https://github.com/rizagp/inferzero/labels/help%20wanted);
- a kernel on a card you own: `bench/` has micro-benchmarks for the GEMMs (`xw_peak`; `gpu_sweep.ps1` / `.sh` sweeps every setting and tile with a model), the GEMVs
  (`gemv_bench`), attention (`attn_bench`) and memory (`bw_bench`);
- Linux: the port works headless (`serverz`); packaging for more distributions, and the Docker / Kubernetes files in
  the repository, are untested by us;
- the web UI (`webui/index.html`, one file) and the launchers (`scripts/`).

**Pull requests.** Small and focused, with the numbers before and after from your hardware in the description. We
merge after checking the results on the hardware we have, and rewrite the CHANGELOG and docs text in the project's own
format, so don't spend time polishing those; the credit stays yours in the CHANGELOG and the commit.

## Talk

Open an issue for anything: a question, a result, an idea. Measurements that disagree with the docs are especially
welcome; they are how most of the tuning here was found.
