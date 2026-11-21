// Tuning switches for GPUs with XMX (Xe2 / Xe3: Arc B580, Pro B60), zero/xe2/: A/B settings that only the owners of
// these cards can measure (docs/xe2-tuning.md). A part of class Engine (zero/engine.h), included inside the class
// body like zero/mgpu/engine_tp_shard.h: not a header of its own.
// On for every GPU that offers XMX: discrete cards (one or the multi-GPU shards) and iGPUs with it (Lunar Lake, Panther
// Lake, Arrow Lake H). Off on the others, as Xe-LPG: there every kernel is built and launched exactly as without this
// file (xe2_ stays false, so the GEMM keeps 256 registers per thread, one column group per work-group, the GEMV its
// default register file). Only discrete B60s have measured these defaults. ARC_XMX_TRY turns it on anywhere, for the
// tests of the switches themselves.

  bool xe2_ = false;  // (set before the XMX check: the GPU offers XMX, or ARC_XMX_TRY)
  // ---- the Arc 140T preset (iGPUs with 8-lane DPAS, weights decoded just in time; issue #5, 2026-11-08). A 140T's 27B
  // prompt with XMX + H8 + 32 tokens per sub-group + 2 column groups + the default register file, against DP4a:
  // 512 tokens 154-159 / 154, 2K 221 / 180, 8K 230-232 / 184-186 tok/s; KL against DP4a 0.00154. Per GEMM: chunks
  // of 512+ tokens (a multiple of 32) on weights decoded just in time take it; shorter chunks and resident weights the
  // DP4a default (256 registers: DP4a spills with 128). The start-time check runs its kernels too. ARC_140T=0: off;
  // ARC_140T=1 with ARC_XMX_TRY=2: on the 125H with DP4a standing in for DPAS (tests)
  bool p140_ = false;
  // (p140_res_: the weights resident in W8 for the chunks below kP140Min, and those of kP140Min+ still decoded just in
  //  time from the raw rows into the staging (an A770's 4B: 128 tokens 1,197 just in time against 2,381 resident, 4K
  //  5,028 against 3,325). ARC_140T_RES=1, engine_setup.h plan_weights)
  bool p140_res_ = false;
  // (from kP140Min tokens XMX takes a chunk; with p140_res_ from kP140MinRes, where the resident DP4a GEMM stops being
  //  faster: an A770, 3 runs each, XMX against resident DP4a, 4B 640 tokens 3,101 / 3,208, 768 3,395 / 3,282; 9B 640
  //  1,757 / 1,847, 768 2,035 / 1,872. ARC_140T_MIN=n: from n)
  static constexpr int kP140Min = 512, kP140MinRes = 768;
  static int p140_min(bool res) {
    static const int e = getenv("ARC_140T_MIN") ? atoi(getenv("ARC_140T_MIN")) : 0;
    return e > 0 ? e : res ? kP140MinRes : kP140Min;
  }
  bool p140_take(const GpuWeight& w, int Mp) const {
    return p140_ && (!w.wq || p140_res_) && w.gemm_qtype >= 100 && w.gemm_qtype <= 103 && Mp >= p140_min(p140_res_) && Mp % 32 == 0 && w.N % 32 == 0;
  }
  // ---- H8 + 4 column groups by default on discrete cards with 16-lane XMX, for the Qwen dense models (issue #4,
  // 2026-11-08, a Pro B60, 27B, package d980970: 512 tokens 615 -> 677 tok/s, 4K 766 -> 1,024, 16K 706 -> 937; KL
  // against the default path 0.00154, max 0.064). Not for MoE models (the 35B: H8 -1 %), Gemma or Flash Next (never
  // measured with H8 on a B60), nor two-pass models. Off again if XMX does not pass its check. ARC_H8=0: off
  bool h8_auto_ = false;
  // (iGPUs with 16-lane XMX too since 2026-11-17, Panther Lake's Arc B390 (gee456, issue #1, release .07): the 27B's 4K
  //  prompt 258-268 -> 546-560 tok/s with H8 on the wide-tile GEMM (gemm_xw, which follows h8_auto_), KL against the
  //  plain XMX path 0.00175 (with attn_fx). There the 4-wide work-groups stay 1 (no gain measured), the decoded weights
  //  just in time (RAM is the GPU's memory), the GEMV rules off (generation 5.81 -> 4.40 tok/s with them). The 125H
  //  (no XMX) and the 140T (8 lanes, its own preset) never take it)
  bool h8_auto_wanted() const {
    if (getenv("ARC_H8") && atoi(getenv("ARC_H8")) == 0) return false;
    return cl_.xmx && !cl_.sg8 && !hp.moe && !hp.gemma && !hp.hc && !hp.qwen3;
  }
  // the H8 GEMM's column groups: 4 with h8_auto_ on a discrete card (unless ARC_GEMM_CG is set), else xe2_cg_for (the
  // B390's measured H8 run had 1: iGPUs keep it; audit 2026-11-17)
  int h8_cg_for(int N) const {
    static const bool cg_env = getenv("ARC_GEMM_CG") != nullptr;
    return h8_auto_ && !zrt::integrated() && !cg_env && (N / 16) % 4 == 0 ? 4 : xe2_cg_for(N);
  }
  bool p140_wanted() const {  // (w8_resident: the weights resident in W8, a member of Engine)
    const char* e = getenv("ARC_140T");
    if (e && atoi(e) == 0) return false;
    const bool test = e && atoi(e) == 1 && getenv("ARC_XMX_TRY");
    // (also discrete cards with 8-lane DPAS, the Alchemist A-series (A770, A750, A580), by default since 2026-11-16:
    //  the same XMX as the 140T, whose Xe-LPG Plus cores are Alchemist's with XMX. Before, such a card took the plain
    //  gemm_x8 or DP4a, whichever won the start-time speed check (engine_setup.h; on a 140T gemm_x8 was 1.6x slower
    //  than DP4a, issue #5), never H8 on gemm_xw8. Not yet measured on an A-series card. Two things differ from the
    //  140T: the speed check is skipped (as there), and h8_splits keeps the discrete cards' 34 superblocks a piece.
    //  The start-time checks still turn it off if its kernels differ or do not build)
    // (the models it was measured on: Qwen dense, as h8_auto_; not MoE, Gemma, Flash Next or Qwen3 dense, and not with
    //  ARC_H8_SB=1, whose staged form it does not track. Self audit 2026-11-08)
    return (cl_.xmx || test) && cl_.sg8 && (!w8_resident || p140_res_) && !hp.moe && !hp.gemma && !hp.hc && !hp.qwen3 && h8_row();
  }
  // (the preset would apply but for the resident weights: the reason the log gives, hansong80's A770, issue #35)
  bool p140_off_resident() const {
    const char* e = getenv("ARC_140T");
    return !(e && atoi(e) == 0) && cl_.xmx && cl_.sg8 && w8_resident && !p140_res_ && !hp.moe && !hp.gemma && !hp.hc && !hp.qwen3 &&
           h8_row();
  }
  // (128 registers only where a register is 64 bytes: on a GPU with 8-lane sub-groups, 32-byte registers, ocloc shows
  //  the XMX GEMM spilling with 128)
  bool xe2_grf128() const {
    static const bool xmx_try = getenv("ARC_XMX_TRY") != nullptr;  // (read once: on the verification GEMVs' path)
    return xe2_ && (!cl_.sg8 || xmx_try);
  }

  // the dense prompt GEMM with the default register file (128) instead of 256 registers per thread: more threads per
  // core; on a Pro B60 the 27B's prompt went 450 -> 617 tok/s (issue #4, 2026-11-02). ARC_GEMM_GRF=256 as before.
  // (On the 125H iGPU 128 made the 35B's prompt 3.4x slower: xe2_ is off there, which keeps 256.)
  // (ARC_GEMM_GRF=128 asked for: 128 on any GPU, the 8-lane ones and the DP4a GEMM of iGPUs without the xe2_ switches
  //  too, for their owners' sweeps (bench\igpu_grf_sweep.ps1); the same sums, other occupancy. The 125H's DP4a GEMM
  //  timed 4.7x slower with it, 2026-11-07, so it stays opt-in)
  std::string xe2_gemm_grf() const {
    static const std::string e = getenv("ARC_GEMM_GRF") ? getenv("ARC_GEMM_GRF") : "";
    if (e == "128") return std::string();
    return xe2_grf128() && e != "256" ? std::string() : std::string("-cl-intel-256-GRF-per-thread ");
  }
  // ARC_X8_W=1: gemm_x8.cl -DX8W, one 8-lane sub-group for both halves of a 16-column group (activations loaded once)
  static std::string x8_defs() {
    static const bool w = getenv("ARC_X8_W") && atoi(getenv("ARC_X8_W")) != 0;
    return w ? " -DX8W" : "";
  }
  // ARC_MOE_GRF=128: the same for the MoE expert GEMMs (256 until measured on a B-series card: their 128 build is the
  // one that spilled on the iGPU)
  // (128 by default on discrete cards with 16-lane XMX since 2026-11-16: a Pro B70's 35B 8K prompt 5,377 / 5,338 ->
  //  5,612 tok/s, +4.4-5.1 % (joshdev-de, issue #27, release 2026.11.16.12); the same source, another register file.
  //  iGPUs keep 256 (the 128 build spilled there). ARC_MOE_GRF=256: as before)
  // A Data Center GPU Max (Xe-HPC) takes its own defaults where hibagus's one-stack Max 1550 sweeps measured a gain
  // (issue #48, 2026-11-20, main 03c310a; every winner's logits the same as the defaults'): 27B generation
  // ARC_GVROWS=1 (27.77 -> 29.70 tok/s); 27B prompt ARC_H8_NORM=0, ARC_H8_REUSE=0, ARC_GEMM_XW_PF=2, ARC_GEMM_XW_WGX=8
  // (together 4K 2,645 -> 2,910, +10.0 %); 35B-A3B generation ARC_MOE_GR_GU=4 + ARC_MOE_GR_DN=8 (81.6 -> 88.0);
  // 35B-A3B prompt ARC_MOE_GRF=256 + ARC_MOE_CG=4 (4K 4,665 -> 5,138). Each switch set by hand decides as before;
  // other GPUs unchanged. (Kept once per engine: each GPU of a split decides for its own)
  mutable int xehpc_ = -1;
  bool xehpc_on() const {
    if (xehpc_ < 0) xehpc_ = !zrt::integrated() && !cl_.sg8 && zrt::xehpc() ? 1 : 0;
    return xehpc_ == 1;
  }
  // A discrete Battlemage card of the B60's die (BMG-G21: the Pro B60, the B580, the B50), by its PCI device ID
  // (zrt::bmg_g21), not its core count: a Pro B65 is the B70's die (BMG-G31, 0xE222) with the B60's 160 EUs, and keeps
  // the B70's defaults until it is measured. 2026-11-21: its own defaults where two B60s measured them, each
  // bit-identical:
  //  - the prompt GEMM's tile 64 x 32 with 8 column sub-groups a work-group, the 2D form's prefetch 1 chunk ahead
  //    (engine_xw.h xw_tm / xw_nc / xw_wgx / xw_pfd_v): skipzx-lab's phase X (issue #57, Linux, ec99a4a), the 27B
  //    512 / 4K / 32K prompt 1,109.9 / 1,872.5 / 1,619.3 -> 1,171.8 / 1,954.2 / 1,680.0 tok/s (+5.6 / +4.4 / +3.7 %),
  //    logits the same at each, 0.0 % drift; the start check proves the form bit for bit as any other;
  //  - the one-token GEMVs' rows by type, not by the thread count (ARC_GEMV_WAVE's single-token part, xe2_gemv_rows):
  //    the 27B's generation +3.5 % (skipzx-lab, Linux, 23.29 -> 24.10 tok/s) and +2.4 % (MaineTim, issue #19,
  //    23.36 -> 23.91), logits the same. Its 2-8-token part (MTP verification) stays, not measured apart.
  // Each switch set by hand still wins.
  mutable int g21_ = -1;
  bool xe2_g21() const {
    if (g21_ < 0) {
      g21_ = !zrt::integrated() && cl_.xmx && !cl_.sg8 && !xehpc_on() && zrt::bmg_g21() ? 1 : 0;
      if (g21_) {
        static std::once_flag said;
        std::call_once(said, [] {
          zlog::eprintf("inferzero: the B60's die (BMG-G21, device 0x%04X): its own prompt tile and GEMV rows\n",
                        zrt::pci_device_id());
        });
      }
    }
    return g21_ == 1;
  }
  std::string xe2_moe_grf() const {
    static const char* e = getenv("ARC_MOE_GRF");  // (read once)
    const bool grf128 = e ? std::string(e) == "128" : (!zrt::integrated() && cl_.xmx && !cl_.sg8 && !xehpc_on());
    return xe2_ && grf128 ? std::string() : std::string("-cl-intel-256-GRF-per-thread ");
  }
  // ARC_GEMM_CG=2 / 4: 16-column groups per work-group of the prompt GEMM (a matrix whose column groups it does not
  // divide keeps 1)
  int xe2_cg_for(int N) const {
    static const int cg = [] {
      const int c = getenv("ARC_GEMM_CG") ? atoi(getenv("ARC_GEMM_CG")) : 1;
      if (c == 1 || c == 2 || c == 4) return c;
      zlog::eprintf("inferzero: ARC_GEMM_CG=%d not supported (1, 2 or 4): 1\n", c);
      return 1;
    }();
    // (asked for with ARC_GEMM_CG: on any GPU, the DP4a GEMM of iGPUs without the xe2_ switches too)
    static const bool asked = getenv("ARC_GEMM_CG") != nullptr;
    return (xe2_ || asked) && (N / 16) % cg == 0 ? cg : 1;
  }
  static std::string xe2_cg_defs(int cg) { return cg != 1 ? " -DCG=" + std::to_string(cg) : std::string(); }
  // ARC_GEMV_GRF=256: the single-token GEMV with the large register file
  // (2026-11-08: the GEMV switches act wherever they are set, not only with xe2_ on: the 140T keeps DP4a for prompts,
  //  so xe2_ is off there, and its owner's generation runs had them silently ignored. Unset, nothing changes)
  std::string xe2_gemv_grf() const {
    static const std::string g = grf_env("ARC_GEMV_GRF");
    return g;
  }
  // ARC_GEMV_PF=1: the single-token GEMV (gemv, codebook / K types) issues every row's reads of a step before decoding
  // the first (GROWS x more loads in flight: a B60's GEMV reached ~55 % of its bandwidth; QUEUE F31.4). Bit-identical
  // On by default on discrete cards since 2026-11-08 for dense models: a Pro B60's 27B 22.39 -> 23.11 and 22.33 ->
  // 23.08 tok/s in two runs (issue #4); 2026-11-09, package 2c132a9: 27B 23.21 with it, 22.45 without. Not for MoE
  // models from 2026-11-09 to 11-12: the same run's 35B 78.77 with it, 85.15 without (+8 %; 2026-11-06 it was 84.08 /
  // 83.88). MoE models too again since 2026-11-12: on package fffaf1a (the small work-groups off again, the code of
  // today) the 35B 88.91 with it, 79.93 without (+11 %; the 27B 23.72 with, 22.46 without), issue #4.
  // iGPUs keep it off (the 125H 8 % slower, a 140T 1-2 %). ARC_GEMV_PF=0 / 1
  // ARC_GEMV_DB=1 (2026-11-15): gemv's window reads one window ahead instead (decode.cl -DGEMV_DB), in place of the
  // prefetch; bit-identical. Not with ARC_GV_PROBE. On by default where the prefetch was (discrete cards) since
  // 2026-11-15: a Pro B70's 27B 31.88 / 31.90 / 31.84 -> 32.25 / 32.32 / 32.32 tok/s in three alternated rounds (+1.4 %;
  // median step 31.2 -> 30.8 ms), logits identical (joshdev-de, issue #27, 6466672). Dense models only: the MoE models' gains of the prefetch (the 35B +11 % on a B60) are not measured with it. ARC_GEMV_DB=0: the prefetch
  // (decided once per engine: it is on every GEMV's path, and its getenv calls take the C runtime's lock, which a
  //  split model's shard threads share; issue #4, 2026-11-21)
  mutable int gemv_pf_kind_ = -1;  // 0 none, 1 -DGEMV_DB, 2 -DGEMV_PF
  const std::string& xe2_gemv_pf() const {
    static const std::string defs[3] = {std::string(), std::string(" -DGEMV_DB"), std::string(" -DGEMV_PF")};
    if (gemv_pf_kind_ < 0) {
      static const int dbe = getenv("ARC_GEMV_DB") ? (atoi(getenv("ARC_GEMV_DB")) != 0 ? 1 : 0) : -1;
      static const bool probe = getenv("ARC_GV_PROBE") != nullptr, pf_set = getenv("ARC_GEMV_PF") != nullptr;
      const bool db = !probe && (dbe < 0 ? (cl_.xmx && !cl_.sg8 && !zrt::integrated() && hp.n_expert == 0 && !pf_set) : dbe == 1);
      static const int pf = getenv("ARC_GEMV_PF") ? (atoi(getenv("ARC_GEMV_PF")) != 0 ? 1 : 0) : -1;
      const bool on = pf < 0 ? (cl_.xmx && !cl_.sg8 && !zrt::integrated()) : pf == 1;
      gemv_pf_kind_ = db ? 1 : on ? 2 : 0;
    }
    return defs[gemv_pf_kind_];
  }
  // ARC_GEMV_WAVE (2026-11-15): the single-token GEMV's rows per sub-group (gemv and gemv_v) by the GPU's thread
  // count T (EUs x threads per EU: a Pro B70 2,048, a B60 1,280) instead of by type: a matrix of T < N <= 4T rows takes
  // 4 rows a sub-group (its N / 4 sub-groups in one round of the GPU's threads), N > 4T one row (many rounds, so the
  // last, part-filled one costs little); N <= T and the output heads (N > 65,536) as before. The rule by type gave a
  // 27B's 5,120-row matrices (down, ssm_out, o) 2 rows on gemv and on IQ4_XS (2,560 sub-groups: 1.25 rounds on a B70,
  // the second a quarter full) and its 17,408-row gate / up 4 on gemv's types and the codebook types (2.1 rounds).
  // A Pro B70 (joshdev-de, issue #27), 27B IQ3_S generation 32.3 -> 35.2 tok/s, Q4_K_M 26.0 -> 27.8. Bit-identical: a
  // row's sums do not depend on which sub-group computes it. Discrete cards only (iGPUs unchanged); with several GPUs
  // the count of the first one asked. ARC_GEMV_WAVE=0: off; =1: on on any GPU. ARC_GVROWS (one count for gemv_v's
  // types, a test) turns it off
  mutable int gemv_threads_ = -1;
  int xe2_gemv_threads() const {  // (the GPU's thread count where ARC_GEMV_WAVE is on, else 0)
    if (gemv_threads_ < 0) {
      static const int e = getenv("ARC_GEMV_WAVE") ? (atoi(getenv("ARC_GEMV_WAVE")) != 0 ? 1 : 0) : -1;
      const bool on = e < 0 ? (cl_.xmx && !cl_.sg8 && !zrt::integrated()) : e == 1;
      gemv_threads_ = on ? (int)zrt::hw_threads() : 0;
    }
    return gemv_threads_;
  }
  int xe2_gemv_rows(const GpuWeight& w, int rows) const {
    static const bool fixed = getenv("ARC_GVROWS") != nullptr;
    static const bool wave_set = getenv("ARC_GEMV_WAVE") != nullptr;
    // (a Max: one row, as ARC_GVROWS=1, see xehpc_on; the B60's die: by type, see xe2_g21)
    const int t = fixed || xehpc_on() || (!wave_set && xe2_g21()) ? 0 : xe2_gemv_threads();
    if (t <= 0 || w.N <= t || w.N > 65536) return rows;
    return w.N <= 4 * t ? 4 : 1;
  }
  // ARC_GEMV_WAVE for 2-8 tokens (gemv_mt / gemv_mt_v: MTP verification, 2026-11-15): rows per sub-group 3, 4 or 5,
  // whichever fills the last round of the GPU's T threads best (sub-groups ceil(N / rows), filled share = sub-groups /
  // (rounds x T)); where 4 rows take one round or less (N <= 4T), 4 up to 5 tokens and 3 from 6 (8 tokens' sums need
  // more registers a row). 27B on a Pro B70 (T = 2,048): qkv 10,240 rows 5 (2,048 sub-groups, one full round; 2 or 4
  // gave 1.25 / 2.5), q 12,288 rows 3 (4,096: two full rounds), gate / up 17,408 rows 3 (5,803: 2.83 rounds, against
  // 2.1 at 4 and 1.7 at 5), 5,120 / 6,144 rows 4 or 3; the rule by type gave 2 or 4. Measured per matrix height in the
  // GPU profile with every height at one count (3, 5 and 8 tokens): 6 rows and more were slower everywhere (registers).
  // MTP with 4 drafts 58.9 -> 61.7 tok/s (code prompt, 200 tokens). Bit-identical (a row's sums do not depend on its
  // sub-group). Not for N <= T (k / v, 1,024 rows: 3-5 rows were much slower there), the heads, Q6_K (1 row, untested)
  // or the fused two-pass GEMV (twice the tokens' registers, untested). ARC_GEMV_WAVE_MT=0: off (the single-token
  // part stays); ARC_GEMV_WAVE=0 turns off both
  int xe2_gemv_mt_rows(const GpuWeight& w, int ntok, int rows, bool fused) const {
    static const bool off = getenv("ARC_GEMV_WAVE_MT") && atoi(getenv("ARC_GEMV_WAVE_MT")) == 0;
    const int t = off ? 0 : xe2_gemv_threads();
    if (t <= 0 || fused || w.type == GGML_Q6_K || w.N <= t || w.N > 65536) return rows;
    if ((w.N + 3) / 4 <= t) return ntok >= 6 ? 3 : 4;
    int best = rows;
    double bf = -1;
    for (int r = 3; r <= 5; ++r) {
      const long sg = (w.N + r - 1) / r, rounds = (sg + t - 1) / t;
      const double f = (double)sg / ((double)rounds * t);
      if (f > bf + 1e-9) bf = f, best = r;
    }
    return best;
  }
  // ARC_GEMV_SMALL_WG: the single-token GEMV (gemv) of a matrix with few rows, at most 512 sub-groups (DeltaNet's
  // alpha / beta projection, 96 rows; MoE routers, 256-512 rows), in work-groups of 4 sub-groups instead of 16, so
  // they spread over more Xe cores: a Pro B60's 27B ran gemm_ab's 96 rows in 6 work-groups at 60 GB/s, 14 % of its
  // bandwidth (issue #4, 2026-11-09). Each row's sums are the same: bit-identical. On by default on discrete cards from 2026-11-10;
  // off again since 2026-11-12 (issue #4, package 18acafc, a Pro B60, best of 2): the 27B gained nothing (23.52 on,
  // 23.57 off; 23.72 / 23.71 on 97d225b), and the 35B lost 30 % (55.85 on, 80.35 off; with ARC_GEMV_PF=1 79.12, so
  // only the plain gemv's 64-item work-groups were slow there). iGPUs: 256 as always. ARC_GEMV_SMALL_WG=1 forces it
  size_t xe2_gemv_wg(size_t subgroups) const {
    static const bool on = getenv("ARC_GEMV_SMALL_WG") && atoi(getenv("ARC_GEMV_SMALL_WG")) != 0;
    return on && subgroups <= 512 ? 64 : 256;
  }
  // ARC_XALL (zero/engine_moe.h): a token's MoE layers skip reading their experts' ids and the cache's ensure() when
  // every expert is cached (XCache::all_in, checked at each token's first MoE layer). On by default on discrete cards
  // since 2026-11-09: a Pro B60 holding the 35B's 14.7 GiB of experts, package 2c132a9, 78.77 -> 88.05 tok/s, host
  // waits 43 -> 3 a token, the same tokens (issue #4). Off on iGPUs (they never hold a 35B's experts; unchanged there).
  // ARC_XALL=0 / 1
  // ARC_GDN_L1PF (zero/engine_load.h, kernels/ops.cl gdn_recur): token t+D's DeltaNet operands prefetched into L1.
  // 2 by default on discrete cards since 2026-11-15: a Pro B70's 27B 4K prompt, gdn_recur 293 -> 263 ms, 2,216 ->
  // 2,247 tok/s, logits identical over 18,566 tokens (joshdev-de, issue #27; 2 / 4 the same, further ahead slower).
  // Where the driver lacks cl_intel_subgroup_buffer_prefetch the kernel ignores it. iGPUs: 0 (unchanged).
  // ARC_GDN_L1PF=0 (or 1: the kernel loads one token ahead anyway) turns it off, =D (2-64) sets the distance; with
  // ARC_GDN_PF >= 2 it does nothing (the register-prefetch loop has no L1 prefetch)
  // The default MTP drafts per step on discrete cards: 4 where the laptop's choice is 7 (the 27B). A Pro B70's 27B,
  // serverz --mtp, greedy, three 400-token answers (code / prose / json): 7 drafts 57.9 / 49.6 / 79.1 tok/s, 4 drafts
  // 64.9 / 53.4 / 79.1 (mean 62.2 -> 65.8, +6 %); each draft past 3 adds ~6-7 ms of verification a pass there, and the
  // later drafts are accepted less (joshdev-de, issue #27, a51686e). The iGPU keeps 7 (laptop: 3 -> 7 drafts 12.6 ->
  // 14.7 tok/s). --mtp-drafts N overrides either; DFlash2 keeps its 7.
  // 3 since 2026-11-18: the same B70, runz --mtp, a code prompt, 256 asked: drafts 2 / 3 / 4 58.2 / 59.8 / 57.5 tok/s,
  // accepted 75.0 / 70.6 / 60.4 %, 2.38 / 2.79 / 2.88 tokens a pass (joshdev-de, issue #27, 7368768): the 4th draft is
  // accepted too seldom to pay its verification
  // Alchemist (discrete, 8-lane XMX) 4 since 2026-11-21: an A770 (xe, 150 W), the 27B on a coding question, 256 tokens,
  // the full sweep's D phase: plain 22.02 tok/s; drafts 2 / 3 / 4 / 5 / 6 / 7 42.06 / 45.56 / 46.08 / 45.11 / 43.74 /
  // 40.75 (7 accepted 60.4 %, 4 76.8 %), every text the plain one's (skipzx-lab, issue #45). Its verify pass costs more
  // than a B-series card's (83.7 ms against a B60's 53.6), so its best count sits between theirs and the laptop's
  int xe2_mtp_drafts(int d) const {
    if (d != 7 || zrt::integrated() || !cl_.xmx) return d;
    return cl_.sg8 ? 4 : 3;
  }
  int xe2_gdn_l1pf() const {
    static const int e = getenv("ARC_GDN_L1PF") ? std::clamp(atoi(getenv("ARC_GDN_L1PF")), 0, 64) : -1;
    return e >= 0 ? e : (zrt::integrated() ? 0 : 2);
  }
  bool xe2_xall() const {
    static const int e = getenv("ARC_XALL") ? (atoi(getenv("ARC_XALL")) != 0 ? 1 : 0) : -1;
    return e < 0 ? !zrt::integrated() : e == 1;
  }
  // the 2-8 token GEMV (gemv_mt / gemv_mt_v: MTP and DFlash2 verification, batched slots) with the default register
  // file (128) on discrete cards: a Pro B60's 27B MTP went 43.3 -> 47.1 tok/s (issue #4, 2026-11-02); ARC_GEMV_MT_GRF=256
  // as before. The iGPUs keep 256 (the 125H forced to 128: 27B MTP 11.78 -> 9.57 tok/s, 2026-11-03)
  std::string xe2_gemv_mt_grf() const {
    static const std::string e = getenv("ARC_GEMV_MT_GRF") ? getenv("ARC_GEMV_MT_GRF") : "";
    if (e == "128") return std::string();  // (asked for: any GPU, for its owner's sweep)
    return xe2_grf128() && e != "256" ? std::string() : std::string("-cl-intel-256-GRF-per-thread ");
  }

  // QUEUE 31.8: the XMX prompt GEMM's activations with 2D block loads (cl_intel_subgroup_2d_block_io, gemm.cl -DX2D): a
  // step's 8 token rows x 64 bytes in one 8b_8r32x2c load, the DPAS A operands as the hardware lays them out, instead of
  // 16 one-row block reads; the prefetch (-DX2D_PF) also asks for the next 64 bytes of those rows. Dense GEMMs only (the
  // MoE tile kernel reads rows through a row map). On by default wherever the driver offers them with XMX: a Pro B60's
  // 27B prompt went 608 -> 752 tok/s, 763 with the prefetch (issue #4, 2026-11-06). ARC_GEMM_2D=0 / ARC_GEMM_2D_PF=0 turn
  // them off. The start-time XMX check runs with them, and if they do not build or differ, XMX goes on without them
  bool x2d_ = false;
  static bool x2d_switch(const char* name) {
    const char* e = getenv(name);
    return e != nullptr ? atoi(e) != 0 : true;
  }
  std::string xe2_x2d() const {
    static const bool pf = x2d_switch("ARC_GEMM_2D_PF");
    return x2d_ ? std::string(pf ? " -DX2D -DX2D_PF" : " -DX2D") : std::string();
  }
  bool x2d_wanted() const { return xe2_ && cl_.block2d && x2d_switch("ARC_GEMM_2D"); }
  // ARC_GEMM_WPF=n (1..16): the XMX prompt GEMM prefetches its weights n 64-value chunks ahead (4: the next
  // superblock); off by default, a B60 to measure. Bit-identical
  std::string xe2_wpf() const {
    static const int n = getenv("ARC_GEMM_WPF") ? atoi(getenv("ARC_GEMM_WPF")) : 0;
    return xe2_ && !cl_.sg8 && n >= 1 && n <= 16 ? " -DWPF=" + std::to_string(n) : std::string();
  }

  // ---- attn_q8 on XMX (QUEUE F33.2, ARC_ATTN_XMX=1; zero/kernels/attn_x.cl in front of the ops parts): the prompt
  // attention's two integer dot products per 16-key tile as int8 DPAS, the rest attn_q8's; bit-identical. Discrete
  // Xe2 / Xe3 cards (16-lane DPAS), Qwen (one scale per row), up to 4 q heads per sub-group; ARC_ATTN_XEMU=1: DPAS
  // emulated on any GPU (tests). Checked once per device and shape at start against attn_q8 bit for bit on random data;
  // any difference keeps attn_q8 and says so. On by default on discrete cards since 2026-11-10: a Pro B60's prompts,
  // 64K tokens, 4B 2,017 -> 2,253, 9B 1,871 -> 2,028, 27B 605 -> 618, 35B 1,814 -> 2,000 tok/s, 4K +0.2-1 %, logits
  // bit-identical (issue #4, package bb8952f, bench\f33_sweep.ps1). On iGPUs with 16-lane XMX too since 2026-11-16:
  // an Arc B390 (Panther Lake, gee456, issue #31, release 2026.11.16.07, f33_sweep) at 65K tokens 4B 950 -> 1,079, 27B
  // 206 -> 215, 35B 795 -> 904 tok/s, 4K within 1 %, logits bit-identical. The 125H (no XMX) and the 140T (8 lanes)
  // never take it. ARC_ATTN_XMX=0 / 1
  std::string attnx_src_, attn_q8_defs_;
  static bool attn_xemu() {
    static const bool e = getenv("ARC_ATTN_XEMU") && atoi(getenv("ARC_ATTN_XEMU")) != 0;
    return e;
  }
  bool attnx_wanted() const {
    static const int e = getenv("ARC_ATTN_XMX") ? (atoi(getenv("ARC_ATTN_XMX")) != 0 ? 1 : 0) : -1;
    // (an iGPU only where XMX passed its start-time checks, speed included (xmx_): one whose check kept DP4a runs the
    //  GPU's own defaults; audit 2026-11-17)
    const bool on = e < 0 ? (!zrt::integrated() || (cl_.xmx && !cl_.sg8 && xmx_)) : e == 1;
    return on && ((cl_.xmx && !cl_.sg8) || attn_xemu()) && !hp.gemma && !hp.swa_rings && attn_hs() <= 4 &&
           !attn_q8_defs_.empty();
  }
  std::string attnx_defs() const { return attn_q8_defs_ + " -DATTN_XMX" + (attn_xemu() ? " -DATTN_XMX_EMU" : ""); }
  void attnx_init() {
    if (!attnx_wanted()) return;
    static std::mutex mu;
    static std::map<std::string, bool> known;
    const std::string key = cl_.device_id + "|" + attnx_defs();
    std::lock_guard<std::mutex> g(mu);
    auto it = known.find(key);
    if (it == known.end()) {
      const bool ok = attnx_check();
      it = known.emplace(key, ok).first;
      if (ok)
        zlog::eprintf("inferzero: prompt attention on XMX%s (bit-identical to attn_q8, checked)\n",
                attn_xemu() ? " (DPAS emulated: tests)" : "");
    }
    if (it->second) k_attn_q8_ = kernel("attnx", attnx_defs(), "attn_q8");
  }
  bool attnx_check() {
    // one kv head, its q heads, 40 tokens at position 0 over a 48-slot cache: three tiles, the last one partial
    const int hpg = hp.n_head / hp.n_head_kv, hd = hp.qwen3 ? hp.head_dim : 256, nh = hpg, M = 40, cap = 48;
    const int nbc = cap / 16, blk_u = 4 * hd, tpw = attn_tpw();
    uint32_t seed = 4243;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    std::vector<uint32_t> qi((size_t)M * nh * hd / 4), kq((size_t)nbc * blk_u), vq((size_t)nbc * blk_u);
    for (auto& v : qi) v = rnd();
    for (auto& v : kq) v = rnd();
    for (auto& v : vq) v = rnd();
    std::vector<float> qsc((size_t)M * nh), ksc((size_t)nbc * 16), vsc((size_t)nbc * 16), out((size_t)M * nh * hd);
    // (scores of a few log2 units, so P takes mid-range bytes too: random int8 rows of 256 dims sum to ~1e5, so q scales
    //  of ~1e-5 with k scales of ~5e-2; 10x larger made P nearly 0 or 255 everywhere)
    for (auto& v : qsc) v = 1e-5f * (1 + rnd() % 100);
    for (auto& v : ksc) v = 1e-3f * (1 + rnd() % 100);
    for (auto& v : vsc) v = 1e-3f * (1 + rnd() % 100);
    std::vector<cl_mem> bufs;
    auto mk = [&](const void* p, size_t bytes) { cl_mem m = buf(bytes, p); bufs.push_back(m); return m; };
    struct Soft { bool& f; bool was; ~Soft() { f = was; } } soft{cl_.soft_fail, cl_.soft_fail};
    cl_.soft_fail = true;
    bool ok = true, built = true;
    try {
      const cl_mem bq = mk(qi.data(), qi.size() * 4), bqs = mk(qsc.data(), qsc.size() * 4), bk = mk(kq.data(), kq.size() * 4),
                   bks = mk(ksc.data(), ksc.size() * 4), bv = mk(vq.data(), vq.size() * 4), bvs = mk(vsc.data(), vsc.size() * 4);
      std::vector<float> res[2];
      for (int x = 0; x < 2; ++x) {
        const cl_mem bo = mk(out.data(), out.size() * 4);
        cl_kernel k = x == 0 ? kernel("ops", attn_q8_defs_, "attn_q8") : kernel("attnx", attnx_defs(), "attn_q8");
        int cx = cap, nn = nh, p0 = 0, mm = M, win = 0, nlo = -1, nhi = -1;
        set_args(k, 0, bq, bqs, bk, bks, bv, bvs, cx, nn, p0, mm, bo, win, nlo, nhi);
        size_t gws[2] = {(size_t)(M + tpw - 1) / tpw * 512, 1}, lws[2] = {512, 1};
        run("attnx_check", k, 2, gws, lws);
        res[x].resize(out.size());
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, bo, CL_TRUE, 0, res[x].size() * 4, res[x].data(), 0, nullptr, nullptr));
      }
      size_t diff = 0;
      for (size_t i = 0; i < out.size(); ++i) diff += memcmp(&res[0][i], &res[1][i], 4) != 0;
      if (diff) {
        zlog::eprintf("inferzero: prompt attention on XMX differs from attn_q8 in %zu of %zu values: attn_q8 "
                        "(please report)\n", diff, out.size());
        ok = false;
      }
    } catch (const zrt::GpuWaitError&) {
      for (cl_mem& m : bufs) free_buf(m);
      throw;
    } catch (const std::exception& e) {
      zlog::eprintf("inferzero: prompt attention on XMX does not build here (%s): attn_q8\n", e.what());
      built = false;
    }
    for (cl_mem& m : bufs) free_buf(m);
    return ok && built;
  }

  // ---- attn_fx (issue #27, 2026-11-15; zero/kernels/attn_fx.cl): the plain causal prompt attention as flash attention
  // on XMX for discrete Xe2 / Xe3 cards: 8 query tokens per sub-group (full 8-row int8 DPAS; attn_q8: 1 token x <= 4
  // heads), the same int8 K / V cache and output, P quantized per 32 keys instead of 16 (not bit-identical: checked at
  // start against attn_q8 on random data within a tolerance; KL in docs/before-after.md). A Pro B70's 27B: attention
  // 4K 118 -> 45 ms, 16K 1,709 -> 618 ms, 64K 26.96 -> 9.64 s (prompt 1,284 -> 1,944 tok/s). Windows, non-causal
  // spans and the sparse masks stay on attn_q8. On by default on discrete cards (16-lane XMX); ARC_ATTN_FX=0: attn_q8,
  // =1 forces it, =2 without the check (the kernel's timing ablations); ARC_ATTN_FX_WG="NT,NH": token groups of 8 x q
  // heads per work-group; ARC_ATTN_FX_PF=0: no L1 prefetch; ARC_ATTN_FX_DEFS: further -D options (tests)
  //
  // attn_x8 (QUEUE A6 / T4, 2026-11-20; zero/kernels/attn_x8.cl): the same flash attention for GPUs whose DPAS takes
  // 8-lane sub-groups (Alchemist A770 / A750, the Arc 140T), whose prompts ran attn_q8 on DP4a. Opt-in until owners
  // measure it: ARC_ATTN_X8=1 where the driver offers XMX, =2 with the DPAS emulated by DP4a dot products on any GPU
  // with 8-lane sub-groups (tests: the 125H). The same start-up check against attn_q8; ARC_ATTN_X8_WG="NT,NH" (heads of
  // more than 128 dims take two sub-groups a token group: at most 32 sub-groups in all); ARC_ATTN_FX_PF and
  // ARC_ATTN_FX_DEFS apply to it too
  std::string attnfx_src_, attnx8_src_;
  cl_kernel k_attn_fx_ = nullptr;
  int fx_nt_ = 0, fx_nh_ = 0;
  bool fx_x8_ = false;  // (k_attn_fx_ is attn_x8)
  static int attnx8_mode() {
    static const int e = getenv("ARC_ATTN_X8") ? atoi(getenv("ARC_ATTN_X8")) : 0;
    return e == 1 || e == 2 ? e : 0;
  }
  // (=1 on an iGPU only where XMX passed its start-time checks (xmx_), as attnfx_wanted: a 140T whose check failed,
  //  e.g. with an old driver, keeps attn_q8; the Researcher's audit half, 2026-11-21. =2: DPAS emulated, tests)
  bool attnfx8_on() const {
    return cl_.sg8 && (attnx8_mode() == 2 || (attnx8_mode() == 1 && cl_.xmx && (xmx_ || !zrt::integrated())));
  }
  bool attnfx_wanted() const {
    static const int e = getenv("ARC_ATTN_FX") ? (atoi(getenv("ARC_ATTN_FX")) != 0 ? 1 : 0) : -1;
    // (on by default on discrete cards, and on iGPUs with 16-lane XMX since 2026-11-17: a B390's 27B 4K prompt attention
    //  377 -> 160 ms, issue #1; GPUs with 8-lane DPAS only through ARC_ATTN_X8, attn_x8)
    // (an iGPU only where XMX passed its start-time checks, speed included (xmx_): one whose check kept DP4a runs the
    //  GPU's own defaults; audit 2026-11-17)
    const bool on = e < 0 ? (!zrt::integrated() || (cl_.xmx && !cl_.sg8 && xmx_)) : e == 1;
    const int hd = hp.qwen3 ? hp.head_dim : 256;
    const bool model = !hp.gemma && !hp.swa_rings && hd % 32 == 0 && hd <= 256 && !attn_q8_defs_.empty();
    if (cl_.sg8) return e != 0 && attnfx8_on() && model;
    return on && cl_.xmx && model;
  }
  std::string attnfx_defs() const {
    const int hd = hp.qwen3 ? hp.head_dim : 256, hpg = hp.n_head / hp.n_head_kv;
    static const bool pf = !getenv("ARC_ATTN_FX_PF") || atoi(getenv("ARC_ATTN_FX_PF")) != 0;
    // (ARC_ATTN_FX_DEFS: further -D options for the kernel's variants, tests)
    static const std::string extra = getenv("ARC_ATTN_FX_DEFS") ? std::string(" ") + getenv("ARC_ATTN_FX_DEFS") : "";
    const std::string pfd = fx_x8_ ? (pf ? " -DX8_PF" : "") + std::string(attnx8_mode() == 2 ? " -DX8_EMU" : "")
                                   : (pf ? " -DFX_PF" : "");
    return "-cl-intel-256-GRF-per-thread -DHD=" + std::to_string(hd) + " -DHPG=" + std::to_string(hpg) +
           " -DNT=" + std::to_string(fx_nt_) + " -DNH=" + std::to_string(fx_nh_) + pfd + extra;
  }
  const char* attnfx_which() const { return fx_x8_ ? "attnx8" : "attnfx"; }
  const char* attnfx_name() const { return fx_x8_ ? "attn_x8" : "attn_fx"; }
  void attnfx_init() {
    if (!attnfx_wanted()) return;
    const int hpg = hp.n_head / hp.n_head_kv;
    fx_x8_ = cl_.sg8;
    // (attn_x8: two sub-groups a token group for heads of more than 128 dims)
    const int ds = fx_x8_ && (hp.qwen3 ? hp.head_dim : 256) > 128 ? 2 : 1;
    // (attn_fx: 32 sub-groups, every thread slot of an Xe-core with 256 registers; two q heads where the GQA ratio is
    //  even: a B70's 27B 16K prompt, attention 965 ms at 4 x 6, 772 at 5 x 6, 707 at 16 x 2, 710 at 32 x 1.
    //  attn_x8, not measured yet: all q heads of a KV head in one work-group (with two sub-groups a token group they
    //  run in step, and token groups added run the last one's spans), 16 sub-groups)
    const int cap = 32 / ds;
    int nh = fx_x8_ ? hpg : hpg % 2 == 0 ? 2 : 1, nt = fx_x8_ ? std::max(1, 16 / (nh * ds)) : 32 / nh;
    if (const char* e = getenv(fx_x8_ ? "ARC_ATTN_X8_WG" : "ARC_ATTN_FX_WG")) sscanf(e, "%d,%d", &nt, &nh);
    if (nh < 1 || hpg % nh) nh = hpg;
    while (nh > 1 && nt * nh > cap) {  // (at most 32 sub-groups: the largest divisor of the GQA ratio that fits)
      int d = nh - 1;
      while (hpg % d) --d;
      nh = d;
    }
    if (nt < 1 || nt * nh > cap) nt = std::max(1, cap / nh);
    fx_nt_ = nt, fx_nh_ = nh;
    static std::mutex mu;
    static std::map<std::string, double> known;  // (-1: failed)
    const std::string key = cl_.device_id + "|" + attnfx_defs();
    std::lock_guard<std::mutex> g(mu);
    auto it = known.find(key);
    if (it == known.end()) {
      // (ARC_ATTN_FX=2: no check, for the kernel's timing ablations, whose results are wrong on purpose)
      const bool nocheck = getenv("ARC_ATTN_FX") && atoi(getenv("ARC_ATTN_FX")) == 2;
      const double d = nocheck ? 0.0 : attnfx_check();
      it = known.emplace(key, d).first;
      if (d >= 0)
        zlog::eprintf("inferzero: prompt attention as flash attention on XMX (%s%s, %d x 8 tokens x %d heads a work-group; "
                      "against attn_q8: max difference %.1e of the output range)\n", attnfx_name(),
                      fx_x8_ && attnx8_mode() == 2 ? ", DPAS emulated: tests" : "", nt, nh, d);
    }
    if (it->second >= 0) k_attn_fx_ = kernel(attnfx_which(), attnfx_defs(), attnfx_name());
  }
  // gws / lws of attn_fx (attn_x8) for T tokens and nkv KV heads
  void attnfx_grid(int T, int nkv, size_t gws[2], size_t lws[2]) const {
    const int hd = hp.qwen3 ? hp.head_dim : 256, ds = fx_x8_ && hd > 128 ? 2 : 1;
    const int hpg = hp.n_head / hp.n_head_kv, tq = 8 * fx_nt_, lsz = (fx_x8_ ? 8 * ds : 16) * fx_nt_ * fx_nh_;
    gws[0] = (size_t)(T + tq - 1) / tq * lsz, gws[1] = (size_t)nkv * (hpg / fx_nh_);
    lws[0] = (size_t)lsz, lws[1] = 1;
  }
  // attn_fx against the current attn_q8 (attn_x or the DP4a form: bit-identical to each other) on random data: one KV
  // head and its q heads, two cases (40 tokens at 0 over a 48-slot cache; 45 tokens at 27 over 80 slots: a partial span,
  // the causal edge inside a span, earlier keys). Returns the largest difference over the largest output value, or -1
  double attnfx_check() {
    const int hpg = hp.n_head / hp.n_head_kv, hd = hp.qwen3 ? hp.head_dim : 256, nh = hpg, tpw = attn_tpw();
    const double tol = 2e-2;
    uint32_t seed = 4247;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    std::vector<cl_mem> bufs;
    auto mk = [&](const void* p, size_t bytes) { cl_mem m = buf(bytes, p); bufs.push_back(m); return m; };
    struct Soft { bool& f; bool was; ~Soft() { f = was; } } soft{cl_.soft_fail, cl_.soft_fail};
    cl_.soft_fail = true;
    double worst = 0;
    try {
      // (attn_x8 also a third: 77 tokens at 150 over 256 slots, token groups of a work-group past M_valid)
      const struct { int M, p0, cap; } cases[3] = {{40, 0, 48}, {45, 27, 80}, {77, 150, 256}};
      for (int ci = 0; ci < (fx_x8_ ? 3 : 2); ++ci) {
        const auto& c = cases[ci];
        const int nbc = c.cap / 16, blk_u = 4 * hd;
        std::vector<uint32_t> qi((size_t)c.M * nh * hd / 4), kq((size_t)nbc * blk_u), vq((size_t)nbc * blk_u);
        for (auto& v : qi) v = rnd();
        for (auto& v : kq) v = rnd();
        for (auto& v : vq) v = rnd();
        std::vector<float> qsc((size_t)c.M * nh), ksc((size_t)nbc * 16), vsc((size_t)nbc * 16), out((size_t)c.M * nh * hd);
        for (auto& v : qsc) v = 1e-5f * (1 + rnd() % 100);
        for (auto& v : ksc) v = 1e-3f * (1 + rnd() % 100);
        for (auto& v : vsc) v = 1e-3f * (1 + rnd() % 100);
        const cl_mem bq = mk(qi.data(), qi.size() * 4), bqs = mk(qsc.data(), qsc.size() * 4), bk = mk(kq.data(), kq.size() * 4),
                     bks = mk(ksc.data(), ksc.size() * 4), bv = mk(vq.data(), vq.size() * 4), bvs = mk(vsc.data(), vsc.size() * 4);
        std::vector<float> res[2];
        for (int x = 0; x < 2; ++x) {
          const cl_mem bo = mk(out.data(), out.size() * 4);
          cl_kernel k = x == 0 ? k_attn_q8_ : kernel(attnfx_which(), attnfx_defs(), attnfx_name());
          int cx = c.cap, nn = nh, p0 = c.p0, mm = c.M, win = 0, nlo = -1, nhi = -1;
          set_args(k, 0, bq, bqs, bk, bks, bv, bvs, cx, nn, p0, mm, bo, win, nlo, nhi);
          size_t gws[2] = {(size_t)(c.M + tpw - 1) / tpw * 512, 1}, lws[2] = {512, 1};
          if (x == 1) attnfx_grid(c.M, 1, gws, lws);
          run("attnfx_check", k, 2, gws, lws);
          res[x].resize(out.size());
          CL_CHECK(clEnqueueReadBuffer(cl_.queue, bo, CL_TRUE, 0, res[x].size() * 4, res[x].data(), 0, nullptr, nullptr));
        }
        double mref = 0, mdiff = 0;
        for (size_t i = 0; i < out.size(); ++i) {
          mref = std::max(mref, (double)std::fabs(res[0][i]));
          const double d = std::fabs((double)res[0][i] - res[1][i]);
          mdiff = std::isfinite(d) ? std::max(mdiff, d) : 1e30;
        }
        worst = std::max(worst, mref > 0 ? mdiff / mref : 1e30);
      }
      if (worst > tol) {
        zlog::eprintf("inferzero: %s differs from attn_q8 by %.2e of the output range (more than %.0e): attn_q8 "
                      "(please report)\n", attnfx_name(), worst, tol);
        worst = -1;
      }
    } catch (const zrt::GpuWaitError&) {
      for (cl_mem& m : bufs) free_buf(m);
      throw;
    } catch (const std::exception& e) {
      zlog::eprintf("inferzero: %s does not build here (%s): attn_q8\n", attnfx_name(), e.what());
      worst = -1;
    }
    for (cl_mem& m : bufs) free_buf(m);
    return worst;
  }

  // ---- the chunked DeltaNet on XMX (QUEUE F33.1, ARC_GDN_CHUNK=3; zero/kernels/gdn_x.cl): gdn_chunk_x where DPAS
  // takes 16-lane sub-groups (discrete Xe2 / Xe3), gdn_chunk_x8 where it takes 8 (the Arc 140T); without XMX
  // (or when the check fails) layer_gdn runs gdn_recur. ARC_GDN_XEMU=1 / 8 / 16: DPAS emulated on any GPU, in its own / the 8- / 16-lane form
  // (tests). Checked once per device and form at start against gdn_chunk on random data (2 heads, 100 tokens: three
  // chunks and a short one; gated and fused gates); outputs or states more than 1e-4 of their size apart: not used.
  std::string gdnx_src_;
  bool gdnx_ = false;
  bool gdnx_wanted() const { return gdn_chunk_mode() == 3 && (cl_.xmx || gdn_xemu()); }
  bool gdnx_8() const { return gdn_xemu() == 8 || (gdn_xemu() != 16 && cl_.sg8); }
  const char* gdnx_name() const { return gdnx_8() ? "gdn_chunk_x8" : "gdn_chunk_x"; }
  size_t gdnx_wg() const { return gdnx_8() ? 128 : 256; }
  std::string gdnx_defs(bool fuse) const {
    return std::string("-cl-intel-256-GRF-per-thread -DGDN_XMX") + (gdnx_8() ? " -DGDN_XMX8" : "") +
           (gdn_xemu() ? " -DGDN_XMX_EMU" : "") + (fuse ? " -DFUSE_GATES" : "");
  }
  void gdnx_init() {
    if (!gdnx_wanted()) return;
    static std::mutex mu;
    static std::map<std::string, bool> known;
    const std::string key = cl_.device_id + "|" + gdnx_name() + (gdn_xemu() ? "|emulated" : "");
    std::lock_guard<std::mutex> g(mu);
    auto it = known.find(key);
    if (it == known.end()) {
      const bool ok = gdnx_check();
      it = known.emplace(key, ok).first;
      if (ok)
        zlog::eprintf("inferzero: chunked DeltaNet on XMX, %d lanes%s (results checked against gdn_chunk)\n",
                gdnx_8() ? 8 : 16, gdn_xemu() ? " (DPAS emulated: tests)" : "");
    }
    gdnx_ = it->second;
  }
  bool gdnx_check() {
    const int nk = 1, nv = 2, M = 100, C = 2 * nk * 128 + nv * 128;
    uint32_t seed = 977;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float)(seed >> 8) / 16777216.0f; };  // [0, 1)
    auto nrm = [&]() { return (rnd() + rnd() + rnd() - 1.5f) * 2.0f; };
    std::vector<float> xh((size_t)M * C), raw((size_t)M * 2 * nv), gated((size_t)M * 2 * nv), dtb(nv), ssa(nv),
        s0((size_t)nv * 128 * 128);
    for (int t = 0; t < M; ++t) {
      float* r = &xh[(size_t)t * C];
      for (int c = 0; c < C; ++c) r[c] = nrm();
      for (int hd = 0; hd < 2 * nk; ++hd) {  // (q then k rows of unit length, q times 1/sqrt(128), as gdn_conv makes them)
        float ss = 0.0f;
        for (int i = 0; i < 128; ++i) ss += r[hd * 128 + i] * r[hd * 128 + i];
        const float sc = (hd < nk ? 1.0f / std::sqrt(128.0f) : 1.0f) / std::sqrt(ss);
        for (int i = 0; i < 128; ++i) r[hd * 128 + i] *= sc;
      }
      for (int i = 0; i < 2 * nv; ++i) raw[(size_t)t * 2 * nv + i] = nrm() * 2.0f;
    }
    for (int hh = 0; hh < nv; ++hh) { dtb[hh] = nrm(); ssa[hh] = -std::exp(nrm()); }
    for (auto& v : s0) v = nrm() * 0.1f;
    for (int t = 0; t < M; ++t)
      for (int hh = 0; hh < nv; ++hh) {  // (gdn_gates' form)
        const float a = raw[(size_t)t * 2 * nv + hh] + dtb[hh];
        gated[(size_t)t * 2 * nv + hh] = std::exp((a > 20.0f ? a : std::log(1.0f + std::exp(a))) * ssa[hh]);
        gated[(size_t)t * 2 * nv + nv + hh] = 1.0f / (1.0f + std::exp(-raw[(size_t)t * 2 * nv + nv + hh]));
      }
    std::vector<cl_mem> bufs;
    auto mk = [&](const std::vector<float>& v) { cl_mem m = buf(v.size() * 4, v.data()); bufs.push_back(m); return m; };
    struct Soft { bool& f; bool was; ~Soft() { f = was; } } soft{cl_.soft_fail, cl_.soft_fail};
    cl_.soft_fail = true;
    bool ok = true;
    double worst = 0.0;
    try {
      const cl_mem bx = mk(xh), bdt = mk(dtb), bsa = mk(ssa);
      for (int fuse = 0; fuse < 2 && ok; ++fuse) {
        const cl_mem bab = mk(fuse ? raw : gated);
        std::vector<float> out[2], st[2];
        for (int x = 0; x < 2; ++x) {
          const cl_mem bS = mk(s0), bo = mk(std::vector<float>((size_t)M * nv * 128, 0.0f));
          cl_kernel k = x == 0 ? kernel("ops", std::string("-cl-intel-256-GRF-per-thread") + (fuse ? " -DFUSE_GATES" : ""), "gdn_chunk")
                               : kernel("gdnx", gdnx_defs(fuse != 0), gdnx_name());
          int Ci = C, nki = nk, nvi = nv, Mi = M;
          set_args(k, 0, bx, Ci, bab, nki, nvi, bS, Mi, bo, bS);
          if (fuse) set_args(k, 9, bdt, bsa);
          size_t g1 = x == 0 ? (size_t)nv * 2 * 256 : (size_t)nv * gdnx_wg(), l1 = x == 0 ? 256 : gdnx_wg();
          run("gdnx_check", k, 1, &g1, &l1);
          out[x].resize((size_t)M * nv * 128);
          st[x].resize(s0.size());
          CL_CHECK(clEnqueueReadBuffer(cl_.queue, bo, CL_TRUE, 0, out[x].size() * 4, out[x].data(), 0, nullptr, nullptr));
          CL_CHECK(clEnqueueReadBuffer(cl_.queue, bS, CL_TRUE, 0, st[x].size() * 4, st[x].data(), 0, nullptr, nullptr));
        }
        for (const auto* pr : {&out, &st}) {
          double e = 0.0, m = 0.0;
          bool fin = true;
          for (size_t i = 0; i < (*pr)[0].size(); ++i) {
            fin = fin && std::isfinite((*pr)[1][i]);
            e = std::max(e, (double)std::fabs((*pr)[1][i] - (*pr)[0][i]));
            m = std::max(m, (double)std::fabs((*pr)[0][i]));
          }
          const double rel = m > 0.0 ? e / m : 1.0;
          worst = std::max(worst, rel);
          if (!fin || !(rel < 1e-4)) ok = false;
        }
      }
    } catch (const zrt::GpuWaitError&) {
      for (cl_mem& m : bufs) free_buf(m);
      throw;  // (a GPU that hangs: not a verdict on the kernel)
    } catch (const std::exception& e) {
      zlog::eprintf("inferzero: the chunked DeltaNet on XMX does not build here (%s): gdn_chunk instead\n", e.what());
      ok = false;
      worst = -1.0;
    }
    for (cl_mem& m : bufs) free_buf(m);
    if (!ok && worst >= 0.0)
      zlog::eprintf("inferzero: the chunked DeltaNet on XMX differs from gdn_chunk (%.2e of the result): gdn_chunk "
                      "instead (please report)\n", worst);
    else if (ok && gdn_xemu())
      zlog::eprintf("inferzero: chunked DeltaNet XMX check: largest difference %.2e of the result\n", worst);
    return ok;
  }

  // ---- the f16 XMX prompt GEMM (kernels/gemm_x16.cl) for the W8 layouts (the 27B's IQ3_S, decoded just in time).
  // Off until measured on a B60: ARC_X16=1 on a discrete GPU with XMX. ARC_X16_TRY=1 anywhere, DPAS emulated where
  // the GPU has none (tests of the kernel's layouts on this laptop). Taken only once its results match the DP4a
  // kernel's on random data within f16 rounding (not bit for bit: f16 weights, f32 sums in another order).
  // ARC_X16_TM=8 / 16 / 32: tokens per sub-group (default 32); ARC_X16_GRF=128: the default register file (256 else)
  bool x16_ = false, x16_emu_ = false;
  std::string x16_src_;
  cl_mem x16_ah_ = nullptr, x16_rs_ = nullptr;  // (the f16 activations and each row's scale: x16_reserve)
  size_t x16_ah_bytes_ = 0, x16_rs_bytes_ = 0;
  static int x16_tm() {
    static const int tm = [] {
      const int t = getenv("ARC_X16_TM") ? atoi(getenv("ARC_X16_TM")) : 32;
      if (t == 8 || t == 16 || t == 32) return t;
      zlog::eprintf("inferzero: ARC_X16_TM=%d not supported (8, 16 or 32): 32\n", t);
      return 32;
    }();
    return tm;
  }
  // tokens per sub-group for a chunk of Mp rows: x16_tm(), or fewer where Mp is not a multiple of it (short chunks
  // are padded to 16 rows only); 0: none fits (gemm.cl then)
  static int x16_tm_for(int Mp) {
    for (int t = x16_tm(); t >= 8; t /= 2)
      if (Mp % t == 0) return t;
    return 0;
  }
  // ARC_X16_CG=2 / 4: 16-column groups per sub-group (default 1). Each 8-token read of the activations then feeds 2 or
  // 4 DPAS instead of 1 (sycl-tla's BMG GEMM: 32 x 64 per sub-group); a matrix whose groups it does not divide keeps 1
  static int x16_cg_for(int N) {
    static const int cg = [] {
      const int c = getenv("ARC_X16_CG") ? atoi(getenv("ARC_X16_CG")) : 1;
      if (c == 1 || c == 2 || c == 4) return c;
      zlog::eprintf("inferzero: ARC_X16_CG=%d not supported (1, 2 or 4): 1\n", c);
      return 1;
    }();
    return (N / 16) % cg == 0 ? cg : 1;
  }
  cl_kernel x16_kernel(int qtype, int cg, int tm) {
    static const bool g128 = getenv("ARC_X16_GRF") && std::string(getenv("ARC_X16_GRF")) == "128";
    return kernel("x16", std::string(g128 ? "" : "-cl-intel-256-GRF-per-thread ") + "-DQTYPE=" + std::to_string(qtype) +
                             " -DX16_TM=" + std::to_string(tm) + (cg != 1 ? " -DX16_CG=" + std::to_string(cg) : "") +
                             (x16_emu_ ? " -DX16_EMU" : "") + (cl_.sg8 ? " -DX16_SG8" : ""),
                  "gemm_x16");
  }
  bool x16_wanted() const {
    const bool tr = getenv("ARC_X16_TRY") && atoi(getenv("ARC_X16_TRY")) != 0;
    // (iGPUs: only those with 8-lane DPAS, the 140T, where the int8 XMX GEMM lost to DP4a in its rescaling: gemm_x16.cl
    //  -DX16_SG8, 2026-11-08; the 16-lane iGPUs have not been measured with it)
    return tr || (getenv("ARC_X16") && atoi(getenv("ARC_X16")) != 0 && cl_.xmx && (!zrt::integrated() || cl_.sg8));
  }
  void x16_init() {
    if (!x16_wanted()) return;
    x16_emu_ = !cl_.xmx;
    // (once per device and kind of DPAS: a split model's GPUs are all one device (ClRt::open_devices) and need not
    //  check again; an engine on another device in the same process checks for itself)
    static std::mutex mu;
    static std::map<std::string, bool> known;
    const std::string key = cl_.device_id + (x16_emu_ ? "|emulated" : "");
    std::lock_guard<std::mutex> g(mu);
    auto it = known.find(key);
    if (it == known.end()) {
      const bool ok = x16_check();
      it = known.emplace(key, ok).first;
      if (ok)
        zlog::eprintf("inferzero: f16 XMX on for the W8 prompt GEMMs%s (results checked against DP4a)\n",
                x16_emu_ ? " (DPAS emulated: tests)" : "");
    }
    x16_ = it->second;
  }
  // its buffers at their full size with the others (alloc_buffers), so the memory plan counts them (27B, 4096-token
  // chunks: 143 MB) and a prompt never allocates mid-pass; a dry run (--ctx auto's estimate) counts them too
  void x16_reserve(size_t M, int maxK) {
    if (!x16_ && !(opt.dry && x16_wanted())) return;
    free_buf(x16_ah_);
    free_buf(x16_rs_);
    x16_ah_bytes_ = M * (size_t)maxK * 2;
    x16_rs_bytes_ = M * 4;
    x16_ah_ = buf(x16_ah_bytes_);
    x16_rs_ = buf(x16_rs_bytes_);
  }
  // aq / asc (act32: scales per 32 values, else per 256) of Mp rows to f16 in x16_ah_, the rows' scales in x16_rs_
  void x16_act(cl_mem aq, cl_mem asc, int Mp, int K, int act32) {
    const size_t ab = (size_t)Mp * K * 2, rb = (size_t)Mp * 4;
    if (x16_ah_bytes_ < ab) {  // (only the start-up check: the engine's are reserved whole)
      free_buf(x16_ah_);
      x16_ah_ = buf(ab);
      x16_ah_bytes_ = ab;
    }
    if (x16_rs_bytes_ < rb) {
      free_buf(x16_rs_);
      x16_rs_ = buf(rb);
      x16_rs_bytes_ = rb;
    }
    cl_kernel k = kernel("x16", "-DACT_ONLY", "act_half");
    set_args(k, 0, aq, asc, Mp, K, act32, x16_ah_, x16_rs_);
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    run("x16_act", k, 2, gws, lws);
  }
  // one GEMM, as matmul1 launches gemm.cl (row groups, split-K over ~17 superblocks of long rows); false: not a
  // shape or layout it takes (the caller runs gemm.cl)
  bool x16_launch(const char* name, int qtype, cl_mem wq, cl_mem wh, cl_mem aq, cl_mem asc, int Mp, int N, int K,
                  cl_mem C, int ldc, int c_off, int a, int act32) {
    const int tm = x16_tm_for(Mp);
    if (qtype < 100 || qtype > 103 || !tm || N % 16 || K % 256) return false;
    x16_act(aq, asc, Mp, K, act32);
    const int cg = cl_.sg8 ? 1 : x16_cg_for(N);
    cl_kernel k = x16_kernel(qtype, cg, tm);
    static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;
    static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;
    static const int sk_sb = getenv("ARC_GEMM_SPLIT") ? std::clamp(atoi(getenv("ARC_GEMM_SPLIT")), 0, 4096) : 17;  // (clamped: no overflow of the piece count)
    // (the fused SwiGLU needs the whole sum in one piece: silu(gate) * (u1 + u2), not silu(gate) * u1 + u2)
    const int nsb = K / 256, splits = a != 2 && K > 8192 && sk_sb > 0 ? (nsb + sk_sb - 1) / sk_sb : 1, rgroup = rg_env;
    int wgy = 1;
    while (wgy < wgy_max && (Mp / tm) % (wgy * 2) == 0) wgy *= 2;
    size_t gws[2] = {(size_t)N / cg * (Mp / tm / wgy), (size_t)wgy}, lws[2] = {16, (size_t)wgy};
    for (int sp = 0; sp < splits; ++sp) {
      int s0 = nsb * sp / splits, s1 = nsb * (sp + 1) / splits, ac = sp == 0 ? a : 1, M = Mp;
      set_args(k, 0, wq, wh, x16_ah_, x16_rs_, C, M, N, K, ldc, c_off, ac, rgroup, s0, s1);
      run(name, k, 2, gws, lws);
    }
    return true;
  }
  bool x16_matmul(const char* name, const GpuWeight& w, cl_mem wq, cl_mem wh, int Mp, cl_mem C, int ldc, int c_off,
                  int a, bool act32) {
    return x16_launch(name, w.gemm_qtype, wq, wh, aq_, asc_, Mp, w.N, w.K, C, ldc, c_off, a, act32 ? 1 : 0);
  }
  // random W8 weights and int8 activations (with their true per-16 sums, which the DP4a kernel's min term reads),
  // activation scales spread over three decades (each row's largest divides the rest): every W8 layout, per-32 and
  // per-256 activation scales; stored / added / fused SwiGLU through x16_launch (the engine's grid: a partial row
  // group, a column offset and a row stride wider than N; 96 x 64 at x16_tm(), 80 x 48 at 16 tokens with one column
  // group) and split-K pieces; against gemm.cl without XMX. A case passes when its error is below 0.5 % of the
  // result's size (f16 rounding ~0.03 %; a wrong layout ~100 %)
  bool x16_check() {
    const int K = 2048, nsb = K / 256;
    uint32_t seed = 4242;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    auto rndf = [&](float lo, float span) { return lo + (float)(rnd() % 10000) * (span / 10000.0f); };
    std::string bad;
    double worst = 0;
    std::vector<cl_mem> bufs;
    struct Soft { bool& f; bool was; ~Soft() { f = was; } } soft{cl_.soft_fail, cl_.soft_fail};
    cl_.soft_fail = true;
    bool built = true;
    try {
      for (auto [M, N] : {std::pair{96, 64}, std::pair{80, 48}}) {
        const int ng = N / 16, ldc = N + 32, coff = 16;
        std::vector<uint32_t> wq((size_t)ng * nsb * 4 * 16 * 16), wh((size_t)ng * nsb * 10 * 16);
        for (auto& v : wq) v = rnd();
        for (auto& v : wh) {
          const float f = rndf(0.001f, 0.01f);
          memcpy(&v, &f, 4);
        }
        std::vector<int8_t> aq((size_t)M * K);
        for (auto& v : aq) v = (int8_t)(rnd() >> 24);
        std::vector<float> asc((size_t)(K / 32) * M), c0((size_t)M * ldc);
        for (auto& v : asc) v = 0.02f * std::pow(10.0f, -rndf(0.0f, 3.0f));
        for (auto& v : c0) v = rndf(-2.0f, 4.0f);
        std::vector<uint16_t> ab((size_t)(K / 16) * M);
        for (int m = 0; m < M; ++m)
          for (int b = 0; b < K / 16; ++b) {
            int sum = 0;
            for (int i = 0; i < 16; ++i) sum += aq[(size_t)m * K + b * 16 + i];
            ab[(size_t)b * M + m] = (uint16_t)(int16_t)sum;
          }
        cl_mem bq = buf(wq.size() * 4, wq.data()), bh = buf(wh.size() * 4, wh.data()), ba = buf(aq.size(), aq.data()),
               bs = buf(asc.size() * 4, asc.data()), bb = buf(ab.size() * 2, ab.data()), bc = buf(c0.size() * 4);
        bufs.insert(bufs.end(), {bq, bh, ba, bs, bb, bc});
        for (int q : {101, 100, 103, 102})
          for (int a32 = 1; a32 >= 0; --a32)
            for (int mode = 0; mode < 4; ++mode) {  // store, split-K pieces, add, fused SwiGLU
              std::vector<float> out[2];
              const int acc = mode == 2 ? 1 : mode == 3 ? 2 : 0, pieces = mode == 1 ? 3 : 1;
              for (int x = 0; x < 2; ++x) {
                CL_CHECK(clEnqueueWriteBuffer(cl_.queue, bc, CL_TRUE, 0, c0.size() * 4, c0.data(), 0, nullptr, nullptr));
                if (x == 1 && pieces == 1) {
                  if (!x16_launch("x16_check", q, bq, bh, ba, bs, M, N, K, bc, ldc, coff, acc, a32)) throw std::runtime_error("shape not taken");
                } else {
                  for (int sp = 0; sp < pieces; ++sp) {
                    int s0 = nsb * sp / pieces, s1 = nsb * (sp + 1) / pieces, ac = sp == 0 ? acc : 1;
                    int MM = M, NN = N, KK = K, ld = ldc, co = coff, rg = 2;
                    if (x == 0) {
                      cl_kernel k = gemm_kernel(q, a32 != 0, 0, 1);
                      size_t gws[2] = {(size_t)N * (M / gemm_tm()), 1}, lws[2] = {16, 1};
                      set_args(k, 0, bq, bh, ba, bs, bb, bc, MM, NN, KK, ld, co, ac, rg, s0, s1);
                      run("x16_check", k, 2, gws, lws);
                    } else {
                      x16_act(ba, bs, M, K, a32);
                      const int cg = cl_.sg8 ? 1 : x16_cg_for(N), tm = x16_tm_for(M);
                      cl_kernel k = x16_kernel(q, cg, tm);
                      size_t gws[2] = {(size_t)N / cg * (M / tm), 1}, lws[2] = {16, 1};
                      set_args(k, 0, bq, bh, x16_ah_, x16_rs_, bc, MM, NN, KK, ld, co, ac, rg, s0, s1);
                      run("x16_check", k, 2, gws, lws);
                    }
                  }
                }
                out[x].resize(c0.size());
                CL_CHECK(clEnqueueReadBuffer(cl_.queue, bc, CL_TRUE, 0, out[x].size() * 4, out[x].data(), 0, nullptr, nullptr));
              }
              double e2 = 0, r2 = 0;
              bool fin = true;
              for (size_t i = 0; i < c0.size(); ++i) {  // (the columns outside [coff, coff + N) must stay as they were)
                fin = fin && std::isfinite(out[0][i]) && std::isfinite(out[1][i]);
                e2 += (double)(out[1][i] - out[0][i]) * (out[1][i] - out[0][i]);
                r2 += (double)(out[0][i] - (mode == 2 ? c0[i] : 0.0f)) * (out[0][i] - (mode == 2 ? c0[i] : 0.0f));
              }
              const double rel = r2 > 0 ? std::sqrt(e2 / r2) : 1.0;
              worst = std::max(worst, rel);
              if (!fin || !(rel < 5e-3))
                bad += " " + std::to_string(M) + "x" + std::to_string(N) + "/" + std::to_string(q) + (a32 ? "/act32" : "") +
                       "/" + std::to_string(mode) + " (" + std::to_string(rel) + ")";
            }
      }
    } catch (const zrt::GpuWaitError&) {
      for (cl_mem& m : bufs) free_buf(m);
      throw;  // (a split model's GPU that hangs: not a verdict on the kernels, which is kept per device)
    } catch (const std::exception& e) {
      zlog::eprintf("inferzero: the f16 XMX kernels do not build here (%s): not used\n", e.what());
      built = false;
    }
    for (cl_mem& m : bufs) free_buf(m);
    if (!built) return false;
    if (!bad.empty()) zlog::eprintf("inferzero: f16 XMX results differ from DP4a (%s ): not used (please report)\n", bad.c_str());
    else if (getenv("ARC_X16_TRY")) zlog::eprintf("inferzero: f16 XMX check: largest error %.2e of the result\n", worst);
    return bad.empty();
  }
