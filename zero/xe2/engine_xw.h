// The H8R prompt GEMM on XMX with a wider tile per sub-group (QUEUE 33.4 / 33.4b, kernels/gemm_xw.cl and
// gemm_xw8.cl): a part of class Engine (zero/engine.h), included inside the class body after zero/xe2/engine_xe2.h
// and engine_h8.h; not a header of its own. On by default in the Arc 140T preset and where H8 is the default on a
// discrete card (h8_auto_, below); elsewhere off unless asked for, every path unchanged without the switch.
//
// ARC_GEMM_XW=1: where H8R runs on XMX (h8_ with per-column scales: the Qwen dense models on discrete cards by
// default, the Arc 140T preset's chunks of 512+ tokens, ARC_H8=1 elsewhere), the H8 GEMMs whose shape it takes (M a
// multiple of XW_TM, N of 16 * XW_NC) go through the wide-tile kernel instead of gemm.cl / gemm_x8.cl; the others
// stay. On by default in the Arc 140T preset since 2026-11-11 (32 x 16 tiles, 256 registers: its owner's sweep, issue
// #5, 27B 512 / 4K / 16K prompts 147.6 / 214.1 / 205.4 -> 204.9 / 298.6 / 283.1 tok/s, logits identical; ARC_GEMM_XW=0:
// off). On by default with h8_auto_ (the Qwen dense models on discrete cards with 16-lane XMX) since 2026-11-13: a Pro
// B60's sweep (issue #4, bench\xw_sweep.ps1), 27B 512 / 4K / 16K prompts 691.1 / 1,066.5 / 978.9 -> 716.8 / 1,291.7 /
// 1,169.5 tok/s with 32 x 64 tiles (32 x 32 739.5 / 1,267.5 / 1,147.6; 16 x 64 and 32 x 16 slower than the default),
// logits identical at every shape.
//   16-lane XMX (Xe2 / Xe3): gemm_xw.cl, XW_TM tokens x XW_NC 16-column groups (32 x 4 = 32 x 64).
//   8-lane XMX (the 140T, Alchemist-kind): gemm_xw8.cl, coalesced loads (a group's weights in 4 block reads, an
//   activation row in one), XW_TM x XW_NC (32 x 1 = 32 x 16, both halves, 256 registers; before 2026-11-11 16 x 16 at
//   128, which the 140T measured 26 % slower than 32 x 16 at 256).
//   ARC_GEMM_XW_TM=16 / 32 / 64 (32 by default): tokens per sub-group; ARC_GEMM_XW_NC=1 / 2 / 4, on 16 lanes also 6 / 8
//   (4 on 16 lanes, 1 on 8); ARC_GEMM_XW_GRF=128 / 256: the register file (256 by default; on 8 lanes 32 tokens spill
//   at 128). 64 tokens and 6 / 8 groups (2026-11-13) are for sweeps beyond 32 x 64 (bench\xw_sweep.ps1 -Wide): at
//   256 registers ocloc shows 32 x 96 ~2 bytes spilled, 16 x 128 and 64 x 32 none, 32 x 128 and 64 x 64 heavy spills
//   (B60); NC 6 runs a width's 96-wide part and its rest 64 columns wide (xw_piece).
//   ARC_GEMM_XW_TRY=1: on any GPU with H8R, DPAS emulated where the GPU has none of its kind (tests; no speed
//   meaning; on an iGPU H8 needs ARC_H8=1, or the 140T preset with ARC_140T=1 ARC_XMX_TRY=2).
//   ARC_GEMM_XW_LANES=16: the 16-lane kernel on an 8-lane GPU, DPAS emulated (tests).
// Checked once per device at start against gemm.cl without XMX, bit for bit (xw_check); a difference or a build
// that fails leaves the old kernels and says so.

  bool xw_ = false, xw_emu_ = false;
  std::string xw_src_, xw8_src_;
  // 8 or 16: which kernel (8: gemm_xw8 on GPUs with 8-lane sub-groups, the 140T and the 125H)
  int xw_lanes() const {
    static const int e = getenv("ARC_GEMM_XW_LANES") ? atoi(getenv("ARC_GEMM_XW_LANES")) : 0;
    return cl_.sg8 && e != 16 ? 8 : 16;
  }
  // (8 lanes: 32 x 16 at 256 registers by default: the 140T's sweep (issue #5, 2026-11-11), 27B 4K prompt 16 x 16 at
  //  128 273.7, 32 x 16 at 256 298.6, 16 x 32 at 256 275.9, 32 x 32 at 256 257.9 tok/s; 16 lanes: 32 x 64 at 256)
  int xw_tm() const {
    static const int e = getenv("ARC_GEMM_XW_TM") ? atoi(getenv("ARC_GEMM_XW_TM")) : 0;
    const int def = xw_lanes() == 16 && xe2_g21() ? 64 : 32;  // (the B60's die: 64 x 32, see xe2_g21)
    if (e == 16 || e == 32 || e == 64) return e;
    if (e) { static std::once_flag said; std::call_once(said, [&] { zlog::eprintf("inferzero: ARC_GEMM_XW_TM=%d not supported (16, 32 or 64): %d\n", e, def); }); }
    return def;
  }
  bool xw_grf256() const {
    static const char* e = getenv("ARC_GEMM_XW_GRF");
    if (e && std::string(e) == "128") return false;
    if (e && std::string(e) == "256") return true;
    return true;
  }
  int xw_nc() const {
    static const int e = getenv("ARC_GEMM_XW_NC") ? atoi(getenv("ARC_GEMM_XW_NC")) : 0;
    const int def = xw_lanes() == 8 ? 1 : xe2_g21() ? 2 : 4;
    if (e == 0) return def;
    if (e == 1 || e == 2 || e == 4 || (xw_lanes() == 16 && (e == 6 || e == 8))) return e;
    static std::once_flag said;
    std::call_once(said, [&] { zlog::eprintf("inferzero: ARC_GEMM_XW_NC=%d not supported here: %d\n", e, def); });
    return def;
  }
  static bool xw_try() {
    static const bool t = getenv("ARC_GEMM_XW_TRY") && atoi(getenv("ARC_GEMM_XW_TRY")) != 0;
    return t;
  }
  // (h8_ and xmx_ known: after the XMX and H8 checks of the constructor)
  // (the 140T preset and h8_auto_ (discrete cards): on unless ARC_GEMM_XW=0; elsewhere ARC_GEMM_XW=1)
  bool xw_wanted() const {
    const char* e = getenv("ARC_GEMM_XW");
    const bool on = e ? atoi(e) != 0 : (p140_ || h8_auto_);
    return h8_ && h8_row() && (xw_try() || (on && xmx_));
  }
  // ARC_GEMM_XW_PF=0 / 1 / 2 / 3 (16 lanes; B60 prefill plan 13.2): the weights (1), the activations (2, with 2D
  // loads) or both (3) of the chunk ARC_GEMM_XW_PFD (1 / 2 / 4) chunks ahead prefetched into L1; the same results bit
  // for bit (checked at start like every shape). On by default since 2026-11-14 as weights, 1 chunk ahead: a Pro B60's
  // xw_sweep (manoreken2, issue #4, release 2026.11.14.05), the 27B 4K prompt 1,321.7 -> 1,529.0 tok/s and 512 tokens
  // 737.3 -> 867.2 (pf3 d1 867.4 / 1,478.8; pf2 no gain; deeper is slower), logits identical. =0: off, as before
  static int xw_pf() {
    static const int v = [] {
      const int e = getenv("ARC_GEMM_XW_PF") ? atoi(getenv("ARC_GEMM_XW_PF")) : 1;
      return e >= 0 && e <= 3 ? e : 1;
    }();
    return v;
  }
  static int xw_pfd() {
    static const int v = [] {
      const int e = getenv("ARC_GEMM_XW_PFD") ? atoi(getenv("ARC_GEMM_XW_PFD")) : 1;
      return e == 1 || e == 2 || e == 4 ? e : 1;
    }();
    return v;
  }
  // (with ARC_GEMM_XW_2D's form, and neither ARC_GEMM_XW_PF nor _PFD set: both parts 2 chunks ahead, since 2026-11-17:
  //  a Pro B70's 27B 4K prompt 2,853.2 with 2D alone -> 2,896.7 tok/s, logits identical; the same prefetch without 2D
  //  2,711.4 against 2,771.7, so the plain form keeps weights 1 ahead (joshdev-de, issue #27, 1a7ac94))
  bool xw_pf_2d_def() const { return xw_2d() && !getenv("ARC_GEMM_XW_PF") && !getenv("ARC_GEMM_XW_PFD"); }
  // (a Data Center GPU Max with neither set: the activations' part 1 chunk ahead, as ARC_GEMM_XW_PF=2; see xehpc_on)
  bool xw_pf_max_def() const {
    static const bool set = getenv("ARC_GEMM_XW_PF") || getenv("ARC_GEMM_XW_PFD");  // (once: runs per launch)
    return !set && xehpc_on();
  }
  int xw_pf_v() const { return xw_pf_max_def() ? 2 : xw_pf_2d_def() ? 3 : xw_pf(); }
  // (the B60's die: 1 chunk ahead in the 2D form, see xe2_g21)
  int xw_pfd_v() const { return xw_pf_max_def() ? 1 : xw_pf_2d_def() ? (xe2_g21() ? 1 : 2) : xw_pfd(); }
  // (the activations' part needs 2D loads: without them, or with DPAS emulated, only the weights' part is built)
  int xw_pf_eff() const {
    // (8 lanes, gemm_xw8: the weights' part by default in the 140T preset (p140_, also Alchemist cards) since
    //  2026-11-16: a 140T's 27B 8K prompt 359.0 / 359.3 -> 366.4 / 366.9 tok/s, +2 % (PascalSoftwares, issue #5,
    //  release 2026.11.16.12; 2 and 3 slower: 341.9 / 338.5); a prefetch reads no value into the sums. Otherwise
    //  only when ARC_GEMM_XW_PF is set; the activations' part with the packed layout (XW_PACKA) only)
    if (xw_lanes() != 16) return (getenv("ARC_GEMM_XW_PF") || p140_) ? xw_pf() : 0;
    if (xw_pf_max_def()) return (!x2d_ || xw_emu_) ? 0 : 2;  // (as ARC_GEMM_XW_PF=2 set by hand: 2 & 1 without 2D)
    return (!x2d_ || xw_emu_) ? (xw_pf() & 1) : xw_pf_v();
  }
  std::string xw_pf_defs() const {
    const int pf = xw_pf_eff();
    return pf ? " -DXW_PF=" + std::to_string(pf) + " -DXW_PFD=" + std::to_string(xw_lanes() == 16 ? xw_pfd_v() : xw_pfd()) : "";
  }
  // ARC_GEMM_XW_PACK (8 lanes: the Arc 140T; issue #5, its xw_peak of 2026-11-13: packAW 32 x 16 against the default,
  // the 27B's FFN shapes, 4K 23.4 -> 25.0 TOPS, 512 21.6 -> 24.5): the activations (1), the weights (2) or both (3)
  // written in the DPAS operand order gemm_xw8 -DXW_PACKA / -DXW_PACKW reads, by the kernels that make them (h8r_act;
  // h8r_decode or h8r_weights), so without a repacking pass. Both by default in the 140T preset, 0 off; only where
  // gemm_xw8 takes the GEMM (its shapes; resident weights, converted once in place, stay in the plain order). The same
  // values in another order: the same results, checked at start for every form used (xw_check).
  static int xw_pack_env() {
    static const int v = [] {
      const char* e = getenv("ARC_GEMM_XW_PACK");
      return e ? std::clamp(atoi(e), 0, 3) : -1;
    }();
    return v;
  }
  int xw_pack_cfg() const {  // (the forms this engine may use; xw_ itself is decided by the check)
    // (one column group a sub-group, the default on 8 lanes, or two (32 x 32) with ARC_GEMM_XW_WGX 2 / 4 (2026-11-17:
    //  32 x 32 packed alone was slower on the 140T, 28 % of the XMX peak against 33 %, issue #5 2026-11-13). Every H8
    //  weight's N is a multiple of 16, and on 8 lanes a width that is not one of 32 runs one column group a sub-group
    //  (xw_nc_for): a packed input is never left to a GEMM gemm_xw8 declines; the packed orders do not depend on NC.
    //  Review 2026-11-14)
    if (xw_lanes() != 8 || !h8_row() || !(xw_nc() == 1 || (xw_nc() == 2 && xw_wgx() > 1))) return 0;
    const int e = xw_pack_env();
    return e >= 0 ? e : (p140_ ? 3 : 0);
  }
  bool xw_pack_a(int Mp, int K) const { return xw_ && (xw_pack_cfg() & 1) && Mp % xw_tm() == 0 && K % 256 == 0; }
  bool xw_pack_w(const GpuWeight& w, int Mp) const {
    return xw_ && (xw_pack_cfg() & 2) && (!w.wq || p140_res_) && Mp % xw_tm() == 0 && xw_takes_n(w.N) && w.K % 256 == 0;
  }
  // ARC_GEMM_XW_WGX=1 / 2 / 4 / 8 (16 lanes, 2026-11-15): column sub-groups per work-group next to the WGY token ones,
  // (a work-group of 256 tokens x 128 columns, 32 sub-groups), so the activations of a chunk are shared in L1
  // by the work-group's column blocks; ARC_GEMM_XW_LOCK=1: the work-group in step around each 64-value chunk (a split
  // barrier). Both the same results bit for bit (checked at start). 4 by default on discrete cards since 2026-11-15: a
  // Pro B70's 27B 4K prompt 2,311 -> 2,396 tok/s in two rounds (+3.7 %; gemm_down 215 -> 191 ms; 2: +2.6 %, 8 with
  // LOCK: -1.7 %, LOCK alone +0.2 %), logits identical (joshdev-de, issue #27, 6466672). iGPUs: 1.
  // 8 lanes (gemm_xw8, the Arc 140T; 2026-11-17, opt-in): ARC_GEMM_XW_WGX=2 / 4 and ARC_GEMM_XW_LOCK=1 the same way;
  // 1 and off by default (the 140T's default unchanged). Its xw_peak --probe (issue #5, 2026-11-16) reached 85 % of
  // the XMX peak at 32 x 32 with the operands in L1, ~33 % in the engine; with WGX packed 32 x 32 is allowed too
  int xw_wgx() const {
    static const int v = [] {
      if (!getenv("ARC_GEMM_XW_WGX")) return zrt::integrated() ? 1 : 4;
      const int e = atoi(getenv("ARC_GEMM_XW_WGX"));
      return e == 2 || e == 4 || e == 8 ? e : 1;
    }();
    // (8 lanes: 2 by default in the Arc 140T preset since 2026-11-17: a 140T's 27B 4K prompt 368.3 -> 390.0 tok/s
    //  (4: 383.2; 4 with LOCK 365.8), the same results checked at start (PascalSoftwares, issue #5, release .07))
    static const int v8 = [] {
      const int e = getenv("ARC_GEMM_XW_WGX") ? atoi(getenv("ARC_GEMM_XW_WGX")) : 0;
      return e == 2 || e == 4 ? e : e ? 1 : 0;
    }();
    // (a Data Center GPU Max: 8 by default, see xehpc_on in zero/xe2/engine_xe2.h)
    static const bool set = getenv("ARC_GEMM_XW_WGX") != nullptr;
    if (xw_lanes() == 16) return !set && (xehpc_on() || xe2_g21()) ? 8 : v;  // (the B60's die: 8, see xe2_g21)
    return v8 ? v8 : p140_ ? 2 : 1;
  }
  bool xw_lock() const {
    static const bool v = getenv("ARC_GEMM_XW_LOCK") && atoi(getenv("ARC_GEMM_XW_LOCK")) != 0;
    return v;
  }
  // ARC_GEMM_XW_2D (2026-11-17, 16 lanes with the 2D block loads): gemm_xw.cl -DXW_2D, the same bytes in the
  // same DPAS operands in fewer messages (the 32 activation rows in one read, a column group's weights in one, C in
  // 8-row blocks): 16 -> 9 a chunk in the loop at 32 x 64, the epilogue's 128 -> 16. Checked at start as every shape.
  // On by default on discrete cards since 2026-11-17 (=0 off): a Pro B70's 27B 4K prompt 2,768.1 -> 2,853.2 tok/s,
  // with the prefetch above 2,896.7, logits identical (joshdev-de, issue #27). iGPUs (the B390) opt-in, unmeasured
  // (the default per engine, set in xw_init on its GPU's thread: one process may hold an iGPU's engine and a card's)
  // =1 (the default on discrete cards): the loads and C's 2D reads / writes (-DXW_2D -DXW_2D_C); =2: the loads only,
  // C as in the plain form. A failed start check falls back to the plain form (=2's loads alone were slower than it
  // there, see xw_init). A Pro B60 under Windows
  // (manoreken2, issue #4, release .06) failed the full form's check in the add and split-K modes only, which add C's
  // old value to the product: that driver's compiler fuses `product + old` into a mad differently in the 2D epilogue
  // than in gemm.cl, while the stores, the fused SwiGLU (which 2D-reads C too) and so the loads were right. With C in
  // the plain form the epilogue is gemm.cl's code again
  bool xw_2d_discrete_ = false;
  int xw_2d_fb_ = 0;  // (the start check's fallback: 0 none, 1 C's 2D form dropped, 2 the 2D form dropped)
  // (C in 2D blocks failing the start check in the add modes: tried once more with gemm_xw.cl -DXW_C_RND, the product
  //  rounded before the add, before the 2D form is dropped. A Data Center GPU Max 1550 (driver 25.18) fails by 1-6 ulp
  //  there and passes with it: the 27B 4K prompt on one stack 2,460 -> 2,638 tok/s, logits identical to the plain
  //  form's. ARC_GEMM_XW_C_RND=1: from the start, =0: never)
  bool xw_c_rnd_ = false;
  int xw_2d_level() const {  // (0 off, 1 the loads and C, 2 the loads only, before the fallback)
    static const char* e = getenv("ARC_GEMM_XW_2D");
    const int v = e ? std::clamp(atoi(e), 0, 2) : (xw_2d_discrete_ ? 1 : 0);
    return xw_lanes() == 16 && x2d_ && !xw_emu_ ? v : 0;
  }
  bool xw_2d() const { return xw_2d_level() != 0 && xw_2d_fb_ < 2; }     // (the loads in 2D blocks)
  bool xw_2d_c() const { return xw_2d_level() == 1 && xw_2d_fb_ == 0; }  // (and C's reads / writes)
  // ARC_GEMM_XW_GU (2026-11-20, opt-in at first, the default below; the B70's 3,000 plan): the FFN's gate and up GEMMs as one gemm_xw launch
  // (gemm_xw.cl -DXW_GU), its epilogue writing silu(gate) * up: the gate's f32 values (285 MB a layer for a 4K prompt
  // of the 27B) are neither written to big_ nor read back by the up GEMM. 16 lanes, NC 2 / 4 / 8; the same results bit
  // for bit as the two launches (xw_gu_check at start; a difference or a failed build keeps the two launches)
  // On by default on discrete cards since 2026-11-20: a Pro B60's 27B 4K / 16K prompt 1,805.8 / 1,764.9 -> 1,865.9 /
  // 1,831.6 tok/s (+3.3 / +3.8 %), a 4K code prompt's logits byte for byte the default's (skipzx-lab, issue #57,
  // a62be02). iGPUs (the 140V / B390) opt-in, unmeasured. ARC_GEMM_XW_GU=0: two launches as before. (The default per
  // call: each GPU of a split decides for its own, as ARC_H8_GRF)
  static bool xw_gu_env() {
    static const int e = getenv("ARC_GEMM_XW_GU") ? (atoi(getenv("ARC_GEMM_XW_GU")) != 0 ? 1 : 0) : -1;
    return e >= 0 ? e == 1 : !zrt::integrated();
  }
  int xw_gu_ok_ = -1;  // (-1 not checked yet, 0 differs / fails, 1 checked)
  // ARC_GEMM_XW_DB (8 lanes, 2026-11-20, opt-in at first, the default below; the A770 row's G3): gemm_xw8.cl -DXW_DB, its packed loop with the
  // next token group's and the next chunk's operands read one step ahead of their DPAS (where both operands are
  // packed: the Alchemist / 140T preset). The same DPAS in the same order: bit for bit, checked at start
  // Only for 32-token tiles, and by default on Alchemist (discrete 8-lane) since 2026-11-21: hansong80's A770 (#45,
  // main 2a78721, the 27B 4K prompt) gained on every 32 x 16 GEMM (down -10 %, qkv / z -8 %, q -7 %, up -6 %, gate
  // -4 %) and lost on every 16 x 32 one (ssm_out / o +137-139 %, k / v +72-82 %): -4.9 % in all with it everywhere.
  // ARC_GEMM_XW_DB=0: off, =1: on (any tile, as asked; 2026-11-21) on any 8-lane GPU, the 140T included; =2 (opt-in, 2026-11-21):
  // -DXW_DB2, the next B in place and the chunk loop unrolled by 4 (no spill on ocloc dg2 where the =1 form spills
  // unrolled), on every tile, 16 x 32 included
  int xw_db_env() const {
    static const int e = getenv("ARC_GEMM_XW_DB") ? std::max(0, std::min(2, atoi(getenv("ARC_GEMM_XW_DB")))) : -1;
    return e;
  }
  bool xw_db(int tm) const {
    const int e = xw_db_env();
    if (e == 2 || e == 1) return true;  // (=1 asked: on any tile, as xw_peak --all measures it, 2026-11-21)
    if (tm != 32 || e == 0) return false;
    return cl_.sg8 && !zrt::integrated();
  }
  // ARC_GEMM_XW_SLM=N (opt-in, 8 lanes, 2026-11-21; the A770 plan: B60 speed): gemm_xw8.cl -DXW_SLM, each work-group's
  // packed chunk copied to local memory once and double-buffered, its sub-groups' DPAS fed from there, N token
  // sub-groups a work-group (2, 4 or 8; =1 means 4). Where both operands are packed (the Alchemist / 140T preset) and N
  // divides the token tiles; else the kernel as before. Integer sums in the same order: bit for bit, checked at start
  // (and emulated on the 125H with bench\xw_peak --emu). ocloc dg2-g10 at 256 registers, 32 x 16 with 4 x 4
  // sub-groups: no spill, 24 KB of local memory, 249 memory messages against 299 for the default
  static int xw_slm_wgy() {
    static const int e = [] {
      const int v = getenv("ARC_GEMM_XW_SLM") ? atoi(getenv("ARC_GEMM_XW_SLM")) : 0;
      return v == 1 ? 4 : v == 2 || v == 4 || v == 8 ? v : 0;
    }();
    return e;
  }
  // ARC_GEMM_XW_EPI2=1 (opt-in, 8 lanes, 2026-11-21; the A770 plan): gemm_xw8.cl -DXW_EPI2, the epilogue a row at a
  // time, a row's columns in one block message, SwiGLU's mode decided once. ocloc dg2-g10 at 256 registers, no spill:
  // 32 x 16 w4 with the read-ahead loop 239 -> 175 memory messages, 32 x 32 525 -> 269, 32 x 32 through local memory
  // 457 -> 201. The same operations per value: bit for bit, checked at start (and xw_peak --emu on the 125H)
  // (on by default on Alchemist since 2026-11-21: skipzx-lab's A770, #45, the 27B 4K 1,286 -> 1,322 tok/s on today's
  //  tile, and with the big tile below 1,488; ARC_GEMM_XW_EPI2=0 off, =1 on any 8-lane GPU. The 140T: until measured)
  bool xw_epi2() const {
    static const int e = getenv("ARC_GEMM_XW_EPI2") ? (atoi(getenv("ARC_GEMM_XW_EPI2")) != 0 ? 1 : 0) : -1;
    return e >= 0 ? e == 1 : xw_lanes() == 8 && cl_.sg8 && !zrt::integrated();
  }
  // The big tile on Alchemist, by default since 2026-11-21 (skipzx-lab's A770 at 150 W, #45, b9f011c, the 27B IQ3_S):
  // every prompt GEMM at 32 x 32 (NC 2) with 4 column sub-groups, through local memory with 4 token sub-groups a
  // work-group (gemm_xw8 -DXW_SLM) and the epilogue a row at a time (EPI2): 4K 1,286.0 -> 1,488.4 tok/s (+15.7 %), 16K
  // 1,191.6 -> 1,358.1 (+14.0 %), the GEMMs 2,168 -> 1,690 ms, every GEMM faster, the 16 x 32 ones too (ssm_out
  // 164.2 -> 114.6, o 55.0 -> 38.3); xw_peak 116.7 TOPS, 50 % of its 232. Only where it fits: both operands packed, N
  // a multiple of 128 and the token tiles of 4: 32 x 32 without local memory spills (548.8 tok/s), so the others keep
  // the per-shape tiles above (xw_tile_for). Decided per call, so a split model's GPU decides for its slice of N.
  // The same sums in the same order: bit for bit, checked at start. ARC_GEMM_XW_BIG=0 off, =1 on any 8-lane GPU
  // (tests); any of ARC_GEMM_XW_TM / _NC / _WGX / _SLM set: off, as asked
  // (the big tile's shape for the sweep, 2026-11-21, opt-in: ARC_GEMM_XW_SLM_TILE=TMxCOLS (16 / 32 / 64 each) the sub-group
  //  tile, ARC_GEMM_XW_SLM_WG="X,Y" column x token sub-groups (2 / 4 / 8 each, X x Y <= 32), ARC_GEMM_XW_SLM_CPS=1 / 2
  //  / 4 chunks a barrier stage (gemm_xw8 -DXS_CPS). ocloc dg2-g10 at 256 registers, no spill: 32 x 32 4x4 / 4x8 / 8x4
  //  and 64 x 16 4x4 / 8x4 / 8x2 at CPS 1, 32 x 16 8x4 at CPS 1 or 2; CPS 2 spills at 32 x 32 (3.1-3.6 KB) and 64 x 16
  //  8x2 (0.6 KB). Unset: 32 x 32, 4 x 4, CPS 1, the default)
  struct XwBig { int tm = 32, nc = 2, x = 4, y = 4, cps = 1; };
  static const XwBig& xw_big_cfg() {
    static const XwBig c = [] {
      XwBig b;
      if (const char* t = getenv("ARC_GEMM_XW_SLM_TILE")) {  // (TMxCOLS, each 16 / 32 / 64: xw_peak --all's forms)
        int tm = 0, cols = 0;
        if (sscanf(t, "%dx%d", &tm, &cols) == 2 && (tm == 16 || tm == 32 || tm == 64) && (cols == 16 || cols == 32 || cols == 64))
          b.tm = tm, b.nc = cols / 16;
        else zlog::eprintf("inferzero: ARC_GEMM_XW_SLM_TILE=%s not TMxCOLS with 16 / 32 / 64 each: 32x32\n", t);
      }
      if (const char* w = getenv("ARC_GEMM_XW_SLM_WG")) {
        int x = 0, y = 0;
        if (sscanf(w, "%d,%d", &x, &y) == 2 && (x == 2 || x == 4 || x == 8) && (y == 2 || y == 4 || y == 8) && x * y <= 32)
          b.x = x, b.y = y;
        else zlog::eprintf("inferzero: ARC_GEMM_XW_SLM_WG=%s not X,Y with 2 / 4 / 8 each and X x Y <= 32: 4,4\n", w);
      }
      if (const char* c2 = getenv("ARC_GEMM_XW_SLM_CPS")) {
        const int v = atoi(c2);
        b.cps = v == 2 || v == 4 ? v : 1;
      }
      // (two stages of CPS chunks in local memory, 64 KB a work-group on DG2: else the default tile)
      const long chunk = (long)b.y * (b.tm / 8) * 512 + (long)b.x * b.nc * 1024;
      if (2 * b.cps * chunk > 65536) {
        zlog::eprintf("inferzero: the big tile %dx%d, %d,%d, CPS %d needs %ld KB of local memory (64): 32x32, 4,4, CPS 1\n",
                      b.tm, 16 * b.nc, b.x, b.y, b.cps, 2 * b.cps * chunk / 1024);
        b = XwBig();
      }
      return b;
    }();
    return c;
  }
  std::string xw_big_key() const {
    const XwBig& b = xw_big_cfg();
    return std::to_string(b.tm) + "x" + std::to_string(16 * b.nc) + "," + std::to_string(b.x) + "x" + std::to_string(b.y) + "c" + std::to_string(b.cps);
  }
  // ARC_GEMM_XW_PTR=1 (2026-11-21, opt-in; the kernel-loop audit, docs/xe2-tuning.md 16): the local-memory forms'
  // copy with its addresses set once before the loop (gemm_xw8.cl -DXW_PTR). ocloc dg2-g10, the big tile's K loop:
  // 410 -> 184 instructions for its 33 DPAS (12.4 -> 5.6 each), no spill: the loop was bound by issuing them. The
  // same words to the same places: bit for bit, as the start check confirms
  // On by default on Alchemist since 2026-11-21 (=0 off): skipzx-lab's A770 (xe, 150 W, .21.07, #45), the 27B 4K prompt
  // 1,432.6 / 1,431.8 -> 1,527.8 / 1,528.0 tok/s (+6.7 %, rounds 0 / 1 / 0 / 1), down / up / gate 444.3 / 437.6 / 382.9
  // -> 413.9 / 386.8 / 329.7 ms, xw_peak 103.5 -> 114.9 T "identical", the logits' sha256 the same. The 140T (an iGPU,
  // whose big tile is opt-in): only with =1, until measured
  bool xw_ptr() const {
    static const int e = getenv("ARC_GEMM_XW_PTR") ? (atoi(getenv("ARC_GEMM_XW_PTR")) != 0 ? 1 : 0) : -1;
    return e >= 0 ? e == 1 : cl_.sg8 && !zrt::integrated();
  }
  bool xw_big_on() const {
    static const int env = getenv("ARC_GEMM_XW_BIG") ? atoi(getenv("ARC_GEMM_XW_BIG")) : -1;
    static const bool fixed = getenv("ARC_GEMM_XW_TM") || getenv("ARC_GEMM_XW_NC") || getenv("ARC_GEMM_XW_WGX") ||
                              getenv("ARC_GEMM_XW_SLM");
    if (env == 0 || fixed || xw_nc_e() || xw_lanes() != 8 || xw_tm() != 32) return false;
    return env == 1 || (cl_.sg8 && !zrt::integrated());
  }
  bool xw_gu_shape(int nc) const { return nc == 2 || nc == 4 || nc == 8; }
  // (nc: column groups per sub-group, 0 = xw_nc(); the 32 x 96 tiles' tail runs the nc 4 kernel. pack: gemm_xw8's
  //  packed operands, bit 1 the activations, bit 2 the weights. wgx: column sub-groups per work-group; gu: -DXW_GU)
  //  slm: -DXW_SLM with that many token sub-groups a work-group (0: not), in place of the read-ahead loop)
  cl_kernel xw_kernel(int nc = 0, int pack = 0, int wgx = 1, bool gu = false, int tm = 0, int slm = 0) {
    if (slm > 0 && xw_lanes() == 8 && (pack & 3) == 3 && !gu) {
      const std::string sd = std::string(xw_grf256() ? "-cl-intel-256-GRF-per-thread " : "") + "-DXW_TM=" +
                             std::to_string(tm ? tm : xw_tm()) + " -DXW_NC=" + std::to_string(nc ? nc : xw_nc()) +
                             (xw_emu_ ? " -DXW_EMU" : "") + " -DXW_PACKA -DXW_PACKW" +
                             (wgx > 1 ? " -DXW_WGX=" + std::to_string(wgx) : std::string()) + " -DXW_SLM -DXW_WGY=" +
                             std::to_string(slm) + (xw_epi2() ? " -DXW_EPI2" : "") +
                             (xw_big_on() && xw_big_cfg().cps > 1 ? " -DXS_CPS=" + std::to_string(xw_big_cfg().cps) : std::string()) +
                             (xw_ptr() ? " -DXW_PTR" : "");
      return kernel("xw8", sd, "gemm_xw8");
    }
    const std::string defs = std::string(xw_grf256() ? "-cl-intel-256-GRF-per-thread " : "") + "-DXW_TM=" +
                             std::to_string(tm ? tm : xw_tm()) + " -DXW_NC=" + std::to_string(nc ? nc : xw_nc()) +
                             (xw_emu_ ? " -DXW_EMU" : "") + xw_pf_defs() +
                             (xw_lanes() == 8 && (pack & 1) ? " -DXW_PACKA" : "") +
                             (xw_lanes() == 8 && (pack & 2) ? " -DXW_PACKW" : "") +
                             (wgx > 1 ? " -DXW_WGX=" + std::to_string(wgx) : std::string()) +
                             (xw_lock() ? " -DXW_LOCK" : "") + (gu ? " -DXW_GU" : "") +
                             (xw_lanes() == 8 && (pack & 3) == 3 && xw_db(tm ? tm : xw_tm()) ? (xw_db_env() == 2 ? " -DXW_DB -DXW_DB2" : " -DXW_DB") : "") +
                             (xw_lanes() == 8 && xw_epi2() && !gu ? " -DXW_EPI2" : "");
    if (xw_lanes() == 8) return kernel("xw8", defs, "gemm_xw8");
    // (not the 32 x 96 tiles (NC 6): with the 2D form they spill 2 KB, ocloc bmg / ptl; 32 x 64 none, 489 -> 77 sends)
    const bool d2 = (nc ? nc : xw_nc()) != 6;
    return kernel("xw", defs + (x2d_ && !xw_emu_ ? " -DX2D" : "") + (xw_2d() && d2 ? " -DXW_2D" : "") +
                            (xw_2d_c() && d2 ? " -DXW_2D_C" : "") + (xw_2d_c() && d2 && xw_c_rnd_ ? " -DXW_C_RND" : ""),
                  "gemm_xw");
  }
  // one piece [s0, s1) of K over N columns from 16-column group g_base (16 lanes; 0 on 8): the grid of gemm_xw.cl's /
  // gemm_xw8.cl's header (a sub-group of 16 or 8 work-items). With ARC_GEMM_XW_WGX the widest column count up to it
  // that divides the column blocks, and fewer token sub-groups if the work-group would not fit a Xe-core's threads (32
  // at 256 registers, 64 at 128)
  // (tm / wgx0: the tile's tokens and column sub-groups, 0 = xw_tm() / xw_wgx(): xw_tile_for's per shape)
  void xw_run(const char* name, int nc, int pack, cl_mem wq, cl_mem wh, cl_mem aq, cl_mem asc, int M, int N, int K,
              cl_mem C, int ldc, int c_off, int a, int rgroup, int s0, int s1, int wgy, int g_base, int tm0 = 0,
              int wgx0 = 0, int slm0 = -1) {
    const int tm = tm0 ? tm0 : xw_tm(), nb = N / (16 * nc), ln = xw_lanes();
    int wgx = wgx0 ? wgx0 : xw_wgx();
    while (wgx > 1 && nb % wgx) wgx /= 2;
    const int cap = xw_grf256() ? 32 : 64;
    while (wgx * wgy > cap && wgy > 1) wgy /= 2;
    // (ARC_GEMM_XW_SLM, or the big tile's 4 (slm0): its fixed token sub-groups where they divide the tokens and both
    //  operands are packed; else the kernel as before)
    const int sw = slm0 >= 0 ? slm0 : xw_slm_wgy();
    const bool slm = ln == 8 && (pack & 3) == 3 && sw > 0 && (M / tm) % sw == 0 && wgx * sw <= cap;
    if (slm) wgy = sw;
    cl_kernel k = xw_kernel(nc, pack, wgx, false, tm, slm ? sw : 0);
    size_t gws[2] = {(size_t)ln * nb * (M / tm / wgy), (size_t)wgy}, lws[2] = {(size_t)ln * wgx, (size_t)wgy};
    if (ln == 8) set_args(k, 0, wq, wh, aq, asc, C, M, N, K, ldc, c_off, a, rgroup, s0, s1);
    else set_args(k, 0, wq, wh, aq, asc, C, M, N, K, ldc, c_off, a, rgroup, s0, s1, g_base);
    run(name, k, 2, gws, lws);
  }
  // gate and up in one launch (-DXW_GU): N output columns, the sub-grid of xw_run with NC / 2 output groups a sub-group
  void xw_gu_run(const char* name, cl_mem wqg, cl_mem whg, cl_mem wqu, cl_mem whu, cl_mem aq, cl_mem asc, int M, int N,
                 int K, cl_mem C, int ldc, int c_off, int rgroup, int wgy) {
    const int nc = xw_nc(), tm = xw_tm(), nb = N / (16 * (nc / 2));
    int wgx = xw_wgx();
    while (wgx > 1 && nb % wgx) wgx /= 2;
    const int cap = xw_grf256() ? 32 : 64;
    while (wgx * wgy > cap && wgy > 1) wgy /= 2;
    cl_kernel k = xw_kernel(nc, 0, wgx, true);
    size_t gws[2] = {(size_t)16 * nb * (M / tm / wgy), (size_t)wgy}, lws[2] = {(size_t)16 * wgx, (size_t)wgy};
    int a = 2, s0 = 0, s1 = K / 256, gb = 0;
    set_args(k, 0, wqg, whg, aq, asc, C, M, N, K, ldc, c_off, a, rgroup, s0, s1, gb, wqu, whu);
    run(name, k, 2, gws, lws);
  }
  // the FFN's gate and up (H8, prompt chunks) in one gemm_xw launch into C = big_ (row stride ldc): true when taken;
  // false: the caller runs the two GEMMs as before
  bool xw_gu_matmul(const GpuWeight& gw, const GpuWeight& uw, int Mp, cl_mem C, int ldc, bool act32) {
    if (!xw_gu_env() || !xw_ || xw_lanes() != 16 || !xw_gu_shape(xw_nc()) || !h8_row()) return false;
    if (gw.N != uw.N || gw.K != uw.K || Mp % xw_tm() || gw.K % 256 || gw.N % (16 * (xw_nc() / 2))) return false;
    // (one piece of K only: the gate GEMM alone splits K past ARC_GEMM_SPLIT's superblocks, which the fused launch
    //  does not, so it would no longer be bit-identical there; the caller's fused path needs K <= 8192 anyway. The
    //  Researcher's review, 2026-11-20)
    if (h8_splits(gw.K, 0) > 1) return false;
    if (!h8_takes(gw, Mp) || !h8_takes(uw, Mp) || p140_take(gw, Mp) || p140_take(uw, Mp)) return false;
    if (xw_gu_ok_ < 0) {
      xw_gu_ok_ = xw_gu_check() ? 1 : 0;
      zlog::eprintf(xw_gu_ok_ ? "inferzero: the FFN's gate and up in one gemm_xw launch (ARC_GEMM_XW_GU, results checked bit for bit)\n"
                              : "inferzero: ARC_GEMM_XW_GU: the fused gate / up GEMM differs or does not build: two launches as before\n");
    }
    if (!xw_gu_ok_) return false;
    cl_mem wqg = nullptr, whg = nullptr, wqu = nullptr, whu = nullptr;
    h8_prepare(gw, wqg, whg);
    h8_prepare(uw, wqu, whu);
    h8_act(Mp, gw.K, act32);
    static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;
    static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;
    int wgy = 1;
    while (wgy < wgy_max && (Mp / xw_tm()) % (wgy * 2) == 0) wgy *= 2;
    xw_gu_run("gemm_gateup", wqg, whg, wqu, whu, h8_aq_, h8_asc_, Mp, gw.N, gw.K, C, ldc, 0, rg_env, wgy);
    return true;
  }
  // the fused launch against the two it replaces (gate stored, then up with the fused SwiGLU), both through gemm_xw,
  // on random H8R data: two shapes, work-groups of 1 and 2 token sub-groups; bit for bit
  bool xw_gu_check() {
    // (widths in the two-launch path's tiles, 16 * NC: the fused one's half-width tiles cover them too)
    const int K = 2048, nsb = K / 256, tm = xw_tm(), cols = 16 * xw_nc();
    uint32_t seed = 4242;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    auto rndf = [&](float lo, float span) { return lo + (float)(rnd() % 10000) * (span / 10000.0f); };
    std::vector<cl_mem> bufs;
    struct Soft { bool& f; bool was; ~Soft() { f = was; } } soft{cl_.soft_fail, cl_.soft_fail};
    cl_.soft_fail = true;
    bool ok = true;
    try {
      std::vector<std::pair<int, int>> shapes = {{3 * tm, 2 * cols}, {2 * tm, 3 * cols}};
      if (xw_wgx() > 1) shapes.push_back({2 * tm, 2 * xw_wgx() * cols});
      for (auto [M, N] : shapes) {
        const int ng = N / 16, ldc = N + 32;
        std::vector<uint32_t> wg((size_t)ng * nsb * 4 * 256), wu(wg.size()), hg((size_t)ng * 16), hu(hg.size());
        for (auto* v : {&wg, &wu}) for (auto& x : *v) x = rnd();
        for (auto* v : {&hg, &hu})
          for (auto& x : *v) { const float f = rndf(0.001f, 0.01f); memcpy(&x, &f, 4); }
        std::vector<int8_t> aq((size_t)M * K);
        for (auto& v : aq) v = (int8_t)(rnd() >> 24);
        std::vector<float> asc((size_t)M), c0((size_t)M * ldc, 0.0f);
        for (auto& v : asc) v = 0.02f * std::pow(10.0f, -rndf(0.0f, 3.0f));
        cl_mem bg = buf(wg.size() * 4, wg.data()), bu = buf(wu.size() * 4, wu.data()), bhg = buf(hg.size() * 4, hg.data()),
               bhu = buf(hu.size() * 4, hu.data()), ba = buf(aq.size(), aq.data()), bs = buf(asc.size() * 4, asc.data()),
               bc = buf(c0.size() * 4);
        bufs.insert(bufs.end(), {bg, bu, bhg, bhu, ba, bs, bc});
        for (int wgy : {1, 2}) {
          if ((M / tm) % wgy) continue;
          std::vector<float> out[2];
          for (int x = 0; x < 2; ++x) {
            CL_CHECK(clEnqueueWriteBuffer(cl_.queue, bc, CL_TRUE, 0, c0.size() * 4, c0.data(), 0, nullptr, nullptr));
            if (x == 0) {
              xw_piece("xw_gu_check", bg, bhg, ba, bs, M, N, K, bc, ldc, 0, 0, 2, 0, nsb, wgy);
              xw_piece("xw_gu_check", bu, bhu, ba, bs, M, N, K, bc, ldc, 0, 2, 2, 0, nsb, wgy);
            } else {
              xw_gu_run("xw_gu_check", bg, bhg, bu, bhu, ba, bs, M, N, K, bc, ldc, 0, 2, wgy);
            }
            out[x].resize(c0.size());
            CL_CHECK(clEnqueueReadBuffer(cl_.queue, bc, CL_TRUE, 0, out[x].size() * 4, out[x].data(), 0, nullptr, nullptr));
          }
          bool sane = false;
          for (size_t i = 0; i < c0.size(); ++i) {
            if (!std::isfinite(out[0][i])) { sane = false; break; }
            sane = sane || out[0][i] != 0.0f;
          }
          size_t diff = 0;
          for (size_t i = 0; i < c0.size(); ++i) diff += memcmp(&out[0][i], &out[1][i], 4) != 0;
          if (!sane || diff) {
            zlog::eprintf("inferzero: ARC_GEMM_XW_GU check %dx%d wgy %d: %zu of %zu values differ%s\n", M, N, wgy, diff,
                          c0.size(), sane ? "" : " (or no output)");
            ok = false;
          }
        }
      }
    } catch (const zrt::GpuWaitError&) {
      for (cl_mem& b : bufs) free_buf(b);
      throw;
    } catch (const std::exception& e) {
      zlog::eprintf("inferzero: ARC_GEMM_XW_GU: %s\n", e.what());
      ok = false;
    }
    for (cl_mem& b : bufs) free_buf(b);
    return ok;
  }
  // ARC_GEMM_XW_NC=6 (96 columns, 16 lanes): N's first columns in 96-wide tiles, as many as leave a multiple of 64 for
  // the NC 4 kernel (the 27B's widths 1024, 5120, 17408 are not multiples of 96); -1: N has no such split
  static int xw_cols96(int N) {
    for (int a = N / 96; a >= 0; --a)
      if ((N - 96 * a) % 64 == 0) return 96 * a;
    return -1;
  }
  // (8 lanes: a width that is not a multiple of 16 * NC runs at the widest NC that divides it, down to one column group:
  //  so packed operands, which ARC_GEMM_XW_NC=2 allows with WGX, always reach gemm_xw8. 2026-11-17)
  // ARC_GEMM_XW_NC_E=2 (8 lanes, opt-in, 2026-11-20; the A770 row's G1; 32 x 64 spills, so 2 only): the GEMMs whose N is n_embd (down, o,
  // ssm_out: K up to 17,408 against N 5,120) at that many column groups a sub-group, the others at ARC_GEMM_XW_NC. An
  // A770's profile has those three ~2x a B60's, the others 1.5-1.75x: at 32 x 16 each activation chunk serves 16
  // columns, so a long K's activations are read again for every 16. The same values in each column (checked at start
  // with an n_embd-wide shape). Off (0): as before
  static int xw_nc_e() {
    static const int v = [] {
      const int e = getenv("ARC_GEMM_XW_NC_E") ? atoi(getenv("ARC_GEMM_XW_NC_E")) : 0;
      return e == 2 ? e : 0;
    }();
    return v;
  }
  int xw_nc_for(int N) const {
    int nc = xw_nc();
    if (xw_lanes() == 8 && xw_nc_e() && N == hp.n_embd) nc = xw_nc_e();
    if (xw_lanes() == 8)
      while (nc > 1 && N % (16 * nc)) nc /= 2;
    return nc;
  }
  // A tile per GEMM shape on Alchemist (discrete 8-lane XMX), by default since 2026-11-20: bench/xw_pergemm on two
  // A770s 16 GB, the 27B IQ3_S (#45; skipzx-lab at 150 W, hansong80 at 225 W, 4K and 16K prompts) gave the same split:
  // 32 x 16 with 4 column sub-groups a work-group for the big ones (down, up, gate, qkv, q, z: -1 to -9 % each) and
  // 16 x 32 for those with N <= n_embd and K <= 8192 (ssm_out, o, k, v: -7 to -27 %); all GEMMs 3.5-6.6 % less time
  // than at the one default tile. The same sums in each column (checked at start, with a long-K shape too).
  // ARC_GEMM_XW_SHAPE=0, or any of ARC_GEMM_XW_TM / _NC / _WGX set, or ARC_GEMM_XW_NC_E=2: one tile for all, as before. The Arc 140T
  // (an iGPU) keeps its preset's tile until measured. Not by default on a split model (self audit 2026-11-21): a
  // shard's N and K are sliced while n_embd is not, so with 3 or 4 A770s the 27B's gate / up / down would fall under
  // the 16 x 32 rule that was measured slower for them on one card (and lose ARC_GEMM_XW_DB, 32-token tiles only).
  struct XwTile { int tm, nc, wgx, slm = -1; };  // (slm: token sub-groups through local memory, 0 none, -1 ARC_GEMM_XW_SLM's)
  // (ARC_GEMM_XW_SHAPE=1: on any GPU taking gemm_xw8, the 140T preset and the 125H's emulation included: tests)
  bool xw_shape_on() const {
    static const int env = getenv("ARC_GEMM_XW_SHAPE") ? atoi(getenv("ARC_GEMM_XW_SHAPE")) : -1;
    static const bool fixed = getenv("ARC_GEMM_XW_TM") || getenv("ARC_GEMM_XW_NC") || getenv("ARC_GEMM_XW_WGX");
    if (env == 0 || fixed || xw_nc_e() || xw_lanes() != 8) return false;
    return env == 1 || (cl_.sg8 && !zrt::integrated() && opt.tp_size <= 1);
  }
  XwTile xw_tile_for(int N, int K, int M = 0, int pack = 0) const {
    // (the big tile where it fits, xw_big_on; elsewhere the rule below, never 32 x 32 without local memory)
    if (M > 0 && xw_big_on() && (pack & 3) == 3) {
      const XwBig& b = xw_big_cfg();  // (32 x 32, 4 x 4 unless the sweep's switches say otherwise)
      if (N % (16 * b.nc * b.x) == 0 && M % (b.tm * b.y) == 0) return {b.tm, b.nc, b.x, b.y};
    }
    XwTile t{xw_tm(), xw_nc_for(N), xw_wgx()};
    if (xw_shape_on() && xw_tm() == 32) {
      if (N <= hp.n_embd && K <= 8192) t = {16, 2, xw_wgx()};
      else t = {32, 1, 4};
      while (t.nc > 1 && N % (16 * t.nc)) t.nc /= 2;
    }
    if (xw_big_on()) t.slm = 0;  // (where the big tile does not fit: the plain kernels, never local memory's)
    return t;
  }
  bool xw_takes_n(int N) const { return xw_nc() == 6 ? xw_cols96(N) >= 0 : N % (16 * xw_nc_for(N)) == 0; }
  // one piece of K over all N columns: one launch, or with NC 6 the 96-wide part and the rest in 64-wide tiles
  void xw_piece(const char* name, cl_mem wq, cl_mem wh, cl_mem aq, cl_mem asc, int M, int N, int K, cl_mem C, int ldc,
                int c_off, int a, int rgroup, int s0, int s1, int wgy, int pack = 0) {
    const XwTile t = xw_tile_for(N, K, M, pack);
    const int nc = t.nc;
    if (nc != 6)
      return xw_run(name, nc, pack, wq, wh, aq, asc, M, N, K, C, ldc, c_off, a, rgroup, s0, s1, wgy, 0, t.tm, t.wgx, t.slm);
    if (pack) throw std::logic_error("gemm_xw: packed operands with NC 6");
    const int n96 = xw_cols96(N);
    if (n96 > 0) xw_run(name, 6, 0, wq, wh, aq, asc, M, n96, K, C, ldc, c_off, a, rgroup, s0, s1, wgy, 0);
    if (n96 < N) xw_run(name, 4, 0, wq, wh, aq, asc, M, N - n96, K, C, ldc, c_off, a, rgroup, s0, s1, wgy, n96 / 16);
  }
  // h8_matmul's GEMM through gemm_xw / gemm_xw8, with gemm.cl's split-K pieces and row groups; false: a shape it does
  // not take (the old kernel runs it)
  bool xw_matmul(const char* name, cl_mem wq, cl_mem hh, int Mp, int N, int K, cl_mem C, int ldc, int c_off, int a,
                 int pack = 0) {
    const int tm = xw_tm();
    if (Mp % tm || !xw_takes_n(N) || K % 256) return false;
    static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;
    static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;
    // (the pieces gemm.cl's H8 GEMM takes, h8_matmul's own helper: the same sums per piece, so the same results)
    const int nsb = K / 256, splits = h8_splits(K, a);
    int wgy = 1;
    while (wgy < wgy_max && (Mp / tm) % (wgy * 2) == 0) wgy *= 2;
    for (int sp = 0; sp < splits; ++sp)
      xw_piece(name, wq, hh, h8_aq_, h8_asc_, Mp, N, K, C, ldc, c_off, sp == 0 ? a : 1, rg_env, nsb * sp / splits,
               nsb * (sp + 1) / splits, wgy, pack);
    return true;
  }
  void xw_init() {
    if (!xw_wanted()) return;
    // (8 lanes: emulated where the GPU has no XMX at all (the 125H, ARC_XMX_TRY=2); 16 lanes: where it has no 16-lane
    //  XMX, the 140T's 8-lane DPAS included)
    xw_emu_ = xw_lanes() == 8 ? !cl_.xmx : (!xmx_ || cl_.sg8);
    xw_2d_discrete_ = !zrt::integrated();
    static std::mutex mu;
    static std::map<std::string, bool> known;
    const std::string key = cl_.device_id + "|" + std::to_string(xw_lanes()) + "|" + std::to_string(xw_tm()) + "x" +
                            std::to_string(xw_nc()) + (xw_emu_ ? "|emulated" : "") + (x2d_ ? "|2d" : "") + xw_pf_defs() +
                            "|pack" + std::to_string(xw_pack_cfg()) + "|wgx" + std::to_string(xw_wgx()) +
                            (xw_lock() ? "|lock" : "") + (xw_2d() ? "|xw2d" : "") + (xw_2d_c() ? "c" : "") +
                            (xw_nc_e() ? "|nce" + std::to_string(xw_nc_e()) + "e" + std::to_string(hp.n_embd) : "") +
                            (xw_shape_on() ? "|shape" + std::to_string(hp.n_embd) : "") + (xw_db(32) ? "|db" + std::to_string(xw_db_env()) : "") +
                            (xw_slm_wgy() ? "|slm" + std::to_string(xw_slm_wgy()) : "") + (xw_epi2() ? "|e2" : "") +
                            (xw_big_on() ? "|big" + xw_big_key() : "");
    std::lock_guard<std::mutex> g(mu);
    static std::map<std::string, int> fb2d;  // (per key: the 2D fallback its check ended at, then checked that way)
    static std::map<std::string, bool> crnd;  // (per key: whether its check ended with XW_C_RND)
    if (fb2d.count(key)) xw_2d_fb_ = fb2d[key];
    if (crnd.count(key)) xw_c_rnd_ = crnd[key];
    auto it = known.find(key);
    if (it == known.end()) {
      static const char* rnd_env = getenv("ARC_GEMM_XW_C_RND");
      xw_c_rnd_ = rnd_env && atoi(rnd_env) != 0;
      crnd[key] = xw_c_rnd_;
      bool ok = xw_check();
      // (C in 2D blocks failing: once more with the add's product rounded first, unless asked not to)
      if (!ok && xw_2d_c() && !xw_c_rnd_ && !(rnd_env && atoi(rnd_env) == 0)) {
        xw_c_rnd_ = true;
        ok = xw_check();
        zlog::eprintf("inferzero: ARC_GEMM_XW_2D: C in 2D blocks with the add's product rounded first (XW_C_RND): %s\n",
                      ok ? "the same as gemm.cl, used" : "differs too");
        if (!ok) xw_c_rnd_ = false;
        crnd[key] = xw_c_rnd_;
      }
      // (a 2D form that differs or does not build: the plain form, checked again. The loads alone (=2) only when asked
      //  for: on the B60 under Windows whose full form failed, they gave 1,705 tok/s against the plain form's 1,726 on
      //  the 27B's 4K prompt, logits equal (manoreken2, issue #4, release .08))
      while (!ok && xw_2d()) {
        const bool c = xw_2d_c();
        zlog::eprintf(c ? "inferzero: ARC_GEMM_XW_2D: gemm_xw with C in 2D blocks differs from gemm.cl or does not build "
                          "on this driver: the plain form\n"
                        : "inferzero: ARC_GEMM_XW_2D: gemm_xw's 2D loads differ from gemm.cl or do not build: without them "
                          "(please report)\n");
        xw_2d_fb_ = 2;
        fb2d[key] = xw_2d_fb_;
        ok = xw_check();
      }
      it = known.emplace(key, ok).first;
      if (ok) {
        zlog::eprintf("inferzero: the H8 GEMMs on XMX with %d x %d tiles (%s, results checked bit for bit)%s%s\n",
                xw_tm(), 16 * xw_nc(), xw_lanes() == 8 ? "gemm_xw8, 8 lanes" : "gemm_xw", xw_emu_ ? " (DPAS emulated: tests)" : "",
                xw_nc() == 6 ? "; a width's rest after its 96s 64 columns wide" : "");
        if (xw_2d())
          zlog::eprintf("inferzero: gemm_xw %s in 2D blocks (ARC_GEMM_XW_2D, =0 off)\n",
                        xw_2d_c() ? "loads and stores" : "loads");
        else if (getenv("ARC_GEMM_XW_2D") && atoi(getenv("ARC_GEMM_XW_2D")) != 0 && xw_2d_fb_ == 0)
          zlog::eprintf("inferzero: ARC_GEMM_XW_2D needs 16-lane XMX with 2D block loads: not used\n");
        if (!xw_pf_defs().empty())
          zlog::eprintf("inferzero: gemm_xw prefetches the %s of the chunk %d ahead (ARC_GEMM_XW_PF=%d)\n",
                        xw_pf_eff() == 1 ? "weights" : xw_pf_eff() == 2 ? "activations" : "weights and activations",
                        xw_lanes() == 16 ? xw_pfd_v() : xw_pfd(), xw_pf_eff());
        if (xw_shape_on())
          zlog::eprintf("inferzero: gemm_xw8: a tile per GEMM shape: 16 x 32 where N <= %d and K <= 8192, else 32 x 16 "
                        "with 4 column sub-groups (ARC_GEMM_XW_SHAPE=0: one tile)\n", hp.n_embd);
        if (xw_lanes() == 8 && xw_nc_e())
          zlog::eprintf("inferzero: gemm_xw8: the GEMMs %d columns wide (n_embd) at %d x %d (ARC_GEMM_XW_NC_E)\n", hp.n_embd,
                        xw_tm(), 16 * xw_nc_e());
        if (xw_wgx() > 1 || xw_lock())
          zlog::eprintf("inferzero: gemm_xw work-groups of up to %d column sub-groups%s\n", xw_wgx(),
                        xw_lock() ? ", in step each 64-value chunk (ARC_GEMM_XW_LOCK=1)" : "");
        if (xw_slm_wgy() && xw_lanes() == 8)  // (skipzx-lab, #45: nothing said it was on)
          zlog::eprintf("inferzero: gemm_xw8: operands through local memory, %d token sub-groups a work-group where they "
                        "divide the tokens (ARC_GEMM_XW_SLM)\n", xw_slm_wgy());
        if (xw_big_on()) {
          const XwBig& b = xw_big_cfg();
          zlog::eprintf("inferzero: gemm_xw8: %d x %d tiles through local memory, %d x %d sub-groups, %d chunk%s a stage, "
                        "where they fit (N a multiple of %d, %d-token multiples), the tiles above elsewhere "
                        "(ARC_GEMM_XW_BIG=0: off)\n", b.tm, 16 * b.nc, b.x, b.y, b.cps, b.cps > 1 ? "s" : "",
                        16 * b.nc * b.x, b.tm * b.y);
        }
        if (xw_ptr() && xw_lanes() == 8)
          zlog::eprintf("inferzero: gemm_xw8: the local-memory copy's addresses set once before the loop "
                        "(ARC_GEMM_XW_PTR=1)\n");
        if (xw_epi2() && xw_lanes() == 8)
          zlog::eprintf("inferzero: gemm_xw8: the epilogue a row at a time (ARC_GEMM_XW_EPI2)\n");
      } else
        zlog::eprintf("inferzero: %s differs from gemm.cl or does not build: not used (please report)\n",
                xw_lanes() == 8 ? "gemm_xw8" : "gemm_xw");
    }
    xw_ = it->second;
  }
  // random H8R weights (int8 values, per-column scales), int8 activations with per-token scales over three decades;
  // stored, in 3 split-K pieces, added and with the fused SwiGLU; two shapes (a partial row group, two and three
  // column blocks, a column offset and a row stride wider than N), against gemm.cl without XMX: bit for bit
  bool xw_check() {
    const int K0 = 2048;  // (each shape's K: K0 unless it says, see the per-shape tiles' long-K shape)
    uint32_t seed = 777;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    auto rndf = [&](float lo, float span) { return lo + (float)(rnd() % 10000) * (span / 10000.0f); };
    std::string bad;
    std::vector<cl_mem> bufs;
    struct Soft { bool& f; bool was; ~Soft() { f = was; } } soft{cl_.soft_fail, cl_.soft_fail};
    cl_.soft_fail = true;
    bool built = true;
    try {
      const int tm = xw_tm(), cols = 16 * xw_nc();
      std::vector<std::array<int, 3>> shapes = {{3 * tm, 2 * cols, K0}, {2 * tm, 3 * cols, K0}};
      // (NC 6: 96 + 64 parts, 96 + 128 (the 27B's widths have a 128-wide rest), 96 only, 64 only)
      if (xw_nc() == 6) shapes = {{3 * tm, 256, K0}, {2 * tm, 224, K0}, {2 * tm, 288, K0}, {2 * tm, 128, K0}};
      // (ARC_GEMM_XW_WGX: a shape of two whole work-groups of column blocks, so the column sub-groups are checked)
      if (xw_wgx() > 1) shapes.push_back({2 * tm, 2 * xw_wgx() * (xw_nc() == 6 ? 64 : cols), K0});
      // (8 lanes, NC 2 / 4: a width of three column groups, which runs one a sub-group (xw_nc_for))
      if (xw_lanes() == 8 && xw_nc() > 1) shapes.push_back({2 * tm, 48, K0});
      // (ARC_GEMM_XW_NC_E: an n_embd-wide shape, which runs at its column groups)
      if (xw_lanes() == 8 && xw_nc_e() && hp.n_embd % 32 == 0) shapes.push_back({2 * tm, hp.n_embd, K0});
      // (the per-shape tiles: K0's shapes take the small one; a long K (33 superblocks) the big one, over two of its
      //  work-groups)
      if (xw_shape_on()) shapes.push_back({2 * tm, 128, 8448});
      // (ARC_GEMM_XW_SLM: two work-groups of its token sub-groups over two of column sub-groups, so the local-memory
      //  kernel runs here: the shapes above have 2 or 3 token tiles, which it does not take)
      if (xw_slm_wgy()) shapes.push_back({2 * xw_slm_wgy() * tm, 2 * std::max(1, xw_wgx()) * cols, K0});
      // (the big tile, xw_big_on: two of its work-groups each way, so it runs here; the other shapes take the
      //  fallback tiles, as calls that it does not fit)
      if (xw_big_on()) {
        const XwBig& b = xw_big_cfg();
        shapes.push_back({2 * b.y * b.tm, 2 * b.x * 16 * b.nc, K0});
      }
      for (const auto& sh : shapes) {
        const int M = sh[0], N = sh[1], K = sh[2], nsb = K / 256;
        const int ng = N / 16, ldc = N + 32, coff = 16;
        std::vector<uint32_t> wq((size_t)ng * nsb * 4 * 16 * 16), wh((size_t)ng * 16);
        for (auto& v : wq) v = rnd();
        for (auto& v : wh) {
          const float f = rndf(0.001f, 0.01f);
          memcpy(&v, &f, 4);
        }
        std::vector<int8_t> aq((size_t)M * K);
        for (auto& v : aq) v = (int8_t)(rnd() >> 24);
        std::vector<float> asc((size_t)M), c0((size_t)M * ldc);
        for (auto& v : asc) v = 0.02f * std::pow(10.0f, -rndf(0.0f, 3.0f));
        for (auto& v : c0) v = rndf(-2.0f, 4.0f);
        cl_mem bq = buf(wq.size() * 4, wq.data()), bh = buf(wh.size() * 4, wh.data()), ba = buf(aq.size(), aq.data()),
               bs = buf(asc.size() * 4, asc.data()), bb = buf((size_t)(K / 16) * M * 2), bc = buf(c0.size() * 4);
        bufs.insert(bufs.end(), {bq, bh, ba, bs, bb, bc});
        // (ARC_GEMM_XW_PACK: the same data in gemm_xw8's packed orders, made here as h8r_act / h8r_decode / h8r_weights
        //  make them, and each form used checked too: plain, activations, weights, both)
        const int pcfg = xw_pack_cfg(), nch = K / 64;
        cl_mem bqp = nullptr, bap = nullptr;
        if (pcfg & 2) {  // ([group][chunk][lo0, lo1, hi0, hi1][8 words][8 lanes] from [group][chunk][16 words][16 columns])
          std::vector<uint32_t> wp(wq.size());
          for (size_t gc = 0; gc < (size_t)ng * nch; ++gc)
            for (int j = 0; j < 16; ++j)
              for (int cl = 0; cl < 16; ++cl)
                wp[gc * 256 + ((j / 8) * 2 + cl / 8) * 64 + (j % 8) * 8 + cl % 8] = wq[gc * 256 + j * 16 + cl];
          bqp = buf(wp.size() * 4, wp.data());
          bufs.push_back(bqp);
        }
        if (pcfg & 1) {  // ([M / 8][chunk][2 halves][8 rows][8 words] from [M][K])
          std::vector<uint32_t> aw((size_t)M * K / 4), ap(aw.size());
          memcpy(aw.data(), aq.data(), aq.size());
          for (int m = 0; m < M; ++m)
            for (int c = 0; c < nch; ++c)
              for (int h = 0; h < 2; ++h)
                for (int l = 0; l < 8; ++l)
                  ap[((size_t)(m / 8) * nch + c) * 128 + h * 64 + (m % 8) * 8 + l] = aw[(size_t)m * (K / 4) + c * 16 + h * 8 + l];
          bap = buf(ap.size() * 4, ap.data());
          bufs.push_back(bap);
        }
        std::vector<int> forms = {0};
        for (int p = 1; p <= 3; ++p)
          if ((p & pcfg) == p) forms.push_back(p);
        for (int mode = 0; mode < 4; ++mode) {  // store, split-K pieces, add, fused SwiGLU
          const int acc = mode == 2 ? 1 : mode == 3 ? 2 : 0, pieces = mode == 1 ? 3 : 1;
          for (int wgy : {1, 2}) {
            if ((M / tm) % wgy) continue;
            for (int form : forms) {
            std::vector<float> out[2];
            for (int x = 0; x < 2; ++x) {
              CL_CHECK(clEnqueueWriteBuffer(cl_.queue, bc, CL_TRUE, 0, c0.size() * 4, c0.data(), 0, nullptr, nullptr));
              for (int sp = 0; sp < pieces; ++sp) {
                int s0 = nsb * sp / pieces, s1 = nsb * (sp + 1) / pieces, ac = sp == 0 ? acc : 1;
                int MM = M, NN = N, KK = K, ld = ldc, co = coff, rg = 2;
                if (x == 0) {  // (16 tokens a sub-group, whatever ARC_GEMM_TM: M is a multiple of 16, not of 32)
                  cl_kernel k = gemm_kernel(111, false, 0, 1, 16);
                  size_t gws[2] = {(size_t)N * (M / 16), 1}, lws[2] = {16, 1};
                  set_args(k, 0, bq, bh, ba, bs, bb, bc, MM, NN, KK, ld, co, ac, rg, s0, s1);
                  run("xw_check", k, 2, gws, lws);
                } else {
                  xw_piece("xw_check", (form & 2) ? bqp : bq, bh, (form & 1) ? bap : ba, bs, M, N, K, bc, ld, co, ac, rg, s0,
                           s1, wgy, form);
                }
              }
              out[x].resize(c0.size());
              CL_CHECK(clEnqueueReadBuffer(cl_.queue, bc, CL_TRUE, 0, out[x].size() * 4, out[x].data(), 0, nullptr, nullptr));
            }
            bool sane = false;  // (finite, and some output changed: else the comparison proves nothing)
            for (size_t i = 0; i < c0.size(); ++i) {
              if (!std::isfinite(out[0][i])) { sane = false; break; }
              sane = sane || out[0][i] != c0[i];
            }
            if (!sane || memcmp(out[0].data(), out[1].data(), c0.size() * 4)) {
              size_t diff = 0;
              // (ARC_GEMM_XW_DIAG=1: the first few differences, by how many ulp: a rounding or a wrong value)
              static const bool diag = getenv("ARC_GEMM_XW_DIAG") && atoi(getenv("ARC_GEMM_XW_DIAG")) != 0;
              int shown = 0;
              for (size_t i = 0; i < c0.size(); ++i) {
                const bool d = memcmp(&out[0][i], &out[1][i], 4) != 0;
                diff += d;
                if (d && diag && shown < 6) {
                  int32_t a, b;
                  memcpy(&a, &out[0][i], 4), memcpy(&b, &out[1][i], 4);
                  zlog::eprintf("inferzero: ARC_GEMM_XW_DIAG %dx%d mode %d wgy %d: row %zu col %zu: gemm.cl %.9g gemm_xw %.9g "
                                "(C before %.9g, %d ulp)\n", M, N, mode, wgy, i / ldc, i % ldc, out[0][i], out[1][i], c0[i], b - a);
                  ++shown;
                }
              }
              bad += " " + std::to_string(M) + "x" + std::to_string(N) + "/" +
                     (mode == 0 ? "store" : mode == 1 ? "split" : mode == 2 ? "add" : "swiglu") + "/wgy" +
                     std::to_string(wgy) + (form ? "/pack" + std::to_string(form) : std::string()) + "(" +
                     std::to_string(diff) + " differ)";
            }
            }
          }
        }
      }
    } catch (const zrt::GpuWaitError&) {
      // (a GPU that hangs or is lost: not a verdict on gemm_xw, which would otherwise stay off for the process; as
      //  the other start checks, audit 2026-11-18)
      for (cl_mem& b : bufs) free_buf(b);
      throw;
    } catch (const std::exception& e) {
      built = false;
      bad += std::string(" build: ") + e.what();
    }
    for (cl_mem& b : bufs) free_buf(b);
    if (!bad.empty()) zlog::eprintf("inferzero: gemm_xw check:%s\n", bad.c_str());
    return built && bad.empty();
  }
