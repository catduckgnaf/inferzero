// Mixture-of-experts FFN (qwen35moe, Flash Next, Gemma 4 26B): routing, per-tile kernels, moe_rg, decode + GEMM.
// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from: not a
// header of its own. Split out of engine.h unchanged on 2026-10-23 (QUEUE F26.3).

  // Mixture-of-experts FFN (qwen35moe): x += sigmoid(g . h) * shared(h) + sum_k w_k * expert_k(h), h = norm(x), over
  // the top n_expert_used experts of the router softmax (weights renormalized over the selection). The (token,
  // expert) pairs are grouped by expert into slots; tiles of up to `tm` slots of one expert are the GEMM rows.
  void layer_moe(Layer& L, int T, int Mp) {
    zrt::note_layer = L.il;
    zrt::note_T = T;
    const int E = hp.n_embd, F = hp.n_ff_exp, Fs = hp.n_ff_shexp, NR = hp.router_rows(), NU = hp.n_expert_used;
    const int Fp = hp.n_ff_exp_pad;  // (the experts' h rows and down input width: Flash Next 640 padded to 768)
    const bool g4 = hp.gemma;  // (Gemma 4: no shared expert; the router's input norm of its own, scaled experts)
    if (g4) {  // (the router in f32: its input normed into gmoe_, free until the experts' sum)
      {
        float eps = hp.eps;
        int r0 = 0, EE = E, MM = Mp, TT = T, wh = 1, a32 = 1;
        set_args(k_rmsnorm_, 0, x_, r0, L.rnorm, EE, MM, TT, eps, aq_, asc_, absum_, gmoe_, wh, a32);
        size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
        run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
      }
      {
        // (prompt chunks: 4 rows x 4 tokens per sub-group, gemv_bf16 -DW_F32; single tokens as before)
        static const bool g4_rt = !getenv("ARC_G4_RTILE") || atoi(getenv("ARC_G4_RTILE")) != 0;
        int K = E, N = NR, ld = NR;  // (all the padded rows: the logit past the experts stays 0, not stale)
        if (g4_rt && T > 8) {
          cl_kernel k = kernel("ops", "-DW_F32 -DRT=4 -DTT=4", "gemv_bf16");
          int nt = T;
          set_args(k, 0, gmoe_, K, L.router_f32, N, nt, rl_, ld);
          size_t gws[2] = {(size_t)16 * ((NR + 3) / 4), (size_t)(T + 3) / 4}, lws[2] = {16, 1};
          run("gemm_router", k, 2, gws, lws);
        } else {
          cl_kernel k = kernel("ops", "", "gemv_f32");
          set_args(k, 0, gmoe_, K, L.router_f32, N, rl_, ld);
          size_t gws[2] = {(size_t)16 * NR, (size_t)T}, lws[2] = {16, 1};
          run("gemm_router", k, 2, gws, lws);
        }
      }
      rmsnorm_quant(x_, 0, L.ffn_norm2, E, Mp, T, 1);
    } else if (L.router_bf16) {  // (Flash Next's precise mode: the router in f32 on the mixer's f32 block input, hco_)
      // (prompt chunks: 4 rows x 4 tokens per sub-group, 4K prompt 744 -> 229 ms; single tokens one row each, enough
      //  sub-groups to keep the reads going: 4 rows gave only 68 of them)
      const int rt = T > 8 ? 4 : 1, tt = T > 8 ? 4 : 1;
      cl_kernel k = kernel("ops", "-DRT=" + std::to_string(rt) + " -DTT=" + std::to_string(tt), "gemv_bf16");
      int K = E, N = NR, nt = T, ld = NR;
      set_args(k, 0, hco_, K, L.router_bf16, N, nt, rl_, ld);
      size_t g2[2] = {(size_t)16 * ((NR + rt - 1) / rt), (size_t)(T + tt - 1) / tt}, l2[2] = {16, 1};
      run("gemm_router", k, 2, g2, l2);
    } else if (!hp.hc && !hp.ds41) {  // (DeepSeek-V4.1: ds_collapse wrote the input, f32 in hco_ and int8 in aq_)
      rmsnorm_quant(x_, 0, L.post_norm, E, Mp, T, 1);
    }
    // router + shared expert gate/up (prompt chunks fuse silu(gate) * up into the up GEMM); one token: one GEMV
    // over the stacked rows when the layer has them (router logits at rgu_out_[0], gate / up in rgu_gu_)
    // (two passes: no fused epilogue; DeepSeek-V4.1: none either, its SwiGLU is clamped in swiglu_quant)
    const int fused = cur_T_ > 8 && !q2mm_ && !h8_q2() && !hp.ds41 ? 1 : 0;  // (ARC_H8_Q2: two passes, no fused epilogue)
    cl_mem rl = rl_, gu = big_;
    if (g4) {
      // (done above)
    } else if (T == 1 && L.rgu.raw) {
      matmul("gemm_router_sh", L.rgu, Mp, rgu_out_, 0, 0, 0, true);
      rl = L.router_bf16 ? rl_ : rgu_out_;
      gu = rgu_gu_;
    } else if (fused) {
      if (!L.router_bf16) matmul("gemm_router", L.router, Mp, rl_, NR, 0, 0, true);
      matmul("gemm_gate_sh", L.gate_sh, Mp, big_, Fs, 0, 0, true);
      matmul("gemm_up_sh", L.up_sh, Mp, big_, Fs, 0, 2, true);
    } else {
      if (!L.router_bf16) matmul("gemm_router", L.router, Mp, rl_, NR, 0, 0, true);
      matmul("gemm_gate_sh", L.gate_sh, Mp, big_, 2 * Fs, 0, 0, true);
      matmul("gemm_up_sh", L.up_sh, Mp, big_, 2 * Fs, Fs, 0, true);
    }
    // routing, then the slots grouped by expert and their tiles
    static const int tm_std = getenv("ARC_MOE_TM") ? std::max(1, atoi(getenv("ARC_MOE_TM"))) : 16;  // (standard experts: slots per decoded weight; 2048-token prompt: TM 8 416, 16 473, 32 423, 64 187 tok/s)
    // (Gemma 4, two passes: prompt chunks through the T > 1 routing and moe_rg, whose two passes it runs; generation
    //  and verification (T <= 8) through the per-tile kernels with both passes fused, -DQ2F, as plain decoding)
    static const int rg_min0 = getenv("ARC_MOE_RG_MIN") ? atoi(getenv("ARC_MOE_RG_MIN")) : 320;
    // (fused two-pass tiles hold twice the sums: as many slots as the tokens (an expert gets at most T, here <= 8);
    //  TM 16 spilled 4 KB per thread in verification. Each slot's terms are the same at any TM: bit-identical)
    // (2..8 rows, a batched step or a verification pass: tiles of 2 / 4 / 8 slots, as many as an expert can get; TM 16
    //  ran mostly empty rows: Flash Next 2 slots gate / up 34.8 -> 29.1 ms per step. ARC_MOE_TM_SMALL=n: TM n, 0: TM 16)
    // (audit 2026-11-12 #8: only the tile sizes the kernels were written and measured for; another value is ignored)
    static const int tm_small = [] {
      if (!getenv("ARC_MOE_TM_SMALL")) return -1;
      const int v = atoi(getenv("ARC_MOE_TM_SMALL"));
      if (v == 0 || v == 1 || v == 2 || v == 4 || v == 8 || v == 16) return v;
      zlog::eprintf("inferzero: ARC_MOE_TM_SMALL=%d ignored (0, 1, 2, 4, 8 or 16)\n", v);
      return -1;
    }();
    const int tm_dec = tm_small < 0 ? (T <= 2 ? 2 : T <= 4 ? 4 : 8) : tm_small > 0 ? tm_small : tm_std;
    // (two passes: Gemma 4, Flash Next's precise mode; layers whose expert types moe_rg lacks per tile at every size.
    //  Audit 2026-11-18: only when the engine runs two passes (ARC_G4_ONE=1 leaves q2mm_ off and no residual is
    //  written), and fused per tile wherever moe_rg would not take the experts (its TM 16 and F % 64), which ran one
    //  pass silently)
    const bool q2on = q2_ && q2mm_;
    const bool rg_fits = (T <= 8 ? tm_dec : tm_std) == 16 && F % 64 == 0;
    const bool q2f = q2on && hp.moe_std && ((T <= 8 && T < rg_min0) || !moe_rg_layer_ok(L) || !rg_fits);
    const bool q2g = q2on && !q2f;
    const std::string q2fd = q2f ? " -DQ2F" + q2_defs_ + " -DQ2M=" + std::to_string(q2m_hq_) + " -DQ2MS=" +
                                       std::to_string(q2m_sc_ / 4)
                                 : std::string();
    const int tm = T == 1 && !q2g ? 1 : q2f ? (T <= 4 ? 4 : 8) : hp.moe_std ? (T <= 8 ? tm_dec : tm_std) : 16, n = T * NU;
    const std::string defs = "-cl-intel-256-GRF-per-thread -DNE=" + std::to_string(hp.n_expert) +
                             " -DNU=" + std::to_string(NU) + " -DTM=" + std::to_string(tm) +
                             (Fp != F ? " -DFV=" + std::to_string(F) : std::string()) + (g4 ? " -DESCALE" : "") +
                             (hp.ds41 ? " -DROUTE_DS -DRSCALE=" + flt_def(ds_.wscale) : std::string());
    const std::string hld = (Fp != F ? " -DHLD=" + std::to_string(Fp) : std::string()) + (g4 ? " -DMOE_GELU" : "");
    if (T == 1 && !q2g) {  // one token: its NU experts are the tiles
      cl_kernel k = kernel("moe", defs, "moe_route1");
      set_args(k, 0, rl, sgate_, mtiles_, stok_, swt_, minv_);
      if (g4) set_args(k, 6, L.escale);
      if (hp.ds41) set_args(k, 6, dsl_[L.il].ebias);
      size_t g1 = 16, l1 = 16;
      run("moe_route", k, 1, &g1, &l1);
      if (tp_moe()) tp_moe_tiles1();  // (multi-GPU: this GPU's experts' tiles only, zero/mgpu/engine_tp_shard.h)
    } else {
      {
        cl_kernel k = kernel("moe", defs, "moe_route");
        set_args(k, 0, rl_, NR, T, mids_, mwts_, sgate_, mcount_);
        if (g4) set_args(k, 7, L.escale);
        if (hp.ds41) set_args(k, 7, dsl_[L.il].ebias);
        size_t gws[2] = {16, (size_t)T}, lws[2] = {16, 1};
        run("moe_route", k, 2, gws, lws);
      }
      if (tp_moe()) tp_moe_mask();  // (multi-GPU: the other GPUs' experts get no slots)
      {
        cl_kernel k = kernel("moe", defs, "moe_scan");
        set_args(k, 0, mcount_, moffs_, mcursor_, mtiles_, mtstart_);
        size_t g1 = hp.n_expert, l1 = hp.n_expert;
        run("moe_scan", k, 1, &g1, &l1);
      }
      if (tp_moe()) {
        tp_moe_scatter(n);
      } else {
        cl_kernel k = kernel("moe", defs, "moe_scatter");
        set_args(k, 0, mids_, mwts_, n, mcursor_, stok_, swt_, minv_);
        size_t g1 = ((size_t)n + 255) / 256 * 256, l1 = 256;
        run("moe_scatter", k, 1, &g1, &l1);
      }
    }
    // (ARC_DUMP_ROUTES=file: append the experts every token chose in this layer, for expert-cache studies:
    //  int32 layer, int32 tokens, then tokens x NU int32 expert ids)
    static FILE* routes = getenv("ARC_DUMP_ROUTES") ? fopen(getenv("ARC_DUMP_ROUTES"), "ab") : nullptr;
    if (routes) {
      std::vector<int> ids((size_t)T * NU);
      if (T == 1) {
        std::vector<int> tl(1 + 3 * NU);
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, mtiles_, CL_TRUE, 0, tl.size() * 4, tl.data(), 0, nullptr, nullptr));
        for (int k = 0; k < NU; ++k) ids[k] = tl[1 + 3 * k];
      } else {
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, mids_, CL_TRUE, 0, ids.size() * 4, ids.data(), 0, nullptr, nullptr));
      }
      const int hdr[2] = {L.il, T};
      fwrite(hdr, 4, 2, routes);
      fwrite(ids.data(), 4, ids.size(), routes);
      fflush(routes);
    }
    // the tile count is only known on the GPU: launch for the most tiles possible (extra work-groups exit)
    const size_t max_tiles = tm == 1 ? (size_t)n : (size_t)(n + tm - 1) / tm + std::min(n, hp.n_expert);
    if (hp.moe_std && T == 1) {  // (weight bytes read: the token's NU experts, gate + up + down)
      // (multi-GPU: this GPU's share of them on average, its part of the experts)
      const double share = tp_moe() ? (double)(tpe_.second - tpe_.first) / hp.n_expert : 1.0;
      for (int i = 0; i < 3; ++i)
        rd_bytes += (uint64_t)(share * NU * (i == 2 ? E : F) * ((i == 2 ? F : E) / 256) * gguf_sb_bytes(L.ex_type[i]));
    }
    // (a Data Center GPU Max: 4 and 8 by default, see xehpc_on in zero/xe2/engine_xe2.h)
    static const int gr_gu_e = getenv("ARC_MOE_GR_GU") ? std::clamp(atoi(getenv("ARC_MOE_GR_GU")), 1, 8) : 0,
                     gr_dn_e = getenv("ARC_MOE_GR_DN") ? std::clamp(atoi(getenv("ARC_MOE_GR_DN")), 1, 8) : 0;
    const int gr_gu = gr_gu_e ? gr_gu_e : xehpc_on() ? 4 : 1, gr_dn = gr_dn_e ? gr_dn_e : xehpc_on() ? 8 : 4;  // (rows per sub-group;
    // the down projection's 512-wide rows: 4 per sub-group in generation, 5.1 -> 4.0 ms per token, 1 for prompts)
    // standard experts, chunks of >= gemm_min tokens: decode + int8 GEMM (moe_gemm). Decoding all experts costs ~1 s
    // per chunk, so shorter chunks keep the per-tile kernels (Qwen3.6 35B: 1024 tokens 499 vs 490 tok/s, 2048 tokens
    // 476 -> 642 tok/s; 512: 450 vs 310)
    static const int gemm_min = getenv("ARC_MOE_GEMM_MIN") ? atoi(getenv("ARC_MOE_GEMM_MIN")) : 1152;
    // prompt chunks of rg_min .. rg_max tokens: moe_rg, the int8 GEMM on the raw expert rows (the results of decode +
    // GEMM, bit for bit, so a prompt's logits no longer depend on its chunk size there, without the decode pass).
    // Qwen3.6 35B: 384 / 512 / 768 / 1024 / 1152 tokens 412 -> 424, 440 -> 459, 449 -> 511, 486 -> 520, 529 -> 548
    // tok/s; 256 tokens: 369 vs 359 with the per-tile kernels, 1536: 595 vs 565 with decode + GEMM (each expert's rows
    // are decoded once per 16-token tile here, once per chunk there).
    static const int rg_min = getenv("ARC_MOE_RG_MIN") ? atoi(getenv("ARC_MOE_RG_MIN")) : 320,
                     rg_max = getenv("ARC_MOE_RG_MAX") ? atoi(getenv("ARC_MOE_RG_MAX")) : 1280;
    // Without decode + GEMM (Flash Next: F padded) only the down projection takes moe_rg, at every chunk size from
    // rg_min: Flash Next, 8K prompt in 2048-token chunks: down 13.8 -> 6.2 s GPU time, 145 -> 165 tok/s, and KL to the
    // CPU reference 0.0221 -> 0.0191 (337 tokens; max 0.96 -> 0.27); its gate / up stay per tile (15.4 vs 13.7 s, KL
    // 0.0230 with both).
    // (multi-GPU: not decode + GEMM, which decodes every expert of the layer, the other GPUs' too)
    // (DeepSeek-V4.1: the per-tile kernels only, which have its clamped SwiGLU)
    const bool gemm_ok = moe_group_ > 0 && Fp == F && moe_gemm_layer_ok(L) && !tp_moe() && !hp.ds41;
    const bool rg_ok = hp.moe_std && tm == 16 && (T >= rg_min || q2g) && F % 64 == 0 && moe_rg_layer_ok(L) && !hp.ds41;
    // (Gemma 4 26B-A4B, F 704 padded to 768: no decode + GEMM, so gate / up take moe_rg at every chunk size from
    //  rg_min as the down projection does; ARC_G4_RG_GU=0: per tile)
    static const bool g4_rg_gu = !getenv("ARC_G4_RG_GU") || atoi(getenv("ARC_G4_RG_GU")) != 0;
    // (multi-GPU: no decode + GEMM, so gate / up take moe_rg at every chunk size from rg_min, as the down projection
    //  already did, instead of the per-tile kernel; ARC_TP_RG_GU=0: per tile as before)
    static const bool tp_rg_gu = !getenv("ARC_TP_RG_GU") || atoi(getenv("ARC_TP_RG_GU")) != 0;
    // (two passes: every prompt chunk through moe_rg, which runs both; decode + GEMM has one)
    // (Flash Next's gate / up stay per tile: through moe_rg they took 8.6 s of a 4K prompt against 6.3 s, 174 against
    //  189 tok/s, 2026-11-05)
    // (ARC_MOE_RG_XMX, zero/xe2/engine_moe_xw.h, the plan's X1: moe_rg instead of decode + GEMM, whose results it gives
    //  bit for bit, at every chunk of rg_min tokens or more; its 32-scale types on XMX. Only where decode + GEMM ran:
    //  Flash Next's gate / up through moe_rg are not the per-tile kernel's results (KL 0.0181 against the default, 2 x
    //  2,048 tokens, 2026-11-17), so they stay per tile (or ARC_MOE_GEMM_PAD); its down projection, on moe_rg already,
    //  takes the XMX form as it is)
    const bool rgx = rg_ok && gemm_ok && rgx_layer(L);
    const bool use_rg = rg_ok && (rgx || q2g || (gemm_ok && T < rg_max) || (g4 && !gemm_ok && g4_rg_gu) ||
                                  (tp_moe() && tp_rg_gu)),
               rg_down = rg_ok && (rgx || q2g || T < rg_max || !gemm_ok);
    const bool use_gemm = !use_rg && gemm_ok && tm == 16 && T >= gemm_min;
    // (ARC_MOE_GEMM_PAD: gate / up alone through decode + GEMM, padded experts; their down stays on moe_rg, rg_down)
    const bool use_gemm_gu = use_gemm || (!use_rg && moe_group_ > 0 && Fp != F && moe_gemm_gu_layer_ok(L) && !tp_moe() &&
                                          !hp.ds41 && tm == 16 && T >= gemm_min);
    const std::string xdef = xc_ ? " -DXCACHE" : "";
    // the per-tile kernels' expert table: the layer's row in host memory, or its copy on the GPU for the missed experts
    // of a split token (ARC_XSPLIT_TAB below)
    cl_mem xtab_l = xc_ ? xc_->xtab[L.il] : nullptr;
    // standard expert types, per tile (zero/kernels/decode.cl moe_gateup_q / moe_down_q): the tiles in `tiles`
    auto gateup_q = [&](cl_mem tiles) {
      const int grows = gr_gu;
      // (prompt tiles: the activation scales transposed first, a token's 80 scales then one coalesced read instead of
      //  16 lanes in 16 rows; Flash Next 4K prompt: gate / up 8.6 -> 7.0 s with the scales left out entirely)
      static const bool asct_on = !getenv("ARC_MOE_ASCT") || atoi(getenv("ARC_MOE_ASCT")) != 0;
      const bool asct = asct_on && tm > 1 && !q2f;
      // (tried 2026-11-05: the tile's activations loaded once per step into local memory for the work-group's 16
      //  rows, two work-group barriers a step: bit-identical but gate / up 6.2 -> 9.6 s of a Flash Next 4K prompt)
      static const std::string moe_grf_env = grf_env("ARC_MOE_GRF");  // (read once: every MoE layer of every token)
      cl_kernel k = kernel("decode", (q2f && tm > 1 ? std::string("-cl-intel-256-GRF-per-thread ") : moe_grf_env) +
                                         "-DMOE_Q -DQT=" + std::to_string(L.ex_type[0]) + " -DGROWS=" +
                                         std::to_string(grows) + " -DTM=" + std::to_string(tm) + xdef + hld + q2fd +
                                         (asct ? " -DASCT" : "") + ds_swi_defs(), "moe_gateup_q");
      int M = Mp, K = E, FF = F;
      cl_mem sc = asc_;
      if (asct) {
        cl_kernel t = kernel("ops", "", "asc_transpose");
        int MM = Mp, KB = E / 32;
        set_args(t, 0, asc_, MM, KB, asct_);
        size_t g2[2] = {((size_t)KB + 15) / 16 * 16, (size_t)Mp};
        run("moe_gateup", t, 2, g2, nullptr);
        sc = asct_;
      }
      if (xc_) {
        set_args(k, 0, xtab_l, xtab_l, FF, K, tabs_, aq_, sc, M, tiles, stok_, mh_, xtab_l,
                 xoff(L, 0), xoff(L, 1));
      } else {
        set_args(k, 0, L.ex_q[0], L.ex_q[1], FF, K, tabs_, aq_, sc, M, tiles, stok_, mh_);
      }
      size_t gws[2] = {((size_t)(F + grows - 1) / grows * 16 + 255) / 256 * 256, max_tiles}, lws[2] = {256, 1};
      run("moe_gateup", k, 2, gws, lws);
    };
    auto quant_h = [&] {
      cl_kernel k = kernel("moe", defs + (q2_ && q2mm_ ? " -DQ2M=" + std::to_string(q2m_hq_) + " -DQ2MS=" + std::to_string(q2m_sc_ / 4)
                                             : std::string()), "moe_quant");
      int FF = Fp;
      set_args(k, 0, mh_, FF, slots_, mhq_, mhsc_);
      size_t gws[2] = {(size_t)16 * Fp / 256, (size_t)n}, lws[2] = {16, 1};
      run("moe_quant", k, 2, gws, lws);
    };
    auto down_q = [&](cl_mem tiles) {
      // (2..8 rows: 4 rows per sub-group too, Flash Next 2 slots down 27.5 -> 20.7 ms per step; ARC_MOE_GR_DN_S=n)
      static const int gr_dn_s = getenv("ARC_MOE_GR_DN_S") ? std::clamp(atoi(getenv("ARC_MOE_GR_DN_S")), 1, 8) : 4;
      const int grows = T == 1 ? gr_dn : T <= 8 ? gr_dn_s : 1;
      // (two passes fused: the 256-register file, as gate / up; with 128 the sums spilled, ocloc: 28-61 scratch messages)
      cl_kernel k = kernel("decode", (q2f ? std::string("-cl-intel-256-GRF-per-thread ") : std::string()) + "-DMOE_Q -DQT=" +
                                         std::to_string(L.ex_type[2]) + " -DGROWS=" + std::to_string(grows) +
                                         " -DTM=" + std::to_string(tm) + xdef +
                                         (Fp != F ? " -DFV=" + std::to_string(F) : std::string()) + q2fd, "moe_down_q");
      int NN = E, K = Fp, S = slots_;
      if (xc_)
        set_args(k, 0, xtab_l, NN, K, tabs_, mhq_, mhsc_, S, tiles, swt_, my_, xtab_l, xoff(L, 2));
      else
        set_args(k, 0, L.ex_q[2], NN, K, tabs_, mhq_, mhsc_, S, tiles, swt_, my_);
      size_t gws[2] = {((size_t)(E + grows - 1) / grows * 16 + 255) / 256 * 256, max_tiles}, lws[2] = {256, 1};
      run("moe_down", k, 2, gws, lws);
    };
    // One token whose experts are partly not cached: the cached ones' gate / up and down run while stream_layer reads
    // the others from the SSD, then the others' (the same kernels on per-slot rows: bit-identical). The victims of
    // the reads are never experts of this layer, so the running kernels' experts stay put. Flash Next (22 % cached,
    // 84.5 % hits): 6.6 -> 7.3 tok/s. ARC_XSPLIT=0: all after the reads.
    static const bool xsplit = !getenv("ARC_XSPLIT") || atoi(getenv("ARC_XSPLIT")) != 0;
    cl_mem tiles = mtiles_;
    std::vector<int> tl1;
    // (a batched step of 2 Flash Next slots split the same way: the same tok/s, 2026-10-27: not kept)
    // (multi-GPU: off. With experts missing, a split model's logits changed from run to run with it (35B over 2 GPUs,
    //  ARC_TP_SIM, 2 GiB caches: up to 1.1 apart from step 7 on), never without it or with ARC_XVERIFY's syncs; the
    //  cause is not found yet, QUEUE F30)
    // (ARC_XALL=1: with every expert in the cache, XCache::all_in, checked at each token's first MoE layer, a token's
    //  layers neither read their experts' ids nor call ensure(): ~40 host waits a 35B token fewer, the same kernels
    //  reading the same addresses. On by default on discrete cards, off on iGPUs: xe2_xall, zero/xe2/engine_xe2.h;
    //  not multi-GPU)
    const bool xall = xe2_xall();
    // (a pinned layer, the MTP head from a file of its own: every expert resident, its table row fixed, no ensure())
    bool skip_x = xc_ && xc_->pinned(L.il);
    if (!skip_x && xall && xc_ && T == 1 && !use_gemm && hp.moe_std && !tp_moe()) {
      if (L.il <= xall_prev_il_) xc_->all_in();
      xall_prev_il_ = L.il;
      skip_x = xc_->all_in_ok;
      if (skip_x) xc_->skip_layer(L.il, NU);
    }
    std::vector<std::pair<int, int>> xmiss;  // (ARC_XCPU: the misses (expert, slot) the CPU computes, zero/xe2/engine_xcpu.h)
    if (skip_x) {
    } else if (xc_ && T == 1 && !use_gemm && hp.moe_std && xsplit && !q2g && (!tp_moe() || xsplit_tp())) {
      tl1.resize(1 + 3 * (size_t)NU);
      const bool xcpu = xcpu_layer_ok(L);
      if (xcpu)  // (with the CPU's inputs, one wait: zero/xe2/engine_xcpu.h)
        xcpu_read_tiles(Mp, tl1, q2f && xcpu_q2());  // (two-pass mode: the residual's too, audit 2026-11-20 A2)
      else
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, mtiles_, CL_TRUE, 0, tl1.size() * 4, tl1.data(), 0, nullptr, nullptr));
      xt_hit_h_.assign(1, 0);
      xt_miss_h_.assign(1, 0);
      if (xcpu) {  // (the background fills of this layer from earlier tokens: cached from here on, xcache.h fill_commit_at)
        const auto tf = std::chrono::steady_clock::now();
        if (L.il <= xcpu_prev_il_) ++xcpu_tok_;  // (a layer at or below the last one: the next token)
        xcpu_prev_il_ = L.il;
        xc_->fill_commit_at(L.il, xcpu_tok_);
        xcpu_fill_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf).count();
      }
      for (int t = 0; t < tl1[0] && t < NU; ++t) {
        const int e = tl1[1 + 3 * t];
        const bool hit = xc_->cached(L.il, e);
        // (ARC_XCPU: a miss whose record is in the RAM tier goes to the CPU; one that is not is read and copied as
        //  before, which puts it in the tier for the next time)
        if (xcpu && !hit && xc_->st.ram_src(L.il, e)) {
          xmiss.emplace_back(e, tl1[2 + 3 * t]);
          continue;
        }
        auto& v = hit ? xt_hit_h_ : xt_miss_h_;
        v.insert(v.end(), tl1.begin() + 1 + 3 * t, tl1.begin() + 4 + 3 * t);
        ++v[0];
      }
      const int nh = xt_hit_h_[0], ng = xt_miss_h_[0];
      auto write_lists = [&] {
        if (!xt_hit_) {
          xt_hit_ = buf((1 + 3 * (size_t)NU) * 4);
          xt_miss_ = buf((1 + 3 * (size_t)NU) * 4);
        }
        CL_CHECK(clEnqueueWriteBuffer(cl_.queue, xt_hit_, CL_TRUE, 0, xt_hit_h_.size() * 4, xt_hit_h_.data(), 0, nullptr,
                                      nullptr));
        CL_CHECK(clEnqueueWriteBuffer(cl_.queue, xt_miss_, CL_TRUE, 0, xt_miss_h_.size() * 4, xt_miss_h_.data(), 0,
                                      nullptr, nullptr));
      };
      if (nh > 0 && ng > 0) {
        write_lists();
        gateup_q(xt_hit_);
        quant_h();
        down_q(xt_hit_);
        tiles = xt_miss_;
      } else if (!xmiss.empty() && nh + ng > 0) {  // (the GPU's experts are the hits alone, or the copied misses alone)
        write_lists();
        tiles = nh > 0 ? xt_hit_ : xt_miss_;
      }
    }
    // (decode + GEMM decodes every expert of a group: the layer's experts all resident first)
    // (ARC_XCPU: the GPU's experts, hits and the misses not in the RAM tier, through ensure; the CPU's are not copied)
    const bool xgpu = xmiss.empty() || xt_hit_h_[0] + xt_miss_h_[0] > 0;
    if (!xmiss.empty()) {
      std::vector<int> ids;
      for (const auto* v : {&xt_hit_h_, &xt_miss_h_})
        for (int t = 0; t < (*v)[0]; ++t) ids.push_back((*v)[1 + 3 * t]);
      if (!ids.empty()) xc_->ensure(L.il, ids.data(), ids.size(), false, false);
    } else if (xc_ && !skip_x) {
      stream_layer(L, T, use_gemm_gu, tl1.empty() ? nullptr : tl1.data());
    }
    // ---- (2026-11-19, issue #56) a split token's missed experts read their table from the GPU. Their entries are
    //      written by the host in ensure() just after the cached experts' kernels read that same row of host memory;
    //      on an A770 the missed experts' kernels then read wrong addresses (the 35B's text corrupted, logits changing
    //      from run to run, ARC_XSPLIT=0 right; skipzx-lab). Here the finished row is copied into a GPU buffer,
    //      which the missed experts' kernels read instead: the same addresses, the same results bit for bit; one
    //      host wait more a split layer. ARC_XSPLIT_TAB=1: this; 2: only that wait (the row in host memory, a test);
    //      0: as before. Default 1 on every GPU since 2026-11-21: first only on Alchemist, but a Panther Lake B390 under
    //      Linux showed the same race (issue #74: Flash Next with a cold cache, KL against f32 0.88 mean / 10.3 max,
    //      10 of 13 top tokens; with 1: 0.0075 / 0.026, 13 of 13, as ARC_XSPLIT=0). Nothing in it is Alchemist's: the
    //      read is the same on every GPU. (Also no longer computed from the first engine's GPU for all of a split)
    if (tiles == xt_miss_ && xt_miss_) {
      static const int xs_tab = getenv("ARC_XSPLIT_TAB") ? atoi(getenv("ARC_XSPLIT_TAB")) : 1;
      if (xs_tab == 1) {
        const size_t nb = (size_t)hp.n_expert * 8;
        if (!xt_tab_) xt_tab_ = buf(nb);
        CL_CHECK(clEnqueueWriteBuffer(cl_.queue, xt_tab_, CL_TRUE, 0, nb, xc_->xaddr + (size_t)L.il * hp.n_expert, 0,
                                      nullptr, nullptr));
        xtab_l = xt_tab_;
      } else if (xs_tab == 2) {
        zrt::sync();
      }
    }
    if (use_rg) {
      moe_rg(L, 0, Mp, max_tiles);
    } else if (use_gemm_gu) {
      moe_gemm(L, 0, Mp);
    } else if (hp.moe_std) {
      if (xgpu) gateup_q(tiles);  // (ARC_XCPU: none when the CPU takes all of the layer's experts)
    } else {
      cl_kernel k = kernel("moe", defs, "moe_gateup");
      int M = Mp, K = E;
      set_args(k, 0, L.ex_q[0], L.ex_s[0], L.ex_q[1], L.ex_s[1], aq_, asc_, M, K, F, mtiles_, stok_, mh_);
      size_t gws[2] = {(size_t)F, max_tiles}, lws[2] = {16, 1};
      run("moe_gateup", k, 2, gws, lws);
    }
    if (xgpu) quant_h();
    // shared expert down: its quantized input replaces the FFN input in aq_ (the expert GEMM above has read it)
    if (!g4) {
      int a32 = 1, gelu = 0;
      set_args(k_swiglu_, 0, gu, Fs, Mp, T, aq_, asc_, absum_, a32, fused, gelu);
      size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
      if (wide_rows(T)) run_rows_wide("swiglu_quant", k_swiglu_, Fs, T);
      else run("swiglu_quant", k_swiglu_, 2, gws, lws);
      matmul("gemm_down_sh", L.down_sh, Mp, msh_, E, 0, 0, true);
    }
    if (rg_down) {
      moe_rg(L, 2, Mp, max_tiles);
    } else if (use_gemm) {
      moe_gemm(L, 2, Mp);
    } else if (hp.moe_std) {
      if (xgpu) down_q(tiles);
    } else {
      cl_kernel k = kernel("moe", defs, "moe_down");
      set_args(k, 0, L.ex_q[2], L.ex_s[2], mhq_, mhsc_, slots_, F, E, mtiles_, swt_, my_);
      size_t gws[2] = {(size_t)E, max_tiles}, lws[2] = {16, 1};
      run("moe_down", k, 2, gws, lws);
    }
    if (!xmiss.empty()) xcpu_run(L, xmiss, Mp, q2f && xcpu_q2());  // (ARC_XCPU_Q2=0: one pass on the CPU)  // (the CPU's slots copied into my_ ahead of moe_combine)
    if (tp_moe()) {
      tp_moe_combine(T);  // (this GPU's experts + the shared expert on GPU 0, summed over the GPUs)
    } else {
      cl_kernel k = kernel("moe", defs, "moe_combine");
      // (Flash Next, DeepSeek-V4.1: the block output alone, combined into the streams after)
      if (hp.hc || hp.ds41) fill(hco_, (size_t)T * E * 4);
      if (g4) fill(gmoe_, (size_t)T * E * 4);    // (Gemma 4: the experts' sum alone; msh_ stays zero)
      set_args(k, 0, g4 ? gmoe_ : hp.hc || hp.ds41 ? hco_ : x_, msh_, sgate_, my_, minv_, E);
      size_t gws[2] = {((size_t)E + 255) / 256 * 256, (size_t)T}, lws[2] = {256, 1};  // (moe_combine checks c < N)
      run("moe_combine", k, 2, gws, lws);
    }
    if (!xmiss.empty()) {  // (the CPU's experts into the cache in the background, for the next tokens)
      const auto tf = std::chrono::steady_clock::now();
      std::vector<int> ids;
      for (auto& m : xmiss) ids.push_back(m.first);
      xc_->fill(L.il, ids.data(), ids.size(), xcpu_tok_);
      xcpu_fill_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf).count();
    }
    // (debug, ARC_DBG_MOE=layer: that layer's shared-expert output, its gate and the FFN output of the last row)
    static const int dbg_moe = getenv("ARC_DBG_MOE") ? atoi(getenv("ARC_DBG_MOE")) : -1;
    if (L.il == dbg_moe && hp.hc && T > 1) {
      std::vector<float> a(E), b(E), g(1);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, msh_, CL_TRUE, (size_t)(T - 1) * E * 4, E * 4, a.data(), 0, nullptr, nullptr));
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, sgate_, CL_TRUE, (size_t)(T - 1) * 4, 4, g.data(), 0, nullptr, nullptr));
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, hco_, CL_TRUE, (size_t)(T - 1) * E * 4, E * 4, b.data(), 0, nullptr, nullptr));
      FILE* f = fopen("moe_dbg_gpu.bin", "wb");
      fwrite(a.data(), 4, E, f); fwrite(g.data(), 4, 1, f); fwrite(b.data(), 4, E, f);
      fclose(f);
    }
  }

  // ARC_XSPLIT_TP=1 (2026-11-19, a test): the split above on a split model too. It was turned off there for logits that
  // changed from run to run (QUEUE F30), much like issue #56's on an A770, whose cause was the missed experts reading
  // their table row in host memory just after the hits' kernels read it (ARC_XSPLIT_TAB); this tests whether the
  // table's GPU copy settles F30 as well, before the split comes back for split models
  static bool xsplit_tp() {
    static const bool on = getenv("ARC_XSPLIT_TP") && atoi(getenv("ARC_XSPLIT_TP")) != 0;
    return on;
  }
  static bool moe_gemm_part_ok(const Layer& L, int i) {
    // (the per-tile GEMM has no min term and no native Q8_0 form)
    return is_w8_type(L.ex_type[i]) && L.ex_type[i] != GGML_Q8_0 && L.ex_type[i] != GGML_Q4_0 && w8_variant(L.ex_type[i]) < 2;
  }
  static bool moe_gemm_layer_ok(const Layer& L) {
    return moe_gemm_part_ok(L, 0) && moe_gemm_part_ok(L, 1) && moe_gemm_part_ok(L, 2);
  }
  // ARC_MOE_GEMM_PAD=1 (2026-11-17, opt-in): decode + GEMM for the gate / up of experts whose width is padded (Flash
  // Next: F 640 in rows of Fp 768), their down projection staying on moe_rg. A Pro B70's Flash Next 4K / 16K prompt
  // spent 45 / 42 % of its GPU time in moe_gateup on the per-tile kernel (DP4a, decoding IQ3_S in the GEMM), where the
  // 35B's experts take decode + the XMX GEMM (joshdev-de, issue #27). Gate / up are N = F rows (a multiple of 16) of K =
  // n_embd values (a multiple of 256): the GEMM takes them as they are, writing h at the row stride Fp; moe_quant
  // reads its first F columns (FV) only. Not bit-identical to the per-tile kernel (another summation order): KL-checked
  // On by default since 2026-11-17 on discrete cards with 16-lane XMX: a Pro B70's Flash Next 4K prompt 1,182 -> 1,414
  // tok/s (moe_gateup 1,504 -> 506 ms; joshdev-de, issue #27, release 2026.11.17.03); the laptop's KL against the
  // per-tile kernel 0.00000. iGPUs and 8-lane GPUs keep the per-tile kernel (slower there without 16-lane XMX: the
  // decode of every expert costs more than the GEMM saves). ARC_MOE_GEMM_PAD=0 / 1 force it
  bool moe_gemm_pad_on() const {
    static const char* e = getenv("ARC_MOE_GEMM_PAD");  // (read once: every MoE layer)
    const bool on = e ? atoi(e) != 0 : (!zrt::integrated() && xmx_ && !cl_.sg8);
    // (not Gemma 4 26B (704 padded to 768): its experts take GELU, which moe_gemm's fused epilogue lacks, and its two
    //  passes go to moe_rg anyway; audit 2026-11-17)
    return on && hp.n_ff_exp_pad != hp.n_ff_exp && !hp.gemma;
  }
  bool moe_gemm_gu_layer_ok(const Layer& L) const {
    return hp.n_ff_exp_pad == hp.n_ff_exp ? moe_gemm_layer_ok(L)
                                          : moe_gemm_pad_on() && moe_gemm_part_ok(L, 0) && moe_gemm_part_ok(L, 1);
  }
  bool moe_gemm_ok() const {
    if (!hp.moe_std) return false;
    for (const Layer& L : layers_)
      if (!moe_gemm_gu_layer_ok(L)) return false;
    return true;
  }
  // moe_rg's types: no per-block mins, not requantized by decode_w8 (IQ1_M: its own scale layout)
  static bool moe_rg_type(uint32_t t) {
    return is_per32_type(t) || t == GGML_IQ4_XS || t == GGML_IQ3_S || t == GGML_IQ3_XXS || t == GGML_IQ2_XXS ||
           t == GGML_IQ2_XS || t == GGML_IQ2_S || t == GGML_Q6_K || t == GGML_Q3_K;
  }
  static bool moe_rg_layer_ok(const Layer& L) {
    for (int i = 0; i < 3; ++i)
      if (!moe_rg_type(L.ex_type[i])) return false;
    return true;
  }
  // Experts of a prompt chunk through zero/kernels/decode.cl moe_rg (the int8 GEMM of zero/kernels/gemm.cl on the raw rows,
  // decoded in registers): part 0 = gate, then up with the fused SwiGLU into mh_; part 2 = down, routing-weighted,
  // into my_. All the chunk's tiles in one launch per matrix (work-groups past the tile count exit at once).
  void moe_rg(Layer& L, int part, int Mp, size_t max_tiles) {
    const int E = hp.n_embd, F = hp.n_ff_exp, NE = hp.n_expert;
    static const int ny_env = getenv("ARC_MOE_RG_NY") ? atoi(getenv("ARC_MOE_RG_NY")) : 0;
    const std::string xd = xc_ ? " -DXCACHE" : "";
    for (int i = part; i < (part == 0 ? 2 : 3); ++i) {
      static const std::string imul = getenv("ARC_RG_IMUL") && atoi(getenv("ARC_RG_IMUL")) != 0 ? " -DRG_IMUL" : "";
      const std::string rg_opts = "-cl-intel-256-GRF-per-thread -DMOE_RG -DQT=" + std::to_string(L.ex_type[i]) + xd + imul;
      cl_kernel k = kernel("decode", rg_opts, "moe_rg");
      // (ARC_MOE_RG_XMX, zero/xe2/engine_moe_xw.h: the XMX form for this type; at its first use checked against k)
      const std::string xdefs = rgx_defs(L, i);
      cl_kernel kx = xdefs.empty() ? nullptr : rgx_kernel(L.ex_type[i], rg_opts + xdefs);
      const bool x_check = kx && rgx_state_[L.ex_type[i]] == 0;
      // (Flash Next: h rows of Fp = 768 for F = 640 values; the down projection's rows padded to Fp with zeros)
      const int Fp = hp.n_ff_exp_pad;
      int N = i == 2 ? E : F, K = i == 2 ? Fp : E, M = i == 2 ? slots_ : Mp, ldc = i == 2 ? E : Fp,
          acc = i == 1 ? 2 : 0, ne = NE,
          um = i == 2 ? 0 : 1, us = i == 2 ? 1 : 0;
      int kv = i == 2 ? F : K;
      // (two-pass mode: the matrix on the values, then on their residual, added; the up projection into mh2_ and
      //  the GELU product after both)
      const int passes = q2_ && q2mm_ ? 2 : 1;
      for (int ps = 0; ps < passes; ++ps) {
        cl_mem a = i == 2 ? (ps ? from(mhq_, q2m_hq_) : mhq_) : (ps ? from(aq_, q2_aq_) : aq_);
        cl_mem as = i == 2 ? (ps ? from(mhsc_, q2m_sc_) : mhsc_) : (ps ? from(asc_, q2_sc_) : asc_);
        cl_mem c = i == 2 ? my_ : (q2_ && q2mm_ && i == 1 ? mh2_ : mh_);
        int ac = q2_ && q2mm_ ? ps : acc;
        const size_t ny = ny_env > 0 ? (size_t)ny_env : max_tiles;
        auto launch = [&](cl_kernel kk, cl_mem cc) {
          if (xc_)
            set_args(kk, 0, xc_->xtab[L.il], N, K, tabs_, a, as, cc, M, ldc, ac, mtiles_, mtstart_, ne, stok_, um, swt_, us,
                     kv, xc_->xtab[L.il], xoff(L, i));
          else
            set_args(kk, 0, L.ex_q[i], N, K, tabs_, a, as, cc, M, ldc, ac, mtiles_, mtstart_, ne, stok_, um, swt_, us, kv);
          size_t gws[2] = {(size_t)N, ny}, lws[2] = {16, 1};
          run(i == 2 ? "moe_down" : "moe_gateup", kk, 2, gws, lws);
        };
        if (x_check && ps == 0) {
          // (c whole: a row per slot (gate / up mh_ / mh2_, down my_), slots_ x ldc floats, engine_alloc.h)
          rgx_first_use(L.ex_type[i], c, (size_t)slots_ * ldc * 4, k, kx, launch);
        } else {
          launch(kx && rgx_state_[L.ex_type[i]] == 1 ? kx : k, c);
        }
      }
      if (q2_ && q2mm_ && i == 1) {
        cl_kernel g = kernel("moe", "-DNE=" + std::to_string(NE) + " -DNU=" + std::to_string(hp.n_expert_used) + " -DTM=16" +
                                        (hp.gemma ? "" : " -DSILU"),  // (Flash Next's precise mode: silu(gate) * up)
                             "moe_gelu_mul");
        int ld = Fp, fv = F;
        set_args(g, 0, mh_, mh2_, ld, fv);
        size_t gws[2] = {(size_t)Fp, (size_t)cur_T_ * hp.n_expert_used};  // (the pass's slots: T x NU rows)
        run("moe_gateup", g, 2, gws, nullptr);
      }
    }
  }
  // Experts of a prompt chunk through the int8 GEMM: part 0 = gate + up (h = silu(x Wg^T) * x Wu^T into mh_),
  // part 2 = down (routing-weighted into my_). Per group of moe_group_ experts: decode to W8, then one GEMM over
  // the group's slot tiles (tiles per expert from moe_scan's tstart).
  void moe_gemm(Layer& L, int part, int Mp) {
    const int E = hp.n_embd, F = hp.n_ff_exp, NE = hp.n_expert, G = moe_group_, Fp = hp.n_ff_exp_pad;
    for (int i = part; i < (part == 0 ? 2 : 3); ++i) {
      const int N = i == 2 ? E : F, K = i == 2 ? F : E;
      const size_t eb = (size_t)N * (K / 256) * raw_padded_sb_bytes(L.ex_type[i]);  // raw bytes per expert
      if (L.ex_sub[i].empty() && !xc_) {
        const size_t total = NE * eb + kRawSlack;  // (upload_padded: the superblocks + the slack)
        for (int e0 = 0; e0 < NE; e0 += G) {
          cl_buffer_region reg{e0 * eb, std::min(total - e0 * eb, G * eb + kRawSlack)};
          cl_int err;
          L.ex_sub[i].push_back(clCreateSubBuffer(L.ex_q[i], CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err));
          CL_CHECK(err);
          mems_.push_back(L.ex_sub[i].back());
        }
      }
      cl_kernel k = moe_gemm_kernel(100 + w8_variant(L.ex_type[i]));
      // tile work-groups per launch: about the average tile count of a group (more loop over the tiles)
      // (ARC_MOE_CG / ARC_MOE_WY: sub-groups per work-group along the columns / the tiles, default 1 x 1; the same
      //  sums in the same order whatever the shape)
      // (at most 32 sub-groups per work-group: CG 1 / 2 / 4, WY up to 32 / CG; self audit 2026-11-08)
      // (a Data Center GPU Max: CG 4 by default, see xehpc_on in zero/xe2/engine_xe2.h)
      static const int mcg_e = getenv("ARC_MOE_CG") ? std::clamp(atoi(getenv("ARC_MOE_CG")), 1, 4) : 0;
      const int mcg = mcg_e ? mcg_e : xehpc_on() ? 4 : 1;
      static const int mwy_e = getenv("ARC_MOE_WY") ? std::max(1, atoi(getenv("ARC_MOE_WY"))) : 1;  // (read once)
      const int mwy = std::min(mwy_e, 32 / (mcg == 3 ? 2 : mcg));
      const int cg = N % (16 * mcg) == 0 ? mcg : 1;
      const size_t ny = (std::max<size_t>(64, (size_t)cur_T_ * hp.n_expert_used / 16 / (NE / G) + G) + mwy - 1) / mwy * mwy;
      for (int e0 = 0; e0 < NE; e0 += G) {
        if (xc_)
          decode_w8_x(L, i, e0, G, N, K);
        else
          decode_w8(L.ex_sub[i][e0 / G], L.ex_type[i], G * N, K, moe_wq_, moe_wh_);
        // (gate / up: h rows of Fp, as the per-tile kernels and moe_rg write them; Fp = F unless the experts are padded)
        int M = i == 2 ? slots_ : Mp, NN = N, KK = K, ldc = i == 2 ? N : Fp, acc = i == 1 ? 2 : 0, e1 = e0 + G, um = i == 2 ? 0 : 1,
            us = i == 2 ? 1 : 0;
        set_args(k, 0, moe_wq_, moe_wh_, i == 2 ? mhq_ : aq_, i == 2 ? mhsc_ : asc_, absum_, i == 2 ? my_ : mh_, M, NN,
                 KK, ldc, acc, mtiles_, mtstart_, e0, e1, stok_, um, swt_, us);
        size_t gws[2] = {(size_t)N, ny}, lws[2] = {16 * (size_t)cg, (size_t)mwy};
        run(i == 2 ? "moe_down" : "moe_gateup", k, 2, gws, lws);
      }
    }
  }
