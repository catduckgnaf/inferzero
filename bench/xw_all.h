// bench/xw_peak --all (2026-11-21, the user: "a super complete sweep ... all possible tiles, with as few instructions
// as possible", "for all possible switches", "selectable", "with an untimed warm-up kernel and a progress bar").
// Every gemm_xw8 form the engine can run with both operands packed (the Alchemist / 140T preset), at each token count
// asked:
//   stage 0  build each; a form that spills (registers kept in memory) or does not build is dropped;
//   stage 1  each survivor, every shape: one untimed launch, then 3 timed (the median), quick, so the slow ones cost
//            little;
//   stage 2  the best --top of each token count again with --iters runs, bit for bit against the plain kernel;
//   stage 3  the best 3 with the row-group order (ARC_GEMM_RG 1 / 2 / 4 / 8) and, without local memory, the prefetch
//            (ARC_GEMM_XW_PF 0-3 / _PFD 1, 2, 4).
// Within 1 % the smaller binary ranks first (fewer instructions). The winners go to --env-out as the engine's own
// switches, for bench/gpu_sweep's phase X to try in the model at short, medium, long and the longest context.
// Selectable: --stages 0123 (any of them; 0 and 1 always run when later ones do), --family plain,slm (or one),
// --tm 16,32,64, --nc 1,2,4, --cps 1,2,4, --epi2 0,1, --db 0,1,2, --wg 4x4,8x4 (local memory's column x token
// sub-groups), --limit N (the first N forms: a quick test).
#pragma once
#include <set>
#include <sstream>

struct AllSel {  // (what --all tries; empty = everything)
  std::set<std::string> family;
  std::set<int> tm, nc, cps, epi2, db;
  std::set<std::string> wg;
  std::string stages = "0123";
  int top = 10, limit = 0;
  std::string env_out;
  static std::set<int> ints(const std::string& s) {
    std::set<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.insert(atoi(t.c_str()));
    return v;
  }
  static std::set<std::string> strs(const std::string& s) {
    std::set<std::string> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.insert(t);
    return v;
  }
  template <class T> static bool in(const std::set<T>& s, const T& x) { return s.empty() || s.count(x); }
};

struct Cand {
  std::string name, defs;
  int tm = 32, nc = 1, wgx = 1, wgy = 0, cps = 1, db = 0, pf = -1, pfd = 1;
  bool slm = false, epi2 = false;
  // (16 lanes, gemm_xw.cl: d2 the 2D block forms (0 none, 2 the loads, 1 the loads and C, 3 the loads and C with
  //  XW_C_RND), lock XW_LOCK; pf0 the prefetch the engine builds a form with when none is asked: both parts 2 ahead
  //  with the 2D loads, xw_pf_2d_def)
  int lanes = 8, d2 = 0;
  bool lock = false;
  std::string pf0;
  cl_kernel k = nullptr;
  size_t spill = 0, bin = 0;
  std::vector<double> score;  // (per token count: the shapes' summed median ms; 0 = not run)
  std::vector<size_t> diff;
  std::string env() const {  // (the engine's switches for this form: zero/xe2/engine_xw.h)
    if (lanes == 16)
      return "ARC_GEMM_XW_TM=" + std::to_string(tm) + " ARC_GEMM_XW_NC=" + std::to_string(nc) + " ARC_GEMM_XW_WGX=" +
             std::to_string(wgx) + " ARC_GEMM_XW_LOCK=" + std::to_string(lock ? 1 : 0) + " ARC_GEMM_XW_2D=" +
             std::to_string(d2 == 3 ? 1 : d2) + (d2 == 3 ? " ARC_GEMM_XW_C_RND=1" : d2 == 1 ? " ARC_GEMM_XW_C_RND=0" : "") +
             (pf >= 0 ? " ARC_GEMM_XW_PF=" + std::to_string(pf) + " ARC_GEMM_XW_PFD=" + std::to_string(pfd) : std::string());
    std::string e = "ARC_GEMM_XW_EPI2=" + std::to_string(epi2 ? 1 : 0);
    if (slm)
      e += " ARC_GEMM_XW_BIG=1 ARC_GEMM_XW_SLM_TILE=" + std::to_string(tm) + "x" + std::to_string(16 * nc) +
           " ARC_GEMM_XW_SLM_WG=" + std::to_string(wgx) + "," + std::to_string(wgy) + " ARC_GEMM_XW_SLM_CPS=" + std::to_string(cps) +
           " ARC_GEMM_XW_PTR=1";  // (the forms are timed with -DXW_PTR; an iGPU's engine has it off unless asked)
    else
      e += " ARC_GEMM_XW_BIG=0 ARC_GEMM_XW_TM=" + std::to_string(tm) + " ARC_GEMM_XW_NC=" + std::to_string(nc) +
           " ARC_GEMM_XW_WGX=" + std::to_string(wgx) + " ARC_GEMM_XW_DB=" + std::to_string(db) +
           (pf >= 0 ? " ARC_GEMM_XW_PF=" + std::to_string(pf) + " ARC_GEMM_XW_PFD=" + std::to_string(pfd) : std::string());
    return e;
  }
};

static std::vector<Cand> all_cands(const AllSel& s) {
  std::vector<Cand> v;
  for (int tm : {16, 32, 64})
    for (int nc : {1, 2, 4}) {
      if (!AllSel::in(s.tm, tm) || !AllSel::in(s.nc, nc)) continue;
      const std::string t = std::to_string(tm) + "x" + std::to_string(16 * nc);
      if (AllSel::in(s.family, std::string("plain")))
        for (int wgx : {1, 2, 4})
          for (int db : {0, 1, 2})
            for (int e : {0, 1}) {
              if (!AllSel::in(s.db, db) || !AllSel::in(s.epi2, e)) continue;
              Cand c;
              c.tm = tm, c.nc = nc, c.wgx = wgx, c.db = db, c.epi2 = e;
              c.name = t + " w" + std::to_string(wgx) + (db ? " db" + std::to_string(db) : "") + (e ? " e2" : "");
              c.defs = std::string(" -DXW_PACKA -DXW_PACKW") + (db == 1 ? " -DXW_DB" : db == 2 ? " -DXW_DB -DXW_DB2" : "") +
                       (e ? " -DXW_EPI2" : "");
              v.push_back(c);
            }
      if (AllSel::in(s.family, std::string("slm")))
        for (int x : {2, 4, 8})
          for (int y : {2, 4, 8}) {
            if (x * y > 32 || !AllSel::in(s.wg, std::to_string(x) + "x" + std::to_string(y))) continue;
            for (int cps : {1, 2, 4}) {
              const long chunk = (long)y * (tm / 8) * 512 + (long)x * nc * 1024;  // (a chunk's bytes in local memory)
              if (2 * cps * chunk > 65536 || !AllSel::in(s.cps, cps)) continue;   // (DG2: 64 KB a work-group)
              for (int e : {0, 1}) {
                if (!AllSel::in(s.epi2, e)) continue;
                Cand c;
                c.tm = tm, c.nc = nc, c.wgx = x, c.wgy = y, c.cps = cps, c.slm = true, c.epi2 = e;
                c.name = "slm " + t + " " + std::to_string(x) + "x" + std::to_string(y) + " c" + std::to_string(cps) + (e ? " e2" : "");
                // (-DXW_PTR: the copy's addresses set once, as the engine builds every local-memory form on Alchemist
                //  since 2026.11.21.10; the same results, so the sweep ranks the forms the engine runs)
                c.defs = std::string(" -DXW_PACKA -DXW_PACKW -DXW_SLM -DXW_PTR") + (cps > 1 ? " -DXS_CPS=" + std::to_string(cps) : "") +
                         (e ? " -DXW_EPI2" : "");
                v.push_back(c);
              }
            }
          }
    }
  if (s.limit > 0 && (int)v.size() > s.limit) v.resize(s.limit);
  return v;
}
// 16 lanes (gemm_xw.cl: B-series, the B390 / 140V, a Max): every tile the engine takes (TM 16 / 32 / 64 x NC 1 / 2 /
// 4 / 6 / 8), 1 / 2 / 4 / 8 column sub-groups a work-group, the lock, and where the GPU has 2D block loads (and the
// DPAS is not emulated) each 2D form; the 32 x 96 tiles (NC 6) never in 2D, as the engine (it spills there). --nc and
// --tm select as for 8 lanes; --family plain / 2d
static std::vector<Cand> all_cands16(const AllSel& s, bool can2d) {
  std::vector<Cand> v;
  for (int tm : {16, 32, 64})
    for (int nc : {1, 2, 4, 6, 8}) {
      if (!AllSel::in(s.tm, tm) || !AllSel::in(s.nc, nc)) continue;
      const std::string t = std::to_string(tm) + "x" + std::to_string(16 * nc);
      for (int wgx : {1, 2, 4, 8})
        for (int d2 : {0, 2, 1, 3}) {
          if (d2 && (!can2d || nc == 6 || !AllSel::in(s.family, std::string("2d")))) continue;
          if (!d2 && !AllSel::in(s.family, std::string("plain"))) continue;
          for (int lock : {0, 1}) {
            Cand c;
            c.lanes = 16, c.tm = tm, c.nc = nc, c.wgx = wgx, c.d2 = d2, c.lock = lock;
            static const char* dn[] = {"", " 2dc", " 2d", " 2dc rnd"};
            c.name = t + " w" + std::to_string(wgx) + dn[d2] + (lock ? " L" : "");
            c.defs = std::string(d2 ? " -DX2D -DXW_2D" : "") + (d2 == 1 || d2 == 3 ? " -DXW_2D_C" : "") +
                     (d2 == 3 ? " -DXW_C_RND" : "") + (lock ? " -DXW_LOCK" : "");
            c.pf0 = d2 ? " -DXW_PF=3 -DXW_PFD=2" : "";
            v.push_back(c);
          }
        }
    }
  if (s.limit > 0 && (int)v.size() > s.limit) v.resize(s.limit);
  return v;
}

// a progress bar on stderr, one line redrawn: [#####.....] 12 / 300 stage 1, 40 s, ~15 min left
static void all_bar(const char* what, size_t i, size_t n, std::chrono::steady_clock::time_point t0) {
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const double left = i ? s / i * (n - i) : 0;
  const int w = 30, f = n ? (int)(w * i / n) : w;
  char bar[w + 1];
  for (int j = 0; j < w; ++j) bar[j] = j < f ? '#' : '.';
  bar[w] = 0;
  fprintf(stderr, "\r  [%s] %zu / %zu %s, %.0f s, ~%.0f min left   %s", bar, i, n, what, s, left / 60.0, i == n ? "\n" : "");
  fflush(stderr);
}

// (lanes 16: src is gemm_xw.cl and the forms all_cands16's, on the plain layouts; 8: gemm_xw8.cl, both packed)
static int run_all(ClRt& cl, const std::string& src, bool emu, double peak, const std::vector<Shape>& shapes,
                   const std::vector<int>& Ms, int iters, const AllSel& sel, int lanes = 8) {
  using clock = std::chrono::steady_clock;
  const bool x16 = lanes == 16;
  const char* kname = x16 ? "gemm_xw" : "gemm_xw8";
  const int Mmax = *std::max_element(Ms.begin(), Ms.end());
  // ---- the shapes' data at the largest token count; a smaller count uses its first rows (both packed layouts keep
  //      8 tokens together, so a prefix of rows is a prefix of the buffer)
  struct SD {
    Shape sh;
    cl_mem bwq, bwp, bwh, baq, bap, bas, bc;
    std::vector<float> c0;
    std::vector<std::vector<float>> ref;  // (per token count)
    int splits;
  };
  std::vector<SD> sd;
  for (const Shape& sh : shapes) {
    const int N = sh.N, K = sh.K, nsb = K / 256, nch = K / 64, ng = N / 16, M = Mmax;
    std::vector<uint32_t> wq((size_t)ng * nch * 256), wh((size_t)N), aq((size_t)M * K / 4);
    for (auto& x : wq) x = rnd();
    for (auto& x : wh) { const float f = 0.001f + (rnd() % 10000) * 1e-6f; memcpy(&x, &f, 4); }
    for (auto& x : aq) x = rnd();
    std::vector<float> asc((size_t)M), c0((size_t)M * N);
    for (auto& x : asc) x = 0.002f + (rnd() % 10000) * 2e-6f;
    for (auto& x : c0) x = -2.0f + (rnd() % 10000) * 4e-4f;
    std::vector<uint32_t> ap(aq.size()), wp(wq.size());
    for (int m = 0; m < M; ++m)
      for (int c = 0; c < nch; ++c)
        for (int h = 0; h < 2; ++h)
          for (int l = 0; l < 8; ++l)
            ap[((size_t)(m / 8) * nch + c) * 128 + h * 64 + (m % 8) * 8 + l] = aq[((size_t)m * K + c * 64 + h * 32 + 4 * l) / 4];
    for (size_t gc = 0; gc < (size_t)ng * nch; ++gc)
      for (int b = 0; b < 4; ++b)
        for (int i = 0; i < 8; ++i)
          for (int l = 0; l < 8; ++l)
            wp[gc * 256 + b * 64 + i * 8 + l] = wq[gc * 256 + ((b >> 1) * 8 + i) * 16 + (b & 1) * 8 + l];
    SD d;
    d.sh = sh;
    d.bwq = cl.buffer(CL_MEM_READ_ONLY, wq.size() * 4, wq.data()), d.bwp = cl.buffer(CL_MEM_READ_ONLY, wp.size() * 4, wp.data());
    d.bwh = cl.buffer(CL_MEM_READ_ONLY, wh.size() * 4, wh.data());
    d.baq = cl.buffer(CL_MEM_READ_ONLY, aq.size() * 4, aq.data()), d.bap = cl.buffer(CL_MEM_READ_ONLY, ap.size() * 4, ap.data());
    d.bas = cl.buffer(CL_MEM_READ_ONLY, asc.size() * 4, asc.data());
    d.bc = cl.buffer(CL_MEM_READ_WRITE, c0.size() * 4);
    d.c0 = std::move(c0);
    d.splits = sh.a != 2 && K > 8192 ? (nsb + 16) / 17 : 1;  // (h8_splits: pieces of 17 superblocks)
    sd.push_back(std::move(d));
  }
  const std::string base = "-cl-std=CL3.0 -cl-mad-enable -cl-intel-256-GRF-per-thread";
  // (the grid of zero/xe2/engine_xw.h xw_run: N / (16 nc) column blocks, wgx of them a work-group; token sub-groups as
  //  the engine picks them, or the local-memory form's own)
  auto fits = [&](const Cand* c, const Shape& sh, int M) {
    const int tm = c ? c->tm : 32, nc = c ? c->nc : 1, wgx = c ? c->wgx : 1;
    if (M % tm || sh.N % (16 * nc * wgx)) return false;
    return !(c && c->slm) || (M / tm) % c->wgy == 0;
  };
  auto launch = [&](cl_kernel k, const Cand* c, SD& d, int M, int rg) {
    const int N = d.sh.N, K = d.sh.K, nsb = K / 256, tm = c ? c->tm : 32, nc = c ? c->nc : 1, wgx = c ? c->wgx : 1;
    int wgy = 1;
    while (wgy < 8 && (M / tm) % (wgy * 2) == 0) wgy *= 2;
    while (wgx * wgy > 32 && wgy > 1) wgy /= 2;
    if (c && c->slm) wgy = c->wgy;
    const size_t sg = x16 ? 16 : 8;  // (a sub-group's lanes)
    size_t gws[2] = {sg * (N / (16 * nc)) * (M / tm / wgy), (size_t)wgy}, lws[2] = {sg * wgx, (size_t)wgy};
    const bool packed = c && !x16;  // (16 lanes: the plain layouts, as the engine's gemm_xw reads them)
    for (int sp = 0; sp < d.splits; ++sp) {
      int s0 = nsb * sp / d.splits, s1 = nsb * (sp + 1) / d.splits, ac = sp == 0 ? d.sh.a : 1, MM = M, NN = N, KK = K,
          ldc = N, coff = 0, r = rg, g_base = 0;
      set_args(k, 0, packed ? d.bwp : d.bwq, d.bwh, packed ? d.bap : d.baq, d.bas, d.bc, MM, NN, KK, ldc, coff, ac, r, s0, s1);
      if (x16) set_args(k, 14, g_base);  // (gemm_xw's first 16-column group: 0, the whole N)
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 2, nullptr, gws, lws, 0, nullptr, nullptr));
    }
  };
  // (an untimed launch first, the "warm-up": the first run of a kernel sets up its state and its caches)
  auto median_ms = [&](cl_kernel k, const Cand* c, SD& d, int M, int rg, int n) {
    launch(k, c, d, M, rg);
    zrt::sync();
    std::vector<double> ms;
    for (int it = 0; it < n; ++it) {
      const auto t0 = clock::now();
      launch(k, c, d, M, rg);
      zrt::sync();
      ms.push_back(std::chrono::duration<double, std::milli>(clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
  };
  auto result = [&](cl_kernel k, const Cand* c, SD& d, int M, int rg) {
    CL_CHECK(clEnqueueWriteBuffer(cl.queue, d.bc, CL_TRUE, 0, d.c0.size() * 4, d.c0.data(), 0, nullptr, nullptr));
    launch(k, c, d, M, rg);
    std::vector<float> out((size_t)M * d.sh.N);
    CL_CHECK(clEnqueueReadBuffer(cl.queue, d.bc, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr));
    return out;
  };
  auto diff_of = [&](cl_kernel k, const Cand* c, size_t mi, int M, int rg) {
    size_t diff = 0;
    for (SD& d : sd) {
      const std::vector<float> out = result(k, c, d, M, rg);
      for (size_t j = 0; j < out.size(); ++j) diff += memcmp(&out[j], &d.ref[mi][j], 4) != 0;
    }
    return diff;
  };
  // ---- the reference: the plain kernel (32 x 16, unpacked), the engine's results bit for bit (xw_check)
  cl_kernel kref = cl.kernel(cl.build(src, base + " -DXW_TM=32 -DXW_NC=1" + (emu ? " -DXW_EMU" : "")), kname);
  for (SD& d : sd)
    for (int M : Ms) d.ref.push_back(result(kref, nullptr, d, M, 4));
  const double ops1 = [&] { double o = 0; for (const Shape& s : shapes) o += 2.0 * s.N * s.K; return o; }();

  // ---- stage 0: build every form
  std::vector<Cand> cs = x16 ? all_cands16(sel, cl.block2d && !emu) : all_cands(sel);
  printf("\n--all: %zu forms of %s%s; token counts", cs.size(), kname,
         x16 ? (cl.block2d && !emu ? " (16 lanes, the 2D block forms too)" : " (16 lanes; no 2D block loads here)") : " (both operands packed)");
  for (int M : Ms) printf(" %d", M);
  printf("; %zu shapes; stages %s\n", shapes.size(), sel.stages.c_str());
  std::vector<Cand> ok;
  size_t n_spill = 0, n_fail = 0;
  auto tb = clock::now();
  for (size_t i = 0; i < cs.size(); ++i) {
    Cand& c = cs[i];
    const std::string opts = base + " -DXW_TM=" + std::to_string(c.tm) + " -DXW_NC=" + std::to_string(c.nc) +
                             (c.wgx > 1 ? " -DXW_WGX=" + std::to_string(c.wgx) : std::string()) +
                             (c.slm ? " -DXW_WGY=" + std::to_string(c.wgy) : std::string()) + c.defs + c.pf0 + (emu ? " -DXW_EMU" : "");
    try {
      cl_program p = cl.build(src, opts);
      c.k = cl.kernel(p, kname);
      c.bin = p->bin_bytes;
      ze_kernel_properties_t kp{};
      kp.stype = (ze_structure_type_t)0x1e;
      if (zeKernelGetProperties(c.k->k, &kp) == ZE_RESULT_SUCCESS) c.spill = kp.spillMemSize;
      if (c.spill) ++n_spill;
      else ok.push_back(c);
    } catch (const std::exception&) {
      ++n_fail;
    }
    all_bar("built (stage 0)", i + 1, cs.size(), tb);
  }
  printf("stage 0: %zu build without a spill (%zu spill, %zu do not build)\n", ok.size(), n_spill, n_fail);
  if (sel.stages.find_first_of("123") == std::string::npos) return 0;

  // ---- stage 1: each survivor at each token count, the quick look
  for (Cand& c : ok) c.score.assign(Ms.size(), 0), c.diff.assign(Ms.size(), 0);
  tb = clock::now();
  for (size_t i = 0; i < ok.size(); ++i) {
    Cand& c = ok[i];
    for (size_t mi = 0; mi < Ms.size(); ++mi) {
      bool all_fit = true;
      for (SD& d : sd) all_fit = all_fit && fits(&c, d.sh, Ms[mi]);
      if (!all_fit) continue;
      double ms = 0;
      for (SD& d : sd) ms += median_ms(c.k, &c, d, Ms[mi], 4, 3);
      c.score[mi] = ms;
    }
    all_bar("timed (stage 1)", i + 1, ok.size(), tb);
  }
  auto better = [](double x, double y, size_t bx, size_t by) {  // (the time; within 1 % the smaller binary)
    if (std::fabs(x - y) <= 0.01 * std::min(x, y)) return bx < by;
    return x < y;
  };
  FILE* ef = sel.env_out.empty() ? nullptr : fopen(sel.env_out.c_str(), "w");
  for (size_t mi = 0; mi < Ms.size(); ++mi) {
    const int M = Ms[mi];
    std::vector<Cand*> r;
    for (Cand& c : ok)
      if (c.score[mi] > 0) r.push_back(&c);
    std::sort(r.begin(), r.end(), [&](const Cand* a, const Cand* b) { return better(a->score[mi], b->score[mi], a->bin, b->bin); });
    const bool s2 = sel.stages.find('2') != std::string::npos, s3 = sel.stages.find('3') != std::string::npos;
    if ((int)r.size() > sel.top) r.resize(sel.top);
    // ---- stage 2: the best of this token count again, --iters runs, and bit for bit against the plain kernel
    if (s2) {
      tb = clock::now();
      for (size_t i = 0; i < r.size(); ++i) {
        Cand* c = r[i];
        double ms = 0;
        for (SD& d : sd) ms += median_ms(c->k, c, d, M, 4, iters);
        c->score[mi] = ms, c->diff[mi] = diff_of(c->k, c, mi, M, 4);
        all_bar("checked (stage 2)", i + 1, r.size(), tb);
      }
      std::sort(r.begin(), r.end(), [&](const Cand* a, const Cand* b) {
        if ((a->diff[mi] != 0) != (b->diff[mi] != 0)) return a->diff[mi] == 0;  // (a wrong result never ranks first)
        return better(a->score[mi], b->score[mi], a->bin, b->bin);
      });
    }
    printf("\n%d tokens: the best %zu of stage 1%s (TOPS over the %zu shapes together)\n", M, r.size(),
           s2 ? (", " + std::to_string(iters) + " runs each").c_str() : " (3 runs each)", shapes.size());
    printf("  %-26s %8s %9s %8s %6s  %s\n", "form", "binary", "ms", "TOPS", "% pk", "bits vs plain");
    for (const Cand* c : r) {
      const double t = ops1 * M / c->score[mi] / 1e9;
      printf("  %-26s %5zu KB %9.3f %8.1f %5.0f%%  %s\n", c->name.c_str(), c->bin / 1024, c->score[mi], t,
             peak > 0 ? 100 * t / peak : 0.0,
             !s2 ? "(stage 2 not run)" : c->diff[mi] ? (std::to_string(c->diff[mi]) + " values differ").c_str() : "identical");
    }
    // ---- stage 3: the best 3 with the row-group order and, without local memory, the prefetch
    int shown = 0;
    for (Cand* c : r) {
      if ((s2 && c->diff[mi]) || shown >= 3) continue;
      ++shown;
      Cand best = *c;
      int best_rg = 4;
      double best_ms = c->score[mi];
      if (s3) {
        for (int rg : {1, 2, 8}) {
          double ms = 0;
          for (SD& d : sd) ms += median_ms(c->k, c, d, M, rg, iters);
          if (ms < best_ms) best_ms = ms, best_rg = rg;
        }
        if (!c->slm)
          for (int pf : {0, 1, 2, 3})
            for (int pfd : {1, 2, 4}) {
              if (pf == 0 && pfd > 1) continue;
              if (x16 && !c->d2 && pf > 1) continue;  // (16 lanes without the 2D loads: the weights' part only, as the engine)
              Cand v = *c;
              v.pf = pf, v.pfd = pfd;
              const std::string opts = base + " -DXW_TM=" + std::to_string(v.tm) + " -DXW_NC=" + std::to_string(v.nc) +
                                       (v.wgx > 1 ? " -DXW_WGX=" + std::to_string(v.wgx) : std::string()) + v.defs +
                                       " -DXW_PF=" + std::to_string(pf) + " -DXW_PFD=" + std::to_string(pfd) + (emu ? " -DXW_EMU" : "");
              try {
                v.k = cl.kernel(cl.build(src, opts), kname);
              } catch (const std::exception&) {
                continue;
              }
              double ms = 0;
              for (SD& d : sd) ms += median_ms(v.k, &v, d, M, best_rg, iters);
              if (ms < best_ms && !diff_of(v.k, &v, mi, M, best_rg)) best_ms = ms, best = v;
            }
      }
      printf("  -> %-23s ARC_GEMM_RG=%d%s: %.3f ms (%.1f TOPS)%s\n", c->name.c_str(), best_rg,
             best.pf >= 0 ? (" PF " + std::to_string(best.pf) + " PFD " + std::to_string(best.pfd)).c_str() : "", best_ms,
             ops1 * M / best_ms / 1e9, s3 ? "" : " (stage 3 not run: the default order)");
      if (ef) fprintf(ef, "%d %d %.3f %s ARC_GEMM_RG=%d\n", M, shown, best_ms, best.env().c_str(), best_rg);
    }
  }
  if (ef) fclose(ef);
  return 0;
}
