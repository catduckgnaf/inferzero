// The H8R prompt GEMM on 8-lane XMX (Arrow Lake H's Arc 140T, Alchemist-kind DPAS) with a wider tile and coalesced
// loads (QUEUE 33.4b; zero/xe2/engine_xw.h, ARC_GEMM_XW=1 on a GPU with 8-lane sub-groups): C[M][ldc] (+)= A * W^T at
// column offset c_off, for gemm.cl's H8R layout (QTYPE 111: int8 weights with one scale per column, int8
// activations with one scale per token, a split's dot product one integer sum).
//
// gemm_x8.cl (the 140T's XMX GEMM) reads gemm.cl's 16-lane weight layout with 8 lanes one word a lane at a time: 16
// reads of 32 bytes per column half and 64 values, and each activation row in two 32-byte reads; ocloc arl-h counts
// ~110 instructions and ~13 sends per DPAS (docs/xe2-tuning.md section 12). Here:
//   - a 16-column group's 64 values of a chunk in 4 block reads of 8 words: lane l gets word w of column l (half 0)
//     in component 2k and of column 8 + l (half 1) in 2k + 1, as the layout interleaves them every 16 words;
//   - an activation row's 64 bytes of a chunk in one 2-word block read (values 0..31, 32..63);
//   - a sub-group holds XW_TM tokens x XW_NC column groups (2 * XW_NC DPAS column blocks of 8): each activation word
//     serves 16 * XW_NC columns, each weight word XW_TM tokens.
// The same values reach the same DPAS operands as in gemm_x8 (A: component r = row r's word, lane l = values 4l..4l+3
// of the 32; B: lane n = column n, component w = its values 4w..4w+3), the integer sums are exact, and the epilogue is
// gemm_x8's (token scale * column scale) * (float)sum with the same accumulate modes: bit for bit gemm.cl's DP4a
// results (checked at start, engine_xw.h xw_check). -DXW_EMU: DPAS as the DP4a dot products the extension's text
// gives (tests on a GPU without XMX).
// Grid: (8 * (N / (16 * XW_NC)) * rows, WGY), local (8 * WGX, WGY), rows = M / (XW_TM * WGY); the row-group sweep of
// gemm.cl.
// -DXW_WGX=2 / 4 (2026-11-17, ARC_GEMM_XW_WGX, as gemm_xw.cl): WGX column sub-groups side by side in a work-group (the
// host keeps N / (16 * XW_NC) a multiple of WGX), so they read the same activation chunk while it is in L1; each
// computes what it did before, in the same order: the same results bit for bit. With XW_PF & 2 and XW_PACKA each
// column sub-group prefetches only its share of the token groups (t % WGX == sx).
// -DXW_LOCK (ARC_GEMM_XW_LOCK=1): the work-group in step, a split barrier around each 64-value chunk (arrive before
// its loads, wait after its DPAS), so the sub-groups that share a chunk's lines read them while they are in L1.
// Changes no value.
//
// Layouts in the DPAS operands' order (bench\xw_peak.cpp, 2026-11-12; not used by the engine yet). Off, the kernel is
// the one above, unchanged; on, the same words reach the same DPAS operands (the same sums, bit for bit):
//   -DXW_PACKA: aq in [M / 8 token groups][K / 64 chunks][2 halves][8 rows][8 words] (word w of half h = values
//     32h + 4w .. 32h + 4w + 3 of the row): a token group's chunk in 2 block reads of 8 words instead of 8 of 2; the
//     loop takes a column group's weights first, then streams the token groups (fewer registers held: 32 x 32 fits).
//   -DXW_PACKW: wq in [N / 16 groups][K / 64 chunks][lo0, lo1, hi0, hi1][8 words][8 lanes] (lo0: columns 0..7, words
//     0..7; lo1: columns 8..15; hi: words 8..15): the 4 block reads land in the DPAS B operands as they are, no
//     register shuffles.

#ifdef XW_LOCK
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable
#ifdef XW_EMU
#define XW_ARRIVE()
#define XW_WAIT() barrier(CLK_LOCAL_MEM_FENCE)
#else
#define XW_ARRIVE() intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE)
#define XW_WAIT() intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE)
#endif
#else
#define XW_ARRIVE()
#define XW_WAIT()
#endif

#define SG 8
#define LS 16  // lanes of the weight layout (gemm.cl's)
#ifndef XW_TM
#define XW_TM 16
#endif
#ifndef XW_NC
#define XW_NC 1
#endif
// (2026-11-13, for a sweep beyond 32 x 32: XW_TM 64 and XW_NC 4. ocloc for arl-h at 256 registers: 16 x 64 no spill;
//  64 x 16 spills plain, none with XW_PACKA + XW_PACKW; 32 x 64 and 64 x 32 spill. With these checks relaxed in a copy,
//  48 x 16 built without a spill and 32 x 48 spilled; neither is allowed here)
#if XW_TM != 16 && XW_TM != 32 && XW_TM != 64
#error "XW_TM must be 16, 32 or 64"
#endif
#if XW_NC != 1 && XW_NC != 2 && XW_NC != 4
#error "XW_NC must be 1, 2 or 4"
#endif
#ifndef XW_WGX
#define XW_WGX 1
#endif
// (8: only with -DXW_SLM, its work-groups of up to 32 sub-groups; 2026-11-21, ocloc dg2-g10: 32 x 32 8 x 4 no spill)
#if XW_WGX != 1 && XW_WGX != 2 && XW_WGX != 4 && !(XW_WGX == 8 && defined(XW_SLM))
#error "XW_WGX must be 1, 2 or 4 (8 with XW_SLM)"
#endif
#define XW_TG (XW_TM / 8)   // DPAS row groups of 8 tokens
#define XW_NH (2 * XW_NC)   // DPAS column blocks of 8
#define BCAST(x, i) intel_sub_group_shuffle((x), (i))
// -DXW_PF=1 / 2 / 3 (2026-11-16, opt-in, ARC_GEMM_XW_PF): the chunk XW_PFD ahead prefetched with the portable
// prefetch() built-in while this one's DPAS run: its weights (1; a column group's chunk is 1 KB in one place, 128
// bytes a lane) and, with XW_PACKA, its activations (2; a token group's chunk is 512 bytes). The 140T's xw_peak
// --probe (issue #5, 2026-11-16) reached 85 % of the XMX peak at 32 x 32 with its data in L1 and loads at ~128 B/clk
// per Xe-core, against ~33 % in the engine: the operands are not in L1 when the loop wants them. A prefetch reads no
// value into the sums: bit for bit the same results
#ifndef XW_PF
#define XW_PF 0
#endif
#ifndef XW_PFD
#define XW_PFD 1
#endif

#ifdef XW_EMU
inline int xw8_row(uint a, int8 b, int acc) {
  acc += dot_4x8packed_ss_int(BCAST(a, 0), (uint)b.s0);
  acc += dot_4x8packed_ss_int(BCAST(a, 1), (uint)b.s1);
  acc += dot_4x8packed_ss_int(BCAST(a, 2), (uint)b.s2);
  acc += dot_4x8packed_ss_int(BCAST(a, 3), (uint)b.s3);
  acc += dot_4x8packed_ss_int(BCAST(a, 4), (uint)b.s4);
  acc += dot_4x8packed_ss_int(BCAST(a, 5), (uint)b.s5);
  acc += dot_4x8packed_ss_int(BCAST(a, 6), (uint)b.s6);
  acc += dot_4x8packed_ss_int(BCAST(a, 7), (uint)b.s7);
  return acc;
}
inline int8 xw8_mad(int8 a, int8 b, int8 c) {
  return (int8)(xw8_row((uint)a.s0, b, c.s0), xw8_row((uint)a.s1, b, c.s1), xw8_row((uint)a.s2, b, c.s2),
                xw8_row((uint)a.s3, b, c.s3), xw8_row((uint)a.s4, b, c.s4), xw8_row((uint)a.s5, b, c.s5),
                xw8_row((uint)a.s6, b, c.s6), xw8_row((uint)a.s7, b, c.s7));
}
#else
#define xw8_mad(a, b, c) intel_sub_group_i8_i8_matrix_mad_k32((a), (b), (c))
#endif

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gemm_xw8(__global const uint* restrict wq, __global const uint* restrict wh,
                       __global const char* restrict aq, __global const float* restrict asc,
                       __global float* restrict C, int M, int N, int K, int ldc, int c_off, int accumulate,
                       int rgroup, int s_begin, int s_end) {
  // (XW_WGX column sub-groups a work-group, sx this one's: local size 8 * XW_WGX)
#if XW_WGX > 1
  const int sx = (int)get_local_id(0) / SG;
#else
  const int sx = 0;
#endif
  const int NB = N / (LS * XW_NC * XW_WGX), rows = M / (XW_TM * (int)get_local_size(1));
  const int w = get_group_id(0), grp = w / (rgroup * NB), rem = w % (rgroup * NB);
  const int rin = min(rgroup, rows - grp * rgroup);
  const int cb = rem / rin, row = grp * rgroup + rem % rin;
  const int m0 = (row * (int)get_local_size(1) + (int)get_local_id(1)) * XW_TM;
  const int nsb = K / 256, g0 = (cb * XW_WGX + sx) * XW_NC;
  __global const char* ap = aq + (size_t)m0 * K;

  int8 xacc[XW_TG][XW_NH];
#pragma unroll
  for (int t = 0; t < XW_TG; ++t)
#pragma unroll
    for (int h = 0; h < XW_NH; ++h) xacc[t][h] = (int8)(0);

#if defined(XW_SLM)
  // ---- -DXW_SLM (2026-11-21, opt-in; the A770 plan: its H8 GEMMs at ~28 % of the measured XMX peak, where the 140T's
  //      loop reached 85 % with its operands already in L1, xw_peak --probe, issue #5: the operands arrive late). The
  //      work-group's packed chunk (its XW_WGY x XW_TG token groups' A, its XW_WGX x XW_NC column groups' B) is copied
  //      to local memory by all its sub-groups together, one 64-word block read each unit, and double-buffered: the next
  //      chunk's global reads are issued before this chunk's DPAS and stored after them, one barrier a chunk. Each
  //      sub-group's DPAS then read their operands from local memory. Every byte of A and B is read from L2 once per
  //      work-group, not once per sub-group. The local copy is the global layout word for word, so the same words reach
  //      the same DPAS in the same order: bit for bit the plain loop's (integer sums, the one epilogue).
  //      The host gives an explicit local size (8 * XW_WGX, XW_WGY): get_local_size(1) must be XW_WGY.
#if !defined(XW_PACKA) || !defined(XW_PACKW) || !defined(XW_WGY)
#error "XW_SLM needs XW_PACKA, XW_PACKW and XW_WGY"
#endif
#ifdef XW_GU
#error "XW_SLM is not for the fused gate + up GEMM"
#endif
#define XS_AW (XW_WGY * XW_TG * 128)  // words of A a chunk (token group t's: 2 halves x 64 words)
#define XS_BW (XW_WGX * XW_NC * 256)  // words of B a chunk (column group c's: lo0, lo1, hi0, hi1 x 64 words)
#define XS_UA (XS_AW / 64)
#define XS_U ((XS_AW + XS_BW) / 64)   // 64-word units a chunk
#define XS_NS (XW_WGX * XW_WGY)       // sub-groups a work-group
// -DXS_CPS=2 / 4 (2026-11-21, ARC_GEMM_XW_SLM_CPS): that many 64-value chunks a stage, one barrier a stage instead of
// a chunk; the chunks' DPAS run in their order, so the sums are the same. A piece's chunks are whole superblocks (4
// chunks each), so CPS 1, 2 or 4 always divides them: no partial stage. Local memory 2 x CPS x a chunk (DG2: 64 KB)
#ifndef XS_CPS
#define XS_CPS 1
#endif
#if XS_CPS != 1 && XS_CPS != 2 && XS_CPS != 4
#error "XS_CPS must be 1, 2 or 4"
#endif
#define XS_UPS ((XS_U * XS_CPS + XS_NS - 1) / XS_NS)  // (a stage's units a sub-group: v = sid + i * XS_NS)
  __local uint xs[2][XS_CPS][XS_AW + XS_BW];
  const int nch = K / 64, lane = (int)get_sub_group_local_id();
  const int sy = (int)get_local_id(1), sid = sy * XW_WGX + sx;
  // (the work-group's first token group and column group: this sub-group's are tg0 + sy * XW_TG and cg0 + sx * XW_NC)
  __global const uint* apw = (__global const uint*)aq + (size_t)(row * XW_WGY * XW_TG) * nch * 128;
  __global const uint* wpw = wq + (size_t)(cb * XW_WGX * XW_NC) * nsb * 4 * 256;
  const int ch0 = s_begin * 4, ch1 = s_end * 4;
  int8 st[XS_UPS];
  // unit u of chunk ch: A (u < XS_UA) of the work-group's token group u / 2, half u % 2; B of its column group
  // (u - XS_UA) / 4, part (u - XS_UA) % 4; in local memory at u * 64 either way
#define XS_SRC(u, ch)                                                                         \
  ((u) < XS_UA ? apw + ((size_t)((u) >> 1) * nch + (ch)) * 128 + ((u) & 1) * 64             \
               : wpw + ((size_t)(((u) - XS_UA) >> 2) * nsb * 4 + (ch)) * 256 + (((u) - XS_UA) & 3) * 64)
#define XS_LOAD(ch)                                                                          \
  _Pragma("unroll") for (int i = 0; i < XS_UPS; ++i) {                                        \
    const int v = sid + i * XS_NS;  /* (unit v of the stage: chunk v / XS_U, its unit v % XS_U) */ \
    if (v < XS_U * XS_CPS) st[i] = as_int8(intel_sub_group_block_read8(XS_SRC(v % XS_U, (ch) + v / XS_U))); \
  }
#define XS_STORE(b)                                                                          \
  _Pragma("unroll") for (int i = 0; i < XS_UPS; ++i) {                                        \
    const int v = sid + i * XS_NS;                                                            \
    if (v < XS_U * XS_CPS) {                                                                  \
      XS_WR(xs[b][v / XS_U] + (v % XS_U) * 64, st[i]);                                        \
    }                                                                                         \
  }
#ifdef XS_SCALAR  // (the local reads and writes a word a lane: for a driver without cl_intel_subgroup_local_block_io)
#define XS_RD(p) as_int8((uint8)((p)[lane], (p)[8 + lane], (p)[16 + lane], (p)[24 + lane], (p)[32 + lane], \
                                 (p)[40 + lane], (p)[48 + lane], (p)[56 + lane]))
#define XS_WR(p, x)                                                                                         \
  {                                                                                                         \
    __local uint* d_ = (p) + lane;                                                                          \
    const uint8 v_ = as_uint8(x);                                                                           \
    d_[0] = v_.s0, d_[8] = v_.s1, d_[16] = v_.s2, d_[24] = v_.s3, d_[32] = v_.s4, d_[40] = v_.s5;           \
    d_[48] = v_.s6, d_[56] = v_.s7;                                                                         \
  }
#else  // (one block message each: component w of lane l = word 8 w + l, as the global block reads)
#define XS_RD(p) as_int8(intel_sub_group_block_read8((__local const uint*)(p)))
#define XS_WR(p, x) intel_sub_group_block_write8((__local uint*)(p), as_uint8(x))
#endif
#ifdef XW_PTR
  // -DXW_PTR (ARC_GEMM_XW_PTR, 2026-11-21, opt-in; the kernel-loop audit, docs/xe2-tuning.md 16): the copy's
  // addresses set once. ocloc dg2-g10 counted the big tile's loop at 410 instructions for 33 DPAS, most of them
  // XS_SRC's 64-bit address rebuilt from the chunk index for every unit, and the units' `v < XS_U * XS_CPS` tests in
  // branches. Here each of this sub-group's units gets its source at the first stage, its step a stage (A's chunks 128
  // words apart, B's 256) and its place in local memory before the loop; a stage adds the step. Where the units divide
  // evenly over the sub-groups (the default 32 x 32 with 4 x 4 sub-groups: 64 units, 16 sub-groups) every unit is
  // real and the test goes. The same words from the same addresses to the same places: bit for bit XW_SLM's results
  // (the offsets in words from apw / wpw, 32 bits: a work-group's A and B are well under 2^32 words. Whether unit i is
  //  A or B is fixed at build time where the sub-groups divide both A's units and all units (XS_FIX: the defaults),
  //  so its base is a constant there and one flag a unit otherwise)
#define XS_ALL ((XS_U * XS_CPS) % XS_NS == 0)
#define XS_FIX (XS_ALL && XS_U % XS_NS == 0 && XS_UA % XS_NS == 0)
#define XS_IS_A(i) (XS_FIX ? ((i) * XS_NS) % XS_U < XS_UA : xs_a[i])
  uint xs_off[XS_UPS];
  int xs_dst[XS_UPS];
  bool xs_a[XS_UPS];
#pragma unroll
  for (int i = 0; i < XS_UPS; ++i) {
    const int v = sid + i * XS_NS, u = v % XS_U, j = v / XS_U, c = ch0 + j;
    xs_a[i] = u < XS_UA;  // (a unit past the stage's end, v >= XS_U * XS_CPS, is never read)
    xs_off[i] = xs_a[i] ? (uint)(((u >> 1) * nch + c) * 128 + (u & 1) * 64)
                        : (uint)((((u - XS_UA) >> 2) * nsb * 4 + c) * 256 + ((u - XS_UA) & 3) * 64);
    xs_dst[i] = j * (XS_AW + XS_BW) + u * 64;
  }
#undef XS_LOAD
#undef XS_STORE
#define XS_LOAD(ch)                                                                          \
  _Pragma("unroll") for (int i = 0; i < XS_UPS; ++i) {                                        \
    if (XS_ALL || sid + i * XS_NS < XS_U * XS_CPS)                                            \
      st[i] = as_int8(intel_sub_group_block_read8((XS_IS_A(i) ? apw : wpw) + xs_off[i]));     \
    xs_off[i] += (XS_IS_A(i) ? 128u : 256u) * XS_CPS;                                         \
  }
#define XS_STORE(b)                                                                          \
  _Pragma("unroll") for (int i = 0; i < XS_UPS; ++i) {                                        \
    if (XS_ALL || sid + i * XS_NS < XS_U * XS_CPS) XS_WR(&xs[b][0][0] + xs_dst[i], st[i]);   \
  }
#endif
  if (ch0 < ch1) {  // (s_begin / s_end are the same for the whole work-group: its barriers stay uniform)
    XS_LOAD(ch0)
    XS_STORE(0)
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int ch = ch0; ch < ch1; ch += XS_CPS) {  // (a stage of XS_CPS chunks)
    const int cur = ((ch - ch0) / XS_CPS) & 1;
    const bool more = ch + XS_CPS < ch1;
    if (more) { XS_LOAD(ch + XS_CPS) }
#pragma unroll
    for (int j = 0; j < XS_CPS; ++j) {  // (the stage's chunks in order: the same sums as one chunk a stage)
      __local const uint* xa = xs[cur][j] + sy * XW_TG * 128;
      __local const uint* xb = xs[cur][j] + XS_AW + sx * XW_NC * 256;
      int8 lo0[XW_NC], lo1[XW_NC], hi0[XW_NC], hi1[XW_NC];
#pragma unroll
      for (int c = 0; c < XW_NC; ++c) {
        lo0[c] = XS_RD(xb + c * 256), lo1[c] = XS_RD(xb + c * 256 + 64);
        hi0[c] = XS_RD(xb + c * 256 + 128), hi1[c] = XS_RD(xb + c * 256 + 192);
      }
#pragma unroll
      for (int t = 0; t < XW_TG; ++t) {
        const int8 a0 = XS_RD(xa + t * 128), a1 = XS_RD(xa + t * 128 + 64);
#pragma unroll
        for (int c = 0; c < XW_NC; ++c) {
          xacc[t][2 * c] = xw8_mad(a1, hi0[c], xw8_mad(a0, lo0[c], xacc[t][2 * c]));
          xacc[t][2 * c + 1] = xw8_mad(a1, hi1[c], xw8_mad(a0, lo1[c], xacc[t][2 * c + 1]));
        }
      }
    }
    if (more) { XS_STORE(cur ^ 1) }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#elif defined(XW_DB) && defined(XW_PACKA) && defined(XW_PACKW)
  // ---- -DXW_DB (ARC_GEMM_XW_DB, 2026-11-20, the A770 row's G3; the Alchemist default for 32-token tiles since 2026-11-21): the packed loop with its loads one step ahead. The
  //      next token group's A operands are read before this group's DPAS, and at a chunk's last group the next chunk's B
  //      operands too, so a load's latency is covered by the DPAS before it is needed (the plain loop waits on each
  //      read where it is used). The same operands reach the same DPAS in the same order: bit for bit the plain loop's.
  //      Registers: one token group's A (2 x 8 words a lane) and one chunk's B (4 x 8 x XW_NC) more than the plain loop.
  const int nch = K / 64;
  __global const uint* apk = (__global const uint*)aq + (size_t)(m0 / 8) * nch * 128;
  const int ch0 = s_begin * 4, ch1 = s_end * 4;
  //      -DXW_DB2 (ARC_GEMM_XW_DB=2, 2026-11-21, opt-in): the next chunk's B into the same registers, column group c's
  //      right after c's last DPAS of the chunk, and the chunk loop unrolled by 4 as the plain loop is. ocloc dg2-g10 at
  //      256 registers: the form above spills when unrolled (16 x 32: 5,952 bytes at 2, 5,696 at 4), this one does not
  //      (76 DPAS at 16 x 32, as the plain loop's 65 + its 4 chunks); the same DPAS in the same order, bit for bit.
#ifdef XW_DB2
  int8 lo0[XW_NC], lo1[XW_NC], hi0[XW_NC], hi1[XW_NC];
#else
  int8 lo0[XW_NC], lo1[XW_NC], hi0[XW_NC], hi1[XW_NC], nlo0[XW_NC], nlo1[XW_NC], nhi0[XW_NC], nhi1[XW_NC];
#endif
  int8 na0 = (int8)(0), na1 = (int8)(0);
  if (ch0 < ch1) {
#pragma unroll
    for (int c = 0; c < XW_NC; ++c) {
      __global const uint* dp = wq + ((size_t)(g0 + c) * nsb * 4 + ch0) * (16 * LS);
      lo0[c] = as_int8(intel_sub_group_block_read8(dp));
      lo1[c] = as_int8(intel_sub_group_block_read8(dp + 64));
      hi0[c] = as_int8(intel_sub_group_block_read8(dp + 128));
      hi1[c] = as_int8(intel_sub_group_block_read8(dp + 192));
    }
    __global const uint* tp = apk + (size_t)ch0 * 128;
    na0 = as_int8(intel_sub_group_block_read8(tp));
    na1 = as_int8(intel_sub_group_block_read8(tp + 64));
  }
#ifdef XW_DB2
#pragma unroll 4
#endif
  for (int ch = ch0; ch < ch1; ++ch) {
    XW_ARRIVE();
#if XW_PF
    {
      const int qn = ch + XW_PFD, lane = (int)get_sub_group_local_id();
      if (qn < ch1) {
#if XW_PF & 1
#pragma unroll
        for (int c = 0; c < XW_NC; ++c) prefetch(wq + ((size_t)(g0 + c) * nsb * 4 + qn) * (16 * LS) + lane * 32, 32);
#endif
#if XW_PF & 2
#pragma unroll
        for (int t = 0; t < XW_TG; ++t)
          if (t % XW_WGX == sx) prefetch(apk + ((size_t)t * nch + qn) * 128 + lane * 16, 16);
#endif
      }
    }
#endif
    const bool more = ch + 1 < ch1;  // (uniform over the sub-group)
#pragma unroll
    for (int t = 0; t < XW_TG; ++t) {
      const int8 a0 = na0, a1 = na1;
      // (the next A: token group t + 1 of this chunk, or group 0 of the next chunk)
      if (t + 1 < XW_TG) {
        __global const uint* tp = apk + ((size_t)(t + 1) * nch + ch) * 128;
        na0 = as_int8(intel_sub_group_block_read8(tp));
        na1 = as_int8(intel_sub_group_block_read8(tp + 64));
      } else if (more) {
        __global const uint* tp = apk + (size_t)(ch + 1) * 128;
        na0 = as_int8(intel_sub_group_block_read8(tp));
        na1 = as_int8(intel_sub_group_block_read8(tp + 64));
#ifndef XW_DB2
#pragma unroll
        for (int c = 0; c < XW_NC; ++c) {  // (and the next chunk's B)
          __global const uint* dp = wq + ((size_t)(g0 + c) * nsb * 4 + ch + 1) * (16 * LS);
          nlo0[c] = as_int8(intel_sub_group_block_read8(dp));
          nlo1[c] = as_int8(intel_sub_group_block_read8(dp + 64));
          nhi0[c] = as_int8(intel_sub_group_block_read8(dp + 128));
          nhi1[c] = as_int8(intel_sub_group_block_read8(dp + 192));
        }
#endif
      }
#pragma unroll
      for (int c = 0; c < XW_NC; ++c) {
        xacc[t][2 * c] = xw8_mad(a1, hi0[c], xw8_mad(a0, lo0[c], xacc[t][2 * c]));
        xacc[t][2 * c + 1] = xw8_mad(a1, hi1[c], xw8_mad(a0, lo1[c], xacc[t][2 * c + 1]));
#ifdef XW_DB2
        if (t + 1 == XW_TG && more) {  // (the next chunk's B for column group c, in place)
          __global const uint* dp = wq + ((size_t)(g0 + c) * nsb * 4 + ch + 1) * (16 * LS);
          lo0[c] = as_int8(intel_sub_group_block_read8(dp));
          lo1[c] = as_int8(intel_sub_group_block_read8(dp + 64));
          hi0[c] = as_int8(intel_sub_group_block_read8(dp + 128));
          hi1[c] = as_int8(intel_sub_group_block_read8(dp + 192));
        }
#endif
      }
    }
#ifndef XW_DB2
    if (more) {
#pragma unroll
      for (int c = 0; c < XW_NC; ++c) lo0[c] = nlo0[c], lo1[c] = nlo1[c], hi0[c] = nhi0[c], hi1[c] = nhi1[c];
    }
#endif
    XW_WAIT();
  }
#elif defined(XW_PACKA) || defined(XW_PACKW)
  // (the packed layouts: a column group's 4 B operands, then each token group's 2 A operands)
  const int nch = K / 64;
  __global const uint* apk = (__global const uint*)aq + (size_t)(m0 / 8) * nch * 128;
  for (int s = s_begin; s < s_end; ++s) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int ch = s * 4 + j;
      XW_ARRIVE();
#if XW_PF
      {
        const int qn = ch + XW_PFD, lane = (int)get_sub_group_local_id();
        if (qn < s_end * 4) {
#if XW_PF & 1
#pragma unroll
          for (int c = 0; c < XW_NC; ++c) prefetch(wq + ((size_t)(g0 + c) * nsb * 4 + qn) * (16 * LS) + lane * 32, 32);
#endif
#if (XW_PF & 2) && defined(XW_PACKA)
#pragma unroll
          for (int t = 0; t < XW_TG; ++t)  // (the work-group's XW_WGX column sub-groups share these rows: one each)
            if (t % XW_WGX == sx) prefetch(apk + ((size_t)t * nch + qn) * 128 + lane * 16, 16);
#endif
        }
      }
#endif
#ifndef XW_PACKA
      int8 al[XW_TG], ah[XW_TG];
#pragma unroll
      for (int t = 0; t < XW_TG; ++t) {
        uint2 r[8];
#pragma unroll
        for (int p = 0; p < 8; ++p)
          r[p] = intel_sub_group_block_read2((__global const uint*)(ap + (size_t)(8 * t + p) * K + ch * 64));
        al[t] = (int8)(r[0].s0, r[1].s0, r[2].s0, r[3].s0, r[4].s0, r[5].s0, r[6].s0, r[7].s0);
        ah[t] = (int8)(r[0].s1, r[1].s1, r[2].s1, r[3].s1, r[4].s1, r[5].s1, r[6].s1, r[7].s1);
      }
#endif
      int8 lo0[XW_NC], lo1[XW_NC], hi0[XW_NC], hi1[XW_NC];
#pragma unroll
      for (int c = 0; c < XW_NC; ++c) {
        __global const uint* dp = wq + ((size_t)(g0 + c) * nsb * 4 + ch) * (16 * LS);
#ifdef XW_PACKW
        lo0[c] = as_int8(intel_sub_group_block_read8(dp));
        lo1[c] = as_int8(intel_sub_group_block_read8(dp + 64));
        hi0[c] = as_int8(intel_sub_group_block_read8(dp + 128));
        hi1[c] = as_int8(intel_sub_group_block_read8(dp + 192));
#else
        const uint8 q0 = intel_sub_group_block_read8(dp), q1 = intel_sub_group_block_read8(dp + 4 * LS);
        const uint8 q2 = intel_sub_group_block_read8(dp + 8 * LS), q3 = intel_sub_group_block_read8(dp + 12 * LS);
        lo0[c] = as_int8((uint8)(q0.s0, q0.s2, q0.s4, q0.s6, q1.s0, q1.s2, q1.s4, q1.s6));
        lo1[c] = as_int8((uint8)(q0.s1, q0.s3, q0.s5, q0.s7, q1.s1, q1.s3, q1.s5, q1.s7));
        hi0[c] = as_int8((uint8)(q2.s0, q2.s2, q2.s4, q2.s6, q3.s0, q3.s2, q3.s4, q3.s6));
        hi1[c] = as_int8((uint8)(q2.s1, q2.s3, q2.s5, q2.s7, q3.s1, q3.s3, q3.s5, q3.s7));
#endif
      }
#pragma unroll
      for (int t = 0; t < XW_TG; ++t) {
#ifdef XW_PACKA
        __global const uint* tp = apk + ((size_t)t * nch + ch) * 128;
        const int8 a0 = as_int8(intel_sub_group_block_read8(tp)), a1 = as_int8(intel_sub_group_block_read8(tp + 64));
#else
        const int8 a0 = al[t], a1 = ah[t];
#endif
#pragma unroll
        for (int c = 0; c < XW_NC; ++c) {
          xacc[t][2 * c] = xw8_mad(a1, hi0[c], xw8_mad(a0, lo0[c], xacc[t][2 * c]));
          xacc[t][2 * c + 1] = xw8_mad(a1, hi1[c], xw8_mad(a0, lo1[c], xacc[t][2 * c + 1]));
        }
      }
      XW_WAIT();
    }
  }
#else
  for (int s = s_begin; s < s_end; ++s) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      XW_ARRIVE();
#if XW_PF & 1
      {  // (the weights of the chunk XW_PFD ahead, as above; the activations' rows are not in one place here)
        const int qn = s * 4 + j + XW_PFD, lane = (int)get_sub_group_local_id();
        if (qn < s_end * 4)
#pragma unroll
          for (int c = 0; c < XW_NC; ++c) prefetch(wq + ((size_t)(g0 + c) * nsb * 4 + qn) * (16 * LS) + lane * 32, 32);
      }
#endif
      // ---- the XW_TM tokens' rows of this 64-value chunk: al values 0..31, ah 32..63 (component p = row p's word)
      int8 al[XW_TG], ah[XW_TG];
#pragma unroll
      for (int t = 0; t < XW_TG; ++t) {
        uint2 r[8];
#pragma unroll
        for (int p = 0; p < 8; ++p)
          r[p] = intel_sub_group_block_read2((__global const uint*)(ap + (size_t)(8 * t + p) * K + s * 256 + j * 64));
        al[t] = (int8)(r[0].s0, r[1].s0, r[2].s0, r[3].s0, r[4].s0, r[5].s0, r[6].s0, r[7].s0);
        ah[t] = (int8)(r[0].s1, r[1].s1, r[2].s1, r[3].s1, r[4].s1, r[5].s1, r[6].s1, r[7].s1);
      }
      // ---- each column group's 16 words (lo 0..7, hi 8..15) of both halves in 4 reads, used by every row group
#pragma unroll
      for (int c = 0; c < XW_NC; ++c) {
        __global const uint* dp = wq + ((size_t)(g0 + c) * nsb * 4 + s * 4 + j) * (16 * LS);
        const uint8 q0 = intel_sub_group_block_read8(dp), q1 = intel_sub_group_block_read8(dp + 4 * LS);
        const uint8 q2 = intel_sub_group_block_read8(dp + 8 * LS), q3 = intel_sub_group_block_read8(dp + 12 * LS);
        const int8 lo0 = as_int8((uint8)(q0.s0, q0.s2, q0.s4, q0.s6, q1.s0, q1.s2, q1.s4, q1.s6));
        const int8 lo1 = as_int8((uint8)(q0.s1, q0.s3, q0.s5, q0.s7, q1.s1, q1.s3, q1.s5, q1.s7));
        const int8 hi0 = as_int8((uint8)(q2.s0, q2.s2, q2.s4, q2.s6, q3.s0, q3.s2, q3.s4, q3.s6));
        const int8 hi1 = as_int8((uint8)(q2.s1, q2.s3, q2.s5, q2.s7, q3.s1, q3.s3, q3.s5, q3.s7));
#pragma unroll
        for (int t = 0; t < XW_TG; ++t) {
          xacc[t][2 * c] = xw8_mad(ah[t], hi0, xw8_mad(al[t], lo0, xacc[t][2 * c]));
          xacc[t][2 * c + 1] = xw8_mad(ah[t], hi1, xw8_mad(al[t], lo1, xacc[t][2 * c + 1]));
        }
      }
      XW_WAIT();
    }
  }
#endif

  // ---- the one epilogue, gemm_x8's: the token's scale x the column's x the sum, then the accumulate mode
  float av[XW_TM / SG];
#pragma unroll
  for (int v = 0; v < XW_TM / SG; ++v)
    av[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + m0 + SG * v)));
#ifdef XW_EPI2
  // -DXW_EPI2 (2026-11-21, opt-in; the A770 plan): the same values a row at a time. A sub-group's XW_NH column blocks
  // of a row are one run of 8 * XW_NH floats (column g0 * 16 + 8 h + lane), so a row's C is read and written in one
  // block message of XW_NH words instead of XW_NH messages of 32 bytes spread over the tile, and the accumulate mode
  // is decided once, not per value. On an A770 the 32 x 32 tiles lost most on up + SwiGLU (xw_peak, skipzx-lab, #45:
  // 12.67 ms against 7.19 for gate; at 32 x 16 8.13 against 6.67). Every value gets the same operations in the same
  // order (scale x column scale x sum, then + C or the SwiGLU): bit for bit the epilogue below.
#if XW_NH == 2
#define EPI_T uint2
#define EPI_RD(p) intel_sub_group_block_read2(p)
#define EPI_WR(p, v) intel_sub_group_block_write2((p), (v))
#elif XW_NH == 4
#define EPI_T uint4
#define EPI_RD(p) intel_sub_group_block_read4(p)
#define EPI_WR(p, v) intel_sub_group_block_write4((p), (v))
#else
#define EPI_T uint8
#define EPI_RD(p) intel_sub_group_block_read8(p)
#define EPI_WR(p, v) intel_sub_group_block_write8((p), (v))
#endif
  float dc[XW_NH];
#pragma unroll
  for (int h = 0; h < XW_NH; ++h)
    dc[h] = as_float(intel_sub_group_block_read(wh + (size_t)(g0 + h / 2) * LS + (h & 1) * SG));
#define EPI_ROWS(MODE)                                                                                     \
  _Pragma("unroll") for (int t = 0; t < XW_TG; ++t) {                                                      \
    _Pragma("unroll") for (int r = 0; r < 8; ++r) {                                                        \
      const float as = BCAST(av[t], r);                                                                    \
      __global uint* cp = (__global uint*)(C + c_off + (size_t)(m0 + 8 * t + r) * ldc + (size_t)g0 * LS); \
      EPI_T o;                                                                                             \
      uint* ow = (uint*)&o;                                                                                \
      MODE##_LOAD                                                                                          \
      _Pragma("unroll") for (int h = 0; h < XW_NH; ++h) {                                                  \
        const int8 x = xacc[t][h];                                                                         \
        const int is[8] = {x.s0, x.s1, x.s2, x.s3, x.s4, x.s5, x.s6, x.s7};                                \
        float v = as * dc[h] * (float)is[r];                                                               \
        MODE##_OP                                                                                          \
        ow[h] = as_uint(v);                                                                                \
      }                                                                                                    \
      EPI_WR(cp, o);                                                                                       \
    }                                                                                                      \
  }
#define EPI_SWI_LOAD o = EPI_RD(cp);
#define EPI_SWI_OP                     \
  {                                    \
    const float gt = as_float(ow[h]);  \
    v *= gt / (1.0f + exp(-gt));       \
  }
  // (store and add share one loop with the add under its per-value condition, as in the plain epilogue: a loop for
  //  the add alone let -cl-mad-enable fuse the product into it, and split-K sums differed by an ulp (the 125H's
  //  xw_peak --emu, 23,285 of 65,536 values; FP_CONTRACT OFF does not stop it). XW_EPI2_FMA: that fused form, to
  //  compare)
#define EPI_GEN_LOAD \
  if (accumulate == 1) o = EPI_RD(cp);
#ifdef XW_EPI2_FMA
#define EPI_GEN_OP \
  if (accumulate == 1) v = fma(as * dc[h], (float)is[r], as_float(ow[h]));
#else
#define EPI_GEN_OP \
  if (accumulate == 1) v += as_float(ow[h]);
#endif
  if (accumulate == 2) {  // fused SwiGLU: C holds the gate projection, v the up projection
    EPI_ROWS(EPI_SWI)
  } else {
    EPI_ROWS(EPI_GEN)
  }
#else
#pragma unroll
  for (int h = 0; h < XW_NH; ++h) {
    const int g = g0 + h / 2, hg = h & 1;
    const float dcol = as_float(intel_sub_group_block_read(wh + (size_t)g * LS + hg * SG));
#pragma unroll
    for (int t = 0; t < XW_TG; ++t) {
      const int8 x = xacc[t][h];
      const int is[8] = {x.s0, x.s1, x.s2, x.s3, x.s4, x.s5, x.s6, x.s7};
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int m = 8 * t + r;
        const float res = BCAST(av[m / SG], m % SG) * dcol * (float)is[r];
        __global uint* cp = (__global uint*)(C + c_off + (size_t)(m0 + m) * ldc + (size_t)g * LS + hg * SG);
        float v = res;
        if (accumulate == 1) {
          v += as_float(intel_sub_group_block_read(cp));
        } else if (accumulate == 2) {  // fused SwiGLU: C holds the gate projection, v the up projection
          const float gt = as_float(intel_sub_group_block_read(cp));
          v *= gt / (1.0f + exp(-gt));
        }
        intel_sub_group_block_write(cp, as_uint(v));
      }
    }
  }
#endif
}
