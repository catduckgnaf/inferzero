// H8 (QUEUE 31.12): the prompt GEMM's weights and activations rotated by a normalized Walsh-Hadamard transform in
// blocks of HB values along K (the same rotation on both sides: (Hx).(Hw) = x.w), then rounded to int8 with one scale
// per 256-superblock. The GEMM (gemm.cl QTYPE 110) then runs pure int8 dot products over each superblock, one float
// multiply-add per superblock, instead of rescaling every 16 or 32 values (the scale epilogue: GEMM 9.7 -> 11.0 TOPS
// on the 125H without it; on a B60 the DPAS accumulates 256 values at a time). The rotation spreads the activations'
// outliers: plain per-row int8 had a 27B KL of 0.038 against f32, rotated in blocks of 1024 0.00057 (tools/ref_forward
// REF_HAD, 2026-11-03). Built with -DHB=<256 | 512 | 1024> (a power of two dividing K) and, for the weights,
// -DSRC=<the layout they were decoded into: 8 = Q8_0 repacked, 100..103 = W8>.
// (rot.cl's helpers come first, put in front by the engine since 2026-11-17: a runz / serverz older than these kernels
//  compiles this file alone, which said "use of undeclared identifier 'rot_pack'" (MaineTim, issue #19))
#ifndef ROT_NV
#error "h8.cl needs rot.cl in front of it: bin/runz and bin/serverz are older than zero/kernels; rebuild both"
#endif
//
// Layouts (gemm.cl): per 16-column group g and superblock s, value e (0..255) of column l is byte e % 4 of uint
// ((g * nsb + s) * 64 + e / 4) * 16 + l of wq; the W8 header is HDR uints per lane (word k of lane l at
// ((g * nsb + s) * HDR + k) * 16 + l): Q8_0 8 per-32 float scales; W8 d, dmin, 4 words of int8 scales (per 16, or per
// 32 in bytes 0 and 2 of a word), 4 words of uint8 mins (per 16). H8: the same values layout, one float per
// superblock and lane (header HDR 1).
#define SG 16
#ifndef HB
#error "HB: the rotation block (256, 512 or 1024)"
#endif
#define NW (HB / 4)  // work-items: one int8x4 word each

// in-place FWHT of the HB values in x[], unnormalized (the 1 / sqrt(HB) is folded into the scales)
inline void fwht_local(__local float* x, int t) {
  for (int len = 1; len < HB; len <<= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (int k = 0; k < 2; ++k) {  // (HB / 2 butterflies over HB / 4 items)
      const int p = 2 * t + k, i = (p / len) * 2 * len + (p % len), j = i + len;
      const float a = x[i], b = x[j];
      x[i] = a + b;
      x[j] = a - b;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

// the largest |value| of each 256-superblock of the block: per sub-group, then 4 sub-groups a superblock
inline float superblock_amax(__local float* mx, float m, int t) {
  const float sgm = sub_group_reduce_max(m);
  if (get_sub_group_local_id() == 0) mx[t / SG] = sgm;
  barrier(CLK_LOCAL_MEM_FENCE);
  const int s0 = (t / 64) * 4;
  return fmax(fmax(mx[s0], mx[s0 + 1]), fmax(mx[s0 + 2], mx[s0 + 3]));
}

#ifdef SRC
// The weights of one column and one rotation block, decoded from the W8 / Q8_0 layout, rotated and written back in
// place as H8 (each work-group reads only the words it then writes), their superblock scales into hh.
// grid (N * NW, K / HB), local (NW, 1)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void h8_weights(__global uint* restrict wq, __global const uint* restrict wh, __global float* restrict hh,
                         int N, int K) {
  __local float x[HB];
  __local float mx[NW / SG];
  const int t = get_local_id(0), n = get_group_id(0), blk = get_group_id(1);
  const int g = n / SG, l = n % SG, nsb = K / 256;
  const int e0 = blk * HB + 4 * t, s = e0 / 256, e = e0 % 256;
  __global uint* wp = wq + (((size_t)g * nsb + s) * 64 + e / 4) * SG + l;
  const uint word = *wp;
  float v[4];
#if SRC == 8
  __global const uint* hp = wh + ((size_t)g * nsb + s) * (8 * SG);
  const float d = as_float(hp[(e / 32) * SG + l]);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = d * (float)(char)(word >> (8 * i));
#else
  __global const uint* hp = wh + ((size_t)g * nsb + s) * (10 * SG);
  const float d = as_float(hp[l]), dmin = as_float(hp[SG + l]);
  const int b16 = e / 16;  // (the 4 values share their 16-block)
  const uint hs = hp[(2 + b16 / 4) * SG + l];
#if SRC & 1
  const int sc = (int)(char)(hs >> ((b16 & 2) ? 16 : 0));  // (per-32 scales: bytes 0 and 2 of the word)
#else
  const int sc = (int)(char)(hs >> (8 * (b16 & 3)));
#endif
#if SRC >= 102
  const float mn = (float)((hp[(6 + b16 / 4) * SG + l] >> (8 * (b16 & 3))) & 0xFF);
#else
  const float mn = 0.0f;
#endif
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = d * (float)sc * (float)(char)(word >> (8 * i)) - dmin * mn;
#endif
#pragma unroll
  for (int i = 0; i < 4; ++i) x[4 * t + i] = v[i];
  fwht_local(x, t);
  float m = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) m = fmax(m, fabs(v[i] = x[4 * t + i]));
  const float amax = superblock_amax(mx, m, t);
  *wp = rot_pack(v, amax > 0.0f ? 127.0f / amax : 0.0f);
  if (t % 64 == 0) hh[((size_t)g * nsb + s) * SG + l] = rot_scale(amax);
}
#endif

// The activations of one token row and one rotation block: the engine's int8 rows (aq [M][K], scales asc per 32
// values with ACT32, else per 256: [K / 32][M] or [K / 256][M]) as floats, rotated, rounded again to int8 into aqh
// [M][K] with one scale per 256 values in asch [K / 256][M].
// grid (NW, M, K / HB), local (NW, 1, 1)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void h8_act(__global const uint* restrict aq, __global const float* restrict asc, int act32,
                     __global uint* restrict aqh, __global float* restrict asch, int M, int K) {
  __local float x[HB];
  __local float mx[NW / SG];
  const int t = get_local_id(0), m = get_global_id(1), blk = get_global_id(2);
  const int e0 = blk * HB + 4 * t;
  const uint word = aq[((size_t)m * K + e0) / 4];
  const float sc = act32 ? asc[(size_t)(e0 / 32) * M + m] : asc[(size_t)(e0 / 256) * M + m];
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) x[4 * t + i] = sc * (float)(char)(word >> (8 * i));
  fwht_local(x, t);
  float mm = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) mm = fmax(mm, fabs(v[i] = x[4 * t + i]));
  const float amax = superblock_amax(mx, mm, t);
  aqh[((size_t)m * K + e0) / 4] = rot_pack(v, amax > 0.0f ? 127.0f / amax : 0.0f);
  if (t % 64 == 0) asch[(size_t)(e0 / 256) * M + m] = rot_scale(amax);
}

// ---- H8R (gemm.cl QTYPE 111): one int8 scale per weight column and one per token, the rotation in registers
// (rot.cl, put in front of this file by the engine). One sub-group rotates one block of HB values of one column or one
// token, ROT_NV per lane, with no barrier between stages, where h8_weights / h8_act above take one per stage.
//   weights: a work-group of 16 sub-groups = 16 columns x one block; lane l holds the block's values 4 WPL l ..
//            4 WPL l + ROT_NV - 1 (contiguous). h8r_colmax finds each column's largest rotated value over all its blocks
//            once per weight, then h8r_weights (or decode.cl h8r_decode straight from the raw rows) writes the int8
//            values at that scale, in place.
//   activations: h8r_act, one sub-group per block of a token's row; lane l holds values 64 r + 4 l + b (b < 4), the
//            order block reads give. The token's scale is its largest rotated value over all blocks.
// The two hold a block in different orders; each value keeps its own index, so both compute the same transform.
#define WPL (HB / 64)    // int8x4 words per lane (weights)
#define TP (HB / 4 + 1)  // a column's words in local memory, one word apart from the next column's (no bank clash)

#ifdef SRC
// The work-group's region of the values: its 16 columns x one block, 4 HB uints from the first (a block starts at a
// superblock). Word w of the block of column colm is uint 16 w + colm of the region, in the plain order; gemm_xw8's
// packed order (XW_PACKW_OUT) keeps each 16 words of the 16 columns in their 256 uints, so the same region.
inline size_t h8r_region(int K) { return (size_t)get_group_id(0) * 4 * K + (size_t)get_group_id(1) * (4 * HB); }

// the region into tw, one column after the other (a sub-group then reads its own column's words). Lane lid of the
// work-group takes uints lid, lid + 256, ...: words lid / 16 + 16 i of column lid % 16, every load a whole 1 KB row
inline void h8r_stage(__global const uint* restrict wq, int K, __local uint* tw) {
  const int lid = get_local_id(0), colm = lid % SG, w0 = lid / SG;
  __global const uint* restrict wr = wq + h8r_region(K);
#pragma unroll
  for (int i = 0; i < WPL; ++i) tw[colm * TP + w0 + SG * i] = wr[256 * i + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
}

// this lane's ROT_NV values of its column as floats (the W8 / Q8_0 header's scales and mins applied, then the signs),
// rotated
inline void h8r_column(__global const uint* restrict wh, int K, __local const uint* tw, float* v) {
  const int c = get_sub_group_id(), l = get_sub_group_local_id();
  const int g = get_group_id(0), blk = get_group_id(1), nsb = K / 256;
  const int e0 = blk * HB + 4 * WPL * l;   // (the lane's first value in the row)
  const int s = e0 / 256, es = e0 % 256;    // (its superblock, and where in it)
  const uint i0 = 4 * WPL * l;              // (its index in the block)
#if SRC == 8
  __global const uint* hp = wh + ((size_t)g * nsb + s) * (8 * SG);
#else
  __global const uint* hp = wh + ((size_t)g * nsb + s) * (10 * SG);
  const float d = as_float(hp[c]), dmin = as_float(hp[SG + c]);
  const int b0 = es / 16;                   // (the lane's 16-blocks share one word of scales and one of mins)
  const uint hs = hp[(2 + b0 / 4) * SG + c];
#if SRC >= 102
  const uint hm = hp[(6 + b0 / 4) * SG + c];
#endif
#endif
#pragma unroll
  for (int i = 0; i < WPL; ++i) {
    const uint word = tw[c * TP + WPL * l + i];
#if SRC == 8
    const float dq = as_float(hp[((es + 4 * i) / 32) * SG + c]), off = 0.0f;
#else
    const int b16 = b0 + i / 4;
#if SRC & 1
    const int sc = (int)(char)(hs >> ((b16 & 2) ? 16 : 0));  // (per-32 scales: bytes 0 and 2 of the word)
#else
    const int sc = (int)(char)(hs >> (8 * (b16 & 3)));
#endif
#if SRC >= 102
    const float mn = (float)((hm >> (8 * (b16 & 3))) & 0xFF);
#else
    const float mn = 0.0f;
#endif
    const float dq = d * (float)sc, off = dmin * mn;
#endif
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float q = (float)(char)(word >> (8 * b));
#if SRC == 8
      const float x = dq * q;
#else
      const float x = dq * q - off;
#endif
      v[4 * i + b] = rot_sign(x, i0 + 4 * i + b);
    }
  }
  rot_block(v);
}

// grid (N / 16 * 256, K / HB), local (256, 1). col: N uints, zero before the first block; the largest scale wins
// (positive floats order as their bits)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void h8r_colmax(__global const uint* restrict wq, __global const uint* restrict wh,
                         __global uint* restrict col, int K) {
  __local uint tw[SG * TP];
  float v[ROT_NV];
  h8r_stage(wq, K, tw);
  h8r_column(wh, K, tw, v);
  const float amax = sub_group_reduce_max(rot_amax(v));
  if (get_sub_group_local_id() == 0)
    atomic_max((volatile __global uint*)(col + get_group_id(0) * SG + get_sub_group_id()), as_uint(rot_scale(amax)));
}

// the same grid: the weight rotated and rounded at its column scales, written back over the words it read
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void h8r_weights(__global uint* wq, __global const uint* restrict wh, __global const float* restrict col,
                          int K) {
  __local uint tw[SG * TP];
  __local uint tq[SG * TP];
  float v[ROT_NV];
  h8r_stage(wq, K, tw);
  h8r_column(wh, K, tw, v);
  const int c = get_sub_group_id(), l = get_sub_group_local_id();
  const float cs = col[get_group_id(0) * SG + c];
  const float iv = cs > 0.0f ? 1.0f / (cs * sqrt((float)HB)) : 0.0f;
#pragma unroll
  for (int i = 0; i < WPL; ++i) tq[c * TP + WPL * l + i] = rot_pack(v + 4 * i, iv);
  barrier(CLK_LOCAL_MEM_FENCE);
  // (lane lid writes words lid / 16 + 16 i of column lid % 16, as h8r_stage read them)
  const int lid = get_local_id(0), colm = lid % SG, w0 = lid / SG;
#ifdef XW_PACKW_OUT
  // (gemm_xw8 -DXW_PACKW's order, engine_xw.h ARC_GEMM_XW_PACK: in each 64-value chunk's 256 uints, word j (0..15) of
  //  column colm at ((j / 8) * 2 + colm / 8) * 64 + (j % 8) * 8 + colm % 8 instead of 16 j + colm)
  const int o = ((w0 / 8) * 2 + colm / 8) * 64 + (w0 % 8) * 8 + colm % 8;
#else
  const int o = lid;
#endif
  __global uint* wr = wq + h8r_region(K);
#pragma unroll
  for (int i = 0; i < WPL; ++i) wr[256 * i + o] = tq[colm * TP + w0 + SG * i];
}
#else
// A token's row: its blocks, one sub-group each, into int8 at the token's one scale (asch[M]). The source: the f32
// copy at byte H8F of aq (-DH8F, the values before their first rounding: ops.cl quant_store16), or the int8 rows aq
// with their scales asc (per 32 values with act32, [K / 32][M], else per 256, [K / 256][M]).
// grid (16 * K / HB * M), local (16 * K / HB)
__attribute__((intel_reqd_sub_group_size(SG)))
// -DQ2 (ARC_H8_Q2=1, 2026-11-21, opt-in; QUEUE "H8's flips", step 1): a second int8 plane of what the first leaves
// out, r = v - q * amax / 127, with a per-token scale of its own (the token's largest |r|), the same layout and
// rounding; aqh2 / asch2. The GEMM runs a second time on it and adds (engine_h8.h h8_matmul), so a value is carried in
// ~15 bits instead of 8 (as precise mode's quant_act32_q2 for the plain path)
__kernel void h8r_act(__global const uint* restrict aq, __global const float* restrict asc, int act32,
                      __global uint* restrict aqh, __global float* restrict asch, int M, int K
#ifdef Q2
                      , __global uint* restrict aqh2, __global float* restrict asch2
#endif
                      ) {
  __local float red[32];
  const int m = get_group_id(0), bk = get_sub_group_id();
  const uint l = get_sub_group_local_id();
  float v[ROT_NV];
#ifdef H8F
  __global const float* fr = (__global const float*)((__global const char*)aq + H8F) + (size_t)m * K + bk * HB + 4 * l;
#else
  __global const uint* qr = aq + (size_t)m * (K / 4) + bk * (HB / 4);
  // (the scales of value e of the row at asc[(e / 32) * M + m] with act32, else [(e / 256) * M + m])
  __global const float* sr = asc + (size_t)(act32 ? bk * (HB / 32) : bk * (HB / 256)) * M + m;
#endif
#pragma unroll
  for (int r = 0; r < ROT_NV / 4; ++r) {
    const uint i0 = 64 * r + 4 * l;  // (the lane's 4 values of this 64, in the block)
#ifdef H8F
    const float4 fv = vload4(0, fr + 64 * r);
    v[4 * r + 0] = rot_sign(fv.s0, i0 + 0);
    v[4 * r + 1] = rot_sign(fv.s1, i0 + 1);
    v[4 * r + 2] = rot_sign(fv.s2, i0 + 2);
    v[4 * r + 3] = rot_sign(fv.s3, i0 + 3);
#else
    const uint word = intel_sub_group_block_read(qr + r * SG);
    // (act32: lanes 0-7 hold the first 32 values of the 64, lanes 8-15 the second; each scale one load for the
    //  sub-group, then the lane's pick: a load per lane cost a 1 KB spill on Xe-LPG)
    float sc;
    if (act32) {
      const float s0 = sr[(size_t)(2 * r) * M], s1 = sr[(size_t)(2 * r + 1) * M];
      sc = l < 8 ? s0 : s1;
    } else {
      sc = sr[(size_t)(r / 4) * M];
    }
#pragma unroll
    for (int b = 0; b < 4; ++b) v[4 * r + b] = rot_sign(sc * (float)(char)(word >> (8 * b)), i0 + b);
#endif
  }
  rot_block(v);
  // (the token's largest value over its blocks, one sub-group each, through local memory: a work_group_reduce_max
  //  here cost a 1 KB spill on Xe-LPG, ocloc mtl 2026-11-17)
  float amax = sub_group_reduce_max(rot_amax(v));
  if (l == 0) red[bk] = amax;
  barrier(CLK_LOCAL_MEM_FENCE);
  amax = 0.0f;
  for (int i = 0; i < K / HB; ++i) amax = fmax(amax, red[i]);
  if (bk == 0 && l == 0) asch[m] = rot_scale(amax);
  const float iv = amax > 0.0f ? 127.0f / amax : 0.0f;
#ifdef XW_PACKA_OUT
  // (gemm_xw8 -DXW_PACKA's order, engine_xw.h ARC_GEMM_XW_PACK: [M / 8][K / 64][2 halves][8 rows][8 words]; lane l's
  //  word is word l % 8 of half l / 8 of chunk bk * HB / 64 + r, 128 uints a chunk)
  const size_t ao = ((size_t)(m / 8) * (K / 64) + bk * (HB / 64)) * 128 + (l / 8) * 64 + (m % 8) * 8 + (l % 8);
  __global uint* ar = aqh + ao;
#pragma unroll
  for (int r = 0; r < ROT_NV / 4; ++r) ar[128 * r] = rot_pack(v + 4 * r, iv);
#else
  const size_t ao = (size_t)m * (K / 4) + bk * (HB / 4);
  __global uint* ar = aqh + ao;
#pragma unroll
  for (int r = 0; r < ROT_NV / 4; ++r) intel_sub_group_block_write(ar + r * SG, rot_pack(v + 4 * r, iv));
#endif
#ifdef Q2
  // (the residual of each value against its int8 code, as rot_pack rounds it; then the token's largest one, through
  //  local memory as above: a second slot set, after a barrier so the first reads are done)
  const float st = amax / 127.0f;
#pragma unroll
  for (int j = 0; j < ROT_NV; ++j) v[j] -= (float)clamp((int)rint(v[j] * iv), -127, 127) * st;
  float amax2 = sub_group_reduce_max(rot_amax(v));
  barrier(CLK_LOCAL_MEM_FENCE);
  if (l == 0) red[bk] = amax2;
  barrier(CLK_LOCAL_MEM_FENCE);
  amax2 = 0.0f;
  for (int i = 0; i < K / HB; ++i) amax2 = fmax(amax2, red[i]);
  if (bk == 0 && l == 0) asch2[m] = rot_scale(amax2);
  const float iv2 = amax2 > 0.0f ? 127.0f / amax2 : 0.0f;
  __global uint* ar2 = aqh2 + ao;
#ifdef XW_PACKA_OUT
#pragma unroll
  for (int r = 0; r < ROT_NV / 4; ++r) ar2[128 * r] = rot_pack(v + 4 * r, iv2);
#else
#pragma unroll
  for (int r = 0; r < ROT_NV / 4; ++r) intel_sub_group_block_write(ar2 + r * SG, rot_pack(v + 4 * r, iv2));
#endif
#endif
}
#endif
