// CPU dot products for MoE expert rows (ARC_XCPU, zero/xe2/engine_xcpu.h): one token's activations as the GPU's expert
// kernels read them (int8, one f32 scale per 32 values: aq_ / asc_, moe_quant's hq / hsc) against a weight row in its
// quantized blocks, neither requantized. A row is read
// either as the GGUF holds it (ggml's block layout) or as the expert file / RAM tier hold it (the GPU raw form,
// src/quants.h gpu_raw_sb: superblocks of 256 values, the IQ types' f16 d moved after the rest and the superblock
// padded to 8 bytes, the per-32 / per-64 types' scales first; rows of a K that is no multiple of 256 padded with zero
// blocks). The AVX2 bodies are ggml's (ggml/src/ggml-cpu/arch/x86/quants.c, MIT license, Copyright (c) 2023-2026 The
// ggml authors; NOTICE) with the block fields read through those two layouts; the codebooks src/iq_tables.h's; Q2_0
// (Flash Next's down projections) written here in the same style. Without AVX2: no kernel (ARC_XCPU stays off).
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include "../../src/gguf.h"
#include "../../src/iq_tables.h"
#include "../../src/quants.h"

namespace cpux {

constexpr int QK = 256;

// (quants.h's exact conversion: MSVC has no _cvtsh_ss; a half's float value is the same either way)
inline float h2f(uint16_t h) { return fp16_to_f32(h); }
inline float d16(const uint8_t* p) { uint16_t h; memcpy(&h, p, 2); return h2f(h); }

// Field offsets in a superblock (bytes), ggml's layout (G false) or the GPU raw form (G true); SB: superblock bytes
template <bool G> struct Iq2s { static constexpr size_t SB = G ? 88 : 82, D = G ? 80 : 0, QS = G ? 0 : 2, QH = QS + 64, SC = QS + 72; };
template <bool G> struct Iq3xxs { static constexpr size_t SB = G ? 104 : 98, D = G ? 96 : 0, QS = G ? 0 : 2, GAS = QS + 64; };
template <bool G> struct Iq3s {
  static constexpr size_t SB = G ? 112 : 110, D = G ? 108 : 0, QS = G ? 0 : 2, QH = QS + 64, SG = QS + 72, SC = QS + 104;
};
// per-32 / per-64 types: block b's d and quants
template <bool G> inline const uint8_t* nl_d(const uint8_t* x, int b) { return G ? x + (b >> 3) * 144 + 2 * (b & 7) : x + 18 * b; }
template <bool G> inline const uint8_t* nl_q(const uint8_t* x, int b) { return G ? x + (b >> 3) * 144 + 16 + 16 * (b & 7) : x + 18 * b + 2; }
template <bool G> inline const uint8_t* q20_d(const uint8_t* x, int b) { return G ? x + (b >> 2) * 72 + 2 * (b & 3) : x + 18 * b; }
template <bool G> inline const uint8_t* q20_q(const uint8_t* x, int b) { return G ? x + (b >> 2) * 72 + 8 + 16 * (b & 3) : x + 18 * b + 2; }

// activations: int8 and one f32 scale per 32 values, as moe_quant writes them (amax / 127, round to nearest even)
inline void quant_act32(const float* x, int8_t* q, float* sc, int K) {
  for (int b = 0; b < K / 32; ++b, x += 32, q += 32) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    for (int i = 0; i < 32; ++i) q[i] = (int8_t)(int)std::nearbyint(x[i] * inv);
    sc[b] = amax / 127.0f;
  }
}
// The same, and the rounding residual quantized the same way (moe_quant -DQ2M: v - q * (amax / 127), per 32 values)
inline void quant_act32_q2(const float* x, int8_t* q, float* sc, int8_t* q2, float* sc2, int K) {
  for (int b = 0; b < K / 32; ++b, x += 32, q += 32, q2 += 32) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f, s = amax / 127.0f;
    float r[32], rmax = 0.0f;
    for (int i = 0; i < 32; ++i) {
      const int qi = (int)std::nearbyint(x[i] * inv);
      q[i] = (int8_t)qi;
      r[i] = x[i] - (float)qi * s;
      rmax = std::max(rmax, std::fabs(r[i]));
    }
    const float rinv = rmax > 0.0f ? 127.0f / rmax : 0.0f;
    for (int i = 0; i < 32; ++i) q2[i] = (int8_t)(int)std::nearbyint(r[i] * rinv);
    sc[b] = s;
    sc2[b] = rmax / 127.0f;
  }
}

#ifdef __AVX2__
#define CPUX_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
inline float hsum8(__m256 x) {
  __m128 r = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
  r = _mm_add_ps(r, _mm_movehl_ps(r, r));
  r = _mm_add_ss(r, _mm_movehdup_ps(r));
  return _mm_cvtss_f32(r);
}
inline __m256i mul_add_epi8(__m256i x, __m256i y) { return _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x)); }
inline __m256i scale_shuffle_k4(int i) {
  alignas(32) static const uint8_t k[256] = {
      0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
      2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
      4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
      6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
      8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
  return _mm256_load_si256((const __m256i*)k + i);
}
// (ggml's keven_signs_q2xs: for each 7-bit sign index, 8 bytes of +1 / -1 from ksigns_iq2xs's 8 bits)
inline const uint64_t* keven_signs() {
  static const auto t = [] {
    std::array<uint64_t, 128> a{};
    for (int i = 0; i < 128; ++i) {
      uint64_t v = 0;
      for (int j = 0; j < 8; ++j) v |= (uint64_t)((ksigns_iq2xs[i] >> j) & 1 ? 0xffu : 0x01u) << (8 * j);
      a[i] = v;
    }
    return a;
  }();
  return t.data();
}
alignas(32) inline const uint8_t k_mask1[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
alignas(32) inline const uint8_t k_mask2[32] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
                                                1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};

template <bool G> float dot_iq2_s(int n, const void* vx, const int8_t* aq, const float* as) {
  using B = Iq2s<G>;
  const uint8_t* x = (const uint8_t*)vx;
  const __m128i m4 = _mm_set1_epi8(0xf), m1 = _mm_set1_epi8(1);
  const __m256i mask1 = _mm256_load_si256((const __m256i*)k_mask1), mask2 = _mm256_load_si256((const __m256i*)k_mask2);
  uint64_t aux64;
  __m256 accumf = _mm256_setzero_ps();
  for (int i = 0; i < n / QK; ++i, x += B::SB) {
    const float d = d16(x + B::D);
    const float* sa = as + 8 * i;
    const uint8_t* qs = x + B::QS;
    const uint8_t* qh = x + B::QH;
    const uint16_t* signs = (const uint16_t*)(x + B::QS + QK / 8);
    const int8_t* q8 = aq + QK * i;
    memcpy(&aux64, x + B::SC, 8);
    const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
    const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
    for (int ib32 = 0; ib32 < QK / 32; ib32 += 2) {
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32 + 0] << 2) & 0x300)], iq2s_grid[qs[2] | ((qh[ib32 + 0] << 4) & 0x300)],
                                             iq2s_grid[qs[1] | ((qh[ib32 + 0] << 6) & 0x300)], iq2s_grid[qs[0] | ((qh[ib32 + 0] << 8) & 0x300)]);
      const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32 + 1] << 2) & 0x300)], iq2s_grid[qs[6] | ((qh[ib32 + 1] << 4) & 0x300)],
                                             iq2s_grid[qs[5] | ((qh[ib32 + 1] << 6) & 0x300)], iq2s_grid[qs[4] | ((qh[ib32 + 1] << 8) & 0x300)]);
      qs += 8;
      __m256i aux = _mm256_set1_epi32(signs[0] | ((uint32_t)signs[1] << 16));
      aux = _mm256_and_si256(_mm256_shuffle_epi8(aux, mask1), mask2);
      const __m256i s2_1 = _mm256_cmpeq_epi8(aux, mask2);
      const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
      aux = _mm256_set1_epi32(signs[2] | ((uint32_t)signs[3] << 16));
      aux = _mm256_and_si256(_mm256_shuffle_epi8(aux, mask1), mask2);
      const __m256i s2_2 = _mm256_cmpeq_epi8(aux, mask2);
      const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
      signs += 4;
      const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1), dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
      const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_shuffle_epi8(scales16, scale_shuffle_k4(ib32 + 0)));
      const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_shuffle_epi8(scales16, scale_shuffle_k4(ib32 + 1)));
      accumf = _mm256_fmadd_ps(_mm256_set1_ps(d * sa[ib32]), _mm256_cvtepi32_ps(p1), accumf);
      accumf = _mm256_fmadd_ps(_mm256_set1_ps(d * sa[ib32 + 1]), _mm256_cvtepi32_ps(p2), accumf);
    }
  }
  return 0.125f * hsum8(accumf);
}

template <bool G> float dot_iq3_xxs(int n, const void* vx, const int8_t* aq, const float* as) {
  using B = Iq3xxs<G>;
  const uint8_t* x = (const uint8_t*)vx;
  const uint64_t* signs64 = keven_signs();
  uint32_t aux32[2];
  __m256 accumf = _mm256_setzero_ps();
  for (int i = 0; i < n / QK; ++i, x += B::SB) {
    const float d = d16(x + B::D);
    const float* sa = as + 8 * i;
    const uint8_t* q3 = x + B::QS;
    const uint8_t* gas = x + B::GAS;
    const int8_t* q8 = aq + QK * i;
    for (int ib32 = 0; ib32 < QK / 32; ib32 += 2) {
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                            iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
      q3 += 8;
      const __m256i q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                            iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
      q3 += 8;
      memcpy(aux32, gas, 8); gas += 8;
      const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                             signs64[(aux32[0] >> 7) & 127], signs64[(aux32[0] >> 0) & 127]);
      const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                             signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
      const __m256i dot1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1));
      const __m256i dot2 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2));
      const uint16_t ls1 = aux32[0] >> 28, ls2 = aux32[1] >> 28;
      accumf = _mm256_fmadd_ps(_mm256_set1_ps(d * sa[ib32]), _mm256_cvtepi32_ps(_mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1))), accumf);
      accumf = _mm256_fmadd_ps(_mm256_set1_ps(d * sa[ib32 + 1]), _mm256_cvtepi32_ps(_mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1))), accumf);
    }
  }
  return 0.25f * hsum8(accumf);
}

template <bool G> float dot_iq3_s(int n, const void* vx, const int8_t* aq, const float* as) {
  using B = Iq3s<G>;
  const uint8_t* x = (const uint8_t*)vx;
  const __m256i mask1 = _mm256_load_si256((const __m256i*)k_mask1), mask2 = _mm256_load_si256((const __m256i*)k_mask2);
  const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8), idx_mask = _mm256_set1_epi32(256);
  union { __m256i vec[2]; uint32_t index[16]; } idx;
  __m256 accumf = _mm256_setzero_ps();
  for (int i = 0; i < n / QK; ++i, x += B::SB) {
    const float d = d16(x + B::D);
    const float* sa = as + 8 * i;
    const uint8_t* qs = x + B::QS;
    const uint8_t* qh = x + B::QH;
    const uint8_t* sc = x + B::SC;
    const uint16_t* signs = (const uint16_t*)(x + B::SG);
    const int8_t* q8 = aq + QK * i;
    for (int ib32 = 0; ib32 < QK / 32; ib32 += 2) {
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)qs)); qs += 16;
      idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib32 + 0]), idx_shift), idx_mask);
      idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib32 + 1]), idx_shift), idx_mask);
      idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
      idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
      const __m256i q2_1 = _mm256_set_epi32(iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                                            iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]);
      const __m256i q2_2 = _mm256_set_epi32(iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                                            iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[9]], iq3s_grid[idx.index[8]]);
      __m256i aux = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
      aux = _mm256_and_si256(_mm256_shuffle_epi8(aux, mask1), mask2);
      const __m256i s2_1 = _mm256_cmpeq_epi8(aux, mask2);
      const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
      aux = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
      aux = _mm256_and_si256(_mm256_shuffle_epi8(aux, mask1), mask2);
      const __m256i s2_2 = _mm256_cmpeq_epi8(aux, mask2);
      const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
      signs += 4;
      const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1), dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
      const uint16_t ls1 = sc[ib32 / 2] & 0xf, ls2 = sc[ib32 / 2] >> 4;
      accumf = _mm256_fmadd_ps(_mm256_set1_ps(d * sa[ib32]), _mm256_cvtepi32_ps(_mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1))), accumf);
      accumf = _mm256_fmadd_ps(_mm256_set1_ps(d * sa[ib32 + 1]), _mm256_cvtepi32_ps(_mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1))), accumf);
    }
  }
  return hsum8(accumf);
}

// (IQ4_XS: the same superblock in both layouts: d, scales_h, scales_l[4], qs[128])
inline float dot_iq4_xs(int n, const void* vx, const int8_t* aq, const float* as) {
  const uint8_t* x = (const uint8_t*)vx;
  const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl), m4b = _mm_set1_epi8(0x0f);
  __m256 accum = _mm256_setzero_ps();
  for (int ibl = 0; ibl < n / QK; ++ibl, x += 136) {
    const uint8_t* qs = x + 8;
    const uint8_t* sl = x + 4;
    const int8_t* q8 = aq + QK * ibl;
    const float* sa = as + 8 * ibl;
    const float d = d16(x);
    uint16_t sh;
    memcpy(&sh, x + 2, 2);
    for (int ib = 0; ib < QK / 32; ib += 2) {
      const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)qs); qs += 16;
      const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)qs); qs += 16;
      const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
      const __m256i q4b_1 = CPUX_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                           _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
      const __m256i q4b_2 = CPUX_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                           _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
      const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1), p16_2 = mul_add_epi8(q4b_2, q8b_2);
      const int16_t ls1 = ((sl[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
      const int16_t ls2 = ((sl[ib / 2] >> 4) | ((sh << 2) & 0x30)) - 32;
      sh >>= 4;
      accum = _mm256_fmadd_ps(_mm256_set1_ps(d * sa[ib]), _mm256_cvtepi32_ps(_mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1))), accum);
      accum = _mm256_fmadd_ps(_mm256_set1_ps(d * sa[ib + 1]), _mm256_cvtepi32_ps(_mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2))), accum);
    }
  }
  return hsum8(accum);
}

template <bool G> float dot_iq4_nl(int n, const void* vx, const int8_t* aq, const float* as) {
  const uint8_t* x = (const uint8_t*)vx;
  const int nb = n / 32;
  const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl), m4b = _mm_set1_epi8(0x0f);
  const __m256i mone = _mm256_set1_epi16(1);
  __m256 accum1 = _mm256_setzero_ps(), accum2 = _mm256_setzero_ps();
  int ib = 0;
  for (; ib + 1 < nb; ib += 2) {
    const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)nl_q<G>(x, ib)), q4bits_2 = _mm_loadu_si128((const __m128i*)nl_q<G>(x, ib + 1));
    const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)(aq + 32 * ib)), q8b_2 = _mm256_loadu_si256((const __m256i*)(aq + 32 * ib + 32));
    const __m256i q4b_1 = CPUX_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                         _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
    const __m256i q4b_2 = CPUX_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                         _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
    const __m256i p_1 = _mm256_madd_epi16(mul_add_epi8(q4b_1, q8b_1), mone), p_2 = _mm256_madd_epi16(mul_add_epi8(q4b_2, q8b_2), mone);
    accum1 = _mm256_fmadd_ps(_mm256_set1_ps(as[ib] * d16(nl_d<G>(x, ib))), _mm256_cvtepi32_ps(p_1), accum1);
    accum2 = _mm256_fmadd_ps(_mm256_set1_ps(as[ib + 1] * d16(nl_d<G>(x, ib + 1))), _mm256_cvtepi32_ps(p_2), accum2);
  }
  float s = hsum8(_mm256_add_ps(accum1, accum2));
  for (; ib < nb; ++ib) {  // (an odd block count: ggml's scalar tail)
    const uint8_t* q = nl_q<G>(x, ib);
    int s1 = 0, s2 = 0;
    for (int j = 0; j < 16; ++j) {
      s1 += aq[32 * ib + j] * kvalues_iq4nl[q[j] & 0xf];
      s2 += aq[32 * ib + j + 16] * kvalues_iq4nl[q[j] >> 4];
    }
    s += as[ib] * d16(nl_d<G>(x, ib)) * (s1 + s2);
  }
  return s;
}

// Q2_0 (blocks of 64: f16 d, 2-bit codes, value (q - 1) * d; element 4 b + j in bits 2 j of byte b) against two
// blocks of 32 activations: the codes spread to bytes in element order, then as IQ4_NL's blocks
template <bool G> float dot_q2_0(int n, const void* vx, const int8_t* aq, const float* as) {
  const uint8_t* x = (const uint8_t*)vx;
  const __m128i m3 = _mm_set1_epi8(3), one = _mm_set1_epi8(1);
  const __m256i mone = _mm256_set1_epi16(1);
  __m256 acc = _mm256_setzero_ps();
  for (int i = 0; i < n / 64; ++i) {
    const __m128i q = _mm_loadu_si128((const __m128i*)q20_q<G>(x, i));
    const __m128i c0 = _mm_sub_epi8(_mm_and_si128(q, m3), one), c1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q, 2), m3), one);
    const __m128i c2 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q, 4), m3), one), c3 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q, 6), m3), one);
    // (bytes b of c0..c3 -> elements 4 b + 0..3)
    const __m128i a01l = _mm_unpacklo_epi8(c0, c1), a01h = _mm_unpackhi_epi8(c0, c1);
    const __m128i a23l = _mm_unpacklo_epi8(c2, c3), a23h = _mm_unpackhi_epi8(c2, c3);
    const __m256i w0 = CPUX_SET_M128I(_mm_unpackhi_epi16(a01l, a23l), _mm_unpacklo_epi16(a01l, a23l));  // elements 0..31
    const __m256i w1 = CPUX_SET_M128I(_mm_unpackhi_epi16(a01h, a23h), _mm_unpacklo_epi16(a01h, a23h));  // elements 32..63
    const __m256i p0 = _mm256_madd_epi16(mul_add_epi8(w0, _mm256_loadu_si256((const __m256i*)(aq + 64 * i))), mone);
    const __m256i p1 = _mm256_madd_epi16(mul_add_epi8(w1, _mm256_loadu_si256((const __m256i*)(aq + 64 * i + 32))), mone);
    const float d = d16(q20_d<G>(x, i));
    acc = _mm256_fmadd_ps(_mm256_set1_ps(d * as[2 * i]), _mm256_cvtepi32_ps(p0), acc);
    acc = _mm256_fmadd_ps(_mm256_set1_ps(d * as[2 * i + 1]), _mm256_cvtepi32_ps(p1), acc);
  }
  return hsum8(acc);
}
// ---- (2026-11-19) DeepSeek-V4.1's expert types: Unsloth's UD-Q8_K_XL has MXFP4 experts (some down projections Q5_1),
// other builds Q8_0. Each against the GPU's int8 activations, one scale per 32, as the types above.
// MXFP4 (blocks of 32: an E8M0 scale, then 16 bytes of E2M1 codes, low nibbles elements 0..15; the GPU raw form puts a
// superblock's 8 scale bytes first, then its 8 x 16 code bytes): IQ4_NL's form with the MXFP4 table and the scale
// 2^(e - 128) (e8m0_to_f32_half, the table's values being twice the E2M1 ones)
template <bool G> inline const uint8_t* mx_e(const uint8_t* x, int b) { return G ? x + (b >> 3) * 136 + (b & 7) : x + 17 * b; }
template <bool G> inline const uint8_t* mx_q(const uint8_t* x, int b) { return G ? x + (b >> 3) * 136 + 8 + 16 * (b & 7) : x + 17 * b + 1; }
template <bool G> float dot_mxfp4(int n, const void* vx, const int8_t* aq, const float* as) {
  const uint8_t* x = (const uint8_t*)vx;
  const int nb = n / 32;
  const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_mxfp4), m4b = _mm_set1_epi8(0x0f);
  const __m256i mone = _mm256_set1_epi16(1);
  __m256 acc = _mm256_setzero_ps();
  for (int ib = 0; ib < nb; ++ib) {
    const __m128i q4 = _mm_loadu_si128((const __m128i*)mx_q<G>(x, ib));
    const __m256i w = CPUX_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4, 4), m4b)),
                                     _mm_shuffle_epi8(values128, _mm_and_si128(q4, m4b)));
    const __m256i p = _mm256_madd_epi16(mul_add_epi8(w, _mm256_loadu_si256((const __m256i*)(aq + 32 * ib))), mone);
    acc = _mm256_fmadd_ps(_mm256_set1_ps(as[ib] * e8m0_to_f32_half(*mx_e<G>(x, ib))), _mm256_cvtepi32_ps(p), acc);
  }
  return hsum8(acc);
}
// Q8_0 (blocks of 32: f16 d + 32 int8; the GPU raw form: a superblock's 8 scales, then its 8 x 32 quants)
template <bool G> inline const uint8_t* q8_d(const uint8_t* x, int b) { return G ? x + (b >> 3) * 272 + 2 * (b & 7) : x + 34 * b; }
template <bool G> inline const uint8_t* q8_q(const uint8_t* x, int b) { return G ? x + (b >> 3) * 272 + 16 + 32 * (b & 7) : x + 34 * b + 2; }
template <bool G> float dot_q8_0(int n, const void* vx, const int8_t* aq, const float* as) {
  const uint8_t* x = (const uint8_t*)vx;
  const __m256i mone = _mm256_set1_epi16(1);
  __m256 acc = _mm256_setzero_ps();
  for (int ib = 0; ib < n / 32; ++ib) {
    const __m256i w = _mm256_loadu_si256((const __m256i*)q8_q<G>(x, ib));
    const __m256i p = _mm256_madd_epi16(mul_add_epi8(w, _mm256_loadu_si256((const __m256i*)(aq + 32 * ib))), mone);
    acc = _mm256_fmadd_ps(_mm256_set1_ps(as[ib] * d16(q8_d<G>(x, ib))), _mm256_cvtepi32_ps(p), acc);
  }
  return hsum8(acc);
}
// Q5_1 (blocks of 32 in 24 bytes: f16 d, f16 m, 32 high bits, 16 bytes of low nibbles; value d q + m, q in 0..31).
// The expert file keeps the GGUF layout (G alike). sum(d q a) + m sum(a), each block's int sums exact
template <bool G> float dot_q5_1(int n, const void* vx, const int8_t* aq, const float* as) {
  const uint8_t* x = (const uint8_t*)vx;
  const __m128i m4b = _mm_set1_epi8(0x0f);
  const __m256i mone = _mm256_set1_epi16(1), one8 = _mm256_set1_epi8(1), b5 = _mm256_set1_epi8(0x10);
  const __m256i shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
  const __m256i bitm = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
  __m256 acc = _mm256_setzero_ps();
  for (int ib = 0; ib < n / 32; ++ib) {
    const uint8_t* p = x + 24 * ib;
    const __m128i lo = _mm_loadu_si128((const __m128i*)(p + 8));
    __m256i q = CPUX_SET_M128I(_mm_and_si128(_mm_srli_epi16(lo, 4), m4b), _mm_and_si128(lo, m4b));  // elements 0..31
    uint32_t hb;
    memcpy(&hb, p + 4, 4);
    // (bit j of hb -> byte j all ones, then the 5th bit of element j)
    const __m256i hi = _mm256_cmpeq_epi8(_mm256_or_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)hb), shuf), bitm),
                                         _mm256_set1_epi64x(-1));
    q = _mm256_or_si256(q, _mm256_and_si256(hi, b5));
    const __m256i a = _mm256_loadu_si256((const __m256i*)(aq + 32 * ib));
    const __m256i pq = _mm256_madd_epi16(_mm256_maddubs_epi16(q, a), mone);      // sum q a (q unsigned, a signed)
    const __m256i pa = _mm256_madd_epi16(_mm256_maddubs_epi16(one8, a), mone);   // sum a
    const float d = d16(p), m = d16(p + 2);
    acc = _mm256_fmadd_ps(_mm256_set1_ps(as[ib] * d), _mm256_cvtepi32_ps(pq), acc);
    acc = _mm256_fmadd_ps(_mm256_set1_ps(as[ib] * m), _mm256_cvtepi32_ps(pa), acc);
  }
  return hsum8(acc);
}
#endif  // __AVX2__

// a row kernel for a GGUF type and a layout (gpu: the expert file's GPU raw form): its dot, the values a block holds,
// the bytes a superblock of 256 values takes in that layout
struct Kern {
  float (*dot)(int, const void*, const int8_t*, const float*) = nullptr;
  int blk = 0;
  size_t sb = 0;
};
inline Kern kern(uint32_t t, bool gpu) {
#ifdef __AVX2__
  switch (t) {
    case GGML_IQ2_S: return {gpu ? dot_iq2_s<true> : dot_iq2_s<false>, 256, gpu ? Iq2s<true>::SB : Iq2s<false>::SB};
    case GGML_IQ3_XXS: return {gpu ? dot_iq3_xxs<true> : dot_iq3_xxs<false>, 256, gpu ? Iq3xxs<true>::SB : Iq3xxs<false>::SB};
    case GGML_IQ3_S: return {gpu ? dot_iq3_s<true> : dot_iq3_s<false>, 256, gpu ? Iq3s<true>::SB : Iq3s<false>::SB};
    case GGML_IQ4_XS: return {dot_iq4_xs, 256, 136};
    case GGML_IQ4_NL: return {gpu ? dot_iq4_nl<true> : dot_iq4_nl<false>, 32, 144};
    case GGML_Q2_0: return {gpu ? dot_q2_0<true> : dot_q2_0<false>, 64, 72};
    case GGML_MXFP4: return {gpu ? dot_mxfp4<true> : dot_mxfp4<false>, 32, 136};
    case GGML_Q8_0: return {gpu ? dot_q8_0<true> : dot_q8_0<false>, 32, 272};
    case GGML_Q5_1: return {dot_q5_1<false>, 32, 192};
  }
#else
  (void)t, (void)gpu;
#endif
  return {};
}

}  // namespace cpux
