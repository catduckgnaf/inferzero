// A small DeepSeek-V4.1-Flash (`deepseek41`) GGUF with the real layout and random weights, for building and checking
// the engine's path against tools/ref_forward without the 231 GB model (docs/deepseek41-spec.md).
//
// The same keys and tensor names as the ISTA-DASLab GGUFs, smaller sizes: 10 layers covering every layer role of the
// real 40 (window only; ratio 2 with its own compressor and indexer; a ratio 2 consumer; ratio 1 with the candidate
// pool; a later index layer that uses the pool; consumers of each), two Engram layers (one a KV source, as 14 is), a
// sliding window of 16 so that a ~100-token prompt passes every case (window wrap, more compressed rows than top_k,
// more blocks than the pool keeps). Head size 512 with 64 rotated dimensions and the 128-wide indexer stay as in the
// real model, since kernels may be written for them.
//
// --real: the real model's widths instead (embedding 5120, 64 heads, q LoRA 1280, 8 output groups of 1024, window 128,
// top-k 512, a 2048-block candidate pool of 8 rows, 32 indexer heads, 384 experts with 6 active, Engram rows of 256),
// so that kernels and buffers meet their real sizes; but 4 layers (window only; ratio 2 with Engram; ratio 1 with the
// candidate pool; an index layer using it), an expert FFN of 256 instead of 2304 and a vocabulary of 2048 to keep the
// file at ~3.5 GB (experts in Q4_0). Its experts are made and written one at a time.
//
// --unsloth: the tensor types of Unsloth's UD-Q8_K_XL (DeepSeek-V4.1-Flash-1017-GGUF): routed experts MXFP4, the
// Engram table MXFP8 (type 43 of Unsloth's llama.cpp fork), the router F32, every other matrix BF16.
//
// --q5_1-down: the down experts in Q5_1, as Unsloth's UD-Q4_K_XL Qwen3.8 Flash Next has them (2026-11-14, issue #23).
//
// usage: make_tiny_deepseek41 OUT.gguf [--seed N] [--nu N] [--heads N] [--real] [--unsloth] [--q5_1-down]
//
// The C++ port of the former tools/make_tiny_deepseek41.py (2026-11-21, no Python in the repository): the same keys,
// tensors, shapes, types and encoders (each rounds as the .py's NumPy code did). The random numbers come from this
// file's own generator (xoshiro256**, normals by Box-Muller), not NumPy's, so a seed gives other weights than the .py
// gave: the same kind of model, not the same bytes.
#include "../src/log.h"  // (zlog: first, as gguf_pack.h's headers use it; build.ps1 force-includes it, g++ does not)
#include "../zero/gguf_pack.h"  // (zpack::Writer / put_kv)
#include "../src/quants.h"  // (f32_to_fp16: round to nearest even)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace {

const std::string ARCH = "deepseek41";
int E = 256, NL = 10, H = 8, D = 512, R = 64, QL = 256, G = 2, OL = 128;
int SWA = 16, TOPK = 8, IH = 16, ID = 128;
int NE = 16, NU = 4, FF = 256, NSH = 1;
int HC = 4, PD = 64, NGRAM = 4, HPN = 8;
int VOCAB = 512;
std::vector<int64_t> RATIOS = {0, 0, 2, 2, 2, 2, 1, 1, 1, 1};
std::vector<int64_t> KV_SRC = {2, 4, 6};
std::vector<int64_t> IDX_SRC = {2, 4, 6, 8};
int CAND_SRC = 6, CAND_BLOCKS = 4, CAND_BS = 4;
std::vector<int64_t> PLE_LAYERS = {1, 4};

void real_config() {
  E = 5120, NL = 4, H = 64, D = 512, R = 64, QL = 1280, G = 8, OL = 1024;
  SWA = 128, TOPK = 512, IH = 32, ID = 128;
  NE = 384, NU = 6, FF = 256, NSH = 1;
  HC = 4, PD = 256, NGRAM = 4, HPN = 8;
  VOCAB = 2048;
  RATIOS = {0, 2, 1, 1};
  KV_SRC = {1, 2};
  IDX_SRC = {1, 2, 3};
  CAND_SRC = 2, CAND_BLOCKS = 2048, CAND_BS = 8;
  PLE_LAYERS = {0, 1};
}

// ---- random numbers: xoshiro256** seeded through splitmix64; normals by Box-Muller (both of each pair used)
struct Rng {
  uint64_t s[4];
  bool have = false;
  double spare = 0;
  explicit Rng(uint64_t seed) {
    for (auto& x : s) {
      seed += 0x9E3779B97F4A7C15ull;
      uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
      x = z ^ (z >> 31);
    }
  }
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t next() {
    const uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
  double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }  // [0, 1)
  uint64_t below(uint64_t n) {  // [0, n), without bias
    const uint64_t lim = UINT64_MAX - UINT64_MAX % n;
    uint64_t x;
    do x = next(); while (x >= lim);
    return x % n;
  }
  double normal() {
    if (have) { have = false; return spare; }
    double u1;
    do u1 = uniform(); while (u1 <= 0);
    const double u2 = uniform(), r = std::sqrt(-2.0 * std::log(u1)), a = 6.283185307179586 * u2;
    spare = r * std::sin(a);
    have = true;
    return r * std::cos(a);
  }
};

typedef std::vector<float> F;

// ---- encoders (each as the .py's NumPy code)
const float kFp4[8] = {0, 1, 2, 3, 4, 6, 8, 12};

uint16_t f2bf(float f) {  // round to nearest even
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
double e4m3fn(int c) {  // (NaN for 0x7F / 0xFF)
  const int sg = c >> 7, ex = (c >> 3) & 15, m = c & 7;
  if (ex == 15 && m == 7) return NAN;
  const double x = ex == 0 ? m * std::ldexp(1.0, -9) : (8 + m) * std::ldexp(1.0, ex - 10);
  return sg ? -x : x;
}
template <typename T> void put(std::vector<uint8_t>& o, T v) {
  const size_t n = o.size();
  o.resize(n + sizeof v);
  memcpy(&o[n], &v, sizeof v);
}

std::vector<uint8_t> encode(const F& x, uint32_t type) {
  std::vector<uint8_t> o;
  const size_t nb = x.size() / 32;
  switch (type) {
    case GGML_F32:
      o.resize(x.size() * 4);
      memcpy(o.data(), x.data(), o.size());
      return o;
    case GGML_F16:
      for (float v : x) put<uint16_t>(o, f32_to_fp16(v));
      return o;
    case GGML_BF16:
      for (float v : x) put<uint16_t>(o, f2bf(v));
      return o;
    default: break;
  }
  if (x.size() % 32) throw std::runtime_error("a block type needs a multiple of 32 values");
  o.reserve(nb * 34);
  for (size_t b = 0; b < nb; ++b) {
    const float* v = &x[b * 32];
    if (type == GGML_Q8_0) {
      float amax = 0;
      for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i]));
      const uint16_t dh = f32_to_fp16(amax / 127.0f);
      const float df = fp16_to_f32(dh), inv = df > 0 ? 1.0f / df : 0.0f;
      put<uint16_t>(o, dh);
      for (int i = 0; i < 32; ++i)
        put<int8_t>(o, (int8_t)std::max(-127.0f, std::min(127.0f, std::nearbyint(v[i] * inv))));
    } else if (type == GGML_Q4_0) {  // (as ggml: d = the value of largest magnitude / -8, q = x / d + 8 in 0..15)
      int im = 0;
      for (int i = 1; i < 32; ++i)
        if (std::fabs(v[i]) > std::fabs(v[im])) im = i;
      const uint16_t dh = f32_to_fp16(v[im] / -8.0f);
      const float df = fp16_to_f32(dh), inv = df != 0 ? 1.0f / df : 0.0f;
      uint8_t q[32];
      for (int i = 0; i < 32; ++i) q[i] = (uint8_t)std::max(0.0f, std::min(15.0f, std::trunc(v[i] * inv + 8.5f)));
      put<uint16_t>(o, dh);
      for (int i = 0; i < 16; ++i) put<uint8_t>(o, (uint8_t)(q[i] | (q[i + 16] << 4)));
    } else if (type == GGML_Q5_1) {  // (as ggml: d = (max - min) / 31, m = min, q = round((x - min) / d) in 0..31)
      float lo = v[0], hi = v[0];
      for (int i = 1; i < 32; ++i) lo = std::min(lo, v[i]), hi = std::max(hi, v[i]);
      const uint16_t dh = f32_to_fp16((hi - lo) / 31.0f), mh = f32_to_fp16(lo);
      const float df = fp16_to_f32(dh), mf = fp16_to_f32(mh), inv = df != 0 ? 1.0f / df : 0.0f;
      uint32_t qh = 0;
      uint8_t ql[32];
      for (int i = 0; i < 32; ++i) {
        const uint32_t q = (uint32_t)std::max(0.0f, std::min(31.0f, std::nearbyint((v[i] - mf) * inv)));
        qh |= (q >> 4) << i;
        ql[i] = (uint8_t)(q & 15);
      }
      put<uint16_t>(o, dh);
      put<uint16_t>(o, mh);
      put<uint32_t>(o, qh);
      for (int i = 0; i < 16; ++i) put<uint8_t>(o, (uint8_t)(ql[i] | (ql[i + 16] << 4)));
    } else if (type == GGML_MXFP4) {  // (scale 2^(e - 128) so that the largest magnitude is at most 6; nearest code)
      double amax = 0;
      for (int i = 0; i < 32; ++i) amax = std::max(amax, (double)std::fabs(v[i]));
      const double k = amax > 0 ? std::ceil(std::log2(amax / 6.0)) : -128;
      const int e = (int)std::max(2.0, std::min(254.0, k + 128));
      const double sc = std::ldexp(1.0, e - 128);
      uint8_t q[32];
      for (int i = 0; i < 32; ++i) {
        const double y = v[i] / sc, sg = y > 0 ? 1 : y < 0 ? -1 : 0;
        int best = 0;
        double bd = INFINITY;
        for (int c = 0; c < 8; ++c) {
          const double dd = std::fabs(y - kFp4[c] * sg);
          if (dd < bd) bd = dd, best = c;  // (the first of equals, as argmin)
        }
        q[i] = (uint8_t)(y < 0 && best > 0 ? best + 8 : best);
      }
      put<uint8_t>(o, (uint8_t)e);
      for (int i = 0; i < 16; ++i) put<uint8_t>(o, (uint8_t)(q[i] | (q[i + 16] << 4)));
    } else if (type == GGML_MXFP8) {  // (scale 2^(e - 127) so that the largest magnitude is at most 448; nearest code)
      static std::vector<std::pair<double, int>> sv;  // (the finite codes by value, ties by code: a stable sort)
      if (sv.empty()) {
        for (int c = 0; c < 256; ++c)
          if (!std::isnan(e4m3fn(c))) sv.push_back({e4m3fn(c), c});
        std::stable_sort(sv.begin(), sv.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
          return a.first < b.first;
        });
      }
      double amax = 0;
      for (int i = 0; i < 32; ++i) amax = std::max(amax, (double)std::fabs(v[i]));
      const double k = amax > 0 ? std::ceil(std::log2(amax / 448.0)) : -127;
      const int e = (int)std::max(1.0, std::min(254.0, k + 127));
      const double sc = std::ldexp(1.0, e - 127);
      put<uint8_t>(o, (uint8_t)e);
      for (int i = 0; i < 32; ++i) {
        const double y = v[i] / sc;
        // (searchsorted left: the first value >= y, clipped to 1 .. n - 1; then the nearer one, the lower on a tie)
        size_t j = std::lower_bound(sv.begin(), sv.end(), y,
                                    [](const std::pair<double, int>& a, double t) { return a.first < t; }) - sv.begin();
        j = std::max<size_t>(1, std::min(sv.size() - 1, j));
        const double lo = sv[j - 1].first, hi = sv[j].first;
        put<uint8_t>(o, (uint8_t)(std::fabs(y - lo) <= std::fabs(hi - y) ? sv[j - 1].second : sv[j].second));
      }
    } else {
      throw std::runtime_error("encode: type " + std::to_string(type));
    }
  }
  return o;
}

uint64_t type_bytes(uint64_t n, uint32_t t) {
  switch (t) {
    case GGML_F32: return 4 * n;
    case GGML_F16: case GGML_BF16: return 2 * n;
    case GGML_Q8_0: return n / 32 * 34;
    case GGML_Q4_0: return n / 32 * 18;
    case GGML_MXFP4: return n / 32 * 17;
    case GGML_MXFP8: return n / 32 * 33;
    case GGML_Q5_1: return n / 32 * 24;
  }
  throw std::runtime_error("type_bytes: type " + std::to_string(t));
}

struct Kv { std::string key; uint32_t t, et; GgufValue v; };
struct Tensor {
  std::string name;
  std::vector<uint64_t> dims;  // (ne0 first: the contiguous dimension)
  uint32_t type;
  uint64_t nbytes;
  std::vector<uint8_t> data;                                      // the encoded bytes, or
  std::function<void(const std::function<void(const std::vector<uint8_t>&)>&)> gen;  // pieces made at write time
};

struct Model {
  std::vector<Kv> kv;
  std::vector<Tensor> ts;
  void u32(const std::string& k, int64_t v) { kv.push_back({k, 4, 0, v}); }
  void i32(const std::string& k, int64_t v) { kv.push_back({k, 5, 0, v}); }
  void f32(const std::string& k, double v) { kv.push_back({k, 6, 0, v}); }
  void b(const std::string& k, bool v) { kv.push_back({k, 7, 0, v}); }
  void str(const std::string& k, const std::string& v) { kv.push_back({k, 8, 0, v}); }
  void arr(const std::string& k, uint32_t et, const GgufValue& v) { kv.push_back({k, 9, et, v}); }
  // data in row-major [.., ne1, ne0]; shape outermost first, as NumPy's
  void tensor(const std::string& name, const F& data, uint32_t type, std::vector<uint64_t> shape) {
    Tensor t;
    t.name = name;
    t.dims.assign(shape.rbegin(), shape.rend());
    t.type = type;
    t.data = encode(data, type);
    t.nbytes = t.data.size();
    ts.push_back(std::move(t));
  }
  void write(const std::string& path) {
    zpack::Writer w(path);
    w.put(std::string("GGUF").data(), 4);
    w.put<uint32_t>(3);
    w.put<uint64_t>(ts.size());
    w.put<uint64_t>(kv.size());
    for (auto& k : kv) zpack::put_kv(w, k.key, k.t, k.et, k.v);
    uint64_t off = 0;
    for (auto& t : ts) {
      w.str(t.name);
      w.put<uint32_t>((uint32_t)t.dims.size());
      for (uint64_t d : t.dims) w.put<uint64_t>(d);
      w.put<uint32_t>(t.type);
      w.put<uint64_t>(off);
      off = (off + t.nbytes + 31) / 32 * 32;
    }
    w.pad(32);
    for (auto& t : ts) {
      uint64_t got = 0;
      if (t.gen) {
        t.gen([&](const std::vector<uint8_t>& p) { w.put(p.data(), p.size()); got += p.size(); });
      } else {
        w.put(t.data.data(), t.data.size());
        got = t.data.size();
      }
      if (got != t.nbytes)
        throw std::runtime_error(t.name + ": " + std::to_string(got) + " bytes written, " + std::to_string(t.nbytes) +
                                 " expected");
      w.pad(32);
    }
    w.close();
  }
};

std::vector<int64_t> primes_from(int64_t start, int n, std::set<int64_t>& used) {
  auto isprime = [](int64_t x) {
    if (x < 2) return false;
    for (int64_t i = 2; i * i <= x; ++i)
      if (x % i == 0) return false;
    return true;
  };
  std::vector<int64_t> out;
  for (int64_t x = start; (int)out.size() < n; ++x)
    if (isprime(x) && !used.count(x)) { out.push_back(x); used.insert(x); }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
    printf("usage: make_tiny_deepseek41 OUT.gguf [--seed N] [--nu N] [--heads N] [--real] [--unsloth] [--q5_1-down]\n");
    return argc < 2 ? 1 : 0;
  }
  const std::string out = argv[1];
  uint64_t seed = 41;
  bool real = false, uns = false, q51 = false;
  int heads = -1, nu = -1;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { fprintf(stderr, "make_tiny_deepseek41: %s needs a value\n", a.c_str()); exit(2); }
      return argv[++i];
    };
    if (a == "--seed") seed = strtoull(next(), nullptr, 10);
    else if (a == "--real") real = true;
    else if (a == "--heads") heads = atoi(next());  // (q heads; 16 or more for the 16-head prompt attention, ds_attn_gh)
    else if (a == "--nu") nu = atoi(next());  // (experts per token; --nu 16 = every expert, no selection to flip on near-ties)
    else if (a == "--unsloth") uns = true;
    else if (a == "--q5_1-down") q51 = true;
    else { fprintf(stderr, "make_tiny_deepseek41: unknown option %s\n", a.c_str()); return 2; }
  }
  if (real) real_config();
  if (heads > 0) H = heads;
  if (nu > 0) NU = nu;
  try {
    Rng rng(seed);
    Model w;
    const std::string p = ARCH + ".";
    // a matmul weight [n_out][n_in] (ne0 = n_in), outputs of about unit size
    auto mat = [&](int n_out, int n_in, double scale = 1.0) {
      F v((size_t)n_out * n_in);
      const double s = scale / std::sqrt((double)n_in);
      for (auto& x : v) x = (float)(rng.normal() * s);
      return v;
    };
    auto normw = [&](int n) {
      F v(n);
      for (auto& x : v) x = (float)(1.0 + 0.1 * rng.normal());
      return v;
    };
    auto randn = [&](size_t n, double scale = 1.0) {
      F v(n);
      for (auto& x : v) x = (float)(scale * rng.normal());
      return v;
    };
    typedef std::vector<uint64_t> S;

    w.str("general.architecture", ARCH);
    w.str("general.name", "DeepSeek-V4.1-Flash tiny (random weights, inferzero test)");
    w.u32("general.alignment", 32);
    w.u32(p + "block_count", NL);
    w.u32(p + "context_length", 4096);
    w.u32(p + "embedding_length", E);
    w.u32(p + "attention.head_count", H);
    w.u32(p + "attention.head_count_kv", 1);
    w.u32(p + "attention.key_length", D);
    w.u32(p + "attention.value_length", D);
    w.u32(p + "rope.dimension_count", R);
    w.f32(p + "rope.freq_base", 10000.0);
    w.str(p + "rope.scaling.type", "yarn");
    w.f32(p + "rope.scaling.factor", 16.0);
    w.u32(p + "rope.scaling.original_context_length", 65536);
    w.f32(p + "rope.scaling.yarn_beta_fast", 32.0);
    w.f32(p + "rope.scaling.yarn_beta_slow", 1.0);
    w.f32(p + "attention.layer_norm_rms_epsilon", 1e-20);
    w.u32(p + "attention.q_lora_rank", QL);
    w.u32(p + "attention.sliding_window", SWA);
    w.u32(p + "expert_count", NE);
    w.u32(p + "expert_used_count", NU);
    w.u32(p + "expert_feed_forward_length", FF);
    w.u32(p + "expert_shared_count", NSH);
    w.f32(p + "expert_weights_scale", 1.5);
    w.b(p + "expert_weights_norm", true);
    w.arr(p + "swiglu_clamp_exp", 6, std::vector<double>(NL, 10.0));
    w.arr(p + "swiglu_clamp_shexp", 6, std::vector<double>(NL, 10.0));
    w.u32(p + "attention.output_group_count", G);
    w.u32(p + "attention.output_lora_rank", OL);
    std::vector<int64_t> ratios = RATIOS;
    ratios.insert(ratios.end(), {0, 0, 0});  // (as the real file: the 3 drafter layers' zeros)
    w.arr(p + "attention.compress_ratios", 4, ratios);
    w.f32(p + "attention.compress_rope_freq_base", 160000.0);
    w.u32(p + "attention.indexer.head_count", IH);
    w.u32(p + "attention.indexer.key_length", ID);
    w.u32(p + "attention.indexer.top_k", TOPK);
    w.arr(p + "attention.kv_source_layers", 4, KV_SRC);
    w.arr(p + "attention.indexer.source_layers", 4, IDX_SRC);
    w.i32(p + "attention.indexer.candidate_source_layer", CAND_SRC);
    w.u32(p + "attention.indexer.candidate_top_k_blocks", CAND_BLOCKS);
    w.u32(p + "attention.indexer.candidate_block_size", CAND_BS);
    w.u32(p + "hyper_connection.count", HC);
    w.u32(p + "hyper_connection.sinkhorn_iterations", 20);
    w.f32(p + "hyper_connection.epsilon", 1e-6);
    w.u32(p + "vocab_size", VOCAB);

    // Engram: hash constants as the converter makes them, scaled down (primes from 101 upward, no prime reused)
    const int ncol = (NGRAM - 1) * HPN;
    std::vector<int64_t> mults, primes, offsets;
    std::set<int64_t> used;
    int64_t base = 0;
    for (size_t li = 0; li < PLE_LAYERS.size(); ++li) {
      Rng r2(10007 * (uint64_t)PLE_LAYERS[li]);
      const uint64_t bound = std::max<uint64_t>(1, (((uint64_t)INT64_MAX) / 99092) / 2);
      for (int j = 0; j < NGRAM; ++j) mults.push_back((int64_t)(2 * r2.below(bound) + 1));
      const auto pr = primes_from(101 + 40 * (int64_t)li, ncol, used);
      for (int64_t q : pr) { offsets.push_back(base); base += q; }
      primes.insert(primes.end(), pr.begin(), pr.end());
    }
    const int64_t n_rows = base;
    std::vector<int64_t> token_map(VOCAB);
    for (int i = 0; i < VOCAB; ++i) token_map[i] = (int64_t)i * 7 % (VOCAB - 100);  // (several ids share one)
    w.arr(p + "ple.layers", 4, PLE_LAYERS);
    w.u32(p + "ple.ngram_size", NGRAM);
    w.u32(p + "ple.heads_per_ngram", HPN);
    w.u32(p + "embedding_length_per_layer_input", PD);
    w.arr(p + "ple.layer_multipliers", 10, mults);
    w.arr(p + "ple.head_offsets", 4, offsets);
    w.arr(p + "ple.head_vocab_sizes", 4, primes);
    w.u32(p + "engram.pad_token_id", 2);
    w.arr(p + "engram.token_map", 5, token_map);

    // a token list so that tokenizers load (ids are fed directly)
    std::vector<std::string> toks = {"<bos>", "<eos>", "<pad>"};
    std::vector<int64_t> ttype = {3, 3, 3};
    for (int i = 3; i < VOCAB; ++i) { toks.push_back("t" + std::to_string(i)); ttype.push_back(1); }
    w.str("tokenizer.ggml.model", "gpt2");
    w.arr("tokenizer.ggml.tokens", 8, toks);
    w.arr("tokenizer.ggml.token_type", 5, ttype);
    w.arr("tokenizer.ggml.merges", 8, std::vector<std::string>{});
    w.u32("tokenizer.ggml.bos_token_id", 0);
    w.u32("tokenizer.ggml.eos_token_id", 1);
    w.u32("tokenizer.ggml.padding_token_id", 2);

    uint32_t Q8 = GGML_Q8_0;
    const uint32_t F32 = GGML_F32, BF16 = GGML_BF16, F16 = GGML_F16;
    if (uns) Q8 = BF16;  // (every dense matrix BF16, as Unsloth's UD-Q8_K_XL)
    const uint32_t EXP = uns ? GGML_MXFP4 : GGML_Q4_0;  // (--real's experts)
    const uint64_t uE = E, uV = VOCAB;
    w.tensor("token_embd.weight", randn(uV * uE), Q8, S{uV, uE});
    w.tensor("output_norm.weight", normw(E), F32, S{uE});
    w.tensor("output.weight", mat(VOCAB, E, 1.5), Q8, S{uV, uE});
    w.tensor("per_layer_token_embd.weight", randn((uint64_t)n_rows * PD), uns ? GGML_MXFP8 : F16,
             S{(uint64_t)n_rows, (uint64_t)PD});
    const int hcd = HC * E, mix = (2 + HC) * HC;
    auto has = [](const std::vector<int64_t>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); };
    for (int il = 0; il < NL; ++il) {
      const std::string b = "blk." + std::to_string(il) + ".";
      w.tensor(b + "attn_norm.weight", normw(E), F32, S{uE});
      w.tensor(b + "attn_q_a.weight", mat(QL, E), Q8, S{(uint64_t)QL, uE});
      w.tensor(b + "attn_q_a_norm.weight", normw(QL), F32, S{(uint64_t)QL});
      w.tensor(b + "attn_q_b.weight", mat(H * D, QL), Q8, S{(uint64_t)H * D, (uint64_t)QL});
      w.tensor(b + "attn_kv.weight", mat(D, E), Q8, S{(uint64_t)D, uE});
      w.tensor(b + "attn_kv_a_norm.weight", normw(D), F32, S{(uint64_t)D});
      w.tensor(b + "attn_sinks.weight", randn(H), F32, S{(uint64_t)H});
      w.tensor(b + "attn_output_a.weight", mat(G * OL, H * D / G), Q8, S{(uint64_t)G * OL, (uint64_t)H * D / G});
      w.tensor(b + "attn_output_b.weight", mat(E, G * OL), Q8, S{uE, (uint64_t)G * OL});
      for (const char* site : {"attn", "ffn"}) {
        w.tensor(b + "hc_" + site + "_fn.weight", mat(mix, hcd, 2.0), F32, S{(uint64_t)mix, (uint64_t)hcd});
        w.tensor(b + "hc_" + site + "_base.weight", randn(mix, 0.5), F32, S{(uint64_t)mix});
        F sc = randn(3, 0.2);
        for (auto& x : sc) x += 1.0f;
        w.tensor(b + "hc_" + site + "_scale.weight", sc, F32, S{3});
      }
      w.tensor(b + "ffn_norm.weight", normw(E), F32, S{uE});
      w.tensor(b + "ffn_gate_inp.weight", mat(NE, E, 8.0), uns ? F32 : BF16, S{(uint64_t)NE, uE});  // (decisive)
      w.tensor(b + "exp_probs_b.bias", randn(NE, 0.1), F32, S{(uint64_t)NE});
      if (real) {  // (per expert at write time: 384 experts of 5120 x 256)
        auto gen_exps = [&](const std::string& name, int n_out, int n_in, double scale, uint64_t seed_e) {
          Tensor t;
          t.name = name;
          t.dims = {(uint64_t)n_in, (uint64_t)n_out, (uint64_t)NE};
          t.type = EXP;
          t.nbytes = type_bytes((uint64_t)NE * n_out * n_in, EXP);
          const int ne = NE;
          t.gen = [=](const std::function<void(const std::vector<uint8_t>&)>& emit) {
            Rng r2(seed_e);
            F v((size_t)n_out * n_in);
            const double s = scale / std::sqrt((double)n_in);
            for (int e = 0; e < ne; ++e) {
              for (auto& x : v) x = (float)(r2.normal() * s);
              emit(encode(v, EXP));
            }
          };
          w.ts.push_back(std::move(t));
        };
        const uint64_t sd = rng.below(1u << 30);
        gen_exps(b + "ffn_gate_exps.weight", FF, E, 3.0, sd);
        gen_exps(b + "ffn_up_exps.weight", FF, E, 3.0, sd + 1);
        gen_exps(b + "ffn_down_exps.weight", E, FF, 1.0, sd + 2);
      } else {
        const uint32_t XQ = uns ? GGML_MXFP4 : Q8;
        const uint64_t n3 = (uint64_t)NE * FF * E;
        w.tensor(b + "ffn_gate_exps.weight", randn(n3, 3 / std::sqrt((double)E)), XQ, S{(uint64_t)NE, (uint64_t)FF, uE});
        w.tensor(b + "ffn_up_exps.weight", randn(n3, 3 / std::sqrt((double)E)), XQ, S{(uint64_t)NE, (uint64_t)FF, uE});
        w.tensor(b + "ffn_down_exps.weight", randn(n3, 1 / std::sqrt((double)FF)), q51 ? GGML_Q5_1 : XQ,
                 S{(uint64_t)NE, uE, (uint64_t)FF});
      }
      w.tensor(b + "ffn_gate_shexp.weight", mat(FF * NSH, E, 3.0), Q8, S{(uint64_t)FF * NSH, uE});
      w.tensor(b + "ffn_up_shexp.weight", mat(FF * NSH, E, 3.0), Q8, S{(uint64_t)FF * NSH, uE});
      w.tensor(b + "ffn_down_shexp.weight", mat(E, FF * NSH), Q8, S{uE, (uint64_t)FF * NSH});
      if (has(KV_SRC, il)) {
        w.tensor(b + "attn_compressor_kv.weight", mat(D, E), BF16, S{(uint64_t)D, uE});
        if (RATIOS[il] > 1) w.tensor(b + "attn_compressor_gate.weight", mat(D, E, 2.0), BF16, S{(uint64_t)D, uE});
        w.tensor(b + "attn_compressor_norm.weight", normw(D), F32, S{(uint64_t)D});
        w.tensor(b + "indexer.attn_k.weight", mat(ID, D), BF16, S{(uint64_t)ID, (uint64_t)D});
        w.tensor(b + "indexer.k_norm.weight", normw(ID), F32, S{(uint64_t)ID});
      }
      if (has(IDX_SRC, il)) {
        w.tensor(b + "indexer.attn_q_b.weight", mat(IH * ID, QL), BF16, S{(uint64_t)IH * ID, (uint64_t)QL});
        w.tensor(b + "indexer.proj.weight", mat(IH, E, 4.0), BF16, S{(uint64_t)IH, uE});
      }
      if (has(PLE_LAYERS, il)) {
        w.tensor(b + "engram_kv.weight", mat(E * (HC + 1), ncol * PD), Q8, S{uE * (HC + 1), (uint64_t)ncol * PD});
        w.tensor(b + "engram_gate.weight", normw(HC * E), F32, S{(uint64_t)HC, uE});
      }
    }
    w.write(out);
    printf("%s: %zu tensors, %zu keys, Engram rows %lld\n", out.c_str(), w.ts.size(), w.kv.size(), (long long)n_rows);
  } catch (const std::exception& e) {
    fprintf(stderr, "make_tiny_deepseek41: %s\n", e.what());
    return 1;
  }
  return 0;
}
