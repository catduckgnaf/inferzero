// Expert-cache simulation from routes recorded with ARC_DUMP_ROUTES (zero/engine_moe.h layer_moe).
//
// Each record: int32 layer, int32 tokens, then tokens x NU int32 expert ids. Records with tokens > 1 are prompt
// chunks, tokens == 1 single-token generation steps. The cache holds (layer, expert) entries; its capacity is a share
// of all of them. Hit rate and SSD bytes are counted over the generation steps (the prompt warms the cache).
//
// usage: cache_sim routes.bin [routes2.bin ...] [--layers 40 --experts 256 --nu 8 --mb 1.36 --ssd 3.5
//                             --tgpu 0.0505 --caps 0.5,0.7,0.8,0.88,0.95]
//
// The C++ port of the former zero/cache_sim.py (2026-11-21, no Python in the repository): the same policies and the
// same table. One difference: "lfu" picks among equally used victims the smallest (layer, expert); the .py took the
// first in its hash set's order, so on ties its lfu rows could differ a little. "pin" and "belady" break ties as the
// .py did (the most used first, then the first seen; the farthest next use, then the oldest entry).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

typedef int64_t Key;  // layer * 2^32 + expert
static Key mk(int l, int e) { return ((Key)l << 32) | (uint32_t)e; }
static int layer_of(Key k) { return (int)(k >> 32); }

struct Rec { int layer; std::vector<std::vector<int>> toks; };

static bool load(const char* path, int nu, std::vector<Rec>& prompt, std::vector<Rec>& gen) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  std::vector<char> d;
  char buf[1 << 16];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) d.insert(d.end(), buf, buf + n);
  fclose(f);
  size_t off = 0;
  while (off + 8 <= d.size()) {
    int32_t lt[2];
    memcpy(lt, &d[off], 8);
    off += 8;
    Rec r;
    r.layer = lt[0];
    const int t = lt[1];
    for (int i = 0; i < t; ++i) {
      std::vector<int> ids(nu, 0);
      if (off + (size_t)nu * 4 <= d.size()) memcpy(ids.data(), &d[off], (size_t)nu * 4);
      off += (size_t)nu * 4;
      r.toks.push_back(ids);
    }
    (t > 1 ? prompt : gen).push_back(r);
  }
  return true;
}

// per token, its (layer, expert) accesses (generation records come layer by layer for each token)
static std::vector<std::vector<Key>> token_stream(const std::vector<Rec>& recs, int layers) {
  std::vector<std::vector<Key>> out;
  std::vector<Key> cur;
  int last = -1;
  for (auto& r : recs) {
    if (r.layer >= layers) continue;  // (the MTP block)
    if (r.layer <= last && !cur.empty()) { out.push_back(cur); cur.clear(); }
    last = r.layer;
    if (!r.toks.empty())
      for (int e : r.toks[0]) cur.push_back(mk(r.layer, e));
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

static std::vector<Key> prompt_accesses(const std::vector<Rec>& recs, int layers) {
  std::vector<Key> acc;
  for (auto& r : recs) {
    if (r.layer >= layers) continue;
    for (auto& t : r.toks)
      for (int e : t) acc.push_back(mk(r.layer, e));
  }
  return acc;
}

// an LRU in insertion order: front the oldest
struct Lru {
  std::list<Key> order;
  std::unordered_map<Key, std::list<Key>::iterator> at;
  size_t size() const { return order.size(); }
  bool has(Key k) const { return at.count(k) != 0; }
  void touch(Key k) { order.splice(order.end(), order, at[k]); }
  void add(Key k) { order.push_back(k); at[k] = std::prev(order.end()); }
  void pop_oldest() { at.erase(order.front()); order.pop_front(); }
};

// counts with their first-seen order: the most used first, ties in the order first seen (Counter.most_common)
struct Counts {
  std::unordered_map<Key, int64_t> n;
  std::unordered_map<Key, size_t> first;
  std::vector<Key> seen;
  void add(Key k) {
    if (!first.count(k)) { first[k] = seen.size(); seen.push_back(k); }
    ++n[k];
  }
  int64_t get(Key k) const { auto it = n.find(k); return it == n.end() ? 0 : it->second; }
  std::vector<Key> most_common(size_t m) const {
    std::vector<Key> v = seen;
    std::stable_sort(v.begin(), v.end(), [&](Key a, Key b) { return n.at(a) > n.at(b); });
    if (v.size() > m) v.resize(m);
    return v;
  }
};

static void simulate(const std::string& pol, long cap, const std::vector<Key>& warm,
                     const std::vector<std::vector<Key>>& gen, int64_t& hits, int64_t& miss, double pin_frac = 0.9) {
  hits = miss = 0;
  if (pol == "lru") {
    Lru c;
    auto touch = [&](Key k, int count) {
      if (c.has(k)) { c.touch(k); hits += count; return; }
      miss += count;
      c.add(k);
      if ((long)c.size() > cap) c.pop_oldest();
    };
    for (Key k : warm) touch(k, 0);
    for (auto& t : gen) for (Key k : t) touch(k, 1);
  } else if (pol == "lfu") {
    Counts cnt;
    for (Key k : warm) cnt.add(k);
    std::set<Key> c;
    for (Key k : cnt.most_common(cap)) c.insert(k);
    for (auto& t : gen)
      for (Key k : t) {
        cnt.add(k);
        if (c.count(k)) { ++hits; continue; }
        ++miss;
        if ((long)c.size() >= cap) {
          if (c.empty()) continue;
          Key victim = *c.begin();
          int64_t vc = cnt.get(victim);
          for (Key x : c) { const int64_t xc = cnt.get(x); if (xc < vc) { victim = x; vc = xc; } }
          if (vc < cnt.get(k)) { c.erase(victim); c.insert(k); }
        } else {
          c.insert(k);
        }
      }
  } else if (pol == "pin" || pol == "pin+lru") {
    Counts cnt;
    for (Key k : warm) cnt.add(k);
    const long npin = pol == "pin" ? cap : (long)(cap * pin_frac);
    std::unordered_set<Key> pinned;
    for (Key k : cnt.most_common(npin)) pinned.insert(k);
    Lru dyn;
    const long dcap = cap - npin;
    for (auto& t : gen)
      for (Key k : t) {
        if (pinned.count(k)) { ++hits; continue; }
        if (dyn.has(k)) { dyn.touch(k); ++hits; continue; }
        ++miss;
        if (dcap > 0) {
          dyn.add(k);
          if ((long)dyn.size() > dcap) dyn.pop_oldest();
        }
      }
  } else if (pol == "layerlru") {
    // an LRU per layer with an equal share of the slots (a global LRU evicts the next layer's experts during
    // single-token decoding: it walks the layers in a cycle)
    std::map<int, Lru> caches;
    std::set<int> ls;
    for (Key k : warm) ls.insert(layer_of(k));
    for (auto& t : gen) for (Key k : t) ls.insert(layer_of(k));
    const long nl = std::max<long>(1, (long)ls.size());
    const long lcap = std::max<long>(1, cap / nl);
    auto touch = [&](Key k, int count) {
      Lru& c = caches[layer_of(k)];
      if (c.has(k)) { c.touch(k); hits += count; return; }
      miss += count;
      c.add(k);
      if ((long)c.size() > lcap) c.pop_oldest();
    };
    for (Key k : warm) touch(k, 0);
    for (auto& t : gen) for (Key k : t) touch(k, 1);
  } else if (pol == "belady") {
    std::vector<Key> seq;
    for (auto& t : gen) for (Key k : t) seq.push_back(k);
    std::vector<int64_t> nxt(seq.size());
    std::unordered_map<Key, int64_t> last;
    for (size_t i = seq.size(); i-- > 0;) {
      auto it = last.find(seq[i]);
      nxt[i] = it == last.end() ? ((int64_t)1 << 60) : it->second;
      last[seq[i]] = (int64_t)i;
    }
    // warmed with the most used prompt experts; their next use unknown until seen. Entries in insertion order: the
    // victim is the farthest next use, the oldest entry among equals (max over a dict)
    Counts cnt;
    for (Key k : warm) cnt.add(k);
    std::list<std::pair<Key, int64_t>> c;
    std::unordered_map<Key, std::list<std::pair<Key, int64_t>>::iterator> at;
    for (Key k : cnt.most_common(cap)) { c.push_back({k, (int64_t)1 << 59}); at[k] = std::prev(c.end()); }
    for (size_t i = 0; i < seq.size(); ++i) {
      const Key k = seq[i];
      auto it = at.find(k);
      if (it != at.end()) { ++hits; it->second->second = nxt[i]; continue; }
      ++miss;
      if ((long)c.size() >= cap && !c.empty()) {
        auto v = c.begin();
        for (auto j = c.begin(); j != c.end(); ++j)
          if (j->second > v->second) v = j;
        at.erase(v->first);
        c.erase(v);
      }
      c.push_back({k, nxt[i]});
      at[k] = std::prev(c.end());
    }
  }
}

int main(int argc, char** argv) {
  std::vector<std::string> files;
  int layers = 40, experts = 256, nu = 8;
  double mb = 1.36, ssd = 3.5, tgpu = 1 / 19.8;
  std::string caps = "0.5,0.7,0.8,0.88,0.95";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { fprintf(stderr, "cache_sim: %s needs a value\n", a.c_str()); exit(2); }
      return argv[++i];
    };
    if (a == "--layers") layers = atoi(next());
    else if (a == "--experts") experts = atoi(next());
    else if (a == "--nu") nu = atoi(next());
    else if (a == "--mb") mb = atof(next());
    else if (a == "--ssd") ssd = atof(next());
    else if (a == "--tgpu") tgpu = atof(next());
    else if (a == "--caps") caps = next();
    else if (a == "-h" || a == "--help") {
      printf("usage: cache_sim routes.bin [routes2.bin ...] [--layers 40] [--experts 256] [--nu 8 (experts a token)]\n"
             "       [--mb 1.36 (MB an expert a layer)] [--ssd 3.5 (GB/s)] [--tgpu 0.0505 (s a token, all cached)]\n"
             "       [--caps 0.5,0.7,0.8,0.88,0.95]\n");
      return 0;
    } else if (!a.empty() && a[0] == '-') { fprintf(stderr, "cache_sim: unknown option %s\n", a.c_str()); return 2; }
    else files.push_back(a);
  }
  if (files.empty()) { fprintf(stderr, "usage: cache_sim routes.bin [...] (--help)\n"); return 2; }
  std::vector<double> capf;
  for (size_t p = 0; p <= caps.size();) {
    size_t q = caps.find(',', p);
    if (q == std::string::npos) q = caps.size();
    if (q > p) capf.push_back(atof(caps.substr(p, q - p).c_str()));
    p = q + 1;
  }
  const long total = (long)layers * experts;
  for (auto& path : files) {
    std::vector<Rec> prompt, gen;
    if (!load(path.c_str(), nu, prompt, gen)) { fprintf(stderr, "cache_sim: cannot read %s\n", path.c_str()); return 1; }
    const std::vector<Key> warm = prompt_accesses(prompt, layers);
    const auto toks = token_stream(gen, layers);
    // locality: the share of a layer's experts the previous token used in that layer too
    double reuse = 0;
    size_t nre = 0;
    for (size_t i = 1; i < toks.size(); ++i) {
      std::unordered_set<Key> prev(toks[i - 1].begin(), toks[i - 1].end());
      size_t same = 0;
      for (Key k : toks[i]) same += prev.count(k);
      reuse += (double)same / std::max<size_t>(1, toks[i].size());
      ++nre;
    }
    std::unordered_set<Key> uq;
    for (auto& t : toks) uq.insert(t.begin(), t.end());
    const size_t sl = path.find_last_of('\\');
    const std::string name = sl == std::string::npos ? path : path.substr(sl + 1);
    printf("\n== %s: prompt %ld tokens, generation %zu tokens; distinct experts in generation %zu of %ld (%.0f%%); "
           "reused from the previous token %.0f%%\n",
           name.c_str(), warm.empty() ? 0L : (long)(warm.size() / nu / layers), toks.size(), uq.size(), total,
           100.0 * uq.size() / total, 100.0 * reuse / std::max<size_t>(1, nre));
    printf("%6s %8s %6s %10s %12s %13s\n", "cache", "policy", "hit %", "SSD MB/tok", "tok/s serial", "tok/s overlap");
    for (double cf : capf) {
      const long cap = (long)(total * cf);
      for (const char* pol : {"lru", "layerlru", "lfu", "pin", "pin+lru", "belady"}) {
        int64_t h, m;
        simulate(pol, cap, warm, toks, h, m);
        const double n = (double)std::max<size_t>(1, toks.size());
        const double mbt = m / n * mb;
        const double io = mbt / 1000 / ssd;
        printf("%5.0f%% %8s %6.1f %10.1f %12.1f %13.1f\n", cf * 100, pol, 100.0 * h / std::max<int64_t>(1, h + m), mbt,
               1 / (tgpu + io), 1 / std::max(tgpu, io));
      }
    }
  }
  return 0;
}
