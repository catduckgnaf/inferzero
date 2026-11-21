// Replays an expert-route trace (ARC_XTRACE=file, zero/xcache.h) through cache policies, with the cache's own pools
// and slots, and counts the records each would read from the SSD: generation misses and prompt reads apart.
// usage: xtrace_sim trace
//   lru     the engine's policy (generation hits and misses to the front, a prompt's misses to the back)
//   slru:p  segmented LRU: experts used again in generation move to a protected part (p of the slots); victims come
//           from the other part first. A prompt's hits do not promote (a chunk touches nearly every expert)
//   lfu:w   victims among the w least recent slots: the one with the fewest uses (counts halved every 64 tokens)
//   opt     Belady (the expert used farthest in the future goes): the most any policy could save, not buildable
// The C++ port of the former bench/xtrace_sim.py (2026-11-21, no Python in the repository): the same policies, the
// same tie-breaks and the same report.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <list>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static const int32_t kMagic = 0x58545231;

struct Pool { int n; std::vector<int> keys; };
struct Hdr { int NL = 0, NE = 0; std::vector<Pool> pools; std::vector<int> pool_of; };
struct Call { bool run_start; int l; bool prompt; std::vector<int> ids; };

static bool read_trace(const char* path, Hdr& hdr, std::vector<Call>& calls) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  std::vector<int32_t> v;
  int32_t buf[1 << 14];
  size_t n;
  while ((n = fread(buf, 4, sizeof buf / 4, f)) > 0) v.insert(v.end(), buf, buf + n);
  fclose(f);
  bool have = false;
  size_t i = 0;
  auto at = [&](size_t j) { if (j >= v.size()) throw std::runtime_error("trace cut short"); return v[j]; };
  while (i < v.size()) {
    if (v[i] == kMagic) {
      const int NL = at(i + 1), NE = at(i + 2), G = at(i + 3);
      i += 4;
      std::vector<Pool> pools;
      for (int g = 0; g < G; ++g) {
        Pool p;
        p.n = at(i);
        const int k = at(i + 1);
        for (int j = 0; j < k; ++j) p.keys.push_back(at(i + 2 + j));
        i += 2 + k;
        pools.push_back(p);
      }
      std::vector<int> po;
      for (int j = 0; j < NL; ++j) po.push_back(at(i + j));
      i += NL;
      if (!have) { hdr.NL = NL; hdr.NE = NE; hdr.pools = pools; hdr.pool_of = po; have = true; }
      calls.push_back({true, 0, false, {}});  // (a new process: the run boundary)
      continue;
    }
    if (!have) throw std::runtime_error("no header before the first record");
    const int l = at(i), prompt = at(i + 1), cnt = at(i + 2);
    Call c{false, l, prompt != 0, {}};
    std::unordered_set<int> seen;
    for (int j = 0; j < cnt; ++j) {
      const int e = at(i + 3 + j);
      if (0 <= e && e < hdr.NE && seen.insert(e).second) c.ids.push_back(e);
    }
    i += 3 + cnt;
    calls.push_back(std::move(c));
  }
  return have;
}

typedef std::unordered_set<int> KeySet;

// an ordered set: front the least recent
struct Od {
  std::list<int> order;
  std::unordered_map<int, std::list<int>::iterator> at;
  bool has(int k) const { return at.count(k) != 0; }
  size_t size() const { return order.size(); }
  void to_end(int k) { order.splice(order.end(), order, at[k]); }
  void to_front(int k) { order.splice(order.begin(), order, at[k]); }
  void add(int k) { order.push_back(k); at[k] = std::prev(order.end()); }
  void del(int k) { auto it = at.find(k); order.erase(it->second); at.erase(it); }
  int pop_front() { const int k = order.front(); del(k); return k; }
};

struct Policy {
  virtual ~Policy() {}
  virtual int access(const std::vector<int>& keys, bool prompt, const KeySet& cur) = 0;
};

struct Lru : Policy {
  int n;
  Od od;
  Lru(int n_, const std::vector<int>& keys) : n(n_) {
    // (keys: most recent first)
    const int m = std::min<int>(n, (int)keys.size());
    for (int j = m - 1; j >= 0; --j)
      if (keys[j] >= 0 && !od.has(keys[j])) od.add(keys[j]);  // (a key twice keeps its first place, as before)
  }
  virtual int victim(const KeySet& cur) {
    for (int k : od.order)
      if (!cur.count(k)) return k;
    throw std::runtime_error("no victim");
  }
  int access(const std::vector<int>& keys, bool prompt, const KeySet& cur) override {
    int miss = 0;
    for (int k : keys)
      if (od.has(k)) od.to_end(k);
    for (int k : keys) {
      if (od.has(k)) continue;
      ++miss;
      if ((int)od.size() >= n) od.del(victim(cur));
      od.add(k);
      if (prompt) od.to_front(k);
    }
    return miss;
  }
};

struct Slru : Lru {
  Od prot;
  int cap;
  Slru(int n_, const std::vector<int>& keys, double p) : Lru(n_, keys), cap((int)(n_ * p)) {}
  int access(const std::vector<int>& keys, bool prompt, const KeySet& cur) override {
    int miss = 0;
    for (int k : keys) {
      if (prot.has(k)) {
        prot.to_end(k);
      } else if (od.has(k)) {
        if (prompt) {
          od.to_end(k);
        } else {
          od.del(k);
          prot.add(k);
          while ((int)prot.size() > cap) od.add(prot.pop_front());
        }
      }
    }
    for (int k : keys) {
      if (od.has(k) || prot.has(k)) continue;
      ++miss;
      if ((int)(od.size() + prot.size()) >= n) {
        bool done = false;
        for (Od* d : {&od, &prot}) {
          for (int x : d->order)
            if (!cur.count(x)) { d->del(x); done = true; break; }
          if (done) break;
        }
        if (!done) throw std::runtime_error("no victim");
      }
      od.add(k);
      if (prompt) od.to_front(k);
    }
    return miss;
  }
};

typedef std::unordered_map<int, int64_t> Cnt;

struct Lfu : Lru {
  int w;
  Cnt* cnt;
  Lfu(int n_, const std::vector<int>& keys, double w_, Cnt* c) : Lru(n_, keys), w(std::max(1, (int)(n_ * w_))), cnt(c) {}
  int victim(const KeySet& cur) override {
    bool have = false;
    int best = 0, j = 0;
    int64_t bc = 0;
    for (int k : od.order) {
      if (cur.count(k)) continue;
      auto it = cnt->find(k);
      const int64_t c = it == cnt->end() ? 0 : it->second;
      if (!have || c < bc) { best = k; bc = c; have = true; }
      ++j;
      if (j >= w || bc == 0) break;
    }
    if (!have) throw std::runtime_error("no victim");
    return best;
  }
};

typedef std::function<Policy*(int, const std::vector<int>&)> Make;

// a layer's pool (-1, a layer without one, counted in the last pool as the .py's list index did)
static size_t pool_index(const Hdr& h, int l) {
  const int g = h.pool_of[l];
  return g >= 0 ? (size_t)g : h.pools.size() - (size_t)(-g);
}

static void run(const Hdr& h, const std::vector<Call>& calls, const Make& make, Cnt* counts, int64_t& gen_miss,
                int64_t& prompt_read) {
  std::vector<std::unique_ptr<Policy>> P;
  for (auto& p : h.pools) P.emplace_back(make(p.n, p.keys));
  gen_miss = prompt_read = 0;
  int64_t gen_calls = 0;
  for (auto& c : calls) {
    if (c.run_start) continue;
    std::vector<int> keys;
    for (int e : c.ids) keys.push_back(c.l * h.NE + e);
    const KeySet cur(keys.begin(), keys.end());
    if (counts && !c.prompt) {
      ++gen_calls;
      for (int k : keys) ++(*counts)[k];
      if (gen_calls % (64 * (int64_t)h.NL) == 0)
        for (auto& kv : *counts) kv.second >>= 1;
    }
    const int m = P[pool_index(h, c.l)]->access(keys, c.prompt, cur);
    (c.prompt ? prompt_read : gen_miss) += m;
  }
}

static void opt(const Hdr& h, const std::vector<Call>& calls, int64_t& gen_miss, int64_t& prompt_read) {
  const int64_t INF = (int64_t)1 << 60;
  std::vector<int> seq;
  for (auto& c : calls)
    if (!c.run_start)
      for (int e : c.ids) seq.push_back(c.l * h.NE + e);
  std::vector<int64_t> nxt(seq.size(), INF);
  std::unordered_map<int, int64_t> last, first;
  for (size_t i = seq.size(); i-- > 0;) {
    auto it = last.find(seq[i]);
    nxt[i] = it == last.end() ? INF : it->second;
    last[seq[i]] = (int64_t)i;
  }
  for (size_t i = 0; i < seq.size(); ++i) first.emplace(seq[i], (int64_t)i);
  // a heap per pool of (-next use, key), the smallest first: the farthest next use, then the smallest key
  typedef std::pair<int64_t, int> E;
  typedef std::priority_queue<E, std::vector<E>, std::greater<E>> Heap;
  std::vector<KeySet> cache;
  std::vector<Heap> heaps(h.pools.size());
  std::unordered_map<int, int64_t> nu;  // key -> its next use (current)
  for (size_t g = 0; g < h.pools.size(); ++g) {
    KeySet s;
    const auto& p = h.pools[g];
    for (int j = 0; j < std::min<int>(p.n, (int)p.keys.size()); ++j)
      if (p.keys[j] >= 0) s.insert(p.keys[j]);
    for (int k : s) {
      auto it = first.find(k);
      nu[k] = it == first.end() ? INF : it->second;
      heaps[g].push({-nu[k], k});
    }
    cache.push_back(std::move(s));
  }
  gen_miss = prompt_read = 0;
  size_t i = 0;
  for (auto& c : calls) {
    if (c.run_start) continue;
    const size_t g = pool_index(h, c.l);
    KeySet& S = cache[g];
    Heap& H = heaps[g];
    const int n = h.pools[g].n;
    KeySet cur;
    for (int e : c.ids) cur.insert(c.l * h.NE + e);
    for (int e : c.ids) {
      const int k = c.l * h.NE + e;
      if (!S.count(k)) {
        (c.prompt ? prompt_read : gen_miss) += 1;
        if ((int)S.size() >= n) {
          std::vector<E> keep;
          while (true) {
            if (H.empty()) throw std::runtime_error("no victim");
            const E x = H.top();
            H.pop();
            auto it = nu.find(x.second);
            if (!S.count(x.second) || it == nu.end() || -x.first != it->second) continue;  // (stale)
            if (cur.count(x.second)) { keep.push_back(x); continue; }
            S.erase(x.second);
            break;
          }
          for (auto& x : keep) H.push(x);
        }
        S.insert(k);
      }
      nu[k] = nxt[i];
      H.push({-nu[k], k});
      ++i;
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
    printf("usage: xtrace_sim trace (recorded with ARC_XTRACE=trace)\n");
    return argc < 2 ? 2 : 0;
  }
  const char* path = argv[1];
  Hdr h;
  std::vector<Call> calls;
  try {
    if (!read_trace(path, h, calls)) { fprintf(stderr, "xtrace_sim: cannot read %s (or no header)\n", path); return 1; }
  } catch (const std::exception& e) {
    fprintf(stderr, "xtrace_sim: %s: %s\n", path, e.what());
    return 1;
  }
  int64_t ng = 0, gen_routes = 0, runs = 0;
  for (auto& c : calls) {
    if (c.run_start) { ++runs; continue; }
    if (!c.prompt) { ++ng; gen_routes += (int64_t)c.ids.size(); }
  }
  int64_t used = 0;
  for (int x : h.pool_of) used += x >= 0;
  std::string slots;
  for (size_t g = 0; g < h.pools.size(); ++g) slots += (g ? "/" : "") + std::to_string(h.pools[g].n);
  printf("%s: %d layers x %d experts, %zu pools (%s slots), %lld runs, %lld generation calls (%lld tokens), %lld routes\n",
         path, h.NL, h.NE, h.pools.size(), slots.c_str(), (long long)runs, (long long)ng,
         (long long)(ng / std::max<int64_t>(1, used)), (long long)gen_routes);
  struct Row { std::string name; Make make; std::shared_ptr<Cnt> cnt; };
  std::vector<Row> rows;
  rows.push_back({"lru (engine)", [](int n, const std::vector<int>& k) -> Policy* { return new Lru(n, k); }, nullptr});
  char nm[32];
  for (double p : {0.3, 0.5, 0.7, 0.8}) {
    snprintf(nm, sizeof nm, "slru:%.1f", p);
    rows.push_back({nm, [p](int n, const std::vector<int>& k) -> Policy* { return new Slru(n, k, p); }, nullptr});
  }
  for (double w : {0.05, 0.25, 0.5, 1.0}) {
    auto cnt = std::make_shared<Cnt>();
    snprintf(nm, sizeof nm, "lfu:%.2f", w);
    Cnt* cp = cnt.get();
    rows.push_back({nm, [w, cp](int n, const std::vector<int>& k) -> Policy* { return new Lfu(n, k, w, cp); }, cnt});
  }
  int64_t base = -1;
  auto line = [&](const std::string& name, int64_t gm, int64_t pr) {
    printf("  %-14s generation misses %7lld (%5.1f %% of routes, %+6.1f %% vs lru) | prompt reads %7lld\n", name.c_str(),
           (long long)gm, 100.0 * gm / std::max<int64_t>(1, gen_routes),
           100.0 * (gm - base) / std::max<int64_t>(1, base), (long long)pr);
  };
  try {
    for (auto& r : rows) {
      int64_t gm, pr;
      run(h, calls, r.make, r.cnt.get(), gm, pr);
      if (base < 0) base = gm;
      line(r.name, gm, pr);
    }
    int64_t gm, pr;
    opt(h, calls, gm, pr);
    line("opt (bound)", gm, pr);
  } catch (const std::exception& e) {
    fprintf(stderr, "xtrace_sim: %s\n", e.what());
    return 1;
  }
  return 0;
}
