// inferzero model packages: a MoE model kept without its original GGUF.
//
//   %USERPROFILE%\.inferzero\zero\models\<name>\<name>.gguf     every tensor except the routed experts, all metadata
//                                                              (the experts listed as virtual tensors: name, type,
//                                                              shape) and "inferzero.package" = 1
//   %USERPROFILE%\.inferzero\zero\models\<name>\<name>.experts the routed experts in the expert file layout
//                                                              (expert_store.h), built from the source GGUF
//
// make_package() reads the source GGUF (one file or split parts, from any drive: a USB disk is fine) once and writes
// both; afterwards the source can be moved away. The engine opens the package GGUF like any model; its expert file
// sits next to it.
#pragma once
#include "../src/gguf.h"
#include "expert_store.h"
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace zpack {

inline bool is_expert_tensor(const std::string& name) {
  // (blk.N.ffn_{gate,up,down}_exps.weight, Gemma 4 ffn_gate_up_exps.weight; not its ffn_down_exps.scale)
  return name.find("_exps.weight") != std::string::npos;
}
inline std::string packages_dir() {
  return app_dir() + PLAT_SEP "zero" PLAT_SEP "models";
}

// ---- a GGUF v3 writer for the metadata (with the stored types) and tensors
class Writer {
 public:
  explicit Writer(const std::string& path) {
    f_ = fopen(path.c_str(), "wb");
    if (!f_) throw std::runtime_error("cannot write " + path);
  }
  ~Writer() { if (f_) fclose(f_); }
  void put(const void* p, size_t n) {
    if (n && fwrite(p, 1, n, f_) != n) throw std::runtime_error("write failed");
    pos_ += n;
  }
  template <typename T> void put(T v) { put(&v, sizeof v); }
  void str(const std::string& s) {
    put<uint64_t>(s.size());
    put(s.data(), s.size());
  }
  void pad(uint64_t align) {
    static const char z[64] = {};
    while (pos_ % align) put(z, std::min<uint64_t>(64, align - pos_ % align));
  }
  uint64_t pos() const { return pos_; }
  // (fclose writes the last buffered bytes: a full disk can fail only here, and the file is then short)
  void close() {
    if (!f_) return;
    const int rc = fclose(f_);
    f_ = nullptr;
    if (rc != 0) throw std::runtime_error("write failed (closing the file)");
  }

 private:
  FILE* f_ = nullptr;
  uint64_t pos_ = 0;
};

// one scalar of GGUF type t
inline void put_scalar(Writer& w, uint32_t t, const GgufValue& v, size_t i = 0) {
  auto as_int = [&]() -> int64_t {
    if (auto p = std::get_if<int64_t>(&v)) return *p;
    if (auto p = std::get_if<uint64_t>(&v)) return (int64_t)*p;
    if (auto p = std::get_if<bool>(&v)) return *p;
    if (auto p = std::get_if<std::vector<int64_t>>(&v)) return (*p)[i];
    return 0;
  };
  auto as_f = [&]() -> double {
    if (auto p = std::get_if<double>(&v)) return *p;
    if (auto p = std::get_if<std::vector<double>>(&v)) return (*p)[i];
    return (double)as_int();
  };
  switch (t) {
    case 0: w.put<uint8_t>((uint8_t)as_int()); break;
    case 1: w.put<int8_t>((int8_t)as_int()); break;
    case 2: w.put<uint16_t>((uint16_t)as_int()); break;
    case 3: w.put<int16_t>((int16_t)as_int()); break;
    case 4: w.put<uint32_t>((uint32_t)as_int()); break;
    case 5: w.put<int32_t>((int32_t)as_int()); break;
    case 6: w.put<float>((float)as_f()); break;
    case 7: w.put<uint8_t>((uint8_t)(as_int() != 0)); break;
    case 10: w.put<uint64_t>((uint64_t)as_int()); break;
    case 11: w.put<int64_t>(as_int()); break;
    case 12: w.put<double>(as_f()); break;
    case 8: {
      if (auto p = std::get_if<std::string>(&v)) w.str(*p);
      else if (auto q = std::get_if<std::vector<std::string>>(&v)) w.str((*q)[i]);
      else w.str("");
      break;
    }
    default: throw std::runtime_error("gguf writer: unsupported type " + std::to_string(t));
  }
}
inline void put_kv(Writer& w, const std::string& key, uint32_t t, uint32_t et, const GgufValue& v) {
  w.str(key);
  w.put<uint32_t>(t);
  if (t != 9) return put_scalar(w, t, v);
  w.put<uint32_t>(et);
  size_t n = 0;
  if (auto pi = std::get_if<std::vector<int64_t>>(&v)) n = pi->size();
  else if (auto pd = std::get_if<std::vector<double>>(&v)) n = pd->size();
  else if (auto ps = std::get_if<std::vector<std::string>>(&v)) n = ps->size();
  w.put<uint64_t>(n);
  for (size_t i = 0; i < n; ++i) put_scalar(w, et, v, i);
}

struct Result {
  std::string gguf, experts;
  uint64_t rest_bytes = 0, expert_bytes = 0;
  double seconds = 0;
};

// S8: the package GGUF at `path` against its source: a package, its kept tensors in order with the same names, types,
// shapes and bytes, and the expert tensors listed as virtual. Throws on the first difference, naming it
inline void verify_package(const GgufFile& src, const std::string& path, const std::vector<const GgufTensor*>& keep,
                           const std::vector<std::string>& vnames) {
  const auto t0 = std::chrono::steady_clock::now();
  GgufFile pk(path);
  auto bad = [&](const std::string& what) {
    throw std::runtime_error("the package " + path + " came out wrong (" + what + "): not used; make it again, and if "
                             "this repeats, put the models on another drive");
  };
  if (!pk.has("inferzero.package")) bad("no inferzero.package key");
  if (pk.tensors.size() != keep.size() + vnames.size()) bad("its tensor count");
  uint64_t bytes = 0;
  // (both files' pieces copied out through copy_mapped first, S4: a page that cannot be read is read again, not the
  //  end of the process; and the stall watch over the check, S1. The Researcher's review)
  const size_t piece = 64ull << 20;
  struct Buf { uint8_t* p; size_t n; ~Buf() { if (p) plat::page_free(p, n); } } ba{(uint8_t*)plat::page_alloc(piece), piece},
      bb{(uint8_t*)plat::page_alloc(piece), piece};
  if (!ba.p || !bb.p) throw std::runtime_error("out of memory checking the package");
  BuildWatch watch("the package check", src.path, path);  // (path: the package's .tmp, removed if it hangs for good)
  for (const GgufTensor* t : keep) {
    const GgufTensor* p = pk.find(t->name);
    if (!p || !p->data) bad("tensor " + t->name + " missing");
    if (p->type != t->type || p->dims != t->dims || p->nbytes != t->nbytes) bad("tensor " + t->name + "'s type or shape");
    XBuildStopGuard::check("the package check " + path);
    for (uint64_t o = 0; o < t->nbytes; o += piece) {
      const size_t n = (size_t)std::min<uint64_t>(piece, t->nbytes - o);
      copy_mapped(src, ba.p, t->data + o, n, "the package check", "");
      copy_mapped(pk, bb.p, p->data + o, n, "the package check", "");
      if (memcmp(ba.p, bb.p, n) != 0) bad("tensor " + t->name + "'s bytes from " + std::to_string(o));
      watch.tick();
      XBuildStopGuard::check("the package check " + path);
    }
    plat::release_mapped(t->data, (size_t)t->nbytes);
    bytes += t->nbytes;
  }
  for (const std::string& v : vnames)
    if (!pk.find(v)) bad("expert tensor " + v + " not listed");
  zlog::eprintf("inferzero: the package checked against its source: %zu tensors, %.2f GB the same, %.1f s\n", keep.size(),
                bytes / 1e9, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

// Writes the package of `source` (the first part of a split model, or the only file) under packages_dir().
inline Result make_package(const std::string& source, int n_layer_hint = 0) {
  namespace fs = std::filesystem;
  XBuildStopGuard stop_guard;  // (a stop: the .tmp files removed, then the signal goes on; D1)
  const auto t0 = std::chrono::steady_clock::now();
  GgufFile src(source);
  // (a package as the source: its expert tensors are virtual (no data), so the new GGUF would lack them, nothing would
  //  be built, and the rename below would replace the source package's own GGUF with it, on Linux even while it is
  //  mapped: the model lost if the original was removed. zero-server.ps1 refused it, serverz --pack and the Linux
  //  launcher did not. Self review 2026-11-19, B1)
  if (src.has("inferzero.package"))
    throw std::runtime_error(source + " is already a package (it has no expert data of its own to pack): pack the "
                             "original GGUF instead");
  std::string name = fs::path(source).stem().string();
  const size_t sp = name.find("-00001-of-");
  if (sp != std::string::npos) name.resize(sp);
  const std::string dir = packages_dir() + PLAT_SEP + name;
  // (a package folder of the same name made from another file, e.g. the same model name in two folders: its GGUF
  //  and expert file were rebuilt and renamed over it without a word. Refused, naming both; a re-pack of the same
  //  source goes on and reuses its expert file. A package from before package-source.txt (2026-11-15) says only the
  //  file name, which matches by construction: it goes on, with a line. Self review 2026-11-19, B2)
  {
    std::error_code e;
    const fs::path rec = fs::path(dir) / "package-source.txt";
    if (fs::exists(fs::path(dir) / (name + ".gguf"), e) && fs::exists(rec, e)) {
      std::string prev;
      if (FILE* f = plat::fopen_path(rec, "rb")) {
        std::vector<char> b(4096);
        const size_t n = fread(b.data(), 1, b.size(), f);
        fclose(f);
#ifdef _WIN32
        if (n >= 2 && (unsigned char)b[0] == 0xFF && (unsigned char)b[1] == 0xFE) {  // (UTF-16 with its BOM)
          std::wstring w((const wchar_t*)(b.data() + 2), (n - 2) / sizeof(wchar_t));
          while (!w.empty() && (w.back() == L'\r' || w.back() == L'\n')) w.pop_back();
          prev = fs::path(w).string();
        } else
#endif
        {
          prev.assign(b.data(), n);
          while (!prev.empty() && (prev.back() == '\r' || prev.back() == '\n')) prev.pop_back();
        }
      }
      if (!prev.empty()) {
        const fs::path now = fs::absolute(fs::path(source), e);
        bool same = fs::exists(prev, e) && fs::equivalent(prev, now, e);
        if (!same) {  // (the earlier source gone or moved: compared as paths, case-blind on Windows)
          std::string a = fs::path(prev).lexically_normal().string(), c = now.lexically_normal().string();
#ifdef _WIN32
          for (auto& ch : a) ch = (char)tolower((unsigned char)ch);
          for (auto& ch : c) ch = (char)tolower((unsigned char)ch);
#endif
          same = a == c;
        }
        // (self audit 2026-11-20 A7: the source copied or moved since, D: -> C: for one: another path, the same model.
        //  Taken as the same when the package's tensor table (every tensor's name, type and shape, by name: the package
        //  lists its expert tensors, virtual, after the others) matches this file's; a model of the same name made
        //  otherwise differs in its types or shapes)
        if (!same) {
          try {
            GgufFile pk((fs::path(dir) / (name + ".gguf")).string());
            same = pk.tensors.size() == src.tensors.size();
            for (size_t i = 0; same && i < src.tensors.size(); ++i) {
              const GgufTensor& s = src.tensors[i];
              const GgufTensor* p = pk.find(s.name);
              same = p && p->type == s.type && p->dims == s.dims;
            }
          } catch (const std::exception&) {
            same = false;  // (an unreadable package: refused as before)
          }
          if (same)
            zlog::eprintf("inferzero: %s was made from %s; this file (%s) has the same tensors: taken as the same model, "
                          "moved or copied\n", dir.c_str(), prev.c_str(), now.string().c_str());
        }
        if (!same)
          throw std::runtime_error("a package of the same name, made from another file, is already in " + dir +
                                   " (from " + prev + "; this one is " + now.string() +
                                   "): remove that package first (zero-server, or delete the folder), or rename this file");
      } else {
        zlog::eprintf("inferzero: %s holds a package whose source is not recorded: taken as this file's\n", dir.c_str());
      }
    }
  }
  fs::create_directories(dir);
  Result r;
  r.gguf = dir + PLAT_SEP + name + ".gguf";
  r.experts = dir + PLAT_SEP + name + ".experts";

  // Order: the package GGUF goes to a .tmp, then the expert file is moved or written, then the .tmp takes the
  // package's name. A step that fails leaves no half package: the .tmp is removed, and a package GGUF only appears
  // once its expert file is complete (the engine refuses a package without one).
  const std::string arch = src.get_str("general.architecture");
  // (the engine's layer count: the MTP / nextn blocks are not counted, the extra entry below holds them; with
  //  block_count the 35B's expert file listed one layer more than the engine expects and was refused)
  const int n_layer = n_layer_hint > 0 ? n_layer_hint
                                       : (int)src.get_int(arch + ".block_count", 0) - (int)src.get_int(arch + ".nextn_predict_layers", 0);
  const int n_exp = (int)src.get_int(arch + ".expert_count", 0);
  if (n_exp <= 0) throw std::runtime_error("not a MoE model (no " + arch + ".expert_count)");
  if (n_layer <= 0 || n_layer > 4096) throw std::runtime_error("unexpected layer count " + std::to_string(n_layer) + " (" + arch + ".block_count)");
  // (the same layer list as the engine's: blocks 0..n_layer, the last one the MTP / nextn block when there is one)
  std::vector<std::string> pre(n_layer + 1);
  for (int il = 0; il <= n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    if (src.find(p + "ffn_gate_exps.weight") || src.find(p + "ffn_gate_up_exps.weight")) pre[il] = p;
  }
  // ---- 1. the package GGUF: all metadata + every non-expert tensor; the experts as virtual tensors
  std::vector<const GgufTensor*> keep;
  std::vector<std::string> vnames;
  std::vector<int64_t> vtypes, vdims;
  for (const GgufTensor& t : src.tensors) {
    if (!t.data) continue;  // (already virtual)
    if (is_expert_tensor(t.name)) {
      vnames.push_back(t.name);
      vtypes.push_back(t.type);
      for (int d = 0; d < 4; ++d) vdims.push_back(d < (int)t.dims.size() ? (int64_t)t.dims[d] : 0);
    } else {
      keep.push_back(&t);
    }
  }
  const uint64_t align = (uint64_t)src.get_int("general.alignment", 32);
  const std::string tmp = r.gguf + ".tmp";
  auto drop_tmp = [&] { std::error_code e; fs::remove(tmp, e); };
  plat::KeepAwake awake;  // (S11)
  const plat::BuildLock lock(tmp + ".lock");  // (A9: until the package is complete or its .tmp removed)
  drop_stale_tmp(tmp, lock);                  // (S7)
  {  // (S3: the package GGUF's room first; the expert file's is checked when it is written)
    uint64_t rest = 1ull << 20;
    for (const GgufTensor* t : keep) rest += (t->nbytes + align - 1) / align * align;
    need_free_space(r.gguf, rest, "the package");
  }
  try {
    Writer w(tmp);
    std::vector<std::string> keys;
    for (const std::string& k : src.kv_order)
      if (k.rfind("split.", 0) != 0 && k.rfind("inferzero.", 0) != 0) keys.push_back(k);  // (the package keys are written below)
    w.put<uint32_t>(0x46554747);
    w.put<uint32_t>(3);
    w.put<uint64_t>(keep.size());
    w.put<uint64_t>(keys.size() + 6);
    for (const std::string& k : keys) {
      const auto& ty = src.kv_types.at(k);
      put_kv(w, k, ty.first, ty.second, src.kv.at(k));
    }
    put_kv(w, "inferzero.package", 4, 0, GgufValue((int64_t)1));
    put_kv(w, "inferzero.experts_file", 8, 0, GgufValue(fs::path(r.experts).filename().string()));
    put_kv(w, "inferzero.source", 8, 0, GgufValue(fs::path(source).filename().string()));
    put_kv(w, "inferzero.virtual.names", 9, 8, GgufValue(vnames));
    put_kv(w, "inferzero.virtual.types", 9, 4, GgufValue(vtypes));
    put_kv(w, "inferzero.virtual.dims", 9, 11, GgufValue(vdims));
    uint64_t off = 0;
    for (const GgufTensor* t : keep) {
      w.str(t->name);
      w.put<uint32_t>((uint32_t)t->dims.size());
      for (uint64_t d : t->dims) w.put<uint64_t>(d);
      w.put<uint32_t>(t->type);
      w.put<uint64_t>(off);
      off += (t->nbytes + align - 1) / align * align;
    }
    w.pad(align);
    BuildWatch watch("the package", source, tmp);  // (S1: a drive that stops answering)
    // (ARC_XBUILD_STALL_TEST=1: a test of S1, the build waits past the watch's limit once, as a drive that hangs)
    if (getenv("ARC_XBUILD_STALL_TEST") && atoi(getenv("ARC_XBUILD_STALL_TEST")) == 1) {
      const int lim = getenv("ARC_XBUILD_STALL_S") ? atoi(getenv("ARC_XBUILD_STALL_S")) : 60;
      std::this_thread::sleep_for(std::chrono::seconds(lim + 2));
    }
    // S6 (2026-11-21): each piece copied out of the mapping into a buffer while the next one is prefetched in large
    // requests (the pages read ahead, not faulted in one by one from fwrite), and the copy guarded (S4: a page that
    // cannot be read is read again with read calls). The same bytes in the same order. ARC_XBUILD_PREFETCH=0: as before
    static const bool s6 = !getenv("ARC_XBUILD_PREFETCH") || atoi(getenv("ARC_XBUILD_PREFETCH")) != 0;
    const size_t piece = 64ull << 20;
    struct Buf { uint8_t* p; size_t n; ~Buf() { if (p) plat::page_free(p, n); } } pbuf{s6 ? (uint8_t*)plat::page_alloc(piece) : nullptr, piece};
    for (size_t ti = 0; ti < keep.size(); ++ti) {
      const GgufTensor* t = keep[ti];
      XBuildStopGuard::check("the package " + tmp);
      // (in pieces of 64 MB, each a tick: a 2 GB tensor from a slow USB disk takes longer than the watch's limit)
      for (uint64_t o = 0; o < t->nbytes; o += piece) {
        const size_t n = (size_t)std::min<uint64_t>(piece, t->nbytes - o);
        if (pbuf.p) {
          // (the next piece: of this tensor, else the next tensor's first)
          if (o + piece < t->nbytes) plat::prefetch_mapped(t->data + o + piece, (size_t)std::min<uint64_t>(piece, t->nbytes - o - piece));
          else if (ti + 1 < keep.size()) plat::prefetch_mapped(keep[ti + 1]->data, (size_t)std::min<uint64_t>(piece, keep[ti + 1]->nbytes));
          copy_mapped(src, pbuf.p, t->data + o, n, "the package", "");
          w.put(pbuf.p, n);
        } else {
          w.put(t->data + o, n);
        }
        watch.tick();
        XBuildStopGuard::check("the package " + tmp);  // (S1: a stall seen meanwhile stops here)
      }
      plat::release_mapped(t->data, (size_t)t->nbytes);  // (written: its pages leave the process on Linux; B6)
      w.pad(align);
    }
    r.rest_bytes = w.pos();
    w.close();
  } catch (...) {
    drop_tmp();
    throw;
  }
  // ---- 2. the experts, through the expert store, next to the package (the engine's expert file for the source
  //  moves here when it exists, see ExpertStore::open_or_build)
  try {
    ExpertStore st;
    st.open_or_build(src, source, pre, n_exp, r.experts);
    if (!st.built) zlog::eprintf("inferzero: the expert file was already there: %s\n", r.experts.c_str());
    r.expert_bytes = st.file_bytes;
  } catch (...) {
    drop_tmp();
    throw;
  }
  // S8 (2026-11-21): the package GGUF read back before it takes its name: its keys, its tensor table and every kept
  // tensor's bytes against the source's. A package that differs is never put in place (a write that went wrong
  // silently: a bad drive, a filesystem that dropped bytes). ARC_PACK_CHECK=0 skips it (it reads the kept tensors of
  // both files once more)
  try {
    static const bool check = !getenv("ARC_PACK_CHECK") || atoi(getenv("ARC_PACK_CHECK")) != 0;
    if (check) verify_package(src, tmp, keep, vnames);
  } catch (...) {
    drop_tmp();
    throw;
  }
  std::error_code ec;
  fs::rename(tmp, r.gguf, ec);
  if (ec) {
    drop_tmp();
    throw std::runtime_error("cannot rename " + tmp + ": " + ec.message());
  }
  // (package-source.txt: the full path of the file it was made from; zero-server remove-original deletes that file
  //  only. Two folders' models of the same file name make the same package folder, so the name alone does not say
  //  which. A file of its own: the folder's source.txt is the expert store's (ExpertStore::place), rewritten by each
  //  GGUF outside the data folder whose expert file goes in this folder, a same-named one from elsewhere too)
  {
    std::error_code e2;
    const fs::path full = fs::absolute(fs::path(source), e2);
    const fs::path rec = fs::path(dir) / "package-source.txt";
    bool ok = false;
    if (FILE* f = plat::fopen_path(rec, "wb")) {
#ifdef _WIN32
      // (UTF-16 with its BOM: any path's characters, and PowerShell 5.1's Get-Content reads it as such)
      const std::wstring s = L"\xFEFF" + (e2 ? fs::path(source) : full).wstring() + L"\r\n";
      ok = fwrite(s.data(), sizeof(wchar_t), s.size(), f) == s.size();
#else
      const std::string s = (e2 ? fs::path(source) : full).string() + "\n";  // (UTF-8, as Linux paths are)
      ok = fwrite(s.data(), 1, s.size(), f) == s.size();
#endif
      ok = fclose(f) == 0 && ok;
    }
    if (!ok) {  // (the package works without it; only remove-original refuses then)
      fs::remove(rec, e2);
      zlog::eprintf("inferzero: could not write %s: zero-server remove-original will not delete the original\n",
              rec.string().c_str());
    }
  }
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

}  // namespace zpack
