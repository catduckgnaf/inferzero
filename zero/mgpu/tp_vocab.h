// Deterministic vocabulary-shard candidate selection shared by tensor-parallel MTP.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

struct TpVocabCandidate {
  float logit;
  int token;
};

// Merge each shard's local top-k in rank order. The global order is descending logit, then ascending token ID,
// independent of how vocabulary rows are partitioned. The local top-k property guarantees the global top-k.
inline std::vector<TpVocabCandidate> tp_vocab_topk(const std::vector<std::vector<float>>& shards,
                                                   const std::vector<int>& offsets, int k) {
  if (shards.empty() || shards.size() != offsets.size() || k < 1)
    throw std::invalid_argument("tensor parallel vocabulary: invalid shards or k");
  std::vector<TpVocabCandidate> all;
  for (size_t rank = 0; rank < shards.size(); ++rank) {
    if (offsets[rank] < 0) throw std::invalid_argument("tensor parallel vocabulary: negative offset");
    for (size_t i = 0; i < shards[rank].size(); ++i)
      all.push_back({shards[rank][i], offsets[rank] + (int)i});
  }
  const auto better = [](const TpVocabCandidate& a, const TpVocabCandidate& b) {
    return a.logit != b.logit ? a.logit > b.logit : a.token < b.token;
  };
  const int n = std::min<int>(k, (int)all.size());
  std::partial_sort(all.begin(), all.begin() + n, all.end(), better);
  all.resize(n);
  return all;
}

// Probability of the best candidate under the full vocabulary softmax, not the truncated candidate set.
inline double tp_vocab_best_probability(const std::vector<std::vector<float>>& shards,
                                        const TpVocabCandidate& best) {
  double denom = 0;
  for (const auto& logits : shards)
    for (float x : logits) denom += std::exp((double)x - best.logit);
  return denom > 0 ? 1.0 / denom : 0.0;
}
