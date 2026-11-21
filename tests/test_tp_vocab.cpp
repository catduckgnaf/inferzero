#include "../zero/mgpu/tp_vocab.h"
#include <cstdio>

int main() {
  const std::vector<std::vector<float>> shards{{1.0f, 3.0f, 2.0f}, {3.0f, 0.0f, 4.0f}};
  const auto top = tp_vocab_topk(shards, {0, 3}, 4);
  if (top.size() != 4 || top[0].token != 5 || top[1].token != 1 || top[2].token != 3 || top[3].token != 2) {
    std::puts("FAIL deterministic shard top-k");
    return 1;
  }
  const double p = tp_vocab_best_probability(shards, top[0]);
  if (!(p > 0.50 && p < 0.53)) {
    std::printf("FAIL full-vocabulary softmax probability: %.9f\n", p);
    return 1;
  }
  std::puts("ok deterministic shard top-k and full-vocabulary probability");
  return 0;
}
