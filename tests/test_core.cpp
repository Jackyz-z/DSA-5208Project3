// Unit tests for the building blocks shared by all trainers.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "dt/binning.hpp"
#include "dt/split.hpp"
#include "dt/tree.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++failures;
}

bool close(double a, double b) { return std::fabs(a - b) < 1e-12; }

// Hand-computed example with 4 bins and 2 classes; parent counts (8, 8).
//   bin:      0      1      2      3
//   counts: (4,0)  (3,1)  (0,4)  (1,3)
// Parent Gini = 0.5. Candidate thresholds:
//   t=0: L=(4,0)  R=(4,8)  gain = 0.5 - (4/16*0     + 12/16*0.4444) = 0.1666...
//   t=1: L=(7,1)  R=(1,7)  gain = 0.5 - (8/16*0.21875 + 8/16*0.21875) = 0.28125
//   t=2: L=(7,5)  R=(1,3)  gain = 0.5 - (12/16*0.4861 + 4/16*0.375)  = 0.041666...
const dt::hist_t kHist[] = {4, 0, 3, 1, 0, 4, 1, 3};
const std::uint64_t kParent[dt::kMaxClasses] = {8, 8};

void test_hand_computed_split() {
  const dt::SplitInfo s = dt::best_split_for_feature(kHist, 4, 2, kParent, 1, 7);
  check(s.valid() && s.feature == 7, "split is found on the requested feature");
  check(s.bin == 1, "best threshold is bin 1");
  check(close(s.gain, 0.28125), "gain equals the hand-computed Gini decrease 0.28125");
  check(s.left[0] == 7 && s.left[1] == 1, "left child counts are (7, 1)");
}

void test_min_samples_leaf() {
  // t=0 leaves 4 rows on the left and t=2 leaves 4 on the right.
  const dt::SplitInfo s = dt::best_split_for_feature(kHist, 4, 2, kParent, 5, 0);
  check(s.valid() && s.bin == 1, "min_samples_leaf=5 keeps only threshold 1");
  const dt::SplitInfo none = dt::best_split_for_feature(kHist, 4, 2, kParent, 9, 0);
  check(!none.valid(), "min_samples_leaf=9 admits no split of 16 rows");
}

void test_tie_breaking() {
  dt::SplitInfo a, b;
  a.gain = b.gain = 0.25;
  a.feature = 3;
  b.feature = 5;
  a.bin = 10;
  b.bin = 2;
  check(dt::better(a, b) && !dt::better(b, a), "equal gain prefers the smaller feature");
  b.feature = 3;
  check(dt::better(b, a), "equal gain and feature prefers the smaller bin");
  dt::SplitInfo invalid;
  check(dt::better(a, invalid) && !dt::better(invalid, a), "any valid split beats no split");
}

void test_binning_rule() {
  std::mt19937 rng(42);
  std::normal_distribution<float> normal;
  std::vector<float> sample(10000);
  for (float& v : sample) v = normal(rng);
  const std::vector<float> edges = dt::compute_edges(sample, 32);
  const dt::BinMapper mapper({edges});
  check(mapper.n_bins(0) <= 32, "quantile binning respects max_bins");

  bool rule_holds = true;
  for (int k = 0; k < 2000; ++k) {
    const float x = normal(rng);
    const int b = mapper.bin(0, x);
    for (int t = 0; t + 1 < mapper.n_bins(0); ++t) {
      if ((b <= t) != (x <= mapper.threshold(0, t))) rule_holds = false;
    }
  }
  check(rule_holds, "bin(x) <= t  <=>  x <= threshold(t)");

  const std::vector<float> few = dt::compute_edges({3.f, 1.f, 2.f, 1.f, 3.f}, 256);
  check(few.size() == 2 && few[0] == 1.f && few[1] == 2.f,
        "few distinct values give exact edges");
}

void test_can_split() {
  dt::GrowthParams g;
  g.max_depth = 3;
  const std::uint64_t mixed[dt::kMaxClasses] = {5, 5};
  const std::uint64_t pure[dt::kMaxClasses] = {10, 0};
  check(dt::can_split(0, 10, mixed, 2, g), "mixed node can split");
  check(!dt::can_split(0, 10, pure, 2, g), "pure node cannot split");
  check(!dt::can_split(3, 10, mixed, 2, g), "node at max_depth cannot split");
}

void test_tree_predict() {
  dt::Tree tree(2);
  const std::uint64_t root[dt::kMaxClasses] = {8, 8};
  const std::uint64_t l[dt::kMaxClasses] = {7, 1};
  const std::uint64_t r[dt::kMaxClasses] = {1, 7};
  tree.add_node(-1, 0, root);
  const int li = tree.add_node(0, 1, l);
  const int ri = tree.add_node(0, 1, r);
  tree.set_split(0, 0, 1, 0.5f, li, ri);
  check(tree.predict([](int) { return 0.2f; }) == 0, "value below threshold goes left");
  check(tree.predict([](int) { return 0.9f; }) == 1, "value above threshold goes right");
  check(tree.training_correct() == 14, "training accuracy from leaf counts is 14/16");
}

}  // namespace

int main() {
  test_hand_computed_split();
  test_min_samples_leaf();
  test_tie_breaking();
  test_binning_rule();
  test_can_split();
  test_tree_predict();
  std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL TESTS PASSED", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
