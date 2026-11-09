#pragma once

#include <cstdint>
#include <type_traits>

#include "dt/common.hpp"

namespace dt {

struct GrowthParams {
  int max_depth = 10;
  std::uint64_t min_samples_leaf = 1;
  std::uint64_t min_samples_split = 2;
  double min_gain = 1e-12;
};

// Best split found for one node (or one feature of one node).
struct SplitInfo {
  double gain = 0.0;          // Gini impurity decrease of the node
  std::int32_t feature = -1;  // global feature index, -1 if no valid split
  std::int32_t bin = -1;      // rows with bin <= this go to the left child
  std::uint64_t left[kMaxClasses] = {};  // class counts of the left child

  bool valid() const { return feature >= 0; }
};
static_assert(std::is_trivially_copyable<SplitInfo>::value, "SplitInfo is sent as raw bytes");

// Strict, total preference order on splits: larger gain first, then smaller
// feature index, then smaller bin. Every implementation uses this order, so
// all of them pick the same split even when gains tie exactly.
inline bool better(const SplitInfo& a, const SplitInfo& b) {
  if (!a.valid()) return false;
  if (!b.valid()) return true;
  if (a.gain != b.gain) return a.gain > b.gain;
  if (a.feature != b.feature) return a.feature < b.feature;
  return a.bin < b.bin;
}

// Scans the histogram of one feature of one node (n_bins x n_classes counts,
// row-major) and returns the best threshold. `parent` holds the class counts
// of the node; the histogram must sum to it.
SplitInfo best_split_for_feature(const hist_t* hist, int n_bins, int n_classes,
                                 const std::uint64_t* parent, std::uint64_t min_samples_leaf,
                                 int feature);

// Whether a node with these statistics is a candidate for splitting.
bool can_split(int depth, std::uint64_t n, const std::uint64_t* counts, int n_classes,
               const GrowthParams& params);

}  // namespace dt
