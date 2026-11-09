#include "dt/split.hpp"

namespace dt {

// Gini impurity of a node with counts c and size n is 1 - sum_k c_k^2 / n^2.
// The impurity decrease of splitting it into L and R is
//
//   gain = ( sum_k L_k^2 / n_L + sum_k R_k^2 / n_R - sum_k c_k^2 / n ) / n,
//
// which only needs the class counts on each side of the threshold. With
// integer counts below 2^26 every square is exact in double precision.
SplitInfo best_split_for_feature(const hist_t* hist, int n_bins, int n_classes,
                                 const std::uint64_t* parent, std::uint64_t min_samples_leaf,
                                 int feature) {
  SplitInfo best;

  std::uint64_t n = 0;
  double parent_sq = 0.0;
  for (int c = 0; c < n_classes; ++c) {
    n += parent[c];
    parent_sq += static_cast<double>(parent[c]) * static_cast<double>(parent[c]);
  }
  if (n == 0) return best;
  const double parent_term = parent_sq / static_cast<double>(n);

  std::uint64_t left[kMaxClasses] = {};
  std::uint64_t n_left = 0;
  for (int b = 0; b + 1 < n_bins; ++b) {
    for (int c = 0; c < n_classes; ++c) {
      left[c] += hist[b * n_classes + c];
      n_left += hist[b * n_classes + c];
    }
    if (n_left < min_samples_leaf) continue;
    const std::uint64_t n_right = n - n_left;
    if (n_right < min_samples_leaf) break;

    double left_sq = 0.0, right_sq = 0.0;
    for (int c = 0; c < n_classes; ++c) {
      const double l = static_cast<double>(left[c]);
      const double r = static_cast<double>(parent[c] - left[c]);
      left_sq += l * l;
      right_sq += r * r;
    }
    const double gain = (left_sq / static_cast<double>(n_left) +
                         right_sq / static_cast<double>(n_right) - parent_term) /
                        static_cast<double>(n);

    if (!best.valid() || gain > best.gain) {
      best.gain = gain;
      best.feature = feature;
      best.bin = b;
      for (int c = 0; c < n_classes; ++c) best.left[c] = left[c];
    }
  }
  return best;
}

bool can_split(int depth, std::uint64_t n, const std::uint64_t* counts, int n_classes,
               const GrowthParams& params) {
  if (depth >= params.max_depth) return false;
  if (n < params.min_samples_split || n < 2 * params.min_samples_leaf) return false;
  int non_empty = 0;
  for (int c = 0; c < n_classes; ++c) non_empty += counts[c] > 0;
  return non_empty > 1;
}

}  // namespace dt
