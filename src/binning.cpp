#include "dt/binning.hpp"

#include <iterator>
#include <limits>
#include <stdexcept>

#include "dt/common.hpp"

namespace dt {

BinMapper::BinMapper(std::vector<std::vector<float>> edges) : edges_(std::move(edges)) {
  static_assert(kPadded == kMaxBins, "the search assumes at most 255 edges");
  padded_.assign(static_cast<std::size_t>(n_features()) * kPadded,
                 std::numeric_limits<float>::infinity());
  for (int f = 0; f < n_features(); ++f) {
    if (n_bins(f) > kMaxBins) throw std::runtime_error("too many bins for a uint8 index");
    max_bins_ = std::max(max_bins_, n_bins(f));
    std::copy(edges_[f].begin(), edges_[f].end(), padded_.begin() + f * kPadded);
  }
}

std::vector<float> compute_edges(std::vector<float> sample, int max_bins) {
  if (max_bins < 2 || max_bins > kMaxBins) throw std::runtime_error("max_bins must be in [2, 256]");

  std::vector<float> edges;
  if (sample.empty()) return edges;
  std::sort(sample.begin(), sample.end());

  std::vector<float> distinct;
  std::unique_copy(sample.begin(), sample.end(), std::back_inserter(distinct));
  if (static_cast<int>(distinct.size()) <= max_bins) {
    // Few distinct values: one bin per value, i.e. exact splits.
    edges.assign(distinct.begin(), distinct.end() - 1);
    return edges;
  }

  const std::uint64_t m = sample.size();
  for (int j = 1; j < max_bins; ++j) {
    const float v = sample[static_cast<std::size_t>(j * m / max_bins)];
    if (edges.empty() || v > edges.back()) edges.push_back(v);
  }
  return edges;
}

}  // namespace dt
