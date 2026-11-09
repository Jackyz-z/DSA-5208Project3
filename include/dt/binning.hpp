#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dt {

// Maps raw feature values to bin indices. For feature f with sorted edges
// e[0] < e[1] < ... < e[k-1], a value x falls in bin b = #{ j : e[j] < x }.
// Hence  bin(x) <= t  <=>  x <= e[t],  so a split learnt on bins can be
// applied to raw values with the threshold e[t].
class BinMapper {
 public:
  BinMapper() = default;
  explicit BinMapper(std::vector<std::vector<float>> edges);

  int n_features() const { return static_cast<int>(edges_.size()); }
  int n_bins(int f) const { return static_cast<int>(edges_[f].size()) + 1; }
  int max_bins() const { return max_bins_; }
  const std::vector<float>& edges(int f) const { return edges_[f]; }
  float threshold(int f, int bin) const { return edges_[f][bin]; }

  // Branch-free binary search over the edges padded to 255 entries with +inf:
  // always 8 steps, compiled to conditional moves. On random data this is
  // several times faster than std::lower_bound, whose branches mispredict.
  std::uint8_t bin(int f, float x) const {
    const float* e = padded_.data() + static_cast<std::size_t>(f) * kPadded;
    int pos = 0;
    for (int step = 128; step > 0; step >>= 1) pos += e[pos + step - 1] < x ? step : 0;
    return static_cast<std::uint8_t>(pos);
  }

 private:
  static constexpr int kPadded = 256;
  std::vector<std::vector<float>> edges_;
  std::vector<float> padded_;
  int max_bins_ = 1;
};

// Quantile bin edges from a sample of one feature (at most max_bins bins).
std::vector<float> compute_edges(std::vector<float> sample, int max_bins);

// Rows with global index divisible by the stride form the binning sample, so
// the sample (and therefore the bins) do not depend on the number of ranks.
inline std::uint64_t sample_stride(std::uint64_t n_rows, std::uint64_t target) {
  if (target == 0 || n_rows <= target) return 1;
  return (n_rows + target - 1) / target;
}

}  // namespace dt
