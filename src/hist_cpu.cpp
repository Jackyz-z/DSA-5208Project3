#include "dt/hist_cpu.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace dt {

OmpStrategy parse_omp_strategy(const std::string& name) {
  if (name == "feature") return OmpStrategy::Feature;
  if (name == "sample") return OmpStrategy::Sample;
  throw std::runtime_error("unknown OpenMP strategy: " + name);
}

const char* to_string(OmpStrategy s) { return s == OmpStrategy::Feature ? "feature" : "sample"; }

CpuBackend::CpuBackend(const std::uint8_t* bins, const std::uint8_t* labels, std::uint64_t n_rows,
                       int n_features, int bin_stride, int n_classes, OmpStrategy strategy)
    : bins_(bins),
      labels_(labels),
      n_rows_(n_rows),
      n_features_(n_features),
      bin_stride_(bin_stride),
      n_classes_(n_classes),
      strategy_(strategy),
      node_of_(n_rows, 0),
      rows_(n_rows),
      slots_(n_rows) {
  if (n_rows > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("too many rows on one rank; use more ranks");
  }
}

// Gathers (row, slot) pairs of the rows that contribute to this level, using a
// parallel count / prefix-sum / fill so the list keeps ascending row order.
void CpuBackend::collect(const std::vector<std::int32_t>& slot_of_node) {
  const std::int64_t n = static_cast<std::int64_t>(n_rows_);
  std::vector<std::size_t> offsets(max_threads() + 1, 0);

#pragma omp parallel
  {
#ifdef _OPENMP
    const int t = omp_get_thread_num();
    const int nt = omp_get_num_threads();
#else
    const int t = 0;
    const int nt = 1;
#endif
    const std::int64_t begin = n * t / nt;
    const std::int64_t end = n * (t + 1) / nt;

    std::size_t count = 0;
    for (std::int64_t i = begin; i < end; ++i) {
      const std::int32_t node = node_of_[i];
      count += node >= 0 && slot_of_node[node] >= 0;
    }
    offsets[t + 1] = count;

#pragma omp barrier
#pragma omp single
    for (int k = 0; k < nt; ++k) offsets[k + 1] += offsets[k];

    std::size_t pos = offsets[t];
    for (std::int64_t i = begin; i < end; ++i) {
      const std::int32_t node = node_of_[i];
      if (node < 0) continue;
      const std::int32_t slot = slot_of_node[node];
      if (slot < 0) continue;
      rows_[pos] = static_cast<std::uint32_t>(i);
      slots_[pos] = slot;
      ++pos;
    }
#pragma omp single nowait
    n_listed_ = offsets[nt];
  }
}

void CpuBackend::build(const std::vector<std::int32_t>& slot_of_node, int n_slots, hist_t* out) {
  const std::size_t total = static_cast<std::size_t>(n_slots) * n_features_ * bin_stride_ * n_classes_;
  if (total == 0) return;
  collect(slot_of_node);
  if (strategy_ == OmpStrategy::Feature) {
    build_by_feature(out, total);
  } else {
    build_by_sample(out, total);
  }
}

void CpuBackend::build_by_feature(hist_t* out, std::size_t total) {
  const std::size_t unit = static_cast<std::size_t>(bin_stride_) * n_classes_;
  const std::size_t slot_size = unit * n_features_;
  const std::int64_t m = static_cast<std::int64_t>(n_listed_);

#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (std::int64_t e = 0; e < static_cast<std::int64_t>(total); ++e) out[e] = 0;

#pragma omp for schedule(dynamic, 1)
    for (int f = 0; f < n_features_; ++f) {
      const std::uint8_t* col = bins_ + static_cast<std::size_t>(f) * n_rows_;
      hist_t* base = out + f * unit;
      for (std::int64_t k = 0; k < m; ++k) {
        const std::uint32_t i = rows_[k];
        base[slots_[k] * slot_size + col[i] * n_classes_ + labels_[i]]++;
      }
    }
  }
}

void CpuBackend::build_by_sample(hist_t* out, std::size_t total) {
  const std::size_t unit = static_cast<std::size_t>(bin_stride_) * n_classes_;
  const std::size_t slot_size = unit * n_features_;
  const std::int64_t m = static_cast<std::int64_t>(n_listed_);
  const std::size_t needed = total * max_threads();
  if (private_.size() < needed) private_.resize(needed);

#pragma omp parallel
  {
#ifdef _OPENMP
    const int t = omp_get_thread_num();
    const int nt = omp_get_num_threads();
#else
    const int t = 0;
    const int nt = 1;
#endif
    hist_t* mine = private_.data() + t * total;
    std::fill(mine, mine + total, 0);

#pragma omp for schedule(static)
    for (std::int64_t k = 0; k < m; ++k) {
      const std::uint32_t i = rows_[k];
      hist_t* base = mine + slots_[k] * slot_size + labels_[i];
      for (int f = 0; f < n_features_; ++f) {
        base[f * unit + bins_[static_cast<std::size_t>(f) * n_rows_ + i] * n_classes_]++;
      }
    }

    // Implicit barrier above; now sum the thread-private copies.
#pragma omp for schedule(static)
    for (std::int64_t e = 0; e < static_cast<std::int64_t>(total); ++e) {
      hist_t sum = 0;
      for (int k = 0; k < nt; ++k) sum += private_[k * total + e];
      out[e] = sum;
    }
  }
}

void CpuBackend::route(const std::vector<Route>& routes) {
  const std::int64_t n = static_cast<std::int64_t>(n_rows_);
#pragma omp parallel for schedule(static)
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int32_t node = node_of_[i];
    if (node < 0) continue;
    const Route& r = routes[node];
    if (r.feature < 0) {
      node_of_[i] = -1;
      continue;
    }
    const std::uint8_t b = bins_[static_cast<std::size_t>(r.feature) * n_rows_ + i];
    node_of_[i] = b <= r.bin ? r.left : r.right;
  }
}

void CpuBackend::mark_left(const std::vector<Route>& owned, std::uint64_t* bits) const {
  const std::int64_t words = static_cast<std::int64_t>((n_rows_ + 63) / 64);
  // One word per iteration, so no two threads ever write the same word.
#pragma omp parallel for schedule(static)
  for (std::int64_t w = 0; w < words; ++w) {
    std::uint64_t word = 0;
    const std::uint64_t begin = static_cast<std::uint64_t>(w) * 64;
    const std::uint64_t end = std::min(begin + 64, n_rows_);
    for (std::uint64_t i = begin; i < end; ++i) {
      const std::int32_t node = node_of_[i];
      if (node < 0) continue;
      const Route& r = owned[node];
      if (r.feature < 0) continue;
      if (bins_[static_cast<std::size_t>(r.feature) * n_rows_ + i] <= r.bin) {
        word |= std::uint64_t{1} << (i - begin);
      }
    }
    bits[w] = word;
  }
}

void CpuBackend::route_bitmap(const std::vector<Route>& routes, const std::uint64_t* bits) {
  const std::int64_t n = static_cast<std::int64_t>(n_rows_);
#pragma omp parallel for schedule(static)
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int32_t node = node_of_[i];
    if (node < 0) continue;
    const Route& r = routes[node];
    if (r.feature < 0) {
      node_of_[i] = -1;
      continue;
    }
    const bool go_left = (bits[i >> 6] >> (i & 63)) & 1u;
    node_of_[i] = go_left ? r.left : r.right;
  }
}

}  // namespace dt
