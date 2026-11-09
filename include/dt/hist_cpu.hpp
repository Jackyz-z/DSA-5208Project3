#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dt/hist_backend.hpp"

namespace dt {

// How OpenMP threads share the work of building histograms.
//   Feature: each thread owns whole features; no write conflicts, but at most
//            n_features threads can be busy.
//   Sample:  each thread owns a block of rows and a private histogram copy;
//            perfectly balanced, but needs memory and a reduction per thread.
enum class OmpStrategy { Feature, Sample };

OmpStrategy parse_omp_strategy(const std::string& name);
const char* to_string(OmpStrategy s);

class CpuBackend : public HistBackend {
 public:
  // bins is column-major (n_features x n_rows); both arrays must outlive this.
  CpuBackend(const std::uint8_t* bins, const std::uint8_t* labels, std::uint64_t n_rows,
             int n_features, int bin_stride, int n_classes, OmpStrategy strategy);

  void build(const std::vector<std::int32_t>& slot_of_node, int n_slots, hist_t* out) override;
  void route(const std::vector<Route>& routes) override;

  // Feature-parallel support. mark_left sets bit i of `bits` when row i sits
  // in a node whose split feature is local (owned[node].feature >= 0) and the
  // row goes left. route_bitmap then routes every row using the combined bits.
  void mark_left(const std::vector<Route>& owned, std::uint64_t* bits) const;
  void route_bitmap(const std::vector<Route>& routes, const std::uint64_t* bits);

 private:
  void collect(const std::vector<std::int32_t>& slot_of_node);
  void build_by_feature(hist_t* out, std::size_t total);
  void build_by_sample(hist_t* out, std::size_t total);

  const std::uint8_t* bins_;
  const std::uint8_t* labels_;
  std::uint64_t n_rows_;
  int n_features_;
  int bin_stride_;
  int n_classes_;
  OmpStrategy strategy_;

  std::vector<std::int32_t> node_of_;  // current node of each row, -1 if inactive

  // Compact list of the rows that contribute to this level's histograms.
  std::vector<std::uint32_t> rows_;
  std::vector<std::int32_t> slots_;
  std::size_t n_listed_ = 0;

  std::vector<hist_t> private_;  // per-thread histograms (Sample strategy)
};

}  // namespace dt
