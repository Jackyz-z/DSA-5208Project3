#pragma once

#include <cstdint>
#include <vector>

#include "dt/common.hpp"

namespace dt {

// How the rows of one tree node move to its children after a split.
//   feature < 0          : the node became a leaf, its rows are deactivated
//   feature >= 0         : rows with bins[feature][i] <= bin go to `left`,
//                          the others to `right` (feature is a LOCAL column)
//   left/right == -1     : that child is a leaf, its rows are deactivated
struct Route {
  std::int32_t feature = -1;
  std::int32_t bin = -1;
  std::int32_t left = -1;
  std::int32_t right = -1;
};

// Owns the local binned rows and their current node assignment, and builds
// local class histograms. Implemented for CPU (OpenMP) and GPU (CUDA).
//
// Histogram layout: out[((slot * n_features + f) * bin_stride + bin) * n_classes + c]
class HistBackend {
 public:
  virtual ~HistBackend() = default;

  // Overwrites out[0 .. n_slots * slot_size) with the histograms of the active
  // rows whose node has slot_of_node[node] >= 0.
  virtual void build(const std::vector<std::int32_t>& slot_of_node, int n_slots, hist_t* out) = 0;

  // Moves every active row to a child node; routes is indexed by node id.
  virtual void route(const std::vector<Route>& routes) = 0;
};

}  // namespace dt
