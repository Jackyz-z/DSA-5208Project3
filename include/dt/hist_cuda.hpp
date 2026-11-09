#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "dt/hist_backend.hpp"

namespace dt {

// GPU histogram backend. The binned rows, labels and node assignment live in
// device memory for the whole run; per level only the slot map (host -> GPU)
// and the finished local histograms (GPU -> host) cross the PCIe bus.
//
// Two kernels are used:
//   * shared-memory privatisation when all slots of one feature fit in
//     shared memory (shallow levels, many rows per histogram bin, hence many
//     colliding atomics), and
//   * direct global atomics otherwise (deep levels, few rows per bin).
class CudaBackend : public HistBackend {
 public:
  CudaBackend(const std::uint8_t* bins, const std::uint8_t* labels, std::uint64_t n_rows,
              int n_features, int bin_stride, int n_classes, int device);
  ~CudaBackend() override;

  void build(const std::vector<std::int32_t>& slot_of_node, int n_slots, hist_t* out) override;
  void route(const std::vector<Route>& routes) override;

  static int device_count();

  // Number of levels that used the shared-memory kernel (for reporting).
  int shared_kernel_levels() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dt
