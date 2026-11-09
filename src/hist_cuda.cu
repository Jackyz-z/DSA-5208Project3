#include "dt/hist_cuda.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                   \
  do {                                                                                     \
    cudaError_t err_ = (call);                                                             \
    if (err_ != cudaSuccess) {                                                             \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) +   \
                               " (" __FILE__ ":" + std::to_string(__LINE__) + ")");        \
    }                                                                                      \
  } while (0)

namespace dt {

namespace {

static_assert(sizeof(hist_t) == sizeof(unsigned int), "atomicAdd works on unsigned int");

constexpr int kThreads = 256;
constexpr std::uint64_t kRowsPerBlock = 1 << 16;
constexpr std::size_t kSharedLimit = 48 * 1024;

// grid = (feature, row chunk). Each block accumulates the histograms of all
// slots for one feature in shared memory, then flushes them with one global
// atomic per non-zero bin.
__global__ void hist_shared_kernel(const std::uint8_t* __restrict__ bins,
                                   const std::uint8_t* __restrict__ labels,
                                   const std::int32_t* __restrict__ node_of,
                                   const std::int32_t* __restrict__ slot_of_node,
                                   std::uint64_t n_rows, int n_features, int unit, int n_classes,
                                   int n_slots, unsigned int* __restrict__ hist) {
  extern __shared__ unsigned int local[];
  const int f = blockIdx.x;
  const int size = n_slots * unit;
  for (int e = threadIdx.x; e < size; e += blockDim.x) local[e] = 0;
  __syncthreads();

  const std::uint64_t begin = static_cast<std::uint64_t>(blockIdx.y) * kRowsPerBlock;
  const std::uint64_t end = min(begin + kRowsPerBlock, n_rows);
  const std::uint8_t* col = bins + static_cast<std::uint64_t>(f) * n_rows;
  for (std::uint64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    const int node = node_of[i];
    if (node < 0) continue;
    const int slot = slot_of_node[node];
    if (slot < 0) continue;
    atomicAdd(&local[slot * unit + col[i] * n_classes + labels[i]], 1u);
  }
  __syncthreads();

  for (int e = threadIdx.x; e < size; e += blockDim.x) {
    const unsigned int v = local[e];
    if (v == 0) continue;
    const int slot = e / unit;
    const int rest = e - slot * unit;
    atomicAdd(&hist[(static_cast<std::uint64_t>(slot) * n_features + f) * unit + rest], v);
  }
}

__global__ void hist_global_kernel(const std::uint8_t* __restrict__ bins,
                                   const std::uint8_t* __restrict__ labels,
                                   const std::int32_t* __restrict__ node_of,
                                   const std::int32_t* __restrict__ slot_of_node,
                                   std::uint64_t n_rows, int n_features, int unit, int n_classes,
                                   unsigned int* __restrict__ hist) {
  const int f = blockIdx.x;
  const std::uint64_t begin = static_cast<std::uint64_t>(blockIdx.y) * kRowsPerBlock;
  const std::uint64_t end = min(begin + kRowsPerBlock, n_rows);
  const std::uint8_t* col = bins + static_cast<std::uint64_t>(f) * n_rows;
  for (std::uint64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    const int node = node_of[i];
    if (node < 0) continue;
    const int slot = slot_of_node[node];
    if (slot < 0) continue;
    atomicAdd(&hist[(static_cast<std::uint64_t>(slot) * n_features + f) * unit +
                    col[i] * n_classes + labels[i]],
              1u);
  }
}

__global__ void route_kernel(const std::uint8_t* __restrict__ bins, const Route* __restrict__ routes,
                             std::int32_t* __restrict__ node_of, std::uint64_t n_rows) {
  const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_rows) return;
  const int node = node_of[i];
  if (node < 0) return;
  const Route r = routes[node];
  if (r.feature < 0) {
    node_of[i] = -1;
    return;
  }
  const std::uint8_t b = bins[static_cast<std::uint64_t>(r.feature) * n_rows + i];
  node_of[i] = b <= r.bin ? r.left : r.right;
}

template <class T>
void ensure_capacity(T*& ptr, std::size_t& capacity, std::size_t count) {
  if (count <= capacity) return;
  if (ptr) CUDA_CHECK(cudaFree(ptr));
  CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
  capacity = count;
}

}  // namespace

struct CudaBackend::Impl {
  int device = 0;
  std::uint64_t n_rows = 0;
  int n_features = 0, bin_stride = 0, n_classes = 0;
  int shared_levels = 0;

  std::uint8_t* bins = nullptr;
  std::uint8_t* labels = nullptr;
  std::int32_t* node_of = nullptr;
  std::int32_t* slot_of_node = nullptr;
  Route* routes = nullptr;
  unsigned int* hist = nullptr;
  std::size_t slot_cap = 0, route_cap = 0, hist_cap = 0;
};

CudaBackend::CudaBackend(const std::uint8_t* bins, const std::uint8_t* labels, std::uint64_t n_rows,
                         int n_features, int bin_stride, int n_classes, int device)
    : impl_(new Impl) {
  Impl& s = *impl_;
  s.device = device;
  s.n_rows = n_rows;
  s.n_features = n_features;
  s.bin_stride = bin_stride;
  s.n_classes = n_classes;
  if ((n_rows + kRowsPerBlock - 1) / kRowsPerBlock > 65535) {
    throw std::runtime_error("too many rows for one GPU; use more ranks");
  }

  CUDA_CHECK(cudaSetDevice(device));
  const std::size_t n_bins = static_cast<std::size_t>(n_features) * n_rows;
  CUDA_CHECK(cudaMalloc(&s.bins, n_bins));
  CUDA_CHECK(cudaMalloc(&s.labels, n_rows));
  CUDA_CHECK(cudaMalloc(&s.node_of, n_rows * sizeof(std::int32_t)));
  CUDA_CHECK(cudaMemcpy(s.bins, bins, n_bins, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(s.labels, labels, n_rows, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(s.node_of, 0, n_rows * sizeof(std::int32_t)));  // all rows at the root
}

CudaBackend::~CudaBackend() {
  Impl& s = *impl_;
  cudaSetDevice(s.device);
  cudaFree(s.bins);
  cudaFree(s.labels);
  cudaFree(s.node_of);
  cudaFree(s.slot_of_node);
  cudaFree(s.routes);
  cudaFree(s.hist);
}

int CudaBackend::device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
  return count;
}

int CudaBackend::shared_kernel_levels() const { return impl_->shared_levels; }

void CudaBackend::build(const std::vector<std::int32_t>& slot_of_node, int n_slots, hist_t* out) {
  Impl& s = *impl_;
  const int unit = s.bin_stride * s.n_classes;
  const std::size_t total = static_cast<std::size_t>(n_slots) * s.n_features * unit;
  if (total == 0) return;

  CUDA_CHECK(cudaSetDevice(s.device));
  ensure_capacity(s.slot_of_node, s.slot_cap, slot_of_node.size());
  ensure_capacity(s.hist, s.hist_cap, total);
  CUDA_CHECK(cudaMemcpy(s.slot_of_node, slot_of_node.data(),
                        slot_of_node.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(s.hist, 0, total * sizeof(unsigned int)));

  const dim3 grid(s.n_features, static_cast<unsigned>((s.n_rows + kRowsPerBlock - 1) / kRowsPerBlock));
  const std::size_t shared_bytes = static_cast<std::size_t>(n_slots) * unit * sizeof(unsigned int);
  if (shared_bytes <= kSharedLimit) {
    hist_shared_kernel<<<grid, kThreads, shared_bytes>>>(s.bins, s.labels, s.node_of, s.slot_of_node,
                                                         s.n_rows, s.n_features, unit, s.n_classes,
                                                         n_slots, s.hist);
    ++s.shared_levels;
  } else {
    hist_global_kernel<<<grid, kThreads>>>(s.bins, s.labels, s.node_of, s.slot_of_node, s.n_rows,
                                           s.n_features, unit, s.n_classes, s.hist);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(out, s.hist, total * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void CudaBackend::route(const std::vector<Route>& routes) {
  Impl& s = *impl_;
  CUDA_CHECK(cudaSetDevice(s.device));
  ensure_capacity(s.routes, s.route_cap, routes.size());
  CUDA_CHECK(cudaMemcpy(s.routes, routes.data(), routes.size() * sizeof(Route),
                        cudaMemcpyHostToDevice));
  const unsigned blocks = static_cast<unsigned>((s.n_rows + kThreads - 1) / kThreads);
  route_kernel<<<blocks, kThreads>>>(s.bins, s.routes, s.node_of, s.n_rows);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace dt
