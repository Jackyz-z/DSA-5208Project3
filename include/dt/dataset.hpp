#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dt {

// Binary dataset format ("DTREEBIN"), little endian:
//
//   char[8]  magic = "DTREEBIN"
//   uint64   n_rows
//   uint32   n_features
//   uint32   n_classes
//   uint8    labels[n_rows]
//   float32  values[n_features][n_rows]      (column-major)
//
// Column-major storage lets a rank read either a block of rows (data-parallel)
// or a block of features (feature-parallel) with a few contiguous reads.
struct DataHeader {
  std::uint64_t n_rows = 0;
  std::uint32_t n_features = 0;
  std::uint32_t n_classes = 0;
};

// Rows [row_begin, row_begin + n_rows) and features
// [feat_begin, feat_begin + n_features) of a dataset, stored column-major.
struct RawBlock {
  std::uint64_t row_begin = 0;
  std::uint64_t n_rows = 0;
  std::uint32_t feat_begin = 0;
  std::uint32_t n_features = 0;
  std::vector<float> values;         // values[f * n_rows + i]
  std::vector<std::uint8_t> labels;  // labels[i]

  float at(std::uint32_t f, std::uint64_t i) const { return values[f * n_rows + i]; }
};

DataHeader read_header(const std::string& path);

RawBlock read_block(const std::string& path, const DataHeader& header,
                    std::uint64_t row_begin, std::uint64_t row_end,
                    std::uint32_t feat_begin, std::uint32_t feat_end);

// Contiguous, balanced partition of [0, n) into `parts` ranges.
inline std::pair<std::uint64_t, std::uint64_t> block_range(std::uint64_t n, int parts, int part) {
  return {n * part / parts, n * (part + 1) / parts};
}

}  // namespace dt
