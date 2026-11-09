#pragma once

#include <mpi.h>

#include <cstdint>
#include <string>
#include <vector>

#include "dt/binning.hpp"
#include "dt/hist_cpu.hpp"
#include "dt/split.hpp"
#include "dt/stats.hpp"
#include "dt/tree.hpp"

namespace dt {

// Collective used by the data-parallel trainer to combine local histograms.
//   Allreduce:     every rank receives every global histogram and evaluates
//                  all splits redundantly. With subtraction, only the smaller
//                  child of each sibling pair is communicated.
//   ReduceScatter: each rank receives the global histograms of a disjoint
//                  set of (node, feature) pairs, evaluates only those, and the
//                  per-node winners are exchanged with an allgather.
enum class CommPattern { Allreduce, ReduceScatter };

CommPattern parse_comm_pattern(const std::string& name);
const char* to_string(CommPattern c);

struct TrainOptions {
  GrowthParams growth;
  bool subtract = true;  // histogram subtraction: sibling = parent - smaller child
  CommPattern comm = CommPattern::Allreduce;
};

// Rows are partitioned across ranks; every rank holds all features of its rows.
Tree train_data_parallel(MPI_Comm comm, HistBackend& backend, const std::uint8_t* labels,
                         std::uint64_t n_local, const BinMapper& mapper, int n_classes,
                         const TrainOptions& opt, TrainStats& stats);

// Features are partitioned across ranks; every rank holds all rows of the
// features [feat_begin, feat_begin + feat_count).
Tree train_feature_parallel(MPI_Comm comm, CpuBackend& backend, const std::uint8_t* labels,
                            std::uint64_t n_rows, int feat_begin, int feat_count,
                            const BinMapper& mapper, int n_classes, const TrainOptions& opt,
                            TrainStats& stats);

}  // namespace dt
