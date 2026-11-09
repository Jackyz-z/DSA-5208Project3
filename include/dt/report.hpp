#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "dt/stats.hpp"

namespace dt {

// One row of the results CSV shared by dt_serial and dt_mpi.
struct RunRecord {
  std::string tag, mode, comm, backend, omp;
  int ranks = 1, threads = 1;
  std::uint64_t n_train = 0;
  int n_features = 0, max_depth = 0, bins = 0;
  bool subtract = false;
  double t_read = 0, t_bin = 0, t_hist = 0, t_comm = 0, t_split = 0, t_route = 0, t_train = 0,
         t_eval = 0;
  double comm_mb = 0;
  int nodes = 0, leaves = 0, depth = 0;
  double train_acc = 0, test_acc = -1;
};

inline bool file_is_empty(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  return !in || in.tellg() == 0;
}

inline void append_record(const std::string& path, const RunRecord& r) {
  const bool header = file_is_empty(path);
  std::ofstream out(path, std::ios::app);
  if (header) {
    out << "tag,mode,comm,backend,omp,ranks,threads,n_train,n_features,max_depth,bins,subtract,"
           "t_read,t_bin,t_hist,t_comm,t_split,t_route,t_train,t_eval,comm_mb,nodes,leaves,depth,"
           "train_acc,test_acc\n";
  }
  out << r.tag << ',' << r.mode << ',' << r.comm << ',' << r.backend << ',' << r.omp << ','
      << r.ranks << ',' << r.threads << ',' << r.n_train << ',' << r.n_features << ','
      << r.max_depth << ',' << r.bins << ',' << (r.subtract ? 1 : 0) << ',' << r.t_read << ','
      << r.t_bin << ',' << r.t_hist << ',' << r.t_comm << ',' << r.t_split << ',' << r.t_route
      << ',' << r.t_train << ',' << r.t_eval << ',' << r.comm_mb << ',' << r.nodes << ','
      << r.leaves << ',' << r.depth << ',' << r.train_acc << ',' << r.test_acc << '\n';
}

inline void append_levels(const std::string& path, const RunRecord& r,
                          const std::vector<LevelStats>& levels) {
  const bool header = file_is_empty(path);
  std::ofstream out(path, std::ios::app);
  if (header) {
    out << "tag,mode,comm,backend,omp,ranks,threads,level,frontier,built,t_hist,t_comm,t_split,"
           "t_route,comm_bytes\n";
  }
  for (const LevelStats& lv : levels) {
    out << r.tag << ',' << r.mode << ',' << r.comm << ',' << r.backend << ',' << r.omp << ','
        << r.ranks << ',' << r.threads << ',' << lv.depth << ',' << lv.frontier << ',' << lv.built
        << ',' << lv.hist << ',' << lv.comm << ',' << lv.split << ',' << lv.route << ','
        << lv.comm_bytes << '\n';
  }
}

}  // namespace dt
