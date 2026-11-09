#pragma once

#include <cstdint>
#include <vector>

namespace dt {

struct LevelStats {
  int depth = 0;
  int frontier = 0;  // nodes evaluated at this level
  int built = 0;     // of which histograms were built from rows
  double hist = 0, comm = 0, split = 0, route = 0;
  std::uint64_t comm_bytes = 0;  // bytes contributed by this rank
};

struct TrainStats {
  double hist = 0, comm = 0, split = 0, route = 0;
  std::uint64_t comm_bytes = 0;
  std::vector<LevelStats> levels;

  void add(const LevelStats& lv) {
    hist += lv.hist;
    comm += lv.comm;
    split += lv.split;
    route += lv.route;
    comm_bytes += lv.comm_bytes;
    levels.push_back(lv);
  }
};

}  // namespace dt
