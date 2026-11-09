// Serial reference implementation.
//
// Deliberately written independently of the parallel trainer: the tree grows
// depth-first by recursion, each node keeps an explicit list of its rows, and
// every histogram is built from scratch (no subtraction, no level batching).
// It shares only the file reader, the binning rule, the split criterion and
// the tree container with dt_mpi, so agreement between the two programs is a
// meaningful check of the parallel logic.

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dt/binning.hpp"
#include "dt/common.hpp"
#include "dt/dataset.hpp"
#include "dt/report.hpp"
#include "dt/split.hpp"
#include "dt/tree.hpp"

namespace {

using namespace dt;

const char* kUsage =
    "usage: dt_serial --train FILE [--test FILE] [--max-depth D] [--min-samples-leaf N]\n"
    "                 [--min-samples-split N] [--bins B] [--sample-rows N]\n"
    "                 [--dump-tree FILE] [--csv FILE] [--tag NAME]\n";

class SerialGrower {
 public:
  SerialGrower(const std::vector<std::uint8_t>& bins, const std::vector<std::uint8_t>& labels,
               std::uint64_t n_rows, const BinMapper& mapper, int n_classes, GrowthParams params)
      : bins_(bins),
        labels_(labels),
        n_rows_(n_rows),
        mapper_(mapper),
        n_classes_(n_classes),
        params_(params),
        tree_(n_classes) {}

  Tree grow() {
    std::vector<std::uint64_t> counts(n_classes_, 0);
    for (std::uint64_t i = 0; i < n_rows_; ++i) counts[labels_[i]]++;
    tree_.add_node(-1, 0, counts.data());

    std::vector<std::uint64_t> rows(n_rows_);
    for (std::uint64_t i = 0; i < n_rows_; ++i) rows[i] = i;
    grow_node(0, rows);
    return tree_;
  }

  double t_hist = 0, t_split = 0, t_partition = 0;

 private:
  void grow_node(int id, std::vector<std::uint64_t>& rows) {
    const TreeNode node = tree_.node(id);  // copy: add_node may reallocate
    if (!can_split(node.depth, node.n, node.counts.data(), n_classes_, params_)) return;

    // Histogram of every feature over the rows of this node.
    Timer t;
    const int F = mapper_.n_features();
    const std::size_t unit = static_cast<std::size_t>(mapper_.max_bins()) * n_classes_;
    hist_.assign(unit * F, 0);
    for (int f = 0; f < F; ++f) {
      const std::uint8_t* col = bins_.data() + static_cast<std::size_t>(f) * n_rows_;
      hist_t* h = hist_.data() + f * unit;
      for (std::uint64_t i : rows) h[col[i] * n_classes_ + labels_[i]]++;
    }
    t_hist += t.lap();

    SplitInfo best;
    for (int f = 0; f < F; ++f) {
      const SplitInfo s = best_split_for_feature(hist_.data() + f * unit, mapper_.n_bins(f),
                                                 n_classes_, node.counts.data(),
                                                 params_.min_samples_leaf, f);
      if (better(s, best)) best = s;
    }
    t_split += t.lap();
    if (!best.valid() || !(best.gain > params_.min_gain)) return;

    std::vector<std::uint64_t> left_rows, right_rows;
    const std::uint8_t* col = bins_.data() + static_cast<std::size_t>(best.feature) * n_rows_;
    for (std::uint64_t i : rows) (col[i] <= best.bin ? left_rows : right_rows).push_back(i);
    std::vector<std::uint64_t>().swap(rows);
    t_partition += t.lap();

    std::uint64_t right[kMaxClasses] = {};
    for (int c = 0; c < n_classes_; ++c) right[c] = node.counts[c] - best.left[c];
    const int l = tree_.add_node(id, node.depth + 1, best.left);
    const int r = tree_.add_node(id, node.depth + 1, right);
    tree_.set_split(id, best.feature, best.bin, mapper_.threshold(best.feature, best.bin), l, r);

    grow_node(l, left_rows);
    grow_node(r, right_rows);
  }

  const std::vector<std::uint8_t>& bins_;
  const std::vector<std::uint8_t>& labels_;
  std::uint64_t n_rows_;
  const BinMapper& mapper_;
  int n_classes_;
  GrowthParams params_;
  Tree tree_;
  std::vector<hist_t> hist_;
};

int run(int argc, char** argv) {
  const Args args(argc, argv);
  if (args.has("help")) {
    std::cout << kUsage;
    return 0;
  }
  const std::string train_path = args.required("train");
  const std::string test_path = args.str("test");
  GrowthParams params;
  params.max_depth = static_cast<int>(args.integer("max-depth", 10));
  params.min_samples_leaf = static_cast<std::uint64_t>(args.integer("min-samples-leaf", 1));
  params.min_samples_split = static_cast<std::uint64_t>(args.integer("min-samples-split", 2));
  const int max_bins = static_cast<int>(args.integer("bins", 256));
  const std::uint64_t sample_rows = static_cast<std::uint64_t>(args.integer("sample-rows", 200000));
  if (params.min_samples_leaf < 1) throw std::runtime_error("--min-samples-leaf must be at least 1");

  Timer t;
  const DataHeader h = read_header(train_path);
  RawBlock block = read_block(train_path, h, 0, h.n_rows, 0, h.n_features);
  const double t_read = t.lap();

  const int F = static_cast<int>(h.n_features);
  const std::uint64_t stride = sample_stride(h.n_rows, sample_rows);
  std::vector<std::vector<float>> edges(F);
  for (int f = 0; f < F; ++f) {
    std::vector<float> sample;
    for (std::uint64_t i = 0; i < h.n_rows; i += stride) sample.push_back(block.at(f, i));
    edges[f] = compute_edges(std::move(sample), max_bins);
  }
  const BinMapper mapper(std::move(edges));
  std::vector<std::uint8_t> bins(static_cast<std::size_t>(F) * h.n_rows);
  for (int f = 0; f < F; ++f) {
    for (std::uint64_t i = 0; i < h.n_rows; ++i) {
      bins[static_cast<std::size_t>(f) * h.n_rows + i] = mapper.bin(f, block.at(f, i));
    }
  }
  const std::vector<std::uint8_t> labels = std::move(block.labels);
  std::vector<float>().swap(block.values);
  const double t_bin = t.lap();

  SerialGrower grower(bins, labels, h.n_rows, mapper, static_cast<int>(h.n_classes), params);
  const Tree tree = grower.grow();
  const double t_train = t.lap();

  double test_acc = -1.0;
  if (!test_path.empty()) {
    const DataHeader th = read_header(test_path);
    if (th.n_features != h.n_features) throw std::runtime_error("test set has a different width");
    const RawBlock test = read_block(test_path, th, 0, th.n_rows, 0, th.n_features);
    test_acc = static_cast<double>(tree.correct_on_block(test)) / static_cast<double>(th.n_rows);
  }
  const double t_eval = t.lap();

  RunRecord r;
  r.tag = args.str("tag", "run");
  r.mode = "serial";
  r.comm = "-";
  r.backend = "cpu";
  r.omp = "-";
  r.n_train = h.n_rows;
  r.n_features = F;
  r.max_depth = params.max_depth;
  r.bins = max_bins;
  r.t_read = t_read;
  r.t_bin = t_bin;
  r.t_hist = grower.t_hist;
  r.t_split = grower.t_split;
  r.t_route = grower.t_partition;
  r.t_train = t_train;
  r.t_eval = t_eval;
  r.nodes = tree.size();
  r.leaves = tree.n_leaves();
  r.depth = tree.depth();
  r.train_acc = static_cast<double>(tree.training_correct()) / static_cast<double>(h.n_rows);
  r.test_acc = test_acc;

  std::printf("mode=serial rows=%llu features=%d classes=%u max_bins=%d\n",
              static_cast<unsigned long long>(h.n_rows), F, h.n_classes, mapper.max_bins());
  std::printf("tree: nodes=%d leaves=%d depth=%d\n", r.nodes, r.leaves, r.depth);
  std::printf("time [s]: read=%.3f bin=%.3f train=%.3f (hist=%.3f split=%.3f partition=%.3f) "
              "eval=%.3f\n",
              t_read, t_bin, t_train, r.t_hist, r.t_split, r.t_route, t_eval);
  std::printf("train accuracy=%.6f", r.train_acc);
  if (test_acc >= 0) std::printf(" test accuracy=%.6f", test_acc);
  std::printf("\n");

  const std::string dump = args.str("dump-tree");
  if (!dump.empty()) {
    std::ofstream out(dump);
    tree.write(out);
  }
  const std::string csv = args.str("csv");
  if (!csv.empty()) append_record(csv, r);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n%s", e.what(), kUsage);
    return 1;
  }
}
