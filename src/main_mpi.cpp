// Parallel decision tree training: MPI across ranks, OpenMP or CUDA within a rank.

#include <mpi.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dt/binning.hpp"
#include "dt/common.hpp"
#include "dt/dataset.hpp"
#include "dt/hist_cpu.hpp"
#include "dt/report.hpp"
#include "dt/trainer.hpp"
#include "dt/tree.hpp"

#ifdef DT_HAVE_CUDA
#include "dt/hist_cuda.hpp"
#endif

namespace {

using namespace dt;

const char* kUsage =
    "usage: dt_mpi --train FILE [options]\n"
    "  --test FILE             evaluate accuracy on this dataset\n"
    "  --mode data|feature     partition rows (default) or features across ranks\n"
    "  --comm allreduce|reducescatter   histogram collective in data mode (default allreduce)\n"
    "  --backend cpu|cuda      histogram construction (cuda: data mode only)\n"
    "  --omp feature|sample    OpenMP work sharing for CPU histograms (default feature)\n"
    "  --max-depth D           (default 10)\n"
    "  --min-samples-leaf N    (default 1)\n"
    "  --min-samples-split N   (default 2)\n"
    "  --bins B                maximum bins per feature, 2..256 (default 256)\n"
    "  --sample-rows N         rows used to estimate bin edges (default 200000)\n"
    "  --no-subtract           disable histogram subtraction\n"
    "  --dump-tree FILE        write the canonical tree\n"
    "  --csv FILE              append a summary row\n"
    "  --level-csv FILE        append per-level statistics of rank 0\n"
    "  --tag NAME              label stored in the CSV files\n";

struct Options {
  std::string train, test, mode, comm, backend, omp, dump_tree, csv, level_csv, tag;
  int max_depth = 10, bins = 256;
  std::uint64_t min_leaf = 1, min_split = 2, sample_rows = 200000;
  bool subtract = true;
};

Options parse_options(const Args& a) {
  Options o;
  o.train = a.required("train");
  o.test = a.str("test");
  o.mode = a.str("mode", "data");
  o.comm = a.str("comm", "allreduce");
  o.backend = a.str("backend", "cpu");
  o.omp = a.str("omp", "feature");
  o.dump_tree = a.str("dump-tree");
  o.csv = a.str("csv");
  o.level_csv = a.str("level-csv");
  o.tag = a.str("tag", "run");
  o.max_depth = static_cast<int>(a.integer("max-depth", 10));
  o.bins = static_cast<int>(a.integer("bins", 256));
  o.min_leaf = static_cast<std::uint64_t>(a.integer("min-samples-leaf", 1));
  o.min_split = static_cast<std::uint64_t>(a.integer("min-samples-split", 2));
  o.sample_rows = static_cast<std::uint64_t>(a.integer("sample-rows", 200000));
  o.subtract = !a.has("no-subtract");
  if (o.mode != "data" && o.mode != "feature") throw std::runtime_error("unknown mode " + o.mode);
  if (o.backend != "cpu" && o.backend != "cuda") throw std::runtime_error("unknown backend " + o.backend);
  if (o.backend == "cuda" && o.mode != "data") throw std::runtime_error("cuda backend requires --mode data");
  if (o.min_leaf < 1) throw std::runtime_error("--min-samples-leaf must be at least 1");
  if (o.max_depth < 0 || o.max_depth > 24) throw std::runtime_error("--max-depth must be in [0, 24]");
  return o;
}

// Data mode: each rank samples its own rows, the samples are allgathered and
// every rank computes the same edges for all features.
BinMapper edges_for_row_blocks(MPI_Comm comm, const RawBlock& block, std::uint64_t stride,
                               int max_bins) {
  int n_ranks;
  MPI_Comm_size(comm, &n_ranks);
  const int F = static_cast<int>(block.n_features);

  std::vector<std::uint64_t> picks;
  for (std::uint64_t i = 0; i < block.n_rows; ++i) {
    if ((block.row_begin + i) % stride == 0) picks.push_back(i);
  }
  const int k = static_cast<int>(picks.size());
  std::vector<float> send(static_cast<std::size_t>(F) * k);
  for (int f = 0; f < F; ++f) {
    for (int j = 0; j < k; ++j) send[static_cast<std::size_t>(f) * k + j] = block.at(f, picks[j]);
  }

  std::vector<int> per_rank(n_ranks), counts(n_ranks), displs(n_ranks);
  MPI_Allgather(&k, 1, MPI_INT, per_rank.data(), 1, MPI_INT, comm);
  int total = 0;
  for (int p = 0; p < n_ranks; ++p) {
    counts[p] = per_rank[p] * F;
    displs[p] = total;
    total += counts[p];
  }
  std::vector<float> recv(total);
  MPI_Allgatherv(send.data(), F * k, MPI_FLOAT, recv.data(), counts.data(), displs.data(),
                 MPI_FLOAT, comm);

  std::vector<std::vector<float>> edges(F);
#pragma omp parallel for schedule(dynamic, 1)
  for (int f = 0; f < F; ++f) {
    std::vector<float> sample;
    sample.reserve(total / F);
    for (int p = 0; p < n_ranks; ++p) {
      const float* src = recv.data() + displs[p] + static_cast<std::size_t>(f) * per_rank[p];
      sample.insert(sample.end(), src, src + per_rank[p]);
    }
    edges[f] = compute_edges(std::move(sample), max_bins);
  }
  return BinMapper(std::move(edges));
}

// Feature mode: each rank computes the edges of its own features, then the
// edges are allgathered so every rank knows the thresholds of all features.
BinMapper edges_for_feature_blocks(MPI_Comm comm, const RawBlock& block, std::uint64_t stride,
                                   int max_bins, int total_features) {
  int n_ranks;
  MPI_Comm_size(comm, &n_ranks);
  const int Fl = static_cast<int>(block.n_features);

  std::vector<std::vector<float>> local(Fl);
#pragma omp parallel for schedule(dynamic, 1)
  for (int f = 0; f < Fl; ++f) {
    std::vector<float> sample;
    for (std::uint64_t i = 0; i < block.n_rows; i += stride) sample.push_back(block.at(f, i));
    local[f] = compute_edges(std::move(sample), max_bins);
  }

  // Exchange edge counts, then the edges themselves, in global feature order.
  std::vector<int> feat_counts(n_ranks), feat_displs(n_ranks);
  for (int p = 0; p < n_ranks; ++p) {
    const auto r = block_range(total_features, n_ranks, p);
    feat_displs[p] = static_cast<int>(r.first);
    feat_counts[p] = static_cast<int>(r.second - r.first);
  }
  std::vector<int> my_sizes(Fl), sizes(total_features);
  for (int f = 0; f < Fl; ++f) my_sizes[f] = static_cast<int>(local[f].size());
  MPI_Allgatherv(my_sizes.data(), Fl, MPI_INT, sizes.data(), feat_counts.data(),
                 feat_displs.data(), MPI_INT, comm);

  std::vector<float> send;
  for (const auto& e : local) send.insert(send.end(), e.begin(), e.end());
  std::vector<int> counts(n_ranks, 0), displs(n_ranks, 0);
  int total = 0;
  for (int p = 0; p < n_ranks; ++p) {
    for (int f = feat_displs[p]; f < feat_displs[p] + feat_counts[p]; ++f) counts[p] += sizes[f];
    displs[p] = total;
    total += counts[p];
  }
  std::vector<float> recv(total);
  MPI_Allgatherv(send.data(), static_cast<int>(send.size()), MPI_FLOAT, recv.data(), counts.data(),
                 displs.data(), MPI_FLOAT, comm);

  std::vector<std::vector<float>> edges(total_features);
  std::size_t pos = 0;
  for (int f = 0; f < total_features; ++f) {
    edges[f].assign(recv.begin() + pos, recv.begin() + pos + sizes[f]);
    pos += sizes[f];
  }
  return BinMapper(std::move(edges));
}

std::vector<std::uint8_t> bin_block(const RawBlock& block, const BinMapper& mapper) {
  const std::int64_t n = static_cast<std::int64_t>(block.n_rows);
  std::vector<std::uint8_t> bins(static_cast<std::size_t>(block.n_features) * block.n_rows);
  for (std::uint32_t f = 0; f < block.n_features; ++f) {
    const int gf = static_cast<int>(block.feat_begin + f);
    const float* x = block.values.data() + static_cast<std::size_t>(f) * block.n_rows;
    std::uint8_t* out = bins.data() + static_cast<std::size_t>(f) * block.n_rows;
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < n; ++i) out[i] = mapper.bin(gf, x[i]);
  }
  return bins;
}

#ifdef DT_HAVE_CUDA
// Rank index among the ranks on the same node, used to pick a GPU.
int local_rank(MPI_Comm comm) {
  MPI_Comm node_comm;
  int rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
  int local;
  MPI_Comm_rank(node_comm, &local);
  MPI_Comm_free(&node_comm);
  return local;
}
#endif

double max_over_ranks(MPI_Comm comm, double v) {
  double out = 0;
  MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, comm);
  return out;
}

int run(int argc, char** argv) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, n_ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_ranks);

  const Args args(argc, argv);
  if (args.has("help")) {
    if (rank == 0) std::cout << kUsage;
    return 0;
  }
  const Options o = parse_options(args);
  const bool data_mode = o.mode == "data";

  // ---- read ------------------------------------------------------------
  Timer t;
  const DataHeader h = read_header(o.train);
  const int F = static_cast<int>(h.n_features);
  const int C = static_cast<int>(h.n_classes);
  int feat_begin = 0, feat_count = F;
  RawBlock block;
  if (data_mode) {
    const auto rows = block_range(h.n_rows, n_ranks, rank);
    block = read_block(o.train, h, rows.first, rows.second, 0, h.n_features);
  } else {
    const auto feats = block_range(h.n_features, n_ranks, rank);
    feat_begin = static_cast<int>(feats.first);
    feat_count = static_cast<int>(feats.second - feats.first);
    block = read_block(o.train, h, 0, h.n_rows, feats.first, feats.second);
  }
  const double t_read = t.lap();

  // ---- bin -------------------------------------------------------------
  const std::uint64_t stride = sample_stride(h.n_rows, o.sample_rows);
  const BinMapper mapper = data_mode ? edges_for_row_blocks(comm, block, stride, o.bins)
                                     : edges_for_feature_blocks(comm, block, stride, o.bins, F);
  const std::vector<std::uint8_t> bins = bin_block(block, mapper);
  const std::vector<std::uint8_t> labels = std::move(block.labels);
  const std::uint64_t n_local = block.n_rows;
  std::vector<float>().swap(block.values);  // raw training values are no longer needed
  const double t_bin = t.lap();

  // ---- train -----------------------------------------------------------
  TrainOptions topt;
  topt.growth.max_depth = o.max_depth;
  topt.growth.min_samples_leaf = o.min_leaf;
  topt.growth.min_samples_split = o.min_split;
  topt.subtract = o.subtract;
  topt.comm = parse_comm_pattern(o.comm);
  const OmpStrategy strategy = parse_omp_strategy(o.omp);

  std::unique_ptr<HistBackend> backend;
  CpuBackend* cpu = nullptr;
  if (o.backend == "cuda") {
#ifdef DT_HAVE_CUDA
    const int devices = CudaBackend::device_count();
    if (devices == 0) throw std::runtime_error("no CUDA device visible");
    backend.reset(new CudaBackend(bins.data(), labels.data(), n_local, feat_count,
                                  mapper.max_bins(), C, local_rank(comm) % devices));
#else
    throw std::runtime_error("dt_mpi was built without CUDA support");
#endif
  } else {
    cpu = new CpuBackend(bins.data(), labels.data(), n_local, feat_count, mapper.max_bins(), C,
                         strategy);
    backend.reset(cpu);
  }

  TrainStats stats;
  MPI_Barrier(comm);
  t.reset();
  const Tree tree =
      data_mode ? train_data_parallel(comm, *backend, labels.data(), n_local, mapper, C, topt, stats)
                : train_feature_parallel(comm, *cpu, labels.data(), n_local, feat_begin,
                                         feat_count, mapper, C, topt, stats);
  const double t_train = t.lap();
  backend.reset();

  // ---- evaluate --------------------------------------------------------
  double test_acc = -1.0;
  if (!o.test.empty()) {
    const DataHeader th = read_header(o.test);
    if (th.n_features != h.n_features) throw std::runtime_error("test set has a different width");
    const auto rows = block_range(th.n_rows, n_ranks, rank);
    const RawBlock test = read_block(o.test, th, rows.first, rows.second, 0, th.n_features);
    std::uint64_t correct = tree.correct_on_block(test);
    MPI_Allreduce(MPI_IN_PLACE, &correct, 1, MPI_UINT64_T, MPI_SUM, comm);
    test_acc = static_cast<double>(correct) / static_cast<double>(th.n_rows);
  }
  const double t_eval = t.lap();

  // ---- report ----------------------------------------------------------
  RunRecord r;
  r.tag = o.tag;
  r.mode = o.mode;
  r.comm = data_mode ? o.comm : "allgather+bitmap";
  r.backend = o.backend;
  r.omp = o.backend == "cpu" ? o.omp : "-";
  r.ranks = n_ranks;
  r.threads = max_threads();
  r.n_train = h.n_rows;
  r.n_features = F;
  r.max_depth = o.max_depth;
  r.bins = o.bins;
  r.subtract = o.subtract;
  r.t_read = max_over_ranks(comm, t_read);
  r.t_bin = max_over_ranks(comm, t_bin);
  r.t_hist = max_over_ranks(comm, stats.hist);
  r.t_comm = max_over_ranks(comm, stats.comm);
  r.t_split = max_over_ranks(comm, stats.split);
  r.t_route = max_over_ranks(comm, stats.route);
  r.t_train = max_over_ranks(comm, t_train);
  r.t_eval = max_over_ranks(comm, t_eval);
  r.comm_mb = static_cast<double>(stats.comm_bytes) / 1e6;
  r.nodes = tree.size();
  r.leaves = tree.n_leaves();
  r.depth = tree.depth();
  r.train_acc = static_cast<double>(tree.training_correct()) / static_cast<double>(h.n_rows);
  r.test_acc = test_acc;

  if (rank == 0) {
    std::printf("mode=%s comm=%s backend=%s omp=%s ranks=%d threads=%d subtract=%d\n",
                r.mode.c_str(), r.comm.c_str(), r.backend.c_str(), r.omp.c_str(), r.ranks,
                r.threads, o.subtract ? 1 : 0);
    std::printf("rows=%llu features=%d classes=%d max_bins=%d\n",
                static_cast<unsigned long long>(h.n_rows), F, C, mapper.max_bins());
    std::printf("tree: nodes=%d leaves=%d depth=%d\n", r.nodes, r.leaves, r.depth);
    std::printf("time [s] (max over ranks): read=%.3f bin=%.3f train=%.3f "
                "(hist=%.3f comm=%.3f split=%.3f route=%.3f) eval=%.3f\n",
                r.t_read, r.t_bin, r.t_train, r.t_hist, r.t_comm, r.t_split, r.t_route, r.t_eval);
    std::printf("communication per rank: %.2f MB\n", r.comm_mb);
    std::printf("train accuracy=%.6f", r.train_acc);
    if (test_acc >= 0) std::printf(" test accuracy=%.6f", test_acc);
    std::printf("\n");

    if (!o.dump_tree.empty()) {
      std::ofstream out(o.dump_tree);
      tree.write(out);
    }
    if (!o.csv.empty()) append_record(o.csv, r);
    if (!o.level_csv.empty()) append_levels(o.level_csv, r, stats.levels);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int status = 0;
  try {
    status = run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[rank %d] error: %s\n", rank, e.what());
    if (rank == 0) std::fprintf(stderr, "%s", kUsage);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
  return status;
}
