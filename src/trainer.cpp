#include "dt/trainer.hpp"

#include <climits>
#include <stdexcept>

namespace dt {

CommPattern parse_comm_pattern(const std::string& name) {
  if (name == "allreduce") return CommPattern::Allreduce;
  if (name == "reducescatter") return CommPattern::ReduceScatter;
  throw std::runtime_error("unknown communication pattern: " + name);
}

const char* to_string(CommPattern c) {
  return c == CommPattern::Allreduce ? "allreduce" : "reducescatter";
}

namespace {

int checked_count(std::size_t n) {
  if (n > static_cast<std::size_t>(INT_MAX)) throw std::runtime_error("MPI message too large");
  return static_cast<int>(n);
}

// Histogram slots of one tree level. Nodes whose histogram is built from rows
// come first (slots [0, n_built)), so the built part is one contiguous buffer.
struct LevelPlan {
  std::vector<int> nodes;          // slot -> node id
  int n_built = 0;
  std::vector<int> parent_slot;    // derived slot -> parent's slot at previous level
  std::vector<int> sibling_slot;   // derived slot -> sibling's slot at this level
  std::vector<std::int32_t> slot_of_node;  // node id -> slot, built nodes only
};

LevelPlan plan_level(const Tree& tree, const std::vector<int>& frontier,
                     const std::vector<int>& prev_slot, bool subtract) {
  std::vector<char> in_frontier(tree.size(), 0);
  for (int id : frontier) in_frontier[id] = 1;

  std::vector<int> built, derived;
  for (int id : frontier) {
    const TreeNode& nd = tree.node(id);
    bool derive = false;
    if (subtract && nd.parent >= 0 && prev_slot[nd.parent] >= 0) {
      const TreeNode& p = tree.node(nd.parent);
      const int sib = p.left == id ? p.right : p.left;
      if (in_frontier[sib]) {
        // Build the smaller child (the left one on ties) and derive the other.
        const std::uint64_t n_sib = tree.node(sib).n;
        const bool smaller = nd.n < n_sib || (nd.n == n_sib && p.left == id);
        derive = !smaller;
      }
    }
    (derive ? derived : built).push_back(id);
  }

  LevelPlan plan;
  plan.nodes = built;
  plan.nodes.insert(plan.nodes.end(), derived.begin(), derived.end());
  plan.n_built = static_cast<int>(built.size());

  std::vector<int> slot(tree.size(), -1);
  for (std::size_t k = 0; k < plan.nodes.size(); ++k) slot[plan.nodes[k]] = static_cast<int>(k);

  plan.slot_of_node.assign(tree.size(), -1);
  for (int k = 0; k < plan.n_built; ++k) plan.slot_of_node[plan.nodes[k]] = k;

  for (int id : derived) {
    const TreeNode& p = tree.node(tree.node(id).parent);
    const int sib = p.left == id ? p.right : p.left;
    plan.parent_slot.push_back(prev_slot[tree.node(id).parent]);
    plan.sibling_slot.push_back(slot[sib]);
  }
  return plan;
}

// hist[derived] = prev[parent] - hist[sibling]
void derive_histograms(hist_t* hist, const hist_t* prev, const LevelPlan& plan,
                       std::size_t slot_size) {
  const std::int64_t n_derived = static_cast<std::int64_t>(plan.parent_slot.size());
  const std::int64_t size = static_cast<std::int64_t>(slot_size);
#pragma omp parallel for collapse(2) schedule(static)
  for (std::int64_t d = 0; d < n_derived; ++d) {
    for (std::int64_t e = 0; e < size; ++e) {
      const std::size_t out = (plan.n_built + d) * slot_size + e;
      hist[out] = prev[plan.parent_slot[d] * slot_size + e] - hist[plan.sibling_slot[d] * slot_size + e];
    }
  }
}

// Evaluates the (slot, local feature) units [u0, u1) whose histograms start at
// `hist`, and returns the best split per slot among those units.
std::vector<SplitInfo> evaluate_units(const hist_t* hist, std::int64_t u0, std::int64_t u1,
                                      int n_feat, int feat_begin, const LevelPlan& plan,
                                      const Tree& tree, const BinMapper& mapper, int n_classes,
                                      std::uint64_t min_leaf) {
  const std::size_t unit = static_cast<std::size_t>(mapper.max_bins()) * n_classes;
  std::vector<SplitInfo> per_unit(u1 - u0);

#pragma omp parallel for schedule(dynamic, 4)
  for (std::int64_t u = u0; u < u1; ++u) {
    const int k = static_cast<int>(u / n_feat);
    const int f = feat_begin + static_cast<int>(u % n_feat);
    per_unit[u - u0] = best_split_for_feature(hist + (u - u0) * unit, mapper.n_bins(f), n_classes,
                                              tree.node(plan.nodes[k]).counts.data(), min_leaf, f);
  }

  std::vector<SplitInfo> best(plan.nodes.size());
  for (std::int64_t u = u0; u < u1; ++u) {
    SplitInfo& b = best[u / n_feat];
    if (better(per_unit[u - u0], b)) b = per_unit[u - u0];
  }
  return best;
}

// Every rank contributes its best split per node; all ranks pick the same winner.
std::vector<SplitInfo> allgather_best(MPI_Comm comm, const std::vector<SplitInfo>& mine,
                                      LevelStats& lv) {
  int n_ranks;
  MPI_Comm_size(comm, &n_ranks);
  const std::size_t s = mine.size();
  std::vector<SplitInfo> all(s * n_ranks);
  const int bytes = checked_count(s * sizeof(SplitInfo));
  MPI_Allgather(mine.data(), bytes, MPI_BYTE, all.data(), bytes, MPI_BYTE, comm);
  lv.comm_bytes += bytes;

  std::vector<SplitInfo> best(s);
  for (int p = 0; p < n_ranks; ++p) {
    for (std::size_t k = 0; k < s; ++k) {
      if (better(all[p * s + k], best[k])) best[k] = all[p * s + k];
    }
  }
  return best;
}

// Applies the chosen splits: creates child nodes, fills routes (indexed by
// node id, feature = global index) and returns the next frontier.
std::vector<int> grow(Tree& tree, const LevelPlan& plan, const std::vector<SplitInfo>& best,
                      const BinMapper& mapper, const GrowthParams& g, std::vector<Route>& routes) {
  const int C = tree.n_classes();
  std::vector<int> next;
  std::vector<std::pair<int, Route>> pending;

  for (std::size_t k = 0; k < plan.nodes.size(); ++k) {
    const int id = plan.nodes[k];
    const SplitInfo& s = best[k];
    if (!s.valid() || !(s.gain > g.min_gain)) continue;  // stays a leaf

    const int depth = tree.node(id).depth + 1;
    std::uint64_t right[kMaxClasses] = {};
    std::uint64_t n_left = 0, n_right = 0;
    for (int c = 0; c < C; ++c) {
      right[c] = tree.node(id).counts[c] - s.left[c];
      n_left += s.left[c];
      n_right += right[c];
    }
    const int l = tree.add_node(id, depth, s.left);
    const int r = tree.add_node(id, depth, right);
    tree.set_split(id, s.feature, s.bin, mapper.threshold(s.feature, s.bin), l, r);

    const bool split_l = can_split(depth, n_left, s.left, C, g);
    const bool split_r = can_split(depth, n_right, right, C, g);
    if (split_l) next.push_back(l);
    if (split_r) next.push_back(r);
    pending.push_back({id, Route{s.feature, s.bin, split_l ? l : -1, split_r ? r : -1}});
  }

  routes.assign(tree.size(), Route{});
  for (const auto& p : pending) routes[p.first] = p.second;
  return next;
}

std::vector<std::uint64_t> local_class_counts(const std::uint8_t* labels, std::uint64_t n,
                                              int n_classes) {
  std::vector<std::uint64_t> counts(n_classes, 0);
  for (std::uint64_t i = 0; i < n; ++i) {
    if (labels[i] >= n_classes) throw std::runtime_error("label out of range");
    counts[labels[i]]++;
  }
  return counts;
}

std::vector<int> root_frontier(const Tree& tree, const GrowthParams& g) {
  const TreeNode& root = tree.node(0);
  if (can_split(0, root.n, root.counts.data(), tree.n_classes(), g)) return {0};
  return {};
}

}  // namespace

Tree train_data_parallel(MPI_Comm comm, HistBackend& backend, const std::uint8_t* labels,
                         std::uint64_t n_local, const BinMapper& mapper, int n_classes,
                         const TrainOptions& opt, TrainStats& stats) {
  int rank, n_ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_ranks);

  const int F = mapper.n_features();
  const std::size_t unit = static_cast<std::size_t>(mapper.max_bins()) * n_classes;
  const std::size_t slot_size = unit * F;

  Tree tree(n_classes);
  {
    Timer t;
    std::vector<std::uint64_t> counts = local_class_counts(labels, n_local, n_classes);
    MPI_Allreduce(MPI_IN_PLACE, counts.data(), n_classes, MPI_UINT64_T, MPI_SUM, comm);
    tree.add_node(-1, 0, counts.data());
    stats.comm += t.seconds();
  }

  std::vector<int> frontier = root_frontier(tree, opt.growth);
  std::vector<hist_t> cur, prev, recv;
  std::vector<int> prev_slot(tree.size(), -1);
  std::vector<Route> routes;

  while (!frontier.empty()) {
    LevelStats lv;
    lv.depth = tree.node(frontier[0]).depth;
    Timer t;

    const LevelPlan plan = plan_level(tree, frontier, prev_slot, opt.subtract);
    const int S = static_cast<int>(plan.nodes.size());
    lv.frontier = S;
    lv.built = plan.n_built;
    cur.resize(S * slot_size);
    backend.build(plan.slot_of_node, plan.n_built, cur.data());

    std::vector<SplitInfo> best;
    if (opt.comm == CommPattern::Allreduce) {
      lv.hist += t.lap();
      const std::size_t count = plan.n_built * slot_size;
      MPI_Allreduce(MPI_IN_PLACE, cur.data(), checked_count(count), MPI_UINT32_T, MPI_SUM, comm);
      lv.comm_bytes += count * sizeof(hist_t);
      lv.comm += t.lap();

      derive_histograms(cur.data(), prev.data(), plan, slot_size);  // on global histograms
      lv.hist += t.lap();

      best = evaluate_units(cur.data(), 0, std::int64_t(S) * F, F, 0, plan, tree, mapper, n_classes,
                            opt.growth.min_samples_leaf);
      lv.split += t.lap();
    } else {
      derive_histograms(cur.data(), prev.data(), plan, slot_size);  // on local histograms
      lv.hist += t.lap();

      const std::int64_t units = std::int64_t(S) * F;
      std::vector<int> recv_counts(n_ranks);
      for (int p = 0; p < n_ranks; ++p) {
        recv_counts[p] = checked_count((units * (p + 1) / n_ranks - units * p / n_ranks) * unit);
      }
      const std::int64_t u0 = units * rank / n_ranks;
      const std::int64_t u1 = units * (rank + 1) / n_ranks;
      recv.resize(std::max<std::size_t>(1, (u1 - u0) * unit));
      MPI_Reduce_scatter(cur.data(), recv.data(), recv_counts.data(), MPI_UINT32_T, MPI_SUM, comm);
      lv.comm_bytes += S * slot_size * sizeof(hist_t);
      lv.comm += t.lap();

      const std::vector<SplitInfo> mine = evaluate_units(recv.data(), u0, u1, F, 0, plan, tree,
                                                         mapper, n_classes,
                                                         opt.growth.min_samples_leaf);
      lv.split += t.lap();
      best = allgather_best(comm, mine, lv);
      lv.comm += t.lap();
    }

    // This level's histograms (global or local, matching the pattern) become
    // the parents for subtraction at the next level.
    std::swap(prev, cur);
    frontier = grow(tree, plan, best, mapper, opt.growth, routes);
    prev_slot.assign(tree.size(), -1);
    for (int k = 0; k < S; ++k) prev_slot[plan.nodes[k]] = k;
    lv.split += t.lap();

    backend.route(routes);
    lv.route += t.lap();
    stats.add(lv);
  }
  return tree;
}

Tree train_feature_parallel(MPI_Comm comm, CpuBackend& backend, const std::uint8_t* labels,
                            std::uint64_t n_rows, int feat_begin, int feat_count,
                            const BinMapper& mapper, int n_classes, const TrainOptions& opt,
                            TrainStats& stats) {
  const std::size_t unit = static_cast<std::size_t>(mapper.max_bins()) * n_classes;
  const std::size_t slot_size = unit * feat_count;
  const std::size_t words = (n_rows + 63) / 64;

  // Every rank holds all labels, so the root statistics need no communication.
  Tree tree(n_classes);
  tree.add_node(-1, 0, local_class_counts(labels, n_rows, n_classes).data());

  std::vector<int> frontier = root_frontier(tree, opt.growth);
  std::vector<hist_t> cur, prev;
  std::vector<int> prev_slot(tree.size(), -1);
  std::vector<Route> routes, owned;
  std::vector<std::uint64_t> bits(words);

  while (!frontier.empty()) {
    LevelStats lv;
    lv.depth = tree.node(frontier[0]).depth;
    Timer t;

    const LevelPlan plan = plan_level(tree, frontier, prev_slot, opt.subtract);
    const int S = static_cast<int>(plan.nodes.size());
    lv.frontier = S;
    lv.built = plan.n_built;
    cur.resize(S * slot_size);
    backend.build(plan.slot_of_node, plan.n_built, cur.data());
    derive_histograms(cur.data(), prev.data(), plan, slot_size);
    lv.hist += t.lap();

    const std::vector<SplitInfo> mine =
        evaluate_units(cur.data(), 0, std::int64_t(S) * feat_count, feat_count, feat_begin, plan,
                       tree, mapper, n_classes, opt.growth.min_samples_leaf);
    lv.split += t.lap();
    const std::vector<SplitInfo> best = allgather_best(comm, mine, lv);
    lv.comm += t.lap();

    std::swap(prev, cur);
    frontier = grow(tree, plan, best, mapper, opt.growth, routes);
    prev_slot.assign(tree.size(), -1);
    for (int k = 0; k < S; ++k) prev_slot[plan.nodes[k]] = k;
    lv.split += t.lap();

    // Only the owner of a node's split feature can evaluate the split on the
    // rows; it publishes the outcome as one bit per row.
    owned = routes;
    for (Route& r : owned) {
      const bool mine_f = r.feature >= feat_begin && r.feature < feat_begin + feat_count;
      r.feature = mine_f ? r.feature - feat_begin : -1;
    }
    backend.mark_left(owned, bits.data());
    lv.route += t.lap();

    MPI_Allreduce(MPI_IN_PLACE, bits.data(), checked_count(words), MPI_UINT64_T, MPI_BOR, comm);
    lv.comm_bytes += words * sizeof(std::uint64_t);
    lv.comm += t.lap();

    backend.route_bitmap(routes, bits.data());
    lv.route += t.lap();
    stats.add(lv);
  }
  return tree;
}

}  // namespace dt
