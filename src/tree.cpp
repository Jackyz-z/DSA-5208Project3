#include "dt/tree.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace dt {

int Tree::add_node(int parent, int depth, const std::uint64_t* counts) {
  TreeNode nd;
  nd.parent = parent;
  nd.depth = depth;
  for (int c = 0; c < n_classes_; ++c) {
    nd.counts[c] = counts[c];
    nd.n += counts[c];
  }
  nodes_.push_back(nd);
  return size() - 1;
}

void Tree::set_split(int id, int feature, int bin, float threshold, int left, int right) {
  TreeNode& nd = nodes_[id];
  nd.feature = feature;
  nd.bin = bin;
  nd.threshold = threshold;
  nd.left = left;
  nd.right = right;
}

int Tree::majority(int id) const {
  const TreeNode& nd = nodes_[id];
  int best = 0;
  for (int c = 1; c < n_classes_; ++c) {
    if (nd.counts[c] > nd.counts[best]) best = c;
  }
  return best;
}

std::uint64_t Tree::correct_on_block(const RawBlock& block) const {
  if (block.feat_begin != 0) throw std::runtime_error("prediction needs all features");
  const std::int64_t n = static_cast<std::int64_t>(block.n_rows);
  std::uint64_t correct = 0;
#pragma omp parallel for schedule(static) reduction(+ : correct)
  for (std::int64_t i = 0; i < n; ++i) {
    const int pred = predict([&](int f) { return block.at(f, i); });
    correct += pred == block.labels[i];
  }
  return correct;
}

std::uint64_t Tree::training_correct() const {
  std::uint64_t correct = 0;
  for (int id = 0; id < size(); ++id) {
    if (nodes_[id].is_leaf()) correct += nodes_[id].counts[majority(id)];
  }
  return correct;
}

int Tree::depth() const {
  int d = 0;
  for (const auto& nd : nodes_) d = std::max(d, nd.depth);
  return d;
}

int Tree::n_leaves() const {
  return static_cast<int>(std::count_if(nodes_.begin(), nodes_.end(),
                                        [](const TreeNode& nd) { return nd.is_leaf(); }));
}

void Tree::write(std::ostream& out) const {
  out << "tree classes " << n_classes_ << " nodes " << size() << " leaves " << n_leaves()
      << " depth " << depth() << "\n";
  if (nodes_.empty()) return;

  std::vector<int> order{0};
  std::vector<int> new_id(nodes_.size(), -1);
  new_id[0] = 0;
  for (std::size_t k = 0; k < order.size(); ++k) {
    const TreeNode& nd = nodes_[order[k]];
    if (nd.is_leaf()) continue;
    for (int child : {nd.left, nd.right}) {
      new_id[child] = static_cast<int>(order.size());
      order.push_back(child);
    }
  }

  char buf[64];
  for (std::size_t k = 0; k < order.size(); ++k) {
    const TreeNode& nd = nodes_[order[k]];
    out << "node " << k << " depth " << nd.depth << " n " << nd.n << " counts";
    for (int c = 0; c < n_classes_; ++c) out << ' ' << nd.counts[c];
    if (nd.is_leaf()) {
      out << " leaf class " << majority(order[k]) << "\n";
    } else {
      std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(nd.threshold));
      out << " split feature " << nd.feature << " bin " << nd.bin << " threshold " << buf
          << " left " << new_id[nd.left] << " right " << new_id[nd.right] << "\n";
    }
  }
}

}  // namespace dt
