#pragma once

#include <array>
#include <cstdint>
#include <ostream>
#include <vector>

#include "dt/common.hpp"
#include "dt/dataset.hpp"

namespace dt {

struct TreeNode {
  std::int32_t parent = -1;
  std::int32_t left = -1;
  std::int32_t right = -1;
  std::int32_t depth = 0;
  std::int32_t feature = -1;  // global feature index, -1 for a leaf
  std::int32_t bin = -1;      // training rows with bin <= this went left
  float threshold = 0.0f;     // equivalent raw-value test: x <= threshold
  std::uint64_t n = 0;
  std::array<std::uint64_t, kMaxClasses> counts{};

  bool is_leaf() const { return feature < 0; }
};

class Tree {
 public:
  explicit Tree(int n_classes = 2) : n_classes_(n_classes) {}

  int n_classes() const { return n_classes_; }
  int size() const { return static_cast<int>(nodes_.size()); }
  const TreeNode& node(int id) const { return nodes_[id]; }

  int add_node(int parent, int depth, const std::uint64_t* counts);
  void set_split(int id, int feature, int bin, float threshold, int left, int right);

  int majority(int id) const;

  // Classifies one row; x(f) must return the raw value of feature f.
  template <class Row>
  int predict(Row x) const {
    int id = 0;
    while (!nodes_[id].is_leaf()) {
      const TreeNode& nd = nodes_[id];
      id = x(nd.feature) <= nd.threshold ? nd.left : nd.right;
    }
    return majority(id);
  }

  // Number of correctly classified rows of a block holding all features.
  std::uint64_t correct_on_block(const RawBlock& block) const;

  // Correctly classified training rows, computed exactly from leaf counts.
  std::uint64_t training_correct() const;

  int depth() const;
  int n_leaves() const;

  // Canonical text form: nodes are renumbered breadth-first, so two trees are
  // equal iff their outputs are byte-identical, regardless of creation order.
  void write(std::ostream& out) const;

 private:
  int n_classes_;
  std::vector<TreeNode> nodes_;
};

}  // namespace dt
