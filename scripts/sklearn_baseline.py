"""Train scikit-learn's DecisionTreeClassifier on the same files as an external reference.

scikit-learn searches exact thresholds on raw values while our trainers search
the bin edges, so the trees differ slightly; the accuracies should be close.

Example:
    python scripts/sklearn_baseline.py data/higgs_train.bin data/higgs_test.bin --max-depth 10
"""

import argparse
import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

import dtbin


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("train")
    p.add_argument("test")
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--max-train-rows", type=int, default=0, help="subsample to save time/memory")
    args = p.parse_args()

    _, y, x = dtbin.load(args.train)
    n = len(y) if not args.max_train_rows else min(len(y), args.max_train_rows)
    x_train = np.ascontiguousarray(x[:, :n].T)
    y_train = np.asarray(y[:n])
    _, y_test, x_test = dtbin.load(args.test)
    x_test = np.ascontiguousarray(x_test.T)

    clf = DecisionTreeClassifier(criterion="gini", max_depth=args.max_depth,
                                 min_samples_leaf=args.min_samples_leaf, random_state=0)
    start = time.time()
    clf.fit(x_train, y_train)
    elapsed = time.time() - start

    print(f"scikit-learn DecisionTreeClassifier, {n} training rows, max_depth={args.max_depth}")
    print(f"fit time {elapsed:.2f}s, nodes {clf.tree_.node_count}, leaves {clf.get_n_leaves()}")
    print(f"train accuracy={clf.score(x_train, y_train):.6f} test accuracy={clf.score(x_test, np.asarray(y_test)):.6f}")


if __name__ == "__main__":
    main()
