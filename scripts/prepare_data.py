"""Convert a CSV classification dataset (e.g. UCI HIGGS) into train/test DTREEBIN files.

Example (HIGGS: label in column 0, the last 500,000 rows are the official test set):
    python scripts/prepare_data.py data/HIGGS.csv.gz --train data/higgs_train.bin \
        --test data/higgs_test.bin --test-rows 500000

The CSV is streamed in chunks, so memory use stays small even for the 11M-row
HIGGS file. Labels are mapped to 0..C-1 in sorted order of their values.
"""

import argparse
import time

import numpy as np
import pandas as pd

import dtbin


def scan(path, label_col, chunk):
    """First pass: row count, number of columns and distinct labels."""
    n_rows, n_cols, labels = 0, None, set()
    for df in pd.read_csv(path, header=None, chunksize=chunk, dtype=np.float32):
        n_cols = df.shape[1]
        n_rows += len(df)
        labels.update(np.unique(df.iloc[:, label_col]).tolist())
    return n_rows, n_cols, sorted(labels)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="input CSV (optionally .gz) without a header line")
    p.add_argument("--train", required=True, help="output training file")
    p.add_argument("--test", help="output test file (the last --test-rows rows)")
    p.add_argument("--test-rows", type=int, default=0)
    p.add_argument("--label-col", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="use only the first N training rows")
    p.add_argument("--chunk", type=int, default=1_000_000)
    args = p.parse_args()

    start = time.time()
    n_rows, n_cols, classes = scan(args.csv, args.label_col, args.chunk)
    n_features = n_cols - 1
    if len(classes) > 16:
        raise SystemExit(f"{len(classes)} distinct labels; at most 16 classes are supported")
    print(f"{n_rows} rows, {n_features} features, classes {classes} ({time.time() - start:.1f}s)")

    n_test = args.test_rows if args.test else 0
    n_train = n_rows - n_test
    if args.limit:
        n_train = min(n_train, args.limit)
    lookup = {v: i for i, v in enumerate(classes)}
    feature_cols = [c for c in range(n_cols) if c != args.label_col]

    train = dtbin.create(args.train, n_train, n_features, len(classes))
    test = dtbin.create(args.test, n_test, n_features, len(classes)) if n_test else None

    row = 0
    for df in pd.read_csv(args.csv, header=None, chunksize=args.chunk, dtype=np.float32):
        y = np.array([lookup[v] for v in df.iloc[:, args.label_col].to_numpy()], dtype=np.uint8)
        x = df.iloc[:, feature_cols].to_numpy(dtype=np.float32).T
        begin, end = row, row + len(df)
        row = end

        # Rows [0, n_train) go to the training file, [n_rows - n_test, n_rows) to the test file.
        a, b = begin, min(end, n_train)
        if a < b:
            train[0][a:b] = y[a - begin:b - begin]
            train[1][:, a:b] = x[:, a - begin:b - begin]
        if test is not None:
            t0 = n_rows - n_test
            a, b = max(begin, t0), end
            if a < b:
                test[0][a - t0:b - t0] = y[a - begin:b - begin]
                test[1][:, a - t0:b - t0] = x[:, a - begin:b - begin]
        print(f"  {row}/{n_rows} rows ({time.time() - start:.1f}s)", flush=True)

    for arrays in (train, test):
        if arrays is not None:
            arrays[0].flush()
            arrays[1].flush()
    print(f"wrote {args.train} ({n_train} rows)" + (f" and {args.test} ({n_test} rows)" if n_test else ""))


if __name__ == "__main__":
    main()
