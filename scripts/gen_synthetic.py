"""Generate a synthetic classification dataset in DTREEBIN format.

Used for weak-scaling runs, where the problem size must grow with the number
of ranks, and for quick local tests. Features are standard normal; the label
is a noisy non-linear function of a few of them, so the tree has real
structure to learn at every depth. Generation is chunked and seeded per chunk,
so the same arguments always produce the same file.

Example:
    python scripts/gen_synthetic.py data/synth_4M.bin --rows 4000000 --features 32
"""

import argparse

import numpy as np

import dtbin


def labels_for(x, n_classes, cuts):
    z = (x[:, 0] - 0.7 * x[:, 1] + 0.5 * x[:, 2] * x[:, 3]
         + np.sin(2.0 * x[:, 4]) + 0.8 * (x[:, 5] > 0.5) - 0.6 * np.abs(x[:, 6]))
    return np.searchsorted(cuts, z).astype(np.uint8) if n_classes > 2 else (z > cuts[0]).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--features", type=int, default=32)
    p.add_argument("--classes", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.5, help="std of the label noise")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--chunk", type=int, default=1_000_000)
    args = p.parse_args()
    if args.features < 7:
        raise SystemExit("--features must be at least 7")
    if not 2 <= args.classes <= 16:
        raise SystemExit("--classes must be in [2, 16]")

    # Class boundaries: quantiles of the noise-free score on a pilot sample.
    pilot = np.random.default_rng(args.seed - 1).standard_normal((200_000, args.features))
    z = (pilot[:, 0] - 0.7 * pilot[:, 1] + 0.5 * pilot[:, 2] * pilot[:, 3]
         + np.sin(2.0 * pilot[:, 4]) + 0.8 * (pilot[:, 5] > 0.5) - 0.6 * np.abs(pilot[:, 6]))
    cuts = np.quantile(z, np.arange(1, args.classes) / args.classes)

    labels, values = dtbin.create(args.out, args.rows, args.features, args.classes)
    for k, begin in enumerate(range(0, args.rows, args.chunk)):
        end = min(begin + args.chunk, args.rows)
        rng = np.random.default_rng([args.seed, k])
        x = rng.standard_normal((end - begin, args.features), dtype=np.float32)
        noisy = x.copy()
        noisy[:, :7] += args.noise * rng.standard_normal((end - begin, 7), dtype=np.float32)
        labels[begin:end] = labels_for(noisy, args.classes, cuts)
        values[:, begin:end] = x.T
    labels.flush()
    values.flush()
    print(f"wrote {args.out}: {args.rows} rows, {args.features} features, {args.classes} classes")


if __name__ == "__main__":
    main()
