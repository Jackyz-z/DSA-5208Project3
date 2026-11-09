"""Plot scaling results from the CSV files written by dt_serial / dt_mpi.

Produces, for the runs matching --tag:
  <prefix>_time.png       training time vs. cores, one line per configuration
  <prefix>_speedup.png    speedup vs. cores (baseline: serial run if present,
                          otherwise the smallest run of each configuration)
  <prefix>_efficiency.png parallel efficiency
  <prefix>_breakdown.png  stacked hist / comm / split / route time per run
With --levels, also <prefix>_levels.png: per-level time and message size.

Example:
    python scripts/plot_results.py results/results.csv --tag strong_cpu --out figures/strong_cpu
    python scripts/plot_results.py results/results.csv --tag weak --weak --out figures/weak
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PHASES = ["t_hist", "t_comm", "t_split", "t_route"]


def config_label(row):
    if row["mode"] == "serial":
        return "serial"
    parts = [row["mode"], row["comm"], row["backend"]]
    if row["backend"] == "cpu":
        parts.append(f"omp={row['omp']}")
    if not row["subtract"]:
        parts.append("no-subtract")
    return " / ".join(parts)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--tag", action="append", help="tag(s) to include (default: all)")
    p.add_argument("--out", default="figures/scaling", help="output file prefix")
    p.add_argument("--x", choices=["cores", "ranks", "threads"], default="cores")
    p.add_argument("--weak", action="store_true", help="weak scaling: efficiency = T1 / Tp")
    p.add_argument("--levels", help="per-level CSV from --level-csv")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.tag:
        df = df[df["tag"].isin(args.tag)]
    if df.empty:
        raise SystemExit("no matching runs")
    # Repeated runs of the same configuration: keep the median time.
    df["cores"] = df["ranks"] * df["threads"]
    df["config"] = df.apply(config_label, axis=1)
    keys = ["config", "ranks", "threads", "cores", "n_train"]
    agg = df.groupby(keys, as_index=False)[PHASES + ["t_train"]].median()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    serial = agg[agg["config"] == "serial"]
    serial_time = serial["t_train"].min() if not serial.empty else None
    parallel = agg[agg["config"] != "serial"].sort_values(args.x)

    fig_t, ax_t = plt.subplots(figsize=(6, 4))
    fig_s, ax_s = plt.subplots(figsize=(6, 4))
    fig_e, ax_e = plt.subplots(figsize=(6, 4))
    for config, g in parallel.groupby("config"):
        x = g[args.x].to_numpy()
        t = g["t_train"].to_numpy()
        base = g.iloc[0]
        if args.weak:
            speedup = base["t_train"] / t * (x / x[0])
            eff = base["t_train"] / t
        elif serial_time is not None:
            speedup = serial_time / t
            eff = speedup / x
        else:
            speedup = base["t_train"] / t * x[0]
            eff = speedup / x
        ax_t.plot(x, t, "o-", label=config)
        ax_s.plot(x, speedup, "o-", label=config)
        ax_e.plot(x, eff, "o-", label=config)

    xs = sorted(parallel[args.x].unique())
    ax_s.plot(xs, xs, "k--", lw=1, label="ideal")
    ax_e.axhline(1.0, color="k", ls="--", lw=1)
    if serial_time is not None and not args.weak:
        ax_t.axhline(serial_time, color="gray", ls=":", label="serial reference")
    for ax, ylabel in ((ax_t, "training time [s]"), (ax_s, "speedup"), (ax_e, "parallel efficiency")):
        ax.set_xlabel(args.x)
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    ax_t.set_yscale("log")
    ax_e.set_ylim(0, 1.1)
    save(fig_t, f"{args.out}_time.png")
    save(fig_s, f"{args.out}_speedup.png")
    save(fig_e, f"{args.out}_efficiency.png")

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(parallel)), 4))
    labels = [f"{r.config}\n{r.ranks}x{r.threads}" for r in parallel.itertuples()]
    bottom = None
    for phase in PHASES:
        vals = parallel[phase].to_numpy()
        ax.bar(range(len(vals)), vals, bottom=bottom, label=phase[2:])
        bottom = vals if bottom is None else bottom + vals
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("time [s] (max over ranks)")
    ax.legend()
    save(fig, f"{args.out}_breakdown.png")

    if args.levels:
        lv = pd.read_csv(args.levels)
        if args.tag:
            lv = lv[lv["tag"].isin(args.tag)]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        for (mode, comm, ranks, threads), g in lv.groupby(["mode", "comm", "ranks", "threads"]):
            g = g.groupby("level", as_index=False).median(numeric_only=True)
            name = f"{mode}/{comm} {ranks}x{threads}"
            a1.plot(g["level"], g["t_hist"] + g["t_comm"] + g["t_split"] + g["t_route"], "o-", label=name)
            a2.plot(g["level"], g["comm_bytes"] / 1e6, "o-", label=name)
        a1.set_ylabel("time per level [s]")
        a2.set_ylabel("bytes sent per rank [MB]")
        for a in (a1, a2):
            a.set_xlabel("tree level")
            a.set_yscale("log")
            a.grid(True, alpha=0.3)
            a.legend(fontsize=7)
        save(fig, f"{args.out}_levels.png")


if __name__ == "__main__":
    main()
