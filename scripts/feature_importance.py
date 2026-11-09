"""Gini feature importance of a tree written by --dump-tree.

importance(f) = sum over nodes split on f of  n * gini(node) - n_L * gini(L) - n_R * gini(R),
normalised to sum to one (the same definition scikit-learn uses).

Example:
    python scripts/feature_importance.py results/higgs_tree.txt --names higgs
"""

import argparse

HIGGS_NAMES = [
    "lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_btag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_btag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_btag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_btag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
]


def parse(path):
    nodes = []
    with open(path) as f:
        n_classes = int(f.readline().split()[2])
        for line in f:
            tok = line.split()
            node = {"n": int(tok[5]), "counts": [int(v) for v in tok[7:7 + n_classes]]}
            rest = tok[7 + n_classes:]
            if rest[0] == "split":
                node.update(feature=int(rest[2]), threshold=float(rest[6]),
                            left=int(rest[8]), right=int(rest[10]))
            nodes.append(node)
    return nodes


def weighted_gini(node):
    n = node["n"]
    return n - sum(c * c for c in node["counts"]) / n if n else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tree")
    p.add_argument("--names", choices=["higgs"], help="use known feature names")
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    nodes = parse(args.tree)
    importance, uses = {}, {}
    for nd in nodes:
        if "feature" not in nd:
            continue
        gain = weighted_gini(nd) - weighted_gini(nodes[nd["left"]]) - weighted_gini(nodes[nd["right"]])
        importance[nd["feature"]] = importance.get(nd["feature"], 0.0) + gain
        uses[nd["feature"]] = uses.get(nd["feature"], 0) + 1

    total = sum(importance.values()) or 1.0
    names = HIGGS_NAMES if args.names == "higgs" else None
    print(f"{'rank':>4}  {'feature':<26} {'importance':>10} {'splits':>7}")
    ranked = sorted(importance.items(), key=lambda kv: -kv[1])[:args.top]
    for i, (f, v) in enumerate(ranked, 1):
        label = f"{f} ({names[f]})" if names and f < len(names) else str(f)
        print(f"{i:>4}  {label:<26} {v / total:>10.4f} {uses[f]:>7}")


if __name__ == "__main__":
    main()
