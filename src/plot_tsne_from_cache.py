import os
import argparse
import json
import numpy as np
import torch

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_cache(cache_dir: str, prefix: str):
    Z = torch.load(os.path.join(cache_dir, f"{prefix}_Z.pt"), map_location="cpu")
    gmm_labels = np.load(os.path.join(cache_dir, f"{prefix}_gmm_labels.npy"))
    core_mask = np.load(os.path.join(cache_dir, f"{prefix}_core_mask.npy")).astype(bool)

    y_path = os.path.join(cache_dir, f"{prefix}_y.npy")
    r_path = os.path.join(cache_dir, f"{prefix}_radii.npy")
    y = np.load(y_path) if os.path.exists(y_path) else None
    radii = np.load(r_path) if os.path.exists(r_path) else None

    meta_path = os.path.join(cache_dir, "meta.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return Z, gmm_labels, core_mask, y, radii, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True,
                    help="e.g., artifacts/tsne_cache/cora/aron_desc/seed0/preprune_0.10")
    ap.add_argument("--prefix", type=str, default="Z0", choices=["Z0", "final"])
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--lr", type=float, default=200.0)
    ap.add_argument("--n_iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_points", type=int, default=4000,
                    help="downsample if N is huge; <=0 means use all points")
    ap.add_argument("--out", type=str, default=None,
                    help="output png path; default inside cache_dir")

    # Optional: show cluster ids in legend (top-k largest only)
    ap.add_argument("--legend_clusters", action="store_true",
                    help="Show top-k cluster ids in legend (can be crowded)")
    ap.add_argument("--legend_topk", type=int, default=12,
                    help="How many largest clusters to show in legend when --legend_clusters is set")

    args = ap.parse_args()

    Z, gmm_labels, core_mask, y, radii, meta = load_cache(args.cache_dir, args.prefix)

    X = Z.detach().cpu().numpy().astype(np.float32)
    N = X.shape[0]

    # Optional downsample for speed
    idx = np.arange(N)
    if args.max_points > 0 and N > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(N, size=args.max_points, replace=False)
        X = X[idx]
        gmm_labels = gmm_labels[idx]
        core_mask = core_mask[idx]
        if y is not None:
            y = y[idx]
        if radii is not None:
            radii = radii[idx]

    # TSNE
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate=args.lr,
        n_iter=args.n_iter,
        init="pca",
        random_state=args.seed,
        verbose=1,
    )
    XY = tsne.fit_transform(X)

    # Plot masks
    is_noise = (gmm_labels == -1)
    is_core = core_mask & (~is_noise)
    is_noncore = (~core_mask) & (~is_noise)

    fig = plt.figure(figsize=(10, 8))

    # Keep handles for legend
    h_noise = None
    sc_noncore = None
    sc_core = None

    # Noise
    if np.any(is_noise):
        h_noise = plt.scatter(
            XY[is_noise, 0], XY[is_noise, 1],
            s=6, alpha=0.25, marker="x", label="noise (-1)"
        )

    # Non-core
    if np.any(is_noncore):
        sc_noncore = plt.scatter(
            XY[is_noncore, 0], XY[is_noncore, 1],
            s=8, alpha=0.35, c=gmm_labels[is_noncore],
            label="non-core", cmap="tab20"
        )

    # Core
    if np.any(is_core):
        sc_core = plt.scatter(
            XY[is_core, 0], XY[is_core, 1],
            s=14, alpha=0.8, c=gmm_labels[is_core],
            label="core (c0p)", cmap="tab20",
            edgecolors="k", linewidths=0.2
        )

    # Title
    title = f"TSNE ({args.prefix})"
    if meta is not None:
        ds = meta.get("dataset", "")
        ver = meta.get("ver", "")
        ppf = meta.get("pre_prune_frac", None)
        stage = meta.get("stage", "")
        title += f" | {ds} {ver} preprune={ppf} stage={stage}"
        hit3 = meta.get("test_hit3", None)
        hit10 = meta.get("test_hit10", None)
        if hit3 is not None or hit10 is not None:
            title += f" | hit@3={hit3} hit@10={hit10}"

    plt.title(title)
    plt.axis("off")

    # Legend (only core/noncore/noise) + Colorbar for cluster ids
    legend_handles = []
    legend_labels = []
    if h_noise is not None:
        legend_handles.append(h_noise)
        legend_labels.append("noise (-1)")
    if sc_noncore is not None:
        legend_handles.append(sc_noncore)
        legend_labels.append("non-core")
    if sc_core is not None:
        legend_handles.append(sc_core)
        legend_labels.append("core (c0p)")

    if legend_handles:
        plt.legend(legend_handles, legend_labels, loc="best")

    # Colorbar for cluster ids (exclude noise)
    # Use whichever scatter exists (core preferred, else noncore)
    sc_for_cbar = sc_core if sc_core is not None else sc_noncore
    if sc_for_cbar is not None and np.any(~is_noise):
        cbar = plt.colorbar(sc_for_cbar, fraction=0.046, pad=0.04)
        cbar.set_label("GMM cluster id")

    # Optional: also list top-k cluster ids in legend (crowded but sometimes useful)
    if args.legend_clusters and np.any(~is_noise):
        non_noise = gmm_labels[~is_noise]
        uniq, cnt = np.unique(non_noise, return_counts=True)
        order = np.argsort(-cnt)
        uniq = uniq[order][:args.legend_topk]

        handles = []
        labels_txt = []
        cmap = plt.cm.get_cmap("tab20")
        for c in uniq:
            handles.append(plt.Line2D(
                [0], [0],
                marker='o', linestyle='',
                markersize=6,
                markerfacecolor=cmap(int(c) % 20),
                markeredgecolor='none'
            ))
            labels_txt.append(f"cluster {int(c)} (n={int((gmm_labels==c).sum())})")

        plt.legend(handles, labels_txt, loc="upper right", title="Top clusters")

    if args.out is None:
        args.out = os.path.join(
            args.cache_dir,
            f"tsne_{args.prefix}_p{int(args.perplexity)}_seed{args.seed}.png"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[TSNE] saved plot -> {args.out}")


if __name__ == "__main__":
    main()
