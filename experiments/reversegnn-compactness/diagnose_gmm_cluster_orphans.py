#!/usr/bin/env python3
"""Check GMM label behavior and graph-orphan rates inside predicted clusters.

The GMM labels are produced from embeddings only. This script therefore treats
same-cluster graph isolation as a separate diagnostic: a node is an orphan if it
has zero neighbors inside its assigned non-noise GMM cluster on the original
dataset graph.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_CACHE_ROOT = (
    REPO_ROOT
    / "experiments"
    / "reversegnn-compactness"
    / "results"
    / "artifacts"
    / "tsne_cache"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "experiments"
    / "reversegnn-compactness"
    / "results"
    / "gmm_orphan_diagnostics"
)


def _add_src_path() -> None:
    src = str(SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)


def _majority_label(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values)
    return counts.most_common(1)[0][0]


def run_gmm_self_test() -> dict[str, str]:
    """Exercise the current src.utils.gmm_labels implementation."""
    _add_src_path()

    import torch
    from utils import gmm_labels

    rng = np.random.default_rng(0)

    center_a = np.array([1.0, 0.0, 0.0, 0.0])
    center_b = np.array([0.0, 1.0, 0.0, 0.0])
    clean_a = center_a + 0.02 * rng.normal(size=(40, 4))
    clean_b = center_b + 0.02 * rng.normal(size=(40, 4))
    clean_z = np.vstack([clean_a, clean_b])

    clean_labels = gmm_labels(
        torch.tensor(clean_z, dtype=torch.float32),
        K=2,
        tau=0.55,
        metric="cosine",
    )
    if np.any(clean_labels < 0):
        raise AssertionError("Separated cosine clusters produced noise labels.")

    label_a = _majority_label(clean_labels[:40])
    label_b = _majority_label(clean_labels[40:])
    purity = (
        int(np.sum(clean_labels[:40] == label_a))
        + int(np.sum(clean_labels[40:] == label_b))
    ) / float(clean_labels.size)
    if label_a == label_b or purity < 0.95:
        raise AssertionError(
            f"Separated cosine clusters were not recovered cleanly: purity={purity:.3f}."
        )

    raw_rng = np.random.default_rng(0)
    raw_a = np.column_stack(
        [raw_rng.normal(-1.5, 0.4, 80), raw_rng.normal(0.0, 0.4, 80)]
    )
    raw_b = np.column_stack(
        [raw_rng.normal(1.5, 0.4, 80), raw_rng.normal(0.0, 0.4, 80)]
    )
    boundary = np.array([[0.0, 0.0]])
    boundary_z = np.vstack([raw_a, raw_b, boundary])
    boundary_labels = gmm_labels(
        torch.tensor(boundary_z, dtype=torch.float32),
        K=2,
        tau=0.85,
        metric="raw",
    )
    if int(boundary_labels[-1]) != -1:
        raise AssertionError("Low-confidence boundary point was not marked as -1.")
    if np.any(boundary_labels[:-1] < 0):
        raise AssertionError("Clean raw clusters became noise in the boundary test.")

    params = inspect.signature(gmm_labels).parameters
    graph_like_params = {"adj", "adjacency", "graph", "edge_index"}
    if graph_like_params.intersection(params):
        raise AssertionError("gmm_labels unexpectedly takes graph structure as input.")

    isolated_like_z = np.vstack([clean_z, center_a[None, :]])
    isolated_like_labels = gmm_labels(
        torch.tensor(isolated_like_z, dtype=torch.float32),
        K=2,
        tau=0.55,
        metric="cosine",
    )
    if int(isolated_like_labels[-1]) == -1:
        raise AssertionError(
            "A clear embedding was marked noise; graph isolation is not an input here."
        )

    return {
        "separated_clusters": f"pass purity={purity:.3f}",
        "boundary_noise": "pass tau=0.85 metric=raw",
        "graph_isolation": "pass gmm_labels has no graph input",
    }


def _synthetic_embeddings(
    *,
    seed: int,
    n_clusters: int,
    n_per_cluster: int,
    dim: int,
    separation: float,
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.zeros((n_clusters, dim), dtype=np.float32)
    for c in range(n_clusters):
        centers[c, c % dim] = separation
        centers[c, (c * 3 + 1) % dim] += 0.25 * separation

    true_labels = np.repeat(np.arange(n_clusters, dtype=np.int64), n_per_cluster)
    z = centers[true_labels] + noise * rng.normal(size=(true_labels.size, dim))
    return z.astype(np.float32), true_labels


def _add_undirected_edge(rows: list[int], cols: list[int], i: int, j: int) -> None:
    if i == j:
        return
    rows.extend([i, j])
    cols.extend([j, i])


def _synthetic_adj(
    true_labels: np.ndarray,
    *,
    mode: str,
    mean_intra_degree: float,
    seed: int,
) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    n_nodes = int(true_labels.size)
    rows: list[int] = []
    cols: list[int] = []

    cluster_ids = sorted(int(x) for x in np.unique(true_labels))
    for cluster_id in cluster_ids:
        nodes = np.flatnonzero(true_labels == cluster_id)
        if mode == "ring":
            for pos, node in enumerate(nodes):
                _add_undirected_edge(rows, cols, int(node), int(nodes[(pos + 1) % nodes.size]))
            continue

        p_in = min(1.0, float(mean_intra_degree) / max(1, nodes.size - 1))
        for local_i in range(nodes.size):
            i = int(nodes[local_i])
            rand = rng.random(nodes.size - local_i - 1)
            hits = np.flatnonzero(rand < p_in)
            for hit in hits:
                j = int(nodes[local_i + 1 + hit])
                _add_undirected_edge(rows, cols, i, j)

    adj = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_nodes, n_nodes))
    adj.setdiag(0)
    adj.eliminate_zeros()

    return adj


def _cluster_assignment_purity(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    cp_mask = pred_labels >= 0
    if not np.any(cp_mask):
        return math.nan
    correct = 0
    for pred_cluster in sorted(int(x) for x in np.unique(pred_labels[cp_mask])):
        idx = np.flatnonzero(pred_labels == pred_cluster)
        counts = Counter(int(v) for v in true_labels[idx])
        correct += counts.most_common(1)[0][1]
    return correct / float(np.sum(cp_mask))


def _orphan_rates_for_arrays(
    adj: sp.csr_matrix,
    pred_labels: np.ndarray,
    core_mask: np.ndarray,
) -> dict[str, float]:
    adj = adj.tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj = ((adj + adj.T) > 0).astype(np.int8).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()

    global_deg = np.asarray(adj.sum(axis=1)).ravel()
    global_zero = global_deg == 0
    cp_mask = pred_labels >= 0
    c0p_mask = cp_mask & core_mask.astype(bool)

    cp_degrees: list[np.ndarray] = []
    c0p_degrees: list[np.ndarray] = []
    for cluster_id in sorted(int(x) for x in np.unique(pred_labels[cp_mask])):
        nodes = np.flatnonzero(pred_labels == cluster_id)
        intra = np.asarray(adj[nodes][:, nodes].sum(axis=1)).ravel()
        cp_degrees.append(intra)
        c0p_local = core_mask[nodes].astype(bool)
        if np.any(c0p_local):
            c0p_degrees.append(intra[c0p_local])

    cp_intra = np.concatenate(cp_degrees) if cp_degrees else np.array([], dtype=float)
    c0p_intra = np.concatenate(c0p_degrees) if c0p_degrees else np.array([], dtype=float)

    return {
        "global_zero_rate": _safe_ratio(int(np.sum(global_zero)), int(global_zero.size)),
        "cp_global_zero_rate": _safe_ratio(int(np.sum(global_zero & cp_mask)), int(np.sum(cp_mask))),
        "c0p_global_zero_rate": _safe_ratio(int(np.sum(global_zero & c0p_mask)), int(np.sum(c0p_mask))),
        "cp_intra_orphan_rate": _safe_ratio(int(np.sum(cp_intra == 0)), int(cp_intra.size)),
        "c0p_intra_orphan_rate": _safe_ratio(int(np.sum(c0p_intra == 0)), int(c0p_intra.size)),
        "cp_intra_mean": _mean(cp_intra),
        "c0p_intra_mean": _mean(c0p_intra),
    }


def run_synthetic_orphan_smoke(
    *,
    seeds: int,
    n_clusters: int,
    n_per_cluster: int,
    dim: int,
) -> list[dict[str, object]]:
    """Generate clear embedding clusters and measure predicted-cluster orphans."""
    _add_src_path()

    import torch
    from utils import gmm_labels, select_gmm_cores

    cases = [
        ("connected_ring", "ring", 2.0),
        ("er_mean8", "er", 8.0),
        ("er_mean2", "er", 2.0),
    ]
    rows: list[dict[str, object]] = []

    for case_name, graph_mode, mean_intra_degree in cases:
        expected_er_orphan = (
            0.0
            if graph_mode == "ring"
            else (1.0 - min(1.0, mean_intra_degree / max(1, n_per_cluster - 1))) ** (n_per_cluster - 1)
        )
        case_rows: list[dict[str, object]] = []
        for seed in range(seeds):
            z_np, true_labels = _synthetic_embeddings(
                seed=seed,
                n_clusters=n_clusters,
                n_per_cluster=n_per_cluster,
                dim=dim,
                separation=6.0,
                noise=0.35,
            )
            adj = _synthetic_adj(
                true_labels,
                mode=graph_mode,
                mean_intra_degree=mean_intra_degree,
                seed=10_000 + seed,
            )
            z = torch.tensor(z_np, dtype=torch.float32)
            pred_labels = gmm_labels(z, K=n_clusters, tau=0.55, metric="raw")
            deg = torch.tensor(np.asarray(adj.sum(axis=1)).ravel(), dtype=torch.float32)
            core_mask, _, _, _ = select_gmm_cores(
                z,
                pred_labels,
                deg,
                alpha=0.8,
                gamma=1.0,
                B=dim,
                normalize_cosine=False,
            )
            rates = _orphan_rates_for_arrays(
                adj,
                pred_labels,
                core_mask.detach().cpu().numpy().astype(bool),
            )
            row: dict[str, object] = {
                "case": case_name,
                "seed": seed,
                "n_clusters": n_clusters,
                "n_per_cluster": n_per_cluster,
                "mean_intra_degree_target": mean_intra_degree,
                "expected_er_orphan": expected_er_orphan,
                "gmm_noise_rate": _safe_ratio(int(np.sum(pred_labels < 0)), int(pred_labels.size)),
                "gmm_cluster_purity": _cluster_assignment_purity(pred_labels, true_labels),
            }
            row.update(rates)
            rows.append(row)
            case_rows.append(row)

        def _case_mean(name: str) -> float:
            values = np.array([float(row[name]) for row in case_rows], dtype=float)
            values = values[~np.isnan(values)]
            return _mean(values)

        print(
            "[SYNTH-ORPHAN] "
            f"{case_name}: "
            f"purity={_case_mean('gmm_cluster_purity'):.4f} "
            f"noise={_case_mean('gmm_noise_rate'):.4f} "
            f"global_zero={_case_mean('global_zero_rate'):.4f} "
            f"cp_intra_orphan={_case_mean('cp_intra_orphan_rate'):.4f} "
            f"c0p_intra_orphan={_case_mean('c0p_intra_orphan_rate'):.4f} "
            f"cp_intra_mean={_case_mean('cp_intra_mean'):.4f} "
            f"expected_er_orphan={expected_er_orphan:.4f}"
        )

    return rows


def _synthetic_plot_positions(
    true_labels: np.ndarray,
    *,
    mode: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_nodes = int(true_labels.size)
    pos = np.zeros((n_nodes, 2), dtype=np.float32)
    cluster_ids = sorted(int(x) for x in np.unique(true_labels))
    angles = np.linspace(0.0, 2.0 * np.pi, len(cluster_ids), endpoint=False)
    centers = {
        cluster_id: np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * 4.0
        for cluster_id, angle in zip(cluster_ids, angles)
    }

    for cluster_id in cluster_ids:
        nodes = np.flatnonzero(true_labels == cluster_id)
        if mode == "ring":
            local_angles = np.linspace(0.0, 2.0 * np.pi, nodes.size, endpoint=False)
            local = np.column_stack([np.cos(local_angles), np.sin(local_angles)]) * 0.95
            pos[nodes] = centers[cluster_id] + local.astype(np.float32)
        else:
            local = rng.normal(0.0, 0.55, size=(nodes.size, 2)).astype(np.float32)
            pos[nodes] = centers[cluster_id] + local
    return pos


def _predicted_cluster_intra_degree(adj: sp.csr_matrix, pred_labels: np.ndarray) -> np.ndarray:
    adj = adj.tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    degrees = np.zeros(pred_labels.shape[0], dtype=np.int64)
    cp_mask = pred_labels >= 0
    for cluster_id in sorted(int(x) for x in np.unique(pred_labels[cp_mask])):
        nodes = np.flatnonzero(pred_labels == cluster_id)
        degrees[nodes] = np.asarray(adj[nodes][:, nodes].sum(axis=1)).ravel().astype(np.int64)
    return degrees


def _embedding_2d(X: np.ndarray, *, method: str, seed: int) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D embedding matrix, got shape {X.shape}")
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(5.0, (X.shape[0] - 1) / 3.0))
        return TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=seed,
        ).fit_transform(X)
    raise ValueError("--plot-method must be tsne or pca")


def _sample_segments(
    segments: list[list[np.ndarray]],
    *,
    max_segments: int,
    seed: int,
) -> list[list[np.ndarray]]:
    if max_segments <= 0 or len(segments) <= max_segments:
        return segments
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(segments), size=max_segments, replace=False)
    return [segments[int(i)] for i in idx]


def _load_artifact_arrays(artifact_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    labels = np.load(artifact_dir / "final_gmm_labels.npy").astype(np.int64, copy=False)
    core_mask = np.load(artifact_dir / "final_core_mask.npy").astype(bool, copy=False)
    z = torch.load(artifact_dir / "final_Z.pt", map_location="cpu")
    if hasattr(z, "detach"):
        z_np = z.detach().cpu().float().numpy()
    else:
        z_np = np.asarray(z, dtype=np.float32)
    return labels, core_mask, z_np


def _normalized_binary_adj(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj = ((adj + adj.T) > 0).astype(np.int8).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def _edge_diff_stats(
    base_adj: sp.csr_matrix,
    view_adj: sp.csr_matrix,
    labels: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, int | float]:
    base = _normalized_binary_adj(base_adj)
    view = _normalized_binary_adj(view_adj)
    added = (view - base).tocsr()
    added.data = (added.data > 0).astype(np.int8)
    added.eliminate_zeros()
    removed = (base - view).tocsr()
    removed.data = (removed.data > 0).astype(np.int8)
    removed.eliminate_zeros()

    def _count_upper(mask_adj: sp.csr_matrix, predicate) -> int:
        coo = mask_adj.tocoo()
        total = 0
        for i_raw, j_raw in zip(coo.row, coo.col):
            i = int(i_raw)
            j = int(j_raw)
            if i < j and predicate(i, j):
                total += 1
        return total

    labels_np = np.asarray(labels, dtype=np.int64)
    target_np = np.asarray(target_mask, dtype=bool)
    same = lambda i, j: labels_np[i] >= 0 and labels_np[i] == labels_np[j]
    cp_cross = lambda i, j: labels_np[i] >= 0 and labels_np[j] >= 0 and labels_np[i] != labels_np[j]
    touch_target = lambda i, j: bool(target_np[i] or target_np[j])

    base_edges = int(base.nnz // 2)
    view_edges = int(view.nnz // 2)
    added_edges = int(added.nnz // 2)
    removed_edges = int(removed.nnz // 2)
    return {
        "base_edges": base_edges,
        "view_edges": view_edges,
        "added_edges": added_edges,
        "removed_edges": removed_edges,
        "same_cluster_added": _count_upper(added, same),
        "same_cluster_removed": _count_upper(removed, same),
        "cross_cluster_added": _count_upper(added, cp_cross),
        "cross_cluster_removed": _count_upper(removed, cp_cross),
        "target_touch_added": _count_upper(added, touch_target),
        "target_touch_removed": _count_upper(removed, touch_target),
        "edge_delta": view_edges - base_edges,
    }


def build_reconstructed_repair_view(
    *,
    adj: sp.csr_matrix,
    labels: np.ndarray,
    core_mask: np.ndarray,
    z_np: np.ndarray,
    add_ratio: float,
    remove_ratio: float,
    per_node_cap: float,
    pull_strength: float,
    add_degree_target: int,
    add_degree_target_scope: str,
    add_degree_target_nodes: str,
    require_c0p_endpoint: bool,
    require_c0p_noncompact_endpoint: bool,
    guarantee_degree_target: bool,
) -> tuple[sp.csr_matrix, dict[str, int | float]]:
    """Reconstruct a representative target-degree repair view from saved artifacts.

    This is not the historical temporary decoder view unless decoder weights were saved.
    It uses the same repair constraints, with pulled-dot scores as the edge-confidence
    tie-breaker.
    """
    _add_src_path()

    import torch
    from aron_train_edit_decoder import build_decoded_augmented_graph, direct_pull_latent_per_cluster

    adj = _normalized_binary_adj(adj)
    z = torch.tensor(z_np, dtype=torch.float32)
    cp_mask = labels >= 0
    core_t = torch.tensor(core_mask.astype(bool), dtype=torch.bool)
    cp_t = torch.tensor(cp_mask.astype(bool), dtype=torch.bool)

    pull_mask = cp_t
    z_pull = direct_pull_latent_per_cluster(
        z,
        labels,
        pull_mask,
        pull_strength=float(pull_strength),
    )
    decoded_scores = torch.sigmoid(z_pull @ z_pull.t())
    dense = torch.tensor(adj.toarray(), dtype=torch.float32)
    dense.fill_diagonal_(1.0)
    e0 = int(adj.nnz // 2)
    decoded_bound_eff = None if float(per_node_cap) <= 0.0 else float(per_node_cap)
    g, added, removed = build_decoded_augmented_graph(
        decoded_scores,
        dense,
        labels,
        core_t,
        E0=e0,
        add_ratio=float(add_ratio),
        remove_ratio=float(remove_ratio),
        per_node_cap_frac=decoded_bound_eff,
        add_degree_target=int(add_degree_target),
        add_degree_target_scope=add_degree_target_scope,
        add_degree_target_nodes=add_degree_target_nodes,
        guarantee_degree_target=bool(guarantee_degree_target),
        degree_floor=0,
        same_cluster_only=True,
        require_c0p_endpoint=bool(require_c0p_endpoint),
        require_both_c0p=False,
        require_c0p_noncompact_endpoint=bool(require_c0p_noncompact_endpoint),
    )
    view = sp.csr_matrix((g.detach().cpu().numpy() > 0).astype(np.int8))
    view.setdiag(0)
    view.eliminate_zeros()
    view = _normalized_binary_adj(view)
    stats = _edge_diff_stats(adj, view, labels, cp_mask)
    stats["builder_added"] = int(added)
    stats["builder_removed"] = int(removed)
    return view, stats


def plot_real_orphan_artifact(
    *,
    dataset: str,
    variant: str,
    seed: int,
    artifact_dir: Path,
    adj: sp.csr_matrix,
    out_dir: Path,
    tag: str,
    method: str,
    max_edge_lines: int,
    max_cross_edge_lines: int,
) -> Path:
    """Plot real cached GMM clusters and original-graph intra-cluster orphans."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    labels, core_mask, z_np = _load_artifact_arrays(artifact_dir)

    if labels.shape[0] != adj.shape[0]:
        raise ValueError(f"{artifact_dir}: label count {labels.shape[0]} != graph nodes {adj.shape[0]}")

    adj = _normalized_binary_adj(adj)

    pos = _embedding_2d(z_np, method=method, seed=seed)
    intra_degree = _predicted_cluster_intra_degree(adj, labels)
    global_degree = np.asarray(adj.sum(axis=1)).ravel()
    cp_mask = labels >= 0
    c0p_mask = cp_mask & core_mask
    cp_orphan = cp_mask & (intra_degree == 0)
    c0p_orphan = c0p_mask & (intra_degree == 0)
    global_orphan = global_degree == 0

    rates = _orphan_rates_for_arrays(adj, labels, core_mask)
    cluster_ids = sorted(int(x) for x in np.unique(labels[cp_mask]))
    palette = plt.get_cmap("tab20", max(1, len(cluster_ids)))
    cluster_to_color = {
        cluster_id: palette(i % palette.N)
        for i, cluster_id in enumerate(cluster_ids)
    }
    node_colors = [
        cluster_to_color.get(int(label), (0.75, 0.75, 0.75, 0.45))
        for label in labels
    ]

    coo = adj.tocoo()
    same_segments: list[list[np.ndarray]] = []
    orphan_cross_segments: list[list[np.ndarray]] = []
    for i_raw, j_raw in zip(coo.row, coo.col):
        i = int(i_raw)
        j = int(j_raw)
        if i >= j:
            continue
        same_cluster = labels[i] >= 0 and labels[i] == labels[j]
        segment = [pos[i], pos[j]]
        if same_cluster:
            same_segments.append(segment)
        elif cp_orphan[i] or cp_orphan[j]:
            orphan_cross_segments.append(segment)

    same_segments = _sample_segments(
        same_segments,
        max_segments=max_edge_lines,
        seed=seed,
    )
    orphan_cross_segments = _sample_segments(
        orphan_cross_segments,
        max_segments=max_cross_edge_lines,
        seed=seed + 997,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    if same_segments:
        ax.add_collection(
            LineCollection(
                same_segments,
                colors="#B8BEC6",
                linewidths=0.22,
                alpha=0.20,
                zorder=1,
            )
        )
    if orphan_cross_segments:
        ax.add_collection(
            LineCollection(
                orphan_cross_segments,
                colors="#E4572E",
                linewidths=0.32,
                alpha=0.26,
                zorder=2,
            )
        )

    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=8,
        c=node_colors,
        alpha=0.78,
        edgecolors="none",
        zorder=3,
    )
    if np.any(cp_orphan):
        ax.scatter(
            pos[cp_orphan, 0],
            pos[cp_orphan, 1],
            s=36,
            facecolors="none",
            edgecolors="#D62728",
            linewidths=0.70,
            zorder=4,
        )
    if np.any(c0p_orphan):
        ax.scatter(
            pos[c0p_orphan, 0],
            pos[c0p_orphan, 1],
            s=48,
            marker="s",
            facecolors="none",
            edgecolors="#111111",
            linewidths=0.65,
            zorder=5,
        )
    if np.any(global_orphan):
        ax.scatter(
            pos[global_orphan, 0],
            pos[global_orphan, 1],
            s=52,
            marker="x",
            c="#111111",
            linewidths=0.80,
            zorder=6,
        )

    title = (
        f"{dataset} {variant} seed{seed} ({method.upper()})\n"
        f"CP intra-orphan={100.0 * rates['cp_intra_orphan_rate']:.2f}% "
        f"({int(np.sum(cp_orphan))}/{int(np.sum(cp_mask))}) | "
        f"C0p intra-orphan={100.0 * rates['c0p_intra_orphan_rate']:.2f}% "
        f"({int(np.sum(c0p_orphan))}/{int(np.sum(c0p_mask))}) | "
        f"global degree-0={100.0 * rates['global_zero_rate']:.2f}%"
    )
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    ax.text(
        0.01,
        0.02,
        (
            "color: GMM cluster\n"
            "red circle: CP intra-orphan\n"
            "black square: C0p intra-orphan\n"
            "black x: global degree 0\n"
            "gray line: sampled same-cluster edge\n"
            "orange line: sampled cross-cluster edge touching an orphan"
        ),
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        va="bottom",
        ha="left",
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"real_orphan_{dataset}_{variant}_seed{seed}_{method}_{tag}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[ORPHAN-PLOT] wrote {_relative_to_repo(out_path)}")
    return out_path


def plot_repair_comparison_artifact(
    *,
    dataset: str,
    variant: str,
    seed: int,
    artifact_dir: Path,
    adj: sp.csr_matrix,
    out_dir: Path,
    tag: str,
    method: str,
    max_edge_lines: int,
    max_cross_edge_lines: int,
    max_added_edge_lines: int,
    repair_add_ratio: float,
    repair_remove_ratio: float,
    repair_per_node_cap: float,
    repair_pull_strength: float,
    save_repair_adj: bool,
) -> dict[str, object]:
    """Plot original graph vs reconstructed target-1 repair view."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    labels, core_mask, z_np = _load_artifact_arrays(artifact_dir)
    if labels.shape[0] != adj.shape[0]:
        raise ValueError(f"{artifact_dir}: label count {labels.shape[0]} != graph nodes {adj.shape[0]}")

    adj_before = _normalized_binary_adj(adj)
    adj_after, diff = build_reconstructed_repair_view(
        adj=adj_before,
        labels=labels,
        core_mask=core_mask,
        z_np=z_np,
        add_ratio=repair_add_ratio,
        remove_ratio=repair_remove_ratio,
        per_node_cap=repair_per_node_cap,
        pull_strength=repair_pull_strength,
        add_degree_target=1,
        add_degree_target_scope="intra_cluster",
        add_degree_target_nodes="cp",
        require_c0p_endpoint=True,
        require_c0p_noncompact_endpoint=True,
        guarantee_degree_target=True,
    )
    if save_repair_adj:
        adj_path = out_dir / f"repair_reconstructed_adj_{dataset}_{variant}_seed{seed}_{tag}.npz"
        adj_path.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(adj_path, adj_after)
        print(f"[REPAIR-PLOT] wrote {_relative_to_repo(adj_path)}")

    pos = _embedding_2d(z_np, method=method, seed=seed)
    cp_mask = labels >= 0
    c0p_mask = cp_mask & core_mask
    before_rates = _orphan_rates_for_arrays(adj_before, labels, core_mask)
    after_rates = _orphan_rates_for_arrays(adj_after, labels, core_mask)
    before_intra = _predicted_cluster_intra_degree(adj_before, labels)
    after_intra = _predicted_cluster_intra_degree(adj_after, labels)
    before_global = np.asarray(adj_before.sum(axis=1)).ravel()
    after_global = np.asarray(adj_after.sum(axis=1)).ravel()

    cluster_ids = sorted(int(x) for x in np.unique(labels[cp_mask]))
    palette = plt.get_cmap("tab20", max(1, len(cluster_ids)))
    cluster_to_color = {
        cluster_id: palette(i % palette.N)
        for i, cluster_id in enumerate(cluster_ids)
    }
    node_colors = [
        cluster_to_color.get(int(label), (0.75, 0.75, 0.75, 0.45))
        for label in labels
    ]

    added_adj = (adj_after - adj_before).tocsr()
    added_adj.data = (added_adj.data > 0).astype(np.int8)
    added_adj.eliminate_zeros()

    def _segments_for(
        graph_adj: sp.csr_matrix,
        cp_orphan: np.ndarray,
        *,
        seed_offset: int,
    ) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
        coo = graph_adj.tocoo()
        same_segments: list[list[np.ndarray]] = []
        orphan_cross_segments: list[list[np.ndarray]] = []
        for i_raw, j_raw in zip(coo.row, coo.col):
            i = int(i_raw)
            j = int(j_raw)
            if i >= j:
                continue
            same_cluster = labels[i] >= 0 and labels[i] == labels[j]
            segment = [pos[i], pos[j]]
            if same_cluster:
                same_segments.append(segment)
            elif cp_orphan[i] or cp_orphan[j]:
                orphan_cross_segments.append(segment)
        return (
            _sample_segments(same_segments, max_segments=max_edge_lines, seed=seed + seed_offset),
            _sample_segments(orphan_cross_segments, max_segments=max_cross_edge_lines, seed=seed + seed_offset + 997),
        )

    added_segments: list[list[np.ndarray]] = []
    added_coo = added_adj.tocoo()
    for i_raw, j_raw in zip(added_coo.row, added_coo.col):
        i = int(i_raw)
        j = int(j_raw)
        if i < j:
            added_segments.append([pos[i], pos[j]])
    added_segments = _sample_segments(
        added_segments,
        max_segments=max_added_edge_lines,
        seed=seed + 4242,
    )

    def _draw_panel(
        ax,
        *,
        graph_adj: sp.csr_matrix,
        intra_degree: np.ndarray,
        global_degree: np.ndarray,
        rates: dict[str, float],
        panel_title: str,
        show_added: bool,
        seed_offset: int,
    ) -> None:
        cp_orphan = cp_mask & (intra_degree == 0)
        c0p_orphan = c0p_mask & (intra_degree == 0)
        global_orphan = global_degree == 0
        same_segments, orphan_cross_segments = _segments_for(
            graph_adj,
            cp_orphan,
            seed_offset=seed_offset,
        )

        if same_segments:
            ax.add_collection(
                LineCollection(
                    same_segments,
                    colors="#B8BEC6",
                    linewidths=0.20,
                    alpha=0.18,
                    zorder=1,
                )
            )
        if orphan_cross_segments:
            ax.add_collection(
                LineCollection(
                    orphan_cross_segments,
                    colors="#E4572E",
                    linewidths=0.30,
                    alpha=0.26,
                    zorder=2,
                )
            )
        if show_added and added_segments:
            ax.add_collection(
                LineCollection(
                    added_segments,
                    colors="#2CA02C",
                    linewidths=0.40,
                    alpha=0.42,
                    zorder=3,
                )
            )

        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            s=7,
            c=node_colors,
            alpha=0.76,
            edgecolors="none",
            zorder=4,
        )
        if np.any(cp_orphan):
            ax.scatter(
                pos[cp_orphan, 0],
                pos[cp_orphan, 1],
                s=30,
                facecolors="none",
                edgecolors="#D62728",
                linewidths=0.65,
                zorder=5,
            )
        if np.any(c0p_orphan):
            ax.scatter(
                pos[c0p_orphan, 0],
                pos[c0p_orphan, 1],
                s=40,
                marker="s",
                facecolors="none",
                edgecolors="#111111",
                linewidths=0.60,
                zorder=6,
            )
        if np.any(global_orphan):
            ax.scatter(
                pos[global_orphan, 0],
                pos[global_orphan, 1],
                s=44,
                marker="x",
                c="#111111",
                linewidths=0.75,
                zorder=7,
            )

        ax.set_title(
            (
                f"{panel_title}\n"
                f"CP orphan={100.0 * rates['cp_intra_orphan_rate']:.2f}% "
                f"({int(np.sum(cp_orphan))}/{int(np.sum(cp_mask))}) | "
                f"C0p={100.0 * rates['c0p_intra_orphan_rate']:.2f}% "
                f"({int(np.sum(c0p_orphan))}/{int(np.sum(c0p_mask))}) | "
                f"global-0={100.0 * rates['global_zero_rate']:.2f}%"
            ),
            fontsize=11,
        )
        ax.axis("off")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    _draw_panel(
        axes[0],
        graph_adj=adj_before,
        intra_degree=before_intra,
        global_degree=before_global,
        rates=before_rates,
        panel_title="Original graph",
        show_added=False,
        seed_offset=0,
    )
    _draw_panel(
        axes[1],
        graph_adj=adj_after,
        intra_degree=after_intra,
        global_degree=after_global,
        rates=after_rates,
        panel_title="Reconstructed target-1 repair view",
        show_added=True,
        seed_offset=10,
    )
    fig.suptitle(
        (
            f"{dataset} {variant} seed{seed} original vs reconstructed repair ({method.upper()}) | "
            f"added={int(diff['added_edges'])}, removed={int(diff['removed_edges'])}, "
            f"same-cluster added={int(diff['same_cluster_added'])}"
        ),
        fontsize=14,
        y=0.98,
    )
    fig.text(
        0.01,
        0.02,
        (
            "color: GMM cluster | red circle: CP intra-orphan | black square: C0p intra-orphan | "
            "black x: global degree 0 | gray: sampled same-cluster edge | orange: sampled cross-cluster edge touching orphan | "
            "green: reconstructed added repair edge"
        ),
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"repair_compare_{dataset}_{variant}_seed{seed}_{method}_{tag}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[REPAIR-PLOT] wrote {_relative_to_repo(out_path)}")

    row: dict[str, object] = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "plot_path": _relative_to_repo(out_path),
        "reconstruction_note": "dot-score repair reconstruction; historical temporary decoder graph was not saved",
        "before_cp_orphan_rate": before_rates["cp_intra_orphan_rate"],
        "after_cp_orphan_rate": after_rates["cp_intra_orphan_rate"],
        "before_c0p_orphan_rate": before_rates["c0p_intra_orphan_rate"],
        "after_c0p_orphan_rate": after_rates["c0p_intra_orphan_rate"],
        "before_global_zero_rate": before_rates["global_zero_rate"],
        "after_global_zero_rate": after_rates["global_zero_rate"],
        "repair_add_ratio": repair_add_ratio,
        "repair_remove_ratio": repair_remove_ratio,
        "repair_per_node_cap": repair_per_node_cap,
        "repair_pull_strength": repair_pull_strength,
    }
    row.update(diff)
    return row


def _counter_top(counter: Counter, *, top_n: int = 5) -> str:
    if not counter:
        return ""
    return ";".join(f"{key}:{value}" for key, value in counter.most_common(top_n))


def _tail_cluster_rows(
    *,
    dataset: str,
    variant: str,
    seed: int,
    labels: np.ndarray,
    core_mask: np.ndarray,
    adj: sp.csr_matrix,
    top_k: int,
    min_cluster_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    adj = _normalized_binary_adj(adj)
    labels_np = np.asarray(labels, dtype=np.int64)
    core_np = np.asarray(core_mask, dtype=bool)
    cp_mask = labels_np >= 0
    global_degree = np.asarray(adj.sum(axis=1)).ravel().astype(np.int64)
    intra_degree = _predicted_cluster_intra_degree(adj, labels_np)
    cross_degree = np.maximum(global_degree - intra_degree, 0)

    cluster_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    for cluster_id in sorted(int(x) for x in np.unique(labels_np[cp_mask])):
        nodes = np.flatnonzero(labels_np == cluster_id)
        if nodes.size < int(min_cluster_size):
            continue
        c0p_nodes = nodes[core_np[nodes]]
        orphan_nodes = nodes[intra_degree[nodes] == 0]
        c0p_orphan_nodes = orphan_nodes[core_np[orphan_nodes]]
        non_orphan_nodes = nodes[intra_degree[nodes] > 0]

        sub_adj = adj[nodes][:, nodes]
        intra_edges = int(sub_adj.nnz // 2)
        possible_edges = int(nodes.size * (nodes.size - 1) // 2)
        orphan_neighbor_labels: Counter[str] = Counter()
        orphan_neighbor_nodes: set[int] = set()
        orphan_cross_edges = 0

        for node in orphan_nodes:
            start, end = adj.indptr[int(node)], adj.indptr[int(node) + 1]
            for nbr_raw in adj.indices[start:end]:
                nbr = int(nbr_raw)
                if nbr == int(node) or labels_np[nbr] == cluster_id:
                    continue
                orphan_cross_edges += 1
                orphan_neighbor_nodes.add(nbr)
                nbr_label = int(labels_np[nbr])
                key = "noise" if nbr_label < 0 else str(nbr_label)
                orphan_neighbor_labels[key] += 1

            node_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "cluster_id": cluster_id,
                    "node_id": int(node),
                    "is_c0p": int(bool(core_np[int(node)])),
                    "global_degree": int(global_degree[int(node)]),
                    "intra_degree": int(intra_degree[int(node)]),
                    "cross_degree": int(cross_degree[int(node)]),
                }
            )

        cluster_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "cluster_id": cluster_id,
                "cp_size": int(nodes.size),
                "c0p_size": int(c0p_nodes.size),
                "cp_orphan_count": int(orphan_nodes.size),
                "cp_orphan_ratio": _safe_ratio(int(orphan_nodes.size), int(nodes.size)),
                "c0p_orphan_count": int(c0p_orphan_nodes.size),
                "c0p_orphan_ratio": _safe_ratio(int(c0p_orphan_nodes.size), int(c0p_nodes.size)),
                "cluster_internal_edges": intra_edges,
                "cluster_internal_density": _safe_ratio(intra_edges, possible_edges),
                "cluster_global_degree_mean": _mean(global_degree[nodes].astype(float)),
                "cluster_intra_degree_mean": _mean(intra_degree[nodes].astype(float)),
                "cluster_cross_degree_mean": _mean(cross_degree[nodes].astype(float)),
                "orphan_global_zero_count": int(np.sum(global_degree[orphan_nodes] == 0)),
                "orphan_global_degree_mean": _mean(global_degree[orphan_nodes].astype(float)),
                "orphan_cross_degree_mean": _mean(cross_degree[orphan_nodes].astype(float)),
                "orphan_cross_degree_p90": _percentile(cross_degree[orphan_nodes].astype(float), 90),
                "non_orphan_global_degree_mean": _mean(global_degree[non_orphan_nodes].astype(float)),
                "non_orphan_cross_degree_mean": _mean(cross_degree[non_orphan_nodes].astype(float)),
                "orphan_cross_edges": int(orphan_cross_edges),
                "orphan_unique_cross_neighbors": int(len(orphan_neighbor_nodes)),
                "top_orphan_neighbor_clusters": _counter_top(orphan_neighbor_labels, top_n=5),
            }
        )

    cluster_rows.sort(
        key=lambda row: (
            float(row["cp_orphan_ratio"]),
            int(row["cp_orphan_count"]),
            int(row["cp_size"]),
        ),
        reverse=True,
    )
    selected_cluster_ids = {int(row["cluster_id"]) for row in cluster_rows[: int(top_k)]}
    selected_node_rows = [
        row for row in node_rows if int(row["cluster_id"]) in selected_cluster_ids
    ]
    return cluster_rows[: int(top_k)], selected_node_rows


def plot_tail_clusters_artifact(
    *,
    dataset: str,
    variant: str,
    seed: int,
    artifact_dir: Path,
    adj: sp.csr_matrix,
    out_dir: Path,
    tag: str,
    method: str,
    top_k: int,
    min_cluster_size: int,
    max_neighbor_nodes: int,
    include_repair: bool,
    repair_add_ratio: float,
    repair_remove_ratio: float,
    repair_per_node_cap: float,
    repair_pull_strength: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Plot zoomed views of the highest orphan-ratio clusters."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    labels, core_mask, z_np = _load_artifact_arrays(artifact_dir)
    if labels.shape[0] != adj.shape[0]:
        raise ValueError(f"{artifact_dir}: label count {labels.shape[0]} != graph nodes {adj.shape[0]}")

    adj = _normalized_binary_adj(adj)
    rows, node_rows = _tail_cluster_rows(
        dataset=dataset,
        variant=variant,
        seed=seed,
        labels=labels,
        core_mask=core_mask,
        adj=adj,
        top_k=top_k,
        min_cluster_size=min_cluster_size,
    )
    if not rows:
        print(f"[TAIL] no tail clusters for {dataset}/{variant}/seed{seed}")
        return rows, node_rows

    repair_added_adj: sp.csr_matrix | None = None
    if include_repair:
        repair_adj, _ = build_reconstructed_repair_view(
            adj=adj,
            labels=labels,
            core_mask=core_mask,
            z_np=z_np,
            add_ratio=repair_add_ratio,
            remove_ratio=repair_remove_ratio,
            per_node_cap=repair_per_node_cap,
            pull_strength=repair_pull_strength,
            add_degree_target=1,
            add_degree_target_scope="intra_cluster",
            add_degree_target_nodes="cp",
            require_c0p_endpoint=True,
            require_c0p_noncompact_endpoint=True,
            guarantee_degree_target=True,
        )
        repair_added_adj = (repair_adj - adj).tocsr()
        repair_added_adj.data = (repair_added_adj.data > 0).astype(np.int8)
        repair_added_adj.eliminate_zeros()

    pos = _embedding_2d(z_np, method=method, seed=seed)
    labels_np = np.asarray(labels, dtype=np.int64)
    core_np = np.asarray(core_mask, dtype=bool)
    global_degree = np.asarray(adj.sum(axis=1)).ravel().astype(np.int64)
    intra_degree = _predicted_cluster_intra_degree(adj, labels_np)

    n_panels = len(rows)
    ncols = min(2, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5 * ncols, 6.8 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()

    for panel_idx, (ax, row) in enumerate(zip(axes_arr, rows)):
        cluster_id = int(row["cluster_id"])
        cluster_nodes = np.flatnonzero(labels_np == cluster_id)
        orphan_nodes = cluster_nodes[intra_degree[cluster_nodes] == 0]
        orphan_set = {int(x) for x in orphan_nodes}
        cluster_set = {int(x) for x in cluster_nodes}

        neighbor_set: set[int] = set()
        for node in orphan_nodes:
            start, end = adj.indptr[int(node)], adj.indptr[int(node) + 1]
            for nbr_raw in adj.indices[start:end]:
                nbr = int(nbr_raw)
                if nbr not in cluster_set:
                    neighbor_set.add(nbr)
        if len(neighbor_set) > int(max_neighbor_nodes):
            rng = np.random.default_rng(seed + cluster_id + 10_003)
            sampled = rng.choice(
                np.array(sorted(neighbor_set), dtype=np.int64),
                size=int(max_neighbor_nodes),
                replace=False,
            )
            neighbor_set = {int(x) for x in sampled}

        display_nodes = np.array(sorted(cluster_set | neighbor_set), dtype=np.int64)
        display_set = {int(x) for x in display_nodes}
        outside_nodes = np.array(sorted(neighbor_set), dtype=np.int64)
        non_orphan_cluster_nodes = np.array(
            [int(x) for x in cluster_nodes if int(x) not in orphan_set],
            dtype=np.int64,
        )
        c0p_orphan_nodes = np.array(
            [int(x) for x in orphan_nodes if core_np[int(x)]],
            dtype=np.int64,
        )

        intra_segments: list[list[np.ndarray]] = []
        orphan_cross_segments: list[list[np.ndarray]] = []
        other_cluster_cross_segments: list[list[np.ndarray]] = []
        coo = adj[display_nodes][:, display_nodes].tocoo()
        for local_i, local_j in zip(coo.row, coo.col):
            if int(local_i) >= int(local_j):
                continue
            i = int(display_nodes[int(local_i)])
            j = int(display_nodes[int(local_j)])
            segment = [pos[i], pos[j]]
            if i in cluster_set and j in cluster_set:
                intra_segments.append(segment)
            elif i in orphan_set or j in orphan_set:
                orphan_cross_segments.append(segment)
            elif i in cluster_set or j in cluster_set:
                other_cluster_cross_segments.append(segment)

        repair_segments: list[list[np.ndarray]] = []
        if repair_added_adj is not None:
            added_coo = repair_added_adj[cluster_nodes][:, cluster_nodes].tocoo()
            for local_i, local_j in zip(added_coo.row, added_coo.col):
                if int(local_i) >= int(local_j):
                    continue
                i = int(cluster_nodes[int(local_i)])
                j = int(cluster_nodes[int(local_j)])
                if i in orphan_set or j in orphan_set:
                    repair_segments.append([pos[i], pos[j]])

        if other_cluster_cross_segments:
            ax.add_collection(
                LineCollection(
                    other_cluster_cross_segments,
                    colors="#C9CDD2",
                    linewidths=0.55,
                    alpha=0.35,
                    zorder=1,
                )
            )
        if intra_segments:
            ax.add_collection(
                LineCollection(
                    intra_segments,
                    colors="#7F8C8D",
                    linewidths=0.70,
                    alpha=0.45,
                    zorder=2,
                )
            )
        if orphan_cross_segments:
            ax.add_collection(
                LineCollection(
                    orphan_cross_segments,
                    colors="#E4572E",
                    linewidths=1.0,
                    alpha=0.72,
                    zorder=3,
                )
            )
        if repair_segments:
            ax.add_collection(
                LineCollection(
                    repair_segments,
                    colors="#2CA02C",
                    linewidths=1.05,
                    alpha=0.78,
                    zorder=4,
                )
            )

        if outside_nodes.size:
            ax.scatter(
                pos[outside_nodes, 0],
                pos[outside_nodes, 1],
                s=18,
                c="#B7BDC5",
                alpha=0.55,
                edgecolors="none",
                zorder=5,
            )
        if non_orphan_cluster_nodes.size:
            ax.scatter(
                pos[non_orphan_cluster_nodes, 0],
                pos[non_orphan_cluster_nodes, 1],
                s=30,
                c="#4C78A8",
                alpha=0.90,
                edgecolors="white",
                linewidths=0.35,
                zorder=6,
            )
        if orphan_nodes.size:
            ax.scatter(
                pos[orphan_nodes, 0],
                pos[orphan_nodes, 1],
                s=58,
                c="#D62728",
                alpha=0.92,
                edgecolors="white",
                linewidths=0.45,
                zorder=7,
            )
        if c0p_orphan_nodes.size:
            ax.scatter(
                pos[c0p_orphan_nodes, 0],
                pos[c0p_orphan_nodes, 1],
                s=78,
                marker="s",
                facecolors="none",
                edgecolors="#111111",
                linewidths=0.90,
                zorder=8,
            )

        title = (
            f"cluster {cluster_id}: {100.0 * float(row['cp_orphan_ratio']):.1f}% "
            f"orphans ({int(row['cp_orphan_count'])}/{int(row['cp_size'])})\n"
            f"orphan global deg mean={float(row['orphan_global_degree_mean']):.2f}, "
            f"cross deg mean={float(row['orphan_cross_degree_mean']):.2f}, "
            f"global-0={int(row['orphan_global_zero_count'])}, "
            f"top outside={row['top_orphan_neighbor_clusters'] or 'none'}"
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if display_nodes.size:
            x = pos[display_nodes, 0]
            y = pos[display_nodes, 1]
            x_pad = max(0.5, 0.08 * float(x.max() - x.min()))
            y_pad = max(0.5, 0.08 * float(y.max() - y.min()))
            ax.set_xlim(float(x.min() - x_pad), float(x.max() + x_pad))
            ax.set_ylim(float(y.min() - y_pad), float(y.max() + y_pad))

        print(
            "[TAIL] "
            f"{dataset}/{variant}/seed{seed}/cluster{cluster_id}: "
            f"cp_orphan={_format_float(float(row['cp_orphan_ratio']))} "
            f"orphans={int(row['cp_orphan_count'])}/{int(row['cp_size'])} "
            f"orphan_global0={int(row['orphan_global_zero_count'])} "
            f"orphan_cross_mean={_format_float(float(row['orphan_cross_degree_mean']))} "
            f"top_outside={row['top_orphan_neighbor_clusters']}"
        )

    for ax in axes_arr[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        (
            f"{dataset} {variant} seed{seed}: highest orphan-ratio clusters ({method.upper()})"
        ),
        fontsize=14,
        y=0.98,
    )
    fig.text(
        0.01,
        0.02,
        (
            "blue: non-orphan nodes in focus cluster | red: intra-orphans | "
            "black square: C0p orphan | gray: outside neighbors | gray lines: intra/other cluster edges | "
            "orange: orphan cross-cluster edges | green: reconstructed repair edges"
        ),
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tail_clusters_{dataset}_{variant}_seed{seed}_{method}_{tag}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[TAIL] wrote {_relative_to_repo(out_path)}")
    for row in rows:
        row["tail_plot_path"] = _relative_to_repo(out_path)
        row["tail_include_repair"] = int(include_repair)
    return rows, node_rows


def plot_synthetic_orphan_examples(
    *,
    out_path: Path,
    seed: int,
    n_clusters: int,
    n_per_cluster: int,
    dim: int,
) -> Path:
    """Plot the synthetic graph cases used by the orphan smoke test."""
    _add_src_path()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import torch
    from utils import gmm_labels, select_gmm_cores

    cases = [
        ("connected_ring", "ring", 2.0, "Connected ring"),
        ("er_mean8", "er", 8.0, "Dense ER, mean degree 8"),
        ("er_mean2", "er", 2.0, "Sparse ER, mean degree 2"),
    ]
    palette = np.array(
        [
            "#4C78A8",
            "#F58518",
            "#54A24B",
            "#B279A2",
            "#E45756",
            "#72B7B2",
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for ax, (case_name, graph_mode, mean_intra_degree, title) in zip(np.ravel(axes), cases):
        z_np, true_labels = _synthetic_embeddings(
            seed=seed,
            n_clusters=n_clusters,
            n_per_cluster=n_per_cluster,
            dim=dim,
            separation=6.0,
            noise=0.35,
        )
        adj = _synthetic_adj(
            true_labels,
            mode=graph_mode,
            mean_intra_degree=mean_intra_degree,
            seed=10_000 + seed,
        )
        z = torch.tensor(z_np, dtype=torch.float32)
        pred_labels = gmm_labels(z, K=n_clusters, tau=0.55, metric="raw")
        deg = torch.tensor(np.asarray(adj.sum(axis=1)).ravel(), dtype=torch.float32)
        core_mask, _, _, _ = select_gmm_cores(
            z,
            pred_labels,
            deg,
            alpha=0.8,
            gamma=1.0,
            B=dim,
            normalize_cosine=False,
        )
        core_np = core_mask.detach().cpu().numpy().astype(bool)
        rates = _orphan_rates_for_arrays(adj, pred_labels, core_np)
        intra_deg = _predicted_cluster_intra_degree(adj, pred_labels)
        intra_orphans = (pred_labels >= 0) & (intra_deg == 0)
        global_deg = np.asarray(adj.sum(axis=1)).ravel()
        global_orphans = global_deg == 0
        pos = _synthetic_plot_positions(true_labels, mode=graph_mode, seed=seed)

        coo = adj.tocoo()
        same_segments = []
        cross_segments = []
        for i, j in zip(coo.row, coo.col):
            if int(i) >= int(j):
                continue
            segment = [pos[int(i)], pos[int(j)]]
            if pred_labels[int(i)] >= 0 and pred_labels[int(i)] == pred_labels[int(j)]:
                same_segments.append(segment)
            else:
                cross_segments.append(segment)

        if same_segments:
            ax.add_collection(
                LineCollection(
                    same_segments,
                    colors="#9AA0A6",
                    linewidths=0.55,
                    alpha=0.42,
                    zorder=1,
                )
            )
        if cross_segments:
            ax.add_collection(
                LineCollection(
                    cross_segments,
                    colors="#E4572E",
                    linewidths=0.9,
                    alpha=0.85,
                    zorder=2,
                )
            )

        node_colors = palette[true_labels % len(palette)]
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            s=28,
            c=node_colors,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        if np.any(intra_orphans):
            ax.scatter(
                pos[intra_orphans, 0],
                pos[intra_orphans, 1],
                s=78,
                facecolors="none",
                edgecolors="#D62728",
                linewidths=1.45,
                zorder=4,
            )
        if np.any(global_orphans):
            ax.scatter(
                pos[global_orphans, 0],
                pos[global_orphans, 1],
                s=110,
                marker="x",
                c="#111111",
                linewidths=1.35,
                zorder=5,
            )

        ax.set_title(
            (
                f"{title}\n"
                f"CP intra-orphan={100.0 * rates['cp_intra_orphan_rate']:.1f}% | "
                f"global orphan={100.0 * rates['global_zero_rate']:.1f}%"
            ),
            fontsize=12,
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(pos[:, 0].min() - 1.0, pos[:, 0].max() + 1.0)
        ax.set_ylim(pos[:, 1].min() - 1.0, pos[:, 1].max() + 1.0)

        ax.text(
            0.01,
            0.02,
            (
                f"GMM purity={100.0 * _cluster_assignment_purity(pred_labels, true_labels):.0f}%\n"
                f"red outline: zero same-cluster degree\n"
                f"black x: global degree 0"
            ),
            transform=ax.transAxes,
            fontsize=9,
            color="#333333",
            va="bottom",
            ha="left",
        )

    fig.suptitle(
        "Synthetic Clear-Cluster Graphs: Why Intra-Cluster Orphans Can Appear",
        fontsize=15,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[SYNTH-ORPHAN] plot saved: {_relative_to_repo(out_path)}")
    return out_path


def _parse_seed_dir(path: Path) -> int | None:
    name = path.name
    if not name.startswith("seed"):
        return None
    try:
        return int(name[4:])
    except ValueError:
        return None


def discover_seeds(cache_root: Path, dataset: str, variant: str, preprune: str) -> list[int]:
    variant_dir = cache_root / dataset / variant
    if not variant_dir.exists():
        return []

    seeds: list[int] = []
    for seed_dir in variant_dir.iterdir():
        seed = _parse_seed_dir(seed_dir)
        if seed is None:
            continue
        if (seed_dir / preprune / "final_gmm_labels.npy").exists():
            seeds.append(seed)
    return sorted(seeds)


def _load_dataset_adj(dataset: str) -> sp.csr_matrix:
    _add_src_path()
    from input_data import load_data

    loaded = load_data(dataset)
    if loaded is None:
        raise ValueError(f"load_data returned None for dataset={dataset!r}")

    adj = loaded[0]
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    else:
        adj = adj.tocsr()

    adj = adj.astype(np.int8, copy=False)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj = ((adj + adj.T) > 0).astype(np.int8).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def _safe_ratio(num: int | float, den: int | float) -> float:
    if den == 0:
        return math.nan
    return float(num) / float(den)


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, q))


def _mean(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(np.mean(values))


def _min(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(np.min(values))


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def diagnose_artifact(
    *,
    dataset: str,
    variant: str,
    seed: int,
    preprune: str,
    artifact_dir: Path,
    adj: sp.csr_matrix,
    orphan_ratio_threshold: float,
    min_cluster_size: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    label_path = artifact_dir / "final_gmm_labels.npy"
    core_path = artifact_dir / "final_core_mask.npy"
    meta_path = artifact_dir / "meta.json"

    labels = np.load(label_path).astype(np.int64, copy=False)
    core_mask = np.load(core_path).astype(bool, copy=False)
    if labels.shape[0] != adj.shape[0]:
        raise ValueError(
            f"{artifact_dir}: label count {labels.shape[0]} != graph nodes {adj.shape[0]}"
        )
    if core_mask.shape[0] != adj.shape[0]:
        raise ValueError(
            f"{artifact_dir}: core mask count {core_mask.shape[0]} != graph nodes {adj.shape[0]}"
        )

    meta: dict[str, object] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            loaded_meta = json.load(f)
        if isinstance(loaded_meta, dict):
            meta = loaded_meta

    cp_mask = labels >= 0
    cluster_ids = np.array(sorted(int(x) for x in np.unique(labels[cp_mask])))
    cp_node_degrees: list[np.ndarray] = []
    c0p_node_degrees: list[np.ndarray] = []
    cluster_rows: list[dict[str, object]] = []

    for cluster_id in cluster_ids:
        nodes = np.flatnonzero(labels == cluster_id)
        sub_adj = adj[nodes][:, nodes]
        intra = np.asarray(sub_adj.sum(axis=1)).ravel().astype(np.float64)
        core_local = core_mask[nodes]
        c0p_intra = intra[core_local]

        cp_orphans = int(np.sum(intra == 0))
        c0p_orphans = int(np.sum(c0p_intra == 0))
        cp_orphan_ratio = _safe_ratio(cp_orphans, int(nodes.size))
        c0p_orphan_ratio = _safe_ratio(c0p_orphans, int(np.sum(core_local)))
        suspicious_cp = (
            int(nodes.size) >= min_cluster_size
            and not math.isnan(cp_orphan_ratio)
            and cp_orphan_ratio >= orphan_ratio_threshold
        )
        suspicious_c0p = (
            int(np.sum(core_local)) >= min_cluster_size
            and not math.isnan(c0p_orphan_ratio)
            and c0p_orphan_ratio >= orphan_ratio_threshold
        )

        cp_node_degrees.append(intra)
        if c0p_intra.size:
            c0p_node_degrees.append(c0p_intra)

        cluster_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "preprune": preprune,
                "cluster_id": cluster_id,
                "cp_size": int(nodes.size),
                "c0p_size": int(np.sum(core_local)),
                "cp_orphan_count": cp_orphans,
                "cp_orphan_ratio": cp_orphan_ratio,
                "c0p_orphan_count": c0p_orphans,
                "c0p_orphan_ratio": c0p_orphan_ratio,
                "cp_intra_min": _min(intra),
                "cp_intra_mean": _mean(intra),
                "cp_intra_p10": _percentile(intra, 10),
                "cp_intra_p50": _percentile(intra, 50),
                "cp_intra_p90": _percentile(intra, 90),
                "c0p_intra_min": _min(c0p_intra),
                "c0p_intra_mean": _mean(c0p_intra),
                "c0p_intra_p10": _percentile(c0p_intra, 10),
                "c0p_intra_p50": _percentile(c0p_intra, 50),
                "c0p_intra_p90": _percentile(c0p_intra, 90),
                "suspicious_cp_cluster": int(suspicious_cp),
                "suspicious_c0p_cluster": int(suspicious_c0p),
            }
        )

    cp_degrees = (
        np.concatenate(cp_node_degrees) if cp_node_degrees else np.array([], dtype=float)
    )
    c0p_degrees = (
        np.concatenate(c0p_node_degrees) if c0p_node_degrees else np.array([], dtype=float)
    )
    cp_cluster_ratios = np.array(
        [float(row["cp_orphan_ratio"]) for row in cluster_rows], dtype=float
    )
    c0p_cluster_ratios = np.array(
        [
            float(row["c0p_orphan_ratio"])
            for row in cluster_rows
            if not math.isnan(float(row["c0p_orphan_ratio"]))
        ],
        dtype=float,
    )
    cp_nodes = int(np.sum(cp_mask))
    c0p_nodes = int(np.sum(core_mask & cp_mask))
    cp_orphan_nodes = int(np.sum(cp_degrees == 0))
    c0p_orphan_nodes = int(np.sum(c0p_degrees == 0))

    summary_row: dict[str, object] = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "preprune": preprune,
        "artifact_dir": _relative_to_repo(artifact_dir),
        "n_nodes": int(adj.shape[0]),
        "n_edges_undirected": int(adj.nnz // 2),
        "clusters": int(cluster_ids.size),
        "singleton_clusters": int(sum(int(row["cp_size"]) == 1 for row in cluster_rows)),
        "noise_nodes": int(np.sum(labels < 0)),
        "noise_ratio": _safe_ratio(int(np.sum(labels < 0)), int(labels.size)),
        "cp_nodes": cp_nodes,
        "cp_orphan_nodes": cp_orphan_nodes,
        "cp_orphan_ratio": _safe_ratio(cp_orphan_nodes, cp_nodes),
        "c0p_nodes": c0p_nodes,
        "c0p_orphan_nodes": c0p_orphan_nodes,
        "c0p_orphan_ratio": _safe_ratio(c0p_orphan_nodes, c0p_nodes),
        "cp_intra_node_min": _min(cp_degrees),
        "cp_intra_node_mean": _mean(cp_degrees),
        "cp_intra_node_p10": _percentile(cp_degrees, 10),
        "cp_intra_node_p50": _percentile(cp_degrees, 50),
        "cp_intra_node_p90": _percentile(cp_degrees, 90),
        "c0p_intra_node_min": _min(c0p_degrees),
        "c0p_intra_node_mean": _mean(c0p_degrees),
        "c0p_intra_node_p10": _percentile(c0p_degrees, 10),
        "c0p_intra_node_p50": _percentile(c0p_degrees, 50),
        "c0p_intra_node_p90": _percentile(c0p_degrees, 90),
        "cp_cluster_orphan_ratio_p50": _percentile(cp_cluster_ratios, 50),
        "cp_cluster_orphan_ratio_p90": _percentile(cp_cluster_ratios, 90),
        "cp_cluster_orphan_ratio_max": _percentile(cp_cluster_ratios, 100),
        "c0p_cluster_orphan_ratio_p50": _percentile(c0p_cluster_ratios, 50),
        "c0p_cluster_orphan_ratio_p90": _percentile(c0p_cluster_ratios, 90),
        "c0p_cluster_orphan_ratio_max": _percentile(c0p_cluster_ratios, 100),
        "suspicious_cp_clusters": int(
            sum(int(row["suspicious_cp_cluster"]) for row in cluster_rows)
        ),
        "suspicious_c0p_clusters": int(
            sum(int(row["suspicious_c0p_cluster"]) for row in cluster_rows)
        ),
        "best_epoch": meta.get("best_epoch", ""),
        "test_hit10": meta.get("test_hit10", ""),
        "val_roc": meta.get("val_roc", ""),
        "val_ap": meta.get("val_ap", ""),
    }

    return summary_row, cluster_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["variant"])), []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    metrics = [
        "noise_ratio",
        "cp_orphan_ratio",
        "c0p_orphan_ratio",
        "cp_intra_node_mean",
        "c0p_intra_node_mean",
        "cp_cluster_orphan_ratio_p90",
        "cp_cluster_orphan_ratio_max",
        "c0p_cluster_orphan_ratio_p90",
        "c0p_cluster_orphan_ratio_max",
        "suspicious_cp_clusters",
        "suspicious_c0p_clusters",
    ]
    for (dataset, variant), group_rows in sorted(grouped.items()):
        agg: dict[str, object] = {
            "dataset": dataset,
            "variant": variant,
            "runs": len(group_rows),
        }
        for metric in metrics:
            values = np.array([float(row[metric]) for row in group_rows], dtype=float)
            values = values[~np.isnan(values)]
            agg[f"{metric}_mean"] = _mean(values)
            agg[f"{metric}_min"] = _min(values)
            agg[f"{metric}_max"] = _percentile(values, 100)
        aggregate_rows.append(agg)
    return aggregate_rows


def _print_aggregate(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("[ORPHAN] no rows to aggregate")
        return

    print("[ORPHAN] aggregate by dataset/variant")
    header = (
        "dataset variant runs noise cp_orphan c0p_orphan "
        "cp_intra_mean c0p_intra_mean suspicious_cp suspicious_c0p"
    )
    print(header)
    for row in rows:
        print(
            " ".join(
                [
                    str(row["dataset"]),
                    str(row["variant"]),
                    str(row["runs"]),
                    _format_float(float(row["noise_ratio_mean"])),
                    _format_float(float(row["cp_orphan_ratio_mean"])),
                    _format_float(float(row["c0p_orphan_ratio_mean"])),
                    _format_float(float(row["cp_intra_node_mean_mean"])),
                    _format_float(float(row["c0p_intra_node_mean_mean"])),
                    _format_float(float(row["suspicious_cp_clusters_mean"])),
                    _format_float(float(row["suspicious_c0p_clusters_mean"])),
                ]
            )
        )


def _iter_requested_runs(
    *,
    cache_root: Path,
    datasets: Iterable[str],
    variants: Iterable[str],
    seeds: list[int] | None,
    preprune: str,
) -> Iterable[tuple[str, str, int, Path]]:
    for dataset in datasets:
        for variant in variants:
            discovered = discover_seeds(cache_root, dataset, variant, preprune)
            requested = discovered if seeds is None else [s for s in seeds if s in discovered]
            if not requested:
                print(
                    f"[ORPHAN] skip dataset={dataset} variant={variant}: "
                    f"no matching seeds under {cache_root}"
                )
                continue
            for seed in requested:
                artifact_dir = cache_root / dataset / variant / f"seed{seed}" / preprune
                yield dataset, variant, seed, artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a GMM self-test and diagnose same-cluster degree-0 orphan nodes "
            "from saved tSNE/GMM cache artifacts."
        )
    )
    parser.add_argument("--self-test-gmm", action="store_true")
    parser.add_argument("--self-test-orphans", action="store_true")
    parser.add_argument("--plot-synthetic-orphans", action="store_true")
    parser.add_argument("--plot-real-orphans", action="store_true")
    parser.add_argument("--plot-repair-comparison", action="store_true")
    parser.add_argument("--plot-tail-clusters", action="store_true")
    parser.add_argument("--plot-method", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--max-edge-lines", type=int, default=6000)
    parser.add_argument("--max-cross-edge-lines", type=int, default=1000)
    parser.add_argument("--max-added-edge-lines", type=int, default=2500)
    parser.add_argument("--tail-top-k", type=int, default=4)
    parser.add_argument("--tail-min-cluster-size", type=int, default=5)
    parser.add_argument("--tail-max-neighbor-nodes", type=int, default=250)
    parser.add_argument("--tail-include-repair", action="store_true")
    parser.add_argument("--repair-add-ratio", type=float, default=0.20)
    parser.add_argument("--repair-remove-ratio", type=float, default=0.0)
    parser.add_argument("--repair-per-node-cap", type=float, default=0.10)
    parser.add_argument("--repair-pull-strength", type=float, default=0.25)
    parser.add_argument("--save-repair-adj", action="store_true")
    parser.add_argument("--smoke-seeds", type=int, default=20)
    parser.add_argument("--smoke-n-clusters", type=int, default=4)
    parser.add_argument("--smoke-n-per-cluster", type=int, default=120)
    parser.add_argument("--smoke-dim", type=int, default=8)
    parser.add_argument("--plot-seed", type=int, default=0)
    parser.add_argument("--plot-n-per-cluster", type=int, default=32)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--variants", nargs="+", default=["no", "v6"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--preprune", default="preprune_0.00")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="20260522_initial")
    parser.add_argument("--orphan-ratio-threshold", type=float, default=0.25)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test_gmm:
        results = run_gmm_self_test()
        for name, result in results.items():
            print(f"[GMM-SELF-TEST] {name}: {result}")

    smoke_rows: list[dict[str, object]] = []
    if args.self_test_orphans:
        smoke_rows = run_synthetic_orphan_smoke(
            seeds=args.smoke_seeds,
            n_clusters=args.smoke_n_clusters,
            n_per_cluster=args.smoke_n_per_cluster,
            dim=args.smoke_dim,
        )
        if not args.no_write:
            smoke_path = args.out_dir / f"synthetic_orphan_smoke_{args.tag}.csv"
            _write_csv(smoke_path, smoke_rows)
            print(f"[SYNTH-ORPHAN] wrote {_relative_to_repo(smoke_path)}")

    if args.plot_synthetic_orphans:
        plot_synthetic_orphan_examples(
            out_path=args.out_dir / f"synthetic_orphan_examples_{args.tag}.png",
            seed=args.plot_seed,
            n_clusters=args.smoke_n_clusters,
            n_per_cluster=args.plot_n_per_cluster,
            dim=args.smoke_dim,
        )

    runs = list(
        _iter_requested_runs(
            cache_root=args.cache_root,
            datasets=args.datasets,
            variants=args.variants,
            seeds=args.seeds,
            preprune=args.preprune,
        )
    )
    if not runs:
        if args.self_test_gmm or args.self_test_orphans or args.plot_synthetic_orphans:
            return
        raise SystemExit("No artifact runs found.")

    adj_by_dataset: dict[str, sp.csr_matrix] = {}
    summary_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    repair_rows: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    tail_node_rows: list[dict[str, object]] = []

    for dataset, variant, seed, artifact_dir in runs:
        if dataset not in adj_by_dataset:
            adj_by_dataset[dataset] = _load_dataset_adj(dataset)
        summary_row, run_cluster_rows = diagnose_artifact(
            dataset=dataset,
            variant=variant,
            seed=seed,
            preprune=args.preprune,
            artifact_dir=artifact_dir,
            adj=adj_by_dataset[dataset],
            orphan_ratio_threshold=args.orphan_ratio_threshold,
            min_cluster_size=args.min_cluster_size,
        )
        summary_rows.append(summary_row)
        cluster_rows.extend(run_cluster_rows)
        if args.plot_real_orphans:
            plot_real_orphan_artifact(
                dataset=dataset,
                variant=variant,
                seed=seed,
                artifact_dir=artifact_dir,
                adj=adj_by_dataset[dataset],
                out_dir=args.out_dir,
                tag=args.tag,
                method=args.plot_method,
                max_edge_lines=args.max_edge_lines,
                max_cross_edge_lines=args.max_cross_edge_lines,
            )
        if args.plot_repair_comparison:
            repair_rows.append(
                plot_repair_comparison_artifact(
                    dataset=dataset,
                    variant=variant,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    adj=adj_by_dataset[dataset],
                    out_dir=args.out_dir,
                    tag=args.tag,
                    method=args.plot_method,
                    max_edge_lines=args.max_edge_lines,
                    max_cross_edge_lines=args.max_cross_edge_lines,
                    max_added_edge_lines=args.max_added_edge_lines,
                    repair_add_ratio=args.repair_add_ratio,
                    repair_remove_ratio=args.repair_remove_ratio,
                    repair_per_node_cap=args.repair_per_node_cap,
                    repair_pull_strength=args.repair_pull_strength,
                    save_repair_adj=args.save_repair_adj,
                )
            )
        if args.plot_tail_clusters:
            run_tail_rows, run_tail_node_rows = plot_tail_clusters_artifact(
                dataset=dataset,
                variant=variant,
                seed=seed,
                artifact_dir=artifact_dir,
                adj=adj_by_dataset[dataset],
                out_dir=args.out_dir,
                tag=args.tag,
                method=args.plot_method,
                top_k=args.tail_top_k,
                min_cluster_size=args.tail_min_cluster_size,
                max_neighbor_nodes=args.tail_max_neighbor_nodes,
                include_repair=args.tail_include_repair,
                repair_add_ratio=args.repair_add_ratio,
                repair_remove_ratio=args.repair_remove_ratio,
                repair_per_node_cap=args.repair_per_node_cap,
                repair_pull_strength=args.repair_pull_strength,
            )
            tail_rows.extend(run_tail_rows)
            tail_node_rows.extend(run_tail_node_rows)
        print(
            "[ORPHAN] "
            f"{dataset}/{variant}/seed{seed}: "
            f"cp_orphan={_format_float(float(summary_row['cp_orphan_ratio']))} "
            f"c0p_orphan={_format_float(float(summary_row['c0p_orphan_ratio']))} "
            f"cp_intra_mean={_format_float(float(summary_row['cp_intra_node_mean']))} "
            f"c0p_intra_mean={_format_float(float(summary_row['c0p_intra_node_mean']))}"
        )

    aggregate_rows = _aggregate(summary_rows)
    _print_aggregate(aggregate_rows)

    if not args.no_write:
        summary_path = args.out_dir / f"gmm_orphan_summary_{args.tag}.csv"
        cluster_path = args.out_dir / f"gmm_orphan_clusters_{args.tag}.csv"
        aggregate_path = args.out_dir / f"gmm_orphan_aggregate_{args.tag}.csv"
        _write_csv(summary_path, summary_rows)
        _write_csv(cluster_path, cluster_rows)
        _write_csv(aggregate_path, aggregate_rows)
        print(f"[ORPHAN] wrote {_relative_to_repo(summary_path)}")
        print(f"[ORPHAN] wrote {_relative_to_repo(cluster_path)}")
        print(f"[ORPHAN] wrote {_relative_to_repo(aggregate_path)}")
        if repair_rows:
            repair_path = args.out_dir / f"repair_comparison_{args.tag}.csv"
            _write_csv(repair_path, repair_rows)
            print(f"[REPAIR-PLOT] wrote {_relative_to_repo(repair_path)}")
        if tail_rows:
            tail_path = args.out_dir / f"tail_cluster_summary_{args.tag}.csv"
            tail_nodes_path = args.out_dir / f"tail_cluster_orphan_nodes_{args.tag}.csv"
            _write_csv(tail_path, tail_rows)
            _write_csv(tail_nodes_path, tail_node_rows)
            print(f"[TAIL] wrote {_relative_to_repo(tail_path)}")
            print(f"[TAIL] wrote {_relative_to_repo(tail_nodes_path)}")


if __name__ == "__main__":
    main()
