import os
import time
import math
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

from ogb.linkproppred import Evaluator
from tqdm import tqdm
from preprocessing import *
from model import VGNAE_ENCODER, VGAE_ENCODER, dot_product_decode, MLP, LogReg
from loss import loss_function, inter_view_CL_loss, intra_view_CL_loss, Cluster
from utils import *
from input_data import CalN2V

from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Tuple

# # ---- side experiment: offline C0p "remove until done" curve ----
# # 要跑這個實驗的時候把這個改成 True，就會在 train 結束後多跑一趟 sweep
# ENABLE_C0P_SWEEP = False

# # 每次評估的步長 & 最大刪除比例（以「intra-cluster C0p 邊的數量」為基準）
# C0P_SWEEP_STEP_FRAC = 0.001   # 每刪掉 1% 的 intra C0p 邊評一次
# C0P_SWEEP_MAX_FRAC  = 1.0    # 最多刪到 100% 的 intra C0p 邊

# --- dense helper: works for both sparse and dense tensors
def _to_dense(A: torch.Tensor) -> torch.Tensor:
    return A.to_dense() if getattr(A, "is_sparse", False) else A

def train_encoder(
    dataset_str: str,
    device: torch.device,
    num_epoch: int,
    adj: torch.Tensor,                       # dense [N,N] with self-loops
    features: torch.Tensor,                  # [N,F]
    hidden1: int,
    hidden2: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    aug_graph_weight: float,
    aug_ratio: float,                        # GLOBAL edit budget (× |E0|)
    aug_bound: float,                        # PER-NODE cap fraction (vs deg0)
    alpha: float, beta: float, gamma: float, delta: float,
    temperature: float,
    labels: torch.Tensor,
    idx_train, idx_val, idx_test,
    ver: str,                                # "aron_desc/asc", "aron_desc_intra/inter", "no", ...
    degree_ratio: float,
    loss_ver: str,
    feat_maske_ratio: float,
    pretrain_epochs: int = 100,
    frozen_scores_path: str = "",
    pretrained_ckpt_path: str = "",
    *,
    # ---- clustering / augmentation knobs ----
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: int = 5,
    dbscan_metric: str = "cosine",
    topk_per_node: int = 64,
    charge_both_endpoints: bool = True,
    aug_ratio_epoch: Optional[float] = None,
    # ---- logging / reproducibility ----
    run_tag: str = "",
    seed: Optional[int] = None,
    # NEW cluster controls
    cluster_method: str = "none",
    cluster_mode:   str = "any",        # "any"|"intra"|"inter"
    gmm_k:   int = 16,
    gmm_tau: float = 0.55,
    louvain_resolution: float = 1.0,
    # NEW: restricted (α,γ, d̂) gating for augmentation (avoid name clash with loss α/γ)
    restricted: bool = False,
    restrict_alpha: float = 0.8,
    restrict_gamma: float = 1.0,
    # c0p pruning (removals that also consume the budget)
    c0p_prune_frac: float = 0.0,
    # ===== NEW: online prune knobs (scope is encoded in `ver=prune_*`) =====
    prune_step_frac: float = 0.01, # 0.1% of candidates pruned per epoch
    prune_max_frac:  float = 1.0,     # up to 100% of candidates in total
    **kwargs,
) -> Tuple[torch.Tensor, list, list, torch.Tensor]:
    """
    Train the encoder with budgeted graph augmentation.

    Args:
        dataset_str: Dataset name (e.g., "cora", "citeseer", "Cora_ML", "LastFMAsia").
        device: Torch device.
        num_epoch: Total training epochs AFTER any pretraining.
        adj: Dense adjacency WITH self-loops for the current working graph.
        features: Node feature matrix (dense).
        hidden1/hidden2, dropout, learning_rate, weight_decay: Encoder architecture & optimizer hparams.
        aug_graph_weight: Weight of the augmentation loss term (if applicable).
        aug_ratio: GLOBAL total edit budget as a fraction of original |E0|.
        aug_bound: PER-NODE edit cap fraction relative to ORIGINAL degree deg0 (ceil applied).
        alpha, beta, gamma, delta, temperature: Loss/config knobs (unchanged).
        labels, idx_train/idx_val/idx_test: Supervision / splits.
        ver: Augmentation variant:
             - "aron_desc"/"aron_asc" (score-guided deficit fill),
             - "aron_db_any"/"aron_db_intra"/"aron_db_inter" (density-cluster filtered),
             - "no" (no augmentation).
        degree_ratio: Percentile (incl self-loop) used to derive degree floor (you subtract 1 internally).
        loss_ver: Which loss variant to use.
        feat_maske_ratio: Column-wise feature dropout ratio (0 disables).
        pretrain_epochs: Warm-up epochs on ORIGINAL graph to get frozen scores (if path missing).
        frozen_scores_path: Optional path to load/save [N,N] confidence matrix.
        pretrained_ckpt_path: Optional path to load/save pretrained encoder weights.

    Keyword Args (NEW):
        dbscan_eps: DBSCAN epsilon; None = auto (median kNN distance).
        dbscan_min_samples: DBSCAN min_samples.
        dbscan_metric: "cosine" (recommended) or "euclidean".
        topk_per_node: Per-node candidate pool size to avoid O(N^2).
        charge_both_endpoints: If True, count both endpoints toward per-node cap; else only the lower-degree side.
        aug_ratio_epoch: Optional per-epoch cap (fraction of |E0|); None disables throttle.
        run_tag: Free-form tag for logging (e.g., "1007_s0_intra").
        seed: Optional random seed.

    Returns:
        Z: Final node embeddings [N,d].
        roc_history: List of validation ROC/AUC across epochs (if you track it).
        modification_ratio_history: List of global mod ratios across epochs.
        edge_index: Final edge_index of the augmented graph (COO indices).

    Notes:
        - E0, deg0_excl_self, forbid_mask are computed from ORIGINAL graph once per run.
        - For "aron_desc/asc", if `frozen_scores_path` missing or bad shape, we derive scores via Z·Z^T once.
        - For "aron_db_*", `frozen_scores_path` is optional; falls back to Z·Z^T.
        - Per-node caps are enforced against ORIGINAL degrees; global cap uses ORIGINAL |E0|.
        - Use `charge_both_endpoints=False` if helper saturation becomes a bottleneck.
    """
    torch.cuda.empty_cache()
    device = adj.device if hasattr(adj, "device") else device
    training_time_start = time.time()

    num_nodes = adj.shape[0]
    
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    if dataset_str in ['ogbl-ddi','ogbl-collab']:
        print("ogbl!!")
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_ogbl(adj,dataset_str, idx_train, idx_val, idx_test)
        print(f"{type(adj_train)}, {type(train_edges)}, {type(val_edges)}, {type(val_edges_false)}, {type(test_edges)}, {type(test_edges_false)}")
        print(len(train_edges), len(val_edges_false), len(test_edges))
    else:
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj,dataset_str)
        print(f"{type(adj_train)}, {type(train_edges)}, {type(val_edges)}, {type(val_edges_false)}, {type(test_edges)}, {type(test_edges_false)}")
         
    adj = adj_train
    train_mask = torch.ones(num_nodes*num_nodes, dtype = torch.bool, requires_grad = False).to(device)

    for r, c in val_edges:
        train_mask[num_nodes * r + c] = False
    for r, c in test_edges:
        train_mask[num_nodes * r + c] = False
    training_instance_number = torch.sum(train_mask).item()

    # APPNP
    edge_index = from_scipy_sparse_matrix(adj)[0].to(device)

    if dataset_str in ['USAir', 'PB', 'Celegans', 'Power', 'Router', 'Ecoli', 'Yeast', 'NS','obgl-ddi']:
        print('Training Data Without Init Attr ...')
        features = CalN2V(edge_index, 16, 1)
        features = sp.lil_matrix(features.numpy())

    feat_dim = features.shape[1]
    print(f'Node Nums: {num_nodes}, Init Feature Dim: {feat_dim}')

    # Some preprocessing
    adj_norm = preprocess_graph(adj) #Laplacian Matrix
    # _, features = preprocess_features(features)
    features = sparse_to_tuple(features.tocoo())
    
    # Create Model
    pos_weight = float(training_instance_number - adj.sum()) / adj.sum() # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = training_instance_number / float((training_instance_number - adj.sum()) * 2) # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2])).to(device)
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]), torch.Size(adj_label[2])).to(device)
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]), torch.Size(features[2])).to(device)
    weight_mask = adj_label.to_dense().view(-1)[train_mask] == 1 # weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight

    # init model and optimizer
    encoder = VGNAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device) # encoder = VGAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device)
    optimizer = Adam(encoder.parameters(), lr = learning_rate, weight_decay = weight_decay)

    data_augmenter = MLP(hidden2, hidden2).to(device)
    # data_augmenter = VGAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device)
    data_augmenter_optimizer = Adam(data_augmenter.parameters(), lr = 0.01, weight_decay = weight_decay)

    best_acc = 0.0; best_roc = 0.0; best_ap = 0.0
    best_hit1 = 0.0;best_hit3 = 0.0;best_hit10 = 0.0; best_hit20 = 0.0; best_hit50 = 0.0; best_hit100 = 0.0
    best_hit1_roc = 0.0;best_hit3_roc = 0.0;best_hit10_roc = 0.0; best_hit20_roc = 0.0; best_hit50_roc = 0.0; best_hit100_roc = 0.0
    best_hit1_ep = 0.0;best_hit3_ep = 0.0;best_hit10_ep = 0.0; best_hit20_ep = 0.0; best_hit50_ep = 0.0; best_hit100_ep = 0.0
    minimum_node_degree_history = []
    roc_history = []
    modification_ratio_history = []
    radius_hist = []          # mean compactness per epoch (↓ better)
    add_hist = []             # added edges per epoch
    remove_hist = []          # removed edges per epoch

    # which epoch each radius entry comes from
    radius_epoch_hist = []

    # store validation Hit@K per epoch (each entry is a list of 6 floats)
    val_hit_history = []
    
    # --- PRUNE sweep logging state & helper (for ver starting with "prune_") ---
    prune_state = None

    # --- sweep-style CSV logger for prune_* versions (NOW uses CURRENT Z + drop orphan nodes) ---
    def _append_prune_sweep_row(
        Z_cur: torch.Tensor,
        val_roc: float, val_ap: float, val_hit: list,
        test_roc: float, test_ap: float, test_hit: list,
    ):
        """
        Append one sweep-style row using the *CURRENT* embedding Z_cur
        + FIXED GMM labels.

        New behavior:
          - Nodes with label=-1 are ignored.
          - Nodes that have NO intra-cluster neighbors in the *current* graph
            are also ignored for CP/C0p radius (treated as removed from CP).
        """
        if prune_state is None:
            return

        # 1) Current Z and labels
        Z_cur = Z_cur.detach()
        labels_np: np.ndarray = prune_state["labels"]      # shape [N], -1 for noise
        device = Z_cur.device

        # 2) Per-node radii r[i] = 1 - cos( x_i, c_{label(i)} ) for current Z_cur
        radii_cp_all: torch.Tensor = _cosine_radii_to_centroid(Z_cur, labels_np)  # [N]
        lbl_t = torch.from_numpy(labels_np.astype(np.int64)).to(device)
        mask_non_noise: torch.Tensor = (lbl_t != -1)

        # 3) Compute which nodes still have ≥1 *real* intra-cluster neighbor
        #    (ignore self-loops) using the current graph after pruning.
        g_now = adj_label.to_dense().to(device)  # [N,N], with diag=1
        g_now = g_now.clone()
        g_now.fill_diagonal_(0)                  # <-- critical: do NOT count self-loop

        same = (lbl_t[:, None] == lbl_t[None, :]) & (lbl_t[:, None] != -1)
        intra = (g_now > 0) & same
        deg_intra = intra.sum(dim=1)                # [N]
        has_intra = (deg_intra > 0)                 # bool [N]

        # valid CP nodes: non-noise AND still have at least one intra-cluster neighbor
        mask_valid_cp = mask_non_noise & has_intra

        # ---- CP radius stats (node-level, after dropping orphan nodes) ----
        if mask_valid_cp.any():
            cp_vals = radii_cp_all[mask_valid_cp]
        else:
            # fallback: if all nodes lost intra neighbors, use all nodes to avoid NaNs
            cp_vals = radii_cp_all

        r_cp_mean = float(cp_vals.mean().item())
        r_cp_med  = float(cp_vals.median().item())

        # tail-sensitive stats
        try:
            r_cp_p90 = float(torch.quantile(cp_vals, 0.90).item())
        except Exception:
            r_cp_p90 = float(
                np.quantile(cp_vals.detach().cpu().numpy(), 0.90)
            )
        r_cp_max = float(cp_vals.max().item())

        # ---- C0p core selection on CURRENT graph degrees (excl self) ----
        deg_now = _deg_excl_self(g_now)  # [N], excl self

        core_now, _, _, _ = select_gmm_cores(
            Z_cur, labels_np, degrees_excl_self=deg_now,
            alpha=prune_state["alpha"], gamma=prune_state["gamma"],
            B=Z_cur.size(1),
        )
        # core nodes that are also valid CP nodes (non-noise + have intra neighbors)
        mask_core_now = core_now.bool().to(device) & mask_valid_cp

        if mask_core_now.any():
            c0p_vals   = radii_cp_all[mask_core_now]
            r_c0p_mean = float(c0p_vals.mean().item())
            try:
                r_c0p_p90 = float(torch.quantile(c0p_vals, 0.90).item())
            except Exception:
                r_c0p_p90 = float(
                    np.quantile(c0p_vals.detach().cpu().numpy(), 0.90)
                )
            r_c0p_max = float(c0p_vals.max().item())
        else:
            r_c0p_mean = float("nan")
            r_c0p_p90  = float("nan")
            r_c0p_max  = float("nan")

        # 3) meta info for this sweep point
        removed_edges = int(prune_state["removed"])
        num_cand      = int(max(1, prune_state["num_cand"]))  # avoid /0
        frac_removed  = removed_edges / float(num_cand)
        c0p_key = f"radius_c0p_a{prune_state['alpha']:g}_g{prune_state['gamma']:g}"

        # 4) compose row
        row = {
            "frac_removed": frac_removed,
            "removed_edges": removed_edges,

            # CP radius stats (current Z_cur, non-orphan nodes only)
            "radius_cp_mean":   r_cp_mean,
            "radius_cp_median": r_cp_med,
            "radius_cp_p90":    r_cp_p90,
            "radius_cp_max":    r_cp_max,

            # C0p radius stats (current Z_cur + current degree, non-orphan)
            c0p_key:            r_c0p_mean,
            f"{c0p_key}_p90":   r_c0p_p90,
            f"{c0p_key}_max":   r_c0p_max,

            # metrics at this sweep point
            "val_roc":   float(val_roc),
            "val_ap":    float(val_ap),
            "val_hit1":  float(val_hit[0]),
            "val_hit3":  float(val_hit[1]),
            "val_hit10": float(val_hit[2]),
            "test_roc":   float(test_roc),
            "test_ap":    float(test_ap),
            "test_hit1":  float(test_hit[0]),
            "test_hit3":  float(test_hit[1]),
            "test_hit10": float(test_hit[2]),
        }

        prune_state["rows"].append(row)
        prune_state["last_logged_removed"] = removed_edges




    # train model
    neighbors = {}

    for u, v in zip(edge_index[0], edge_index[1]):
        if u not in neighbors:
            neighbors[u] = set()
        if v not in neighbors:
            neighbors[v] = set()
        neighbors[u].add(v)
        neighbors[v].add(u)

    # common_neighbors_count = {}
    common_neighbors_count = np.zeros((num_nodes, num_nodes), dtype=int)
    total = 0
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):  
            if u in neighbors and v in neighbors:
                common_neighbors = neighbors[u].intersection(neighbors[v])
                common_neighbors_count[u][v] = common_neighbors_count[v][u] = len(common_neighbors)
                total += len(common_neighbors)
    avg_cn_cnt = float(total) / (num_nodes*(num_nodes-1)/2)
    
    feat_sim = cosine_similarity(features.to_dense().cpu())
    
    # Degree percentile -> absolute degree threshold (NB: adj_label currently includes diag)
    degree = np.array(adj_label.to_dense().cpu().sum(0)).squeeze()
    degree = np.array(sorted(degree))
    topk_idx = int(degree.shape[0] * degree_ratio)
    degree_threshold = degree[topk_idx]
    print(f"degree_threshold (incl self-loop) : {degree_threshold}")
    
    # -------------------
    # PRETRAIN (only for frozen-score variants)
    # -------------------
    frozen_scores = None
    need_frozen = ver in ["aron_desc", "aron_asc", "aron_desc_intra", "aron_desc_inter"]
    
    is_remove_only = isinstance(ver, str) and ver.startswith("remove_only_")
    needs_scores   = isinstance(ver, str) and ver.startswith("aron_") and ("desc" in ver or "asc" in ver)
    
    def _iter_edge_pairs(*edge_sets):
        """Yield (r, c) pairs from lists or numpy arrays uniformly."""
        for edges in edge_sets:
            if isinstance(edges, np.ndarray):
                for r, c in edges:
                    yield int(r), int(c)
            elif isinstance(edges, list):
                for r, c in edges:
                    yield int(r), int(c)
            else:
                # fallback: try to iterate whatever it is
                for r, c in list(edges):
                    yield int(r), int(c)

    # --- build a forbid mask for (val + test) edges to avoid leakage ---
    forbid_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    for r, c in _iter_edge_pairs(val_edges, test_edges):
        forbid_mask[r, c] = True
        forbid_mask[c, r] = True
    forbid_mask.fill_diagonal_(True)

    if need_frozen:
        loaded = False
        if frozen_scores_path and os.path.exists(frozen_scores_path):
            try:
                print(f"[pretrain] Loading frozen scores from {frozen_scores_path}")
                frozen_scores = torch.load(frozen_scores_path, map_location=device)
                loaded = True
            except Exception as e:
                print(f"[pretrain] Failed to load frozen scores: {e}")

        if pretrained_ckpt_path and os.path.exists(pretrained_ckpt_path):
            try:
                print(f"[pretrain] Loading encoder checkpoint from {pretrained_ckpt_path}")
                encoder.load_state_dict(torch.load(pretrained_ckpt_path, map_location=device))
                loaded = True  # at least weights are aligned
            except Exception as e:
                print(f"[pretrain] Failed to load encoder checkpoint: {e}")

        if not loaded:
            print(f"[pretrain] Starting pretraining for {pretrain_epochs} epochs on ORIGINAL graph (no augmentation).")
            for pe in range(pretrain_epochs):
                encoder.train()
                optimizer.zero_grad()
                Z0 = encoder(features, edge_index)  # original edges
                A_pred0 = dot_product_decode(Z0)
                recon0 = loss_function(A_pred0, adj_label, encoder.mean, encoder.logstd, norm, weight_tensor, alpha, beta, train_mask)
                # only intra-view CL during pretrain
                if loss_ver == "nei":
                    cl0 = inter_view_CL_loss(device, Z0, Z0, adj_label, gamma, temperature)
                else:
                    cl0 = intra_view_CL_loss(device, Z0, adj_label, gamma, temperature)
                loss0 = recon0 + cl0
                loss0.backward()
                optimizer.step()
                if (pe+1) % 20 == 0 or pe == 0:
                    print(f"[pretrain] epoch {pe+1}/{pretrain_epochs} loss={float(loss0):.4f}")

            with torch.no_grad():
                Z0 = encoder(features, edge_index)
                frozen_scores = dot_product_decode(Z0).detach()

            if frozen_scores_path:
                try:
                    torch.save(frozen_scores.cpu(), frozen_scores_path)
                    print(f"[pretrain] Saved frozen scores -> {frozen_scores_path}")
                except Exception as e:
                    print(f"[pretrain] Save frozen scores failed: {e}")
            # only enforce shape if needed
            if needs_scores:
                if frozen_scores is None or frozen_scores.shape != (N, N):
                    print("[WARN] score-based adds disabled (no valid frozen_scores).")
                    frozen_scores = None
            if pretrained_ckpt_path:
                try:
                    torch.save(encoder.state_dict(), pretrained_ckpt_path)
                    print(f"[pretrain] Saved encoder checkpoint -> {pretrained_ckpt_path}")
                except Exception as e:
                    print(f"[pretrain] Save encoder checkpoint failed: {e}")

        # If we loaded only the scores (not weights), that’s fine.
        # If we loaded weights too, great. Either way, re-init optimizer so both DESC/ASC start fresh.
        optimizer = Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if frozen_scores is None:
            with torch.no_grad():
                Z0 = encoder(features, edge_index)
                frozen_scores = dot_product_decode(Z0).detach()
        print(f"[pretrain] frozen_scores checksum: {float(frozen_scores.sum().item()):.6f}")
    
    # --- dataset-specific init (run this right after you load the graph for *this* dataset) ---
    g0 = adj_label.to_dense().clone()               # current dataset, WITH self-loops
    N  = g0.size(0)

    E0 = int(((g0.sum() - torch.trace(g0)) // 2).item())   # ORIGINAL undirected |E|
    deg0_excl_self = _deg_excl_self(g0)                    # shape [N], ORIGINAL degrees (excl self)

    # cumulative trackers (reset for this dataset/run)
    global_used_adds = 0
    global_used_removals = 0             # count removals across all epochs
    node_used_adds   = torch.zeros(N, dtype=torch.long, device=g0.device)

    # sanity checks (helpful when switching datasets)
    print(f"[AUG-INIT] N={N} | E0={E0} | aug_ratio(global)={aug_ratio} | aug_bound(per-node)={aug_bound}")
    if frozen_scores is not None and frozen_scores.shape != (N, N):
        raise ValueError(f"frozen_scores shape {tuple(frozen_scores.shape)} != {(N,N)} for current dataset")
    # -------------------- PRUNE INIT (for ver: prune_*) --------------------
    prune_state = None
    if isinstance(ver, str) and ver.startswith("prune_"):
        print(f"[PRUNE] initializing '{ver}'")

        # Base embedding on ORIGINAL train graph (edge_index) for ranking
        with torch.no_grad():
            Z_base = encoder(features, edge_index)

        # Fixed GMM labels and fixed core mask from base Z
        labels_fixed = gmm_labels(Z_base.detach(), K=gmm_k, tau=gmm_tau, metric="cosine")
        deg_base = _deg_excl_self(adj_label.to_dense())
        core_mask_fixed, _, _, _ = select_gmm_cores(
            Z_base, labels_fixed, degrees_excl_self=deg_base,
            alpha=restrict_alpha, gamma=restrict_gamma, B=Z_base.size(1)
        )
        core_t = core_mask_fixed.bool()
        lbl_t  = torch.from_numpy(labels_fixed.astype(np.int64)).to(core_t.device)

        # Candidates from CURRENT adj_label (with self-loops)
        g0_dense  = adj_label.to_dense()
        existing  = (g0_dense > 0)
        same_cl   = (lbl_t[:, None] == lbl_t[None, :]) & (lbl_t[:, None] != -1)
        both_core = (core_t[:, None] & core_t[None, :])

        if ver == "prune_cp_all":
            cand = existing.clone()              # (includes inter; switch to &same_cl for intra-only)
        elif ver == "prune_c0p_only":
            cand = existing & same_cl & both_core
        elif ver == "prune_cp_minus_c0p":
            cand = existing & same_cl & (~both_core)
        else:
            raise ValueError(f"[PRUNE] unknown ver={ver}")

        cand.fill_diagonal_(False)
        ii, jj = cand.triu(1).nonzero(as_tuple=True)
        num_cand = int(ii.numel())

        if num_cand == 0:
            print("[PRUNE] no candidates; disabling prune mode.")
        else:
            # Only record how many initial candidates we have; ranking will be done
            # dynamically using CURRENT Z each time we prune.
            step_edges = max(1, int(round(prune_step_frac * num_cand)))
            max_remove = max(1, int(round(prune_max_frac  * num_cand)))

            csv_dir  = os.path.join("logs", "c0p_sweep", dataset_str)
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f"{dataset_str}_{ver}_seed{seed}_idx{run_tag}.csv")

            prune_state = {
                "removed": 0,
                "num_cand": num_cand,
                "step": step_edges,
                "max_remove": max_remove,
                "labels": labels_fixed,                        # numpy
                "core_mask": core_mask_fixed.bool().cpu().numpy(),
                "alpha": float(restrict_alpha),
                "gamma": float(restrict_gamma),
                "rows": [],
                "csv_dir": csv_dir,
                "csv_path": csv_path,
                "last_logged_removed": -1,
            }

            print(f"[PRUNE] candidates={num_cand} step={step_edges} max={max_remove}")
    
    best_val_roc = -float("inf")
    best_epoch = -1
    best_state_cpu = None   # store on CPU to save GPU mem
    best_val_ap = 0.0

    # optional: a convenient on-disk checkpoint path (safe default)
    ckpt_dir = os.path.join("checkpoints", dataset_str)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path_runtime = os.path.join(ckpt_dir, f"{ver}_{run_tag or 'run'}_best.pt")

    for epoch in tqdm(range(num_epoch)):
        t = time.time()
        t1 = time.time()
        encoder.train()
        #print(f"trn time1 {time.time()-t1:.2f} s", flush=True)
        optimizer.zero_grad()
        
        Z = encoder(features, edge_index) # Z = encoder(features, adj_norm)
        hidden_repr = encoder.Z

        # original loss
        A_pred = dot_product_decode(Z)

                # adjusted_weight_tensor = weight_tensor * torch.abs(A_pred.view(-1)[train_mask] - adj_label.to_dense().view(-1)[train_mask]).detach()
                # recon_loss = loss_function(A_pred, adj_label, encoder.mean, encoder.logstd, norm, adjusted_weight_tensor, alpha, beta, train_mask)
        recon_loss = loss_function(A_pred, adj_label, encoder.mean, encoder.logstd, norm, weight_tensor, alpha, beta, train_mask)
        if(loss_ver=="nei"):
            ori_intra_CL = inter_view_CL_loss(device, Z, Z, adj_label, gamma, temperature)
        else:
            ori_intra_CL = intra_view_CL_loss(device, Z, adj_label, gamma, temperature)
        loss = recon_loss + ori_intra_CL
        
        # Generate K graphs
        if epoch % 10 == 0:
            if epoch != 0:
                del aug_edge_index # del aug_edge_weights # del aug_adj_labels # del aug_norms # del aug_weight_tensors
                torch.cuda.empty_cache()
            #print(f'loss: {loss}, recon loss: {recon_loss}')

            ###
            k = (num_nodes-1) * num_nodes * aug_ratio
            if(ver=="origin"):
                Augmentation_Time_start  = time.time()
                print(f"Time {random.random()}")
                g, modification_ratio = Graph_Modify_Constraint(Z.detach(), adj_label.to_dense(), int(k), aug_bound)
                aug_feat = features
                print(f"Augmentation Time {time.time() - Augmentation_Time_start}")
            elif(ver=="thm_exp"):
                g, modification_ratio = Graph_Modify_Constraint_exp(Z.detach(), adj_label.to_dense(), int(k), aug_bound)
            elif(ver=="random"):
                g, modification_ratio = aug_random_edge(adj_label.to_dense(),aug_ratio)
                aug_feat = features
            elif(ver=="local"):
                g, modification_ratio = Graph_Modify_Constraint_local(Z.detach(), adj_label.to_dense(), int(k), aug_bound, common_neighbors_count, avg_cn_cnt)
            elif(ver=="feat"):
                g, modification_ratio = Graph_Modify_Constraint_feat(Z.detach(), adj_label.to_dense(), int(k), aug_bound, feat_sim)
            elif(ver=="uncover"):
                g, modification_ratio = degree_aug(Z.detach(), adj_label.to_dense(),degree, num_nodes, aug_ratio, degree_threshold, epoch)
            elif(ver=="v2"):
                g, modification_ratio = degree_aug_v2(Z.detach(), adj_label.to_dense(),degree, num_nodes, aug_ratio, degree_threshold, epoch) 
                aug_feat = drop_feature(features.to_dense(), feat_maske_ratio)           
            elif(ver=="v3"):
                g, modification_ratio = degree_aug_v3(Z.detach(), adj_label.to_dense(),degree, num_nodes, aug_ratio, degree_threshold, epoch)
            elif(ver=="v4"):
                g, modification_ratio = degree_aug_v4(Z.detach(), adj_label.to_dense(),degree, num_nodes, aug_ratio, degree_threshold, epoch)
                aug_feat = drop_feature(features.to_dense(), feat_maske_ratio)                
            elif(ver=="v5"):
                g, modification_ratio = degree_aug_v5(Z.detach(), adj_label.to_dense(),degree, num_nodes, aug_ratio, degree_threshold, epoch)
                aug_feat = drop_feature(features.to_dense(), feat_maske_ratio)    
            elif(ver=="v6"):
                g, modification_ratio = degree_aug_v6(Z.detach(), adj_label.to_dense(),degree, num_nodes, aug_ratio, degree_threshold, epoch)
                aug_feat = drop_feature(features.to_dense(), feat_maske_ratio)     
            # ***** NEW: frozen-score A/B variants *****
            elif ver in ("aron_desc", "aron_asc"):
                degree_floor = max(0, int(degree_threshold) - 1)   # align with excl-self
                order = "desc" if ver == "aron_desc" else "asc"

                # ---- BEFORE augmentation snapshot ----
                # _ = degree_deficit_snapshot(
                #     adj_label=adj_label.to_dense(),
                #     degree_floor=degree_floor,
                #     out_dir="logs/deg_deficit",
                #     tag="before",
                #     epoch=epoch,                    # assuming you have `epoch` in scope
                #     writer=writer if 'writer' in locals() else None,
                # )

                # ---- epoch-bounded quota (fraction of E0 for THIS epoch) ----
                def _epoch_target_frac(ep: int, T: int, mode: str = "linear") -> float:
                    if mode == "cosine":
                        import math
                        return 0.5 * (1.0 - math.cos(math.pi * float(ep + 1) / max(1, T)))
                    return float(ep + 1) / max(1, T)

                # If user passed aug_ratio_epoch, obey it; else throttle by cumulative schedule.
                if aug_ratio_epoch is None:
                    target_cum_edges = int(round(aug_ratio * E0 * _epoch_target_frac(epoch, num_epoch, mode="linear")))
                    epoch_quota_edges = max(0, target_cum_edges - int(global_used_adds))
                    epoch_quota_frac  = epoch_quota_edges / float(E0)
                else:
                    # explicit per-epoch cap from args, but don’t exceed remaining global budget
                    remaining_global = max(0, int(round(aug_ratio * E0)) - int(global_used_adds))
                    epoch_quota_edges = min(int(round(aug_ratio_epoch * E0)), remaining_global)
                    epoch_quota_frac  = epoch_quota_edges / float(E0)

                g, added_this_epoch, global_used_adds, node_used_adds = degree_aug_fill_deficit_from_scores_budget(
                    scores_frozen=frozen_scores,
                    adj_label=adj_label.to_dense(),
                    num_nodes=num_nodes,
                    degree_floor=degree_floor,
                    order=order,
                    forbid_mask=forbid_mask,
                    # ---- budgets ----
                    E0=E0,
                    aug_ratio=aug_ratio,                     # GLOBAL budget
                    global_used_adds=global_used_adds,       # cumulative count
                    deg0_excl_self=deg0_excl_self,           # ORIGINAL degrees
                    aug_bound=aug_bound,                     # PER-NODE cap fraction
                    node_used_adds=node_used_adds,           # cumulative per-node counts
                    aug_ratio_epoch=epoch_quota_frac,        # <-- epoch-bounded cap
                )

                adj_label = g
                modification_ratio = added_this_epoch / float(E0)  # this-epoch ratio for logging
                print(f"[AUG] this_epoch_add={added_this_epoch} "
                    f"| this_epoch_ratio={modification_ratio:.6f} "
                    f"| global_used_adds={global_used_adds} ({global_used_adds/E0:.4f} of E0)")

                aug_feat = features if feat_maske_ratio <= 0 else drop_feature(features.to_dense(), feat_maske_ratio)

            # ***** NEW: confidence-guided (desc) with cluster constraint *****
            elif ver in ("aron_desc_intra", "aron_desc_inter"):
                # ---- config & prep ----
                degree_floor = max(0, int(degree_threshold) - 1)
                order = "desc"
                Z_for_cluster = Z.detach()
                base_mode = "intra" if ver == "aron_desc_intra" else "inter"

                # ---- clustering (SELECT METHOD) ----
                # (keeps per-epoch behavior; move this before the epoch loop if you want "compute once")
                if cluster_method == "gmm":
                    labels_cluster = gmm_labels(
                        Z_for_cluster,            # torch.Tensor [N,d]
                        K=gmm_k,                  # e.g., 16/32
                        tau=gmm_tau,              # confidence → noise threshold
                        metric="cosine",
                    )
                elif cluster_method == "louvain":
                    A_np = (_to_dense(adj_label) > 0).detach().cpu().numpy().astype(np.int8)
                    np.fill_diagonal(A_np, 0)     # Louvain doesn’t need self-loops
                    labels_cluster = louvain_labels(A_np)
                else:
                    # default: your current HDBSCAN path (unchanged)
                    labels_cluster = density_cluster_embeddings(
                        Z_for_cluster,
                        method="hdbscan",
                        eps=dbscan_eps,
                        min_samples=dbscan_min_samples,
                        metric=dbscan_metric,
                    )

                # ---- (α,γ, d̂) restriction: compute once for this epoch (no in-place shrinking) ----
                allow_restricted = None
                if restricted:
                    deg_now = _deg_excl_self(adj_label.to_dense())
                    allow_restricted = build_alpha_gamma_allow_mask_gmm(
                        Z_for_cluster, labels_cluster, degrees_excl_self=deg_now,
                        alpha=restrict_alpha, gamma=restrict_gamma, B=Z_for_cluster.size(1),
                        exclude_noise=True, require_core_endpoint=True, within_threshold=True,
                    )
                    if allow_restricted is not None and allow_restricted.sum() == 0:
                        print("[RESTRICT] empty eligibility this epoch → disabling restriction")
                        allow_restricted = None

                print(f"[RESTRICT] active={allow_restricted is not None} (mode={base_mode})")

                # ---- epoch-bounded quota (compute remaining from SHARED budget) ----
                def _epoch_target_frac(ep: int, T: int, mode: str = "linear") -> float:
                    if mode == "cosine":
                        import math
                        return 0.5 * (1.0 - math.cos(math.pi * float(ep + 1) / max(1, T)))
                    return float(ep + 1) / max(1, T)

                # Use shared pool = (adds + removes)
                if aug_ratio_epoch is None:
                    target_cum_mods = int(round(aug_ratio * E0 * _epoch_target_frac(epoch, num_epoch, mode="linear")))
                    used_mods_global = int(global_used_adds + global_used_removals)
                    epoch_quota_edges = max(0, target_cum_mods - used_mods_global)
                else:
                    remaining_global = max(0, int(round(aug_ratio * E0)) - int(global_used_adds + global_used_removals))
                    epoch_quota_edges = min(int(round(aug_ratio_epoch * E0)), remaining_global)

                print(f"[AUG-{base_mode}] epoch={epoch}/{num_epoch-1} | E0={E0} | global_cap={int(round(aug_ratio*E0))} "
                    f"| used={global_used_adds + global_used_removals} | epoch_quota_edges={epoch_quota_edges}")

                # ---- EARLY C0p prune (optional): removals spend from the SAME budget ----
                g_work = adj_label.to_dense()
                added_this_epoch = 0
                removed_this_epoch = 0

                if (cluster_method == "gmm") and (c0p_prune_frac > 0.0) and epoch_quota_edges > 0:
                    try:
                        g_work.fill_diagonal_(1)  # pruning helper expects self-loops present
                        g_work, cstats = c0p_prune_outliers(
                            Z=Z_for_cluster,
                            adj_label=g_work,
                            labels=labels_cluster,
                            drop_frac=c0p_prune_frac,
                            alpha=restrict_alpha,
                            gamma=restrict_gamma,
                            prefer_low_sim=True,
                            min_keep=1,
                            degree_floor=degree_floor,
                            metric="cosine",
                        )
                        dropped = int(cstats["dropped_edges"])
                        if dropped > 0:
                            removed_this_epoch += dropped
                            global_used_removals += dropped
                        print(f"[C0P-PRUNE] early prune: drop_frac={c0p_prune_frac:.3f} | dropped_edges={dropped} "
                            f"| min_deg_after={cstats.get('min_deg_after_excl_self', -1)} "
                            f"| isolated={cstats.get('isolated', -1)}")
                    except Exception as e:
                        print(f"[C0P-PRUNE] ERROR during early prune: {e}")

                # ---- recompute remaining epoch quota after early prune (shared pool) ----
                if aug_ratio_epoch is None:
                    target_cum_mods = int(round(aug_ratio * E0 * _epoch_target_frac(epoch, num_epoch, mode="linear")))
                    used_mods_global = int(global_used_adds + global_used_removals)
                    remaining_quota_edges = max(0, target_cum_mods - used_mods_global)
                else:
                    target_epoch_mods = int(round(aug_ratio_epoch * E0))
                    used_mods_epoch = int(added_this_epoch + removed_this_epoch)
                    remaining_quota_edges = max(0, target_epoch_mods - used_mods_epoch)

                # ---- scores for ranking ----
                scores_for_rank = frozen_scores if frozen_scores is not None else _scores_from_Z(Z_for_cluster).detach()

                # ---- nearby mask for Tier C ----
                def _topk_mask_per_row(S: torch.Tensor, k: int) -> torch.Tensor:
                    Nloc = S.size(0)
                    k = max(0, min(k, Nloc))
                    if k == 0:
                        return torch.zeros_like(S, dtype=torch.bool)
                    idx = torch.topk(S, k=k, dim=1, largest=True).indices
                    mask = torch.zeros_like(S, dtype=torch.bool)
                    rows = torch.arange(Nloc, device=S.device).unsqueeze(1).expand_as(idx)
                    mask[rows, idx] = True
                    mask.fill_diagonal_(False)
                    return mask

                nearby_topk = topk_per_node
                nearby_mask = _topk_mask_per_row(scores_for_rank, nearby_topk)

                # ---- Tier D quality gate (score quantile) ----
                late_score_q = 0.90
                base_block = (forbid_mask | (g_work > 0))
                if allow_restricted is not None:             # restrict only in candidate/quantile phase
                    base_block = base_block | (~allow_restricted)

                score_thr = robust_score_quantile_from_scores(
                    scores_for_rank, base_block, q=late_score_q, max_elems=2_000_000,
                )

                # ---- build allow masks for tiers (base) ----
                allow_A = build_cluster_allow_mask(labels_cluster, mode=base_mode, exclude_noise=True,  device=device)
                allow_B = build_cluster_allow_mask(labels_cluster, mode=base_mode, exclude_noise=False, device=device)
                allow_C = build_cluster_allow_mask(labels_cluster, mode="inter",  exclude_noise=False, device=device) & nearby_mask
                allow_D = torch.ones_like(allow_A, dtype=torch.bool, device=device); allow_D.fill_diagonal_(False)

                # IMPORTANT: do NOT intersect allow_A/B/C/D here.
                # The restriction is applied only when merging the forbid mask in _run_tier.

                # ---- epoch-gated tier release ----
                progress = float(epoch + 1) / max(1, num_epoch)
                A_rel, B_rel, C_rel, D_rel = 0.00, 0.30, 0.60, 0.85  # tune if desired
                tiers: list[tuple[str, torch.Tensor]] = []
                if progress >= A_rel: tiers.append(("A", allow_A))
                if progress >= B_rel: tiers.append(("B", allow_B))
                if progress >= C_rel and base_mode == "intra": tiers.append(("C", allow_C))
                if progress >= D_rel: tiers.append(("D", allow_D))
                print(f"[TIER-{base_mode}] progress={progress:.3f} | eligible={[t for t,_ in tiers]}")

                # ---- pure helper: NO nonlocal; returns updated state ----
                def _run_tier(
                    allow_mask: torch.Tensor,
                    tier_name: str,
                    g_in: torch.Tensor,
                    global_used_adds_in: int,
                    node_used_adds_in: torch.Tensor,
                    quota_edges_this_call: int | None,
                ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
                    # 1) Start from tier allow
                    effective_allow = allow_mask

                    # 2) Apply α–γ–d eligibility ONLY for this selection call (if active)
                    if restricted and (allow_restricted is not None):
                        effective_allow = allow_mask & allow_restricted

                        # Make it robust: symmetric + zero diag
                        effective_allow = effective_allow & effective_allow.T
                        effective_allow.fill_diagonal_(False)

                        # If everything is wiped out once current forbids are considered, ignore restriction this call
                        # (so we can still add edges this epoch/tier)
                        if ((effective_allow & (~forbid_mask)).sum().item() == 0):
                            print(f"[RESTRICT] {tier_name}: no eligible pairs after gate → ignoring restriction for this call")
                            effective_allow = allow_mask
                        else:
                            kept_pairs = (effective_allow & (~forbid_mask)).float().mean().item()
                            kept_nodes = (effective_allow.any(dim=1)).float().mean().item()
                            print(f"[RESTRICT] {tier_name}: kept_pairs={kept_pairs:.4f} kept_nodes={kept_nodes:.4f}")

                    # 3) Merge forbids (scores/used edges/etc.) AFTER the local gate
                    forbid_mask_combined = forbid_mask | (~effective_allow)

                    # Extra forbid for Tier-D score-quantile (if you use it)
                    if tier_name == "D" and score_thr is not None:
                        forbid_mask_combined = forbid_mask_combined | (scores_for_rank < score_thr)

                    # 4) Per-call quota as fraction of E0 (optional)
                    quota_frac = None if quota_edges_this_call is None else max(0.0, float(quota_edges_this_call) / float(E0))

                    # 5) Run the degree-aware adder on CURRENT dense adj
                    g_out, added_out, global_used_out, node_used_out = degree_aug_fill_deficit_from_scores_budget(
                        scores_frozen=scores_for_rank,
                        adj_label=g_in,                     # CURRENT adj (dense)
                        num_nodes=num_nodes,
                        degree_floor=degree_floor,
                        order=order,
                        forbid_mask=forbid_mask_combined,   # <- restriction enforced only here
                        # budgets
                        E0=E0,
                        aug_ratio=aug_ratio,
                        global_used_adds=global_used_out if False else global_used_adds_in,  # keep original signature
                        deg0_excl_self=deg0_excl_self,
                        aug_bound=aug_bound,
                        node_used_adds=node_used_adds_in,
                        aug_ratio_epoch=quota_frac,         # per-call epoch slice
                    )
                    return g_out, added_out, global_used_out, node_used_out

                # ---- iterate tiers with exact epoch quota accounting (SHARED pool) ----
                tier_loop_added = 0
                for name, allow in tiers:
                    if remaining_quota_edges <= 0:
                        break

                    tier_added_total = 0
                    # Greedy within tier: keep adding until it stalls or quota is exhausted
                    while remaining_quota_edges > 0:
                        quota_for_call = remaining_quota_edges  # exact remaining for this call
                        g_work, added, global_used_adds, node_used_adds = _run_tier(
                            allow_mask=allow,
                            tier_name=name,
                            g_in=g_work,
                            global_used_adds_in=global_used_adds,
                            node_used_adds_in=node_used_adds,
                            quota_edges_this_call=quota_for_call,
                        )
                        tier_added_total += added
                        added_this_epoch += added
                        tier_loop_added  += added

                        # recompute remaining quota (adds + removals)
                        if aug_ratio_epoch is None:
                            target_cum_mods = int(round(aug_ratio * E0 * _epoch_target_frac(epoch, num_epoch, mode="linear")))
                            used_mods_global = int(global_used_adds + global_used_removals)
                            remaining_quota_edges = max(0, target_cum_mods - used_mods_global)
                        else:
                            target_epoch_mods = int(round(aug_ratio_epoch * E0))
                            used_mods_epoch = int(added_this_epoch + removed_this_epoch)
                            remaining_quota_edges = max(0, target_epoch_mods - used_mods_epoch)

                        print(f"[AUG-{base_mode}|Tier {name}] added={added} | tier_total={tier_added_total} "
                            f"| epoch_quota_left={remaining_quota_edges} "
                            f"| global_used={global_used_adds + global_used_removals}/{int(round(aug_ratio*E0))}")

                        if added == 0:
                            break  # this tier stalled; move to next tier

                # ---- finalize epoch changes ----
                adj_label = g_work
                g = g_work

                used_mods_global = int(global_used_adds + global_used_removals)   # SHARED pool
                modification_ratio = (added_this_epoch + removed_this_epoch) / float(E0)

                print(f"[AUG-{base_mode}] this_epoch_add={added_this_epoch} "
                    f"remove={removed_this_epoch} "
                    f"| this_epoch_ratio={modification_ratio:.6f} "
                    f"| global_used_mods={used_mods_global}/{int(round(aug_ratio*E0))} "
                    f"(adds={global_used_adds}, removes={global_used_removals})")

                aug_feat = features if feat_maske_ratio <= 0 else drop_feature(features.to_dense(), feat_maske_ratio)

            elif ver.startswith("remove_only_"):
                import re

                # per-epoch defaults to avoid UnboundLocalError
                modification_ratio = 0.0
                added_this_epoch = 0
                removed_this_epoch = 0

                # ------------------------------------------------------------
                # Inline parse from `ver`
                # Formats:
                #   remove_only_intra
                #   remove_only_inter
                #   remove_only_both
                #   + optional scope:  _c0p | _noncore | _cp           (for INTRA only)
                #   + optional frac :  _frac0.05                        (steady per-node fraction hint)
                # Examples:
                #   remove_only_intra_c0p
                #   remove_only_intra_cp_frac0.07
                #   remove_only_inter_frac0.03
                # ------------------------------------------------------------
                m = re.match(r"^remove_only_(intra|inter|both)(?:_(c0p|noncore|cp))?(?:_frac([0-9]*\.?[0-9]+))?$", ver)
                if not m:
                    print(f"[REMOVE-ONLY] Unrecognized ver={ver}; skipping this epoch.")
                    # keep graph & features unchanged this epoch
                    g = _to_dense(adj_label)
                    aug_feat = features if feat_maske_ratio <= 0 else drop_feature(features, feat_maske_ratio)
                else:
                    kind  = m.group(1)                  # "intra" | "inter" | "both"
                    scope = m.group(2) or "c0p"         # default scope for INTRA
                    frac_hint = float(m.group(3)) if m.group(3) else None

                    # ---- cluster labels ----
                    if cluster_method == "gmm":
                        labels_cluster = gmm_labels(Z.detach(), K=gmm_k, tau=gmm_tau, metric="cosine")
                    elif cluster_method == "louvain":
                        A_np = (_to_dense(adj_label) > 0).detach().cpu().numpy().astype(np.int8)
                        np.fill_diagonal(A_np, 0)
                        labels_cluster = louvain_labels(A_np)
                    else:
                        labels_cluster = density_cluster_embeddings(Z.detach(), method="hdbscan")

                    # ---- compactness BEFORE ----
                    # (radius uses Z & labels; it won’t change until the next epoch after updates)
                    radii, _, _ = per_cluster_stats_diag(Z.detach(), labels_cluster, normalize_cosine=True)
                    if (labels_cluster != -1).sum() > 0:
                        mean_radius = float(radii[torch.from_numpy(labels_cluster) != -1].mean().item())
                    else:
                        mean_radius = float(radii.mean().item())
                    print(f"[RADIUS] epoch={epoch} mean={mean_radius:.6f}")
                    radius_hist.append(mean_radius)
                    radius_epoch_hist.append(epoch)

                    # ---- epoch quota from GLOBAL pool (adds+removes shared) ----
                    # E0 is ORIGINAL undirected |E|, computed earlier
                    def _epoch_target_frac(ep: int, T: int) -> float:
                        return float(ep + 1) / max(1, T)

                    target_cum_mods = int(round(aug_ratio * E0 * _epoch_target_frac(epoch, num_epoch)))
                    used_mods_global = int(global_used_adds + global_used_removals)
                    epoch_quota_edges = max(0, target_cum_mods - used_mods_global)
                    print(f"[REMOVE-ONLY] epoch_quota_left={epoch_quota_edges} / global_cap={int(round(aug_ratio*E0))}")

                    dropped_intra = 0
                    dropped_inter = 0

                    if epoch_quota_edges > 0:
                        g_dense = _to_dense(adj_label)
                        lbl_t = torch.from_numpy(labels_cluster).to(g_dense.device)

                        same = (lbl_t[:, None] == lbl_t[None, :]) & (lbl_t[:, None] != -1)
                        diff = (lbl_t[:, None] != lbl_t[None, :]) | (lbl_t[:, None] == -1) | (lbl_t[None, :] == -1)

                        # undirected counts
                        intra_edges = int(((g_dense > 0) & same).sum().item() // 2)
                        inter_edges = int(((g_dense > 0) & diff).sum().item() // 2)

                        # split quota by kind
                        if kind == "intra":
                            q_intra, q_inter = epoch_quota_edges, 0
                        elif kind == "inter":
                            q_intra, q_inter = 0, epoch_quota_edges
                        else:
                            # both: 50/50, but shift to the other side if one empty
                            if intra_edges == 0 and inter_edges > 0:
                                q_intra, q_inter = 0, epoch_quota_edges
                            elif inter_edges == 0 and intra_edges > 0:
                                q_intra, q_inter = epoch_quota_edges, 0
                            else:
                                q_intra = epoch_quota_edges // 2
                                q_inter = epoch_quota_edges - q_intra

                        # per-node drop fractions (allow steady override via _fracX)
                        if frac_hint is not None:
                            drop_intra = max(0.0, min(0.9, float(frac_hint)))
                            drop_inter = max(0.0, min(0.9, float(frac_hint)))
                        else:
                            drop_intra = max(0.0, min(0.6, q_intra / max(1, intra_edges)))
                            drop_inter = max(0.0, min(0.6, q_inter / max(1, inter_edges)))

                        degree_floor = max(0, int(degree_threshold) - 1)

                        # ---- INTRA prune ----
                        if q_intra > 0 and intra_edges > 0:
                            new_adj, stats_intra = cp_prune_intra(
                                Z=Z.detach(),
                                adj_label=g_dense,
                                labels=labels_cluster,
                                scope=scope,                      # "c0p" | "noncore" | "cp"
                                drop_frac=drop_intra,
                                prefer_low_sim=True,
                                min_keep=1,
                                degree_floor=degree_floor,
                                metric="cosine",
                            )
                            dropped_intra = int(stats_intra["dropped_edges"])
                            adj_label = new_adj
                            print(f"[REMOVE-ONLY:INTRA] scope={scope} drop_frac≈{drop_intra:.4f} dropped={dropped_intra}")

                        # ---- INTER prune ----
                        if q_inter > 0 and inter_edges > 0:
                            new_adj, stats_inter = cp_prune_inter(
                                Z=Z.detach(),
                                adj_label=adj_label,              # current (may already be dense from INTRA)
                                labels=labels_cluster,
                                drop_frac=drop_inter,
                                prefer_low_sim=True,
                                min_keep=1,
                                degree_floor=degree_floor,
                                metric="cosine",
                                include_noise=True,
                            )
                            dropped_inter = int(stats_inter["dropped_edges"])
                            adj_label = new_adj
                            print(f"[REMOVE-ONLY:INTER] drop_frac≈{drop_inter:.4f} dropped={dropped_inter}")

                    # bookkeeping: adds=0 in remove-only
                    added_this_epoch = 0
                    removed_this_epoch = dropped_intra + dropped_inter
                    global_used_adds     += added_this_epoch
                    global_used_removals += removed_this_epoch

                    # per-epoch modification ratio (relative to ORIGINAL undirected |E| = E0)
                    modification_ratio = (added_this_epoch + removed_this_epoch) / float(max(1, E0))

                    print(f"[BUDGET] add/remove this epoch = {added_this_epoch}/{removed_this_epoch} "
                        f"(intra={dropped_intra}, inter={dropped_inter}) | "
                        f"global_used adds/removes = {global_used_adds}/{global_used_removals}")

                    # keep feature aug behavior consistent with others
                    aug_feat = features if feat_maske_ratio <= 0 else drop_feature(features, feat_maske_ratio)

                    # optional: maintain `g` mirror like other branches
                    g = _to_dense(adj_label)

            # ===== NEW: online prune family =====
            elif ver.startswith("prune_"):
                # measure radius BEFORE pruning this epoch (live radius plot, dropping orphan nodes)
                try:
                    if prune_state is not None and "labels" in prune_state:
                        labels_np = prune_state["labels"]
                        device_r  = Z.device

                        # per-node radii for current Z
                        radii_cur, _, _ = per_cluster_stats_diag(
                            Z.detach(), labels_np, normalize_cosine=True
                        )

                        # current graph (after previous pruning), with self-loops
                        g_now = adj_label.to_dense().to(device_r)
                        g_now = g_now.clone()
                        g_now.fill_diagonal_(0)   # <-- ignore self-loop

                        lbl_t = torch.from_numpy(labels_np.astype(np.int64)).to(device_r)

                        same = (lbl_t[:, None] == lbl_t[None, :]) & (lbl_t[:, None] != -1)
                        intra = (g_now > 0) & same
                        deg_intra = intra.sum(dim=1)
                        has_intra = (deg_intra > 0)

                        mask_non_noise = (lbl_t != -1)
                        mask_valid = mask_non_noise & has_intra

                        if mask_valid.any():
                            r_mean = float(radii_cur[mask_valid].mean().item())
                        else:
                            r_mean = float(radii_cur.mean().item())

                        radius_hist.append(r_mean)
                        radius_epoch_hist.append(epoch)
                except Exception as e:
                    print(f"[RADIUS|prune] skip logging due to error: {e}")


                if (prune_state is None) or (prune_state["removed"] >= prune_state["max_remove"]):
                    g = adj_label.to_dense()
                    modification_ratio = 0.0
                    added_this_epoch = 0
                    removed_this_epoch = 0
                else:
                    # --- NEW: dynamic pruning using CURRENT Z & CURRENT graph ---
                    g = adj_label.to_dense()
                    drop_quota = min(
                        prune_state["step"],
                        prune_state["max_remove"] - prune_state["removed"],
                    )

                    if drop_quota <= 0:
                        added_this_epoch = 0
                        removed_this_epoch = 0
                        modification_ratio = 0.0
                    else:
                        # 1) Build candidate mask on CURRENT graph
                        labels_np = prune_state["labels"]           # fixed clusters
                        core_np   = prune_state["core_mask"]        # fixed core mask
                        device_r  = g.device

                        lbl_t  = torch.from_numpy(labels_np.astype(np.int64)).to(device_r)
                        core_t = torch.from_numpy(core_np.astype(np.bool_)).to(device_r)

                        existing  = (g > 0)
                        same_cl   = (lbl_t[:, None] == lbl_t[None, :]) & (lbl_t[:, None] != -1)
                        both_core = (core_t[:, None] & core_t[None, :])

                        if ver == "prune_cp_all":
                            cand = existing.clone()              # all edges
                        elif ver == "prune_c0p_only":
                            cand = existing & same_cl & both_core
                        elif ver == "prune_cp_minus_c0p":
                            cand = existing & same_cl & (~both_core)
                        else:
                            cand = existing.clone()              # fallback

                        cand.fill_diagonal_(False)
                        ii, jj = cand.triu(1).nonzero(as_tuple=True)

                        if ii.numel() == 0:
                            print("[PRUNE] no more candidate edges; stopping prune here.")
                            added_this_epoch = 0
                            removed_this_epoch = 0
                            modification_ratio = 0.0
                        else:
                            # 2) Compute per-node radius from CURRENT Z
                            radii_cur, _, _ = per_cluster_stats_diag(
                                Z.detach(), labels_np, normalize_cosine=True
                            )
                            radii_cur = radii_cur.to(device_r)

                            r_i = radii_cur[ii]
                            r_j = radii_cur[jj]
                            edge_score = torch.maximum(r_i, r_j)   # high-radius edges first

                            order = torch.argsort(edge_score, descending=True)
                            num_to_drop = min(drop_quota, int(order.numel()))
                            sel = order[:num_to_drop]
                            sel_i = ii[sel]
                            sel_j = jj[sel]

                            # 3) Drop those edges
                            removed_this_epoch = 0
                            for a, b in zip(sel_i.tolist(), sel_j.tolist()):
                                if g[a, b] > 0:
                                    g[a, b] = 0.0
                                    g[b, a] = 0.0
                                    removed_this_epoch += 1

                            prune_state["removed"] += int(removed_this_epoch)
                            added_this_epoch   = 0
                            modification_ratio = removed_this_epoch / float(max(1, E0))

                            adj_label = g  # reflect into training adjacency

                aug_feat = features if feat_maske_ratio <= 0 else drop_feature(features, feat_maske_ratio)
                g = adj_label.to_dense()

            elif(ver=="no"):
                g = adj_label.to_dense()
                modification_ratio = 0
                aug_feat = features
            else:
                raise NotImplementedError(
                    f"[AUG] ver='{ver}' is not implemented. "
                )
                            
            aug_edge_index = g.to_sparse().indices()

            # ★★ 只有 remove_only_* 和 prune_* 才更新 base view graph
            if (ver.startswith("remove_only_")) or (ver.startswith("prune_")):
                edge_index = aug_edge_index
            
            # Aron (for minimum node degree)
            # Node degree calculation (ignoring self-loops)
            node_degrees = torch.sum(g, dim=1) - torch.diag(g)

            # Calculate minimum and average node degrees
            node_degrees = torch.sum(g, dim=1) - torch.diag(g)
            min_degree = torch.min(node_degrees)
            #print(f'Minimum node degree: {min_degree.item()}')
            minimum_node_degree_history.append(min_degree.item())

        modification_ratio_history.append(modification_ratio)
        add_hist.append(int(added_this_epoch))
        remove_hist.append(int(removed_this_epoch))

        # Calcualte Augment View
        # bias_Z = encoder(features, aug_edge_index)
        bias_Z = encoder(aug_feat, aug_edge_index)

        # Overall losses        
        loss += inter_view_CL_loss(device, hidden_repr, encoder.Z.detach(), adj_label, delta, temperature)
        aug_loss = loss_function(dot_product_decode(bias_Z), adj_label, encoder.mean, encoder.logstd, norm, weight_tensor, alpha, beta, train_mask) # aug_loss = loss_function(dot_product_decode(bias_Z), aug_adj_labels[i], encoder.mean, encoder.logstd, aug_norms[i], aug_weight_tensors[i], alpha, beta, train_mask)
        # if(loss_ver=="nei"):
            # intra_CL = inter_view_CL_loss(device, bias_Z, bias_Z, adj_label, gamma, temperature)
        # else:
        intra_CL = intra_view_CL_loss(device, bias_Z, adj_label, gamma, temperature)
        aug_losses = aug_loss  + intra_CL
        loss += aug_losses * aug_graph_weight
        #print(f'aug_loss: {aug_loss}, intra_CL: {intra_CL}')

        # Update Model
        loss.backward()
        optimizer.step()
        
        del bias_Z
        del encoder.Z
        del encoder.mean
        del encoder.logstd
        del Z
        del hidden_repr
        torch.cuda.empty_cache()
        #print(f"trn time2 {time.time()-t1:.2f} s", flush=True)
        ########################################################
        # Evaluate edge prediction
        t1 = time.time()
        encoder.eval()
        # print(f"test time {time.time()-t1:.2f} s")
        with torch.no_grad():
            inference_time_start = time.time()
            Z = encoder(features, edge_index) # Z = encoder(features, adj_norm)
            A_pred = dot_product_decode(Z)
        
        # A_pred = train_decoder(device, encoder.Z.clone().detach(), adj_label, weight_tensor, norm, train_mask)
        # print(A_pred.shape)
        train_acc = get_acc(A_pred.data.cpu(), adj_label.data.cpu())
        val_roc, val_ap, val_hit = get_scores(dataset_str, val_edges, val_edges_false,
                                      A_pred.data.cpu().numpy(), adj_orig)
        test_roc, test_ap, test_hit = get_scores(dataset_str, test_edges, test_edges_false,
                                                A_pred.data.cpu().numpy(), adj_orig)

        # track validation curve instead of test
        roc_history.append(val_roc)              # per-epoch val ROC
        val_hit_history.append([float(h) for h in val_hit])   # per-epoch val Hit@K
        
        # --- NEW: sweep-style logging for prune_* ---
        if isinstance(ver, str) and ver.startswith("prune_") and (prune_state is not None):
            if epoch % 10 == 0:
                with torch.no_grad():
                    if prune_state["last_logged_removed"] != int(prune_state["removed"]):
                        _append_prune_sweep_row(Z, val_roc, val_ap, val_hit, test_roc, test_ap, test_hit)

        # val_acc, test_acc = logist_regressor_classification(device = device, Z = encoder.Z.clone().detach(), labels = labels, idx_train = idx_train, idx_val = idx_val, idx_test = idx_test)
        # print(f'Epoch: {epoch + 1}, train_loss= {loss.item():.4f}, train_acc= {train_acc:.4f}, val_roc= {val_roc:.4f}, val_ap= {val_ap:.4f}, test_roc= {test_roc:.4f}, test_ap= {test_ap:.4f}, time= {time.time() - t:.4f}')
        # print(f'Hit@K for val: 1={val_hit[0]}, 3={val_hit[1]}, 10={val_hit[2]}, 20={val_hit[3]}, 100={val_hit[4]}')
        # print(f'Hit@K for test: 1={test_hit[0]}, 3={test_hit[1]}, 10={test_hit[2]}, 20={test_hit[3]}, 100={test_hit[4]}')
        # print(f'Hit@K for test: 10={test_hit[0]}, 20={test_hit[1]}, 50={test_hit[2]}')
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     best_test_acc = test_acc
        #     best_classi_epoch = epoch
        #     print(f'Update Best Acc, Epoch = {epoch+1}, Val_acc = {best_acc:.3f}, Test_acc = {test_acc:.3f}')
        if test_hit[0] > best_hit1:
            best_hit1 = test_hit[0]
            best_hit1_roc = test_roc
            best_hit1_ep = epoch 
        if test_hit[1] > best_hit3:
            best_hit3 = test_hit[1]
            best_hit3_roc = test_roc
            best_hit3_ep = epoch 
        if test_hit[2] > best_hit10:
            best_hit10 = test_hit[2]
            best_hit10_roc = test_roc
            best_hit10_ep = epoch
        if test_hit[3] > best_hit20:
            best_hit20 = test_hit[3]
            best_hit20_roc = test_roc
            best_hit20_ep = epoch 
        if test_hit[4] > best_hit50:
            best_hit50 = test_hit[4]
            best_hit50_roc = test_roc
            best_hit50_ep = epoch 
        if test_hit[5] > best_hit100:
            best_hit100 = test_hit[5]
            best_hit100_roc = test_roc
            best_hit100_ep = epoch    
        if test_roc > best_roc:
            best_roc = test_roc
            best_test_roc = test_roc
            best_ap = val_ap
            best_test_ap = test_ap
            best_link_epoch = epoch
            #print(f'Update Best Roc, Epoch = {epoch+1}, Val_roc = {best_roc:.3f}, val_ap = {best_ap:.3f}, test_roc = {best_test_roc:.3f}, test_ap = {best_test_ap:.3f}')
        #print('-' * 100)
        
        # --------- SELECT BEST BY VALIDATION ROC (NO TEST LEAKAGE) ---------
        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_val_ap = val_ap
            best_epoch = epoch

            # store CPU copy of the best weights
            best_state_cpu = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}

            # also save to disk (optional, helpful for crashes / later reuse)
            try:
                torch.save(best_state_cpu, best_ckpt_path_runtime)
                print(f"[CKPT] Saved best-by-val ROC at epoch {epoch} -> {best_ckpt_path_runtime}")
            except Exception as e:
                print(f"[CKPT] Warning: failed to save best checkpoint: {e}")
                
        # --------- OPTIONAL: offline C0p sweep with fixed encoder & fixed GMM ---------
    # # side experiment：只在你開 ENABLE_C0P_SWEEP 時啟用
    # if ENABLE_C0P_SWEEP and is_remove_only and (cluster_method == "gmm"):
    #     # For offline sweep we don't enforce min-degree; we want to see the full curve
    #     sweep_degree_floor = 0
    #     print(f"[CP-SWEEP] start | ver={ver}, degree_floor={sweep_degree_floor} (no floor for sweep)")

    #     _ = run_c0p_sweep_static(
    #         dataset_str=dataset_str,
    #         ver=ver,
    #         seed=seed,
    #         run_tag=str(run_tag),
    #         device=device,
    #         encoder=encoder,
    #         features=features,
    #         adj_train=adj_train,
    #         adj_orig=adj_orig,
    #         val_edges=val_edges,
    #         val_edges_false=val_edges_false,
    #         test_edges=test_edges,
    #         test_edges_false=test_edges_false,
    #         gmm_k=gmm_k,
    #         gmm_tau=gmm_tau,
    #         alpha_c0p=restrict_alpha,
    #         gamma_c0p=restrict_gamma,
    #         degree_floor=sweep_degree_floor,  # = 0 → enforce_floor=False
    #         step_frac=C0P_SWEEP_STEP_FRAC,
    #         max_frac=C0P_SWEEP_MAX_FRAC,
    #         sweep_scope=sweep_scope,    # <-- NEW
    #         out_dir=None,
    #     )

    # print(f'best classification epoch = {best_classi_epoch+1}, val_acc = {best_acc:.3f}, test_acc = {best_test_acc:.3f}')
    print(f'best link prediction epoch = {best_link_epoch+1}, Val_roc = {best_roc:.3f}, val_ap = {best_ap:.3f}, test_roc = {best_test_roc:.3f}, test_ap = {best_test_ap:.3f}')
    print(f'best hit@1 epoch = {best_hit1_ep}, hit@1 = {best_hit1}, val = {best_hit1_roc}')
    print(f'best hit@3 epoch = {best_hit3_ep}, hit@3 = {best_hit3}, val = {best_hit3_roc}')
    print(f'best hit@10 epoch = {best_hit10_ep}, hit@10 = {best_hit10}, val = {best_hit10_roc}')
    print(f'best hit@20 epoch = {best_hit20_ep}, hit@20 = {best_hit20}, val = {best_hit20_roc}')
    print(f'best hit@50 epoch = {best_hit50_ep}, hit@50 = {best_hit50}, val = {best_hit50_roc}')
    print(f'best hit@100 epoch = {best_hit100_ep}, hit@100 = {best_hit100}, val = {best_hit100_roc}')

    # --------- RELOAD BEST MODEL BEFORE FINAL TEST ---------
    if best_state_cpu is not None:
        encoder.load_state_dict(best_state_cpu, strict=True)
        print(f"[CKPT] Reloaded best weights (by val ROC) from epoch {best_epoch}")
    elif os.path.exists(best_ckpt_path_runtime):
        # fallback if only on-disk exists
        try:
            state = torch.load(best_ckpt_path_runtime, map_location=device)
            encoder.load_state_dict(state, strict=True)
            print(f"[CKPT] Reloaded best weights from disk: {best_ckpt_path_runtime}")
        except Exception as e:
            print(f"[CKPT] ERROR loading best checkpoint; using last epoch weights: {e}")
    else:
        print("[CKPT] No best checkpoint found; using last epoch weights.")

    # --------- FINAL TEST EVAL WITH BEST MODEL ---------
    encoder.eval()
    with torch.no_grad():
        Z_best = encoder(features, edge_index)
        A_pred_best = dot_product_decode(Z_best)

    final_test_roc, final_test_ap, final_test_hit = get_scores(
        dataset_str, test_edges, test_edges_false, A_pred_best.data.cpu().numpy(), adj_orig
    )
    print(f"best link prediction epoch (by val ROC) = {best_epoch+1}, "
        f"val_roc = {best_val_roc:.3f}, val_ap = {best_val_ap:.3f}")
    print(f"[FINAL TEST] test_roc = {final_test_roc:.5f}, test_ap = {final_test_ap:.5f}")
    print(f"[FINAL TEST] Hit@K: 1={final_test_hit[0]}, 3={final_test_hit[1]}, 10={final_test_hit[2]}, "
        f"20={final_test_hit[3]}, 50={final_test_hit[4]}, 100={final_test_hit[5]}")
    
    print(f"Total training time {time.time() - training_time_start}")
    print(f"Average minimum node degree {sum(minimum_node_degree_history) / len(minimum_node_degree_history)}")
    print(minimum_node_degree_history)
    
    # ==================== PRUNE SWEEP CSV FINALIZE (online) ====================
    try:
        if isinstance(ver, str) and ver.startswith("prune_") and (prune_state is not None) and len(prune_state["rows"]) > 0:
            df = pd.DataFrame(prune_state["rows"])
            c0p_key = f"radius_c0p_a{prune_state['alpha']:g}_g{prune_state['gamma']:g}"
            cols = [
                "frac_removed","removed_edges",
                "radius_cp_mean","radius_cp_median","radius_cp_p90","radius_cp_max",
                c0p_key, f"{c0p_key}_p90", f"{c0p_key}_max",
                "val_roc","val_ap","val_hit1","val_hit3","val_hit10",
                "test_roc","test_ap","test_hit1","test_hit3","test_hit10",
            ]
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[cols]
            os.makedirs(prune_state["csv_dir"], exist_ok=True)
            df.to_csv(prune_state["csv_path"], index=False)
            print(f"[PRUNE|SWEEP] saved online sweep CSV: {prune_state['csv_path']}")
    except Exception as e:
        print(f"[PRUNE|SWEEP] finalize failed: {e}")
    
    # ==================== RADIUS HISTORY FINALIZE ====================
    try:
        if (not (isinstance(ver, str) and ver.startswith("prune_"))) and (len(radius_hist) > 0):
            out_dir = os.path.join("logs", "radius")
            os.makedirs(out_dir, exist_ok=True)

            # Stable run id (easy to aggregate across runs)
            run_id = f"{dataset_str}_{ver}_seed{seed}_idx{run_tag}"
            csv_path = os.path.join(out_dir, f"{run_id}.csv")

            # Align defensively: radius_hist[i] corresponds to epoch radius_epoch_hist[i]
            if len(radius_hist) != len(radius_epoch_hist):
                print(f"[RADIUS] WARNING: len(radius_hist)={len(radius_hist)} "
                    f"!= len(radius_epoch_hist)={len(radius_epoch_hist)}; "
                    f"using min length for export.")
            L = min(len(radius_hist), len(radius_epoch_hist))

            # Actual epochs where we measured radius (e.g., 0, 10, 20, ...)
            epochs = radius_epoch_hist[:L]

            # Radius values in the same order
            radius_vals = [float(radius_hist[i]) for i in range(L)]

            # Validation Hit@K for those epochs
            # (val_hit_history is indexed by epoch)
            val_hit1  = []
            val_hit3  = []
            val_hit10 = []
            val_hit20 = []
            val_hit50 = []
            val_hit100 = []
            for e in epochs:
                if e < len(val_hit_history):
                    h = val_hit_history[e]
                    val_hit1.append(h[0])
                    val_hit3.append(h[1])
                    val_hit10.append(h[2])
                    val_hit20.append(h[3])
                    val_hit50.append(h[4])
                    val_hit100.append(h[5])
                else:
                    # If for some reason we have fewer val entries, fill with NaN
                    val_hit1.append(float("nan"))
                    val_hit3.append(float("nan"))
                    val_hit10.append(float("nan"))
                    val_hit20.append(float("nan"))
                    val_hit50.append(float("nan"))
                    val_hit100.append(float("nan"))

            # Added/removed/mod_ratio values for those epochs
            added   = [add_hist[e] if e < len(add_hist) else 0 for e in epochs]
            removed = [remove_hist[e] if e < len(remove_hist) else 0 for e in epochs]
            mod     = [modification_ratio_history[e] if e < len(modification_ratio_history) else 0.0 for e in epochs]

            df = pd.DataFrame({
                "epoch": epochs,
                "radius_mean": radius_vals,
                "val_hit@1": val_hit1,
                "val_hit@3": val_hit3,
                "val_hit@10": val_hit10,
                "val_hit@20": val_hit20,
                "val_hit@50": val_hit50,
                "val_hit@100": val_hit100,
                "added": added,
                "removed": removed,
                "mod_ratio": mod,
            })
            df.to_csv(csv_path, index=False)
            print(f"[RADIUS] series CSV saved: {csv_path}")

            # Plot radius over epochs
            try:
                fig = plt.figure(figsize=(7.5, 4.5), dpi=130)
                plt.plot(epochs, radius_vals, marker="o")
                plt.xlabel("Epoch")
                plt.ylabel("Mean radius (↓ better)")
                plt.title(run_id)
                plt.tight_layout()
                png_path = os.path.join(out_dir, f"{run_id}.png")
                plt.savefig(png_path, bbox_inches="tight")
                plt.close(fig)
                print(f"[RADIUS] plot saved: {png_path}")
            except Exception as e:
                print(f"[RADIUS] plot skipped: {e}")

            # Log summary stats for easy grepping later
            start_r = float(radius_vals[0])
            end_r   = float(radius_vals[-1])
            delta_r = end_r - start_r
            best_r  = float(min(radius_vals))
            best_idx = int(radius_vals.index(best_r))
            best_ep = epochs[best_idx]
            print(f"[RADIUS] summary start={start_r:.6f} end={end_r:.6f} Δ={delta_r:+.6f} best={best_r:.6f}@e{best_ep}")
        else:
            print("[RADIUS] no radius history recorded for this run.")
    except Exception as e:
        print(f"[RADIUS] finalize failed: {e}")
    # ================================================================

    
    # return the best embeddings, not the last ones
    return Z_best.clone().detach(), roc_history, modification_ratio_history, edge_index



def get_scores(dataset_str,edges_pos, edges_neg, adj_rec, adj_orig):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # Predict on test set of edges
    preds = []
    pos = []
    
    pos_for_hitsk = []
    neg_for_hitsk = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        # print(sigmoid(adj_rec[e[0], e[1]].item()))
        preds.append(adj_rec[e[0], e[1]].item())
        pos.append(adj_orig[e[0], e[1]])
        
        
        # pos_for_hitsk.append(adj_rec[e[0], e[1]].item())
        
        
    preds_neg = []
    neg = []
    for e in edges_neg:
        # preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        preds_neg.append(adj_rec[e[0], e[1]].item())
        neg.append(adj_orig[e[0], e[1]])
        # neg_for_hitsk.append(adj_rec[e[0], e[1]].item())
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    hitk = []
    # evaluator = Evaluator(name=dataset_str)
    for k in [1,3,10,20,50,100]:
        hitk.append(eval_hits(torch.tensor(preds), torch.tensor(preds_neg),k))
        # evaluator.K = k
        # hits = evaluator.eval({
            # 'y_pred_pos': torch.tensor(preds),
            # 'y_pred_neg': torch.tensor(preds_neg),
        # })[f'hits@{k}']
        # hitk.append(hits)
    return roc_score, ap_score, hitk

def eval_hits(y_pred_pos, y_pred_neg, K):
    '''
    compute Hits@K
    For each positive target node, the negative target nodes are the same.
    y_pred_neg is an array.
    rank y_pred_pos[i] against y_pred_neg for each i
    From:
    https://github.com/snap-stanford/ogb/blob/1c875697fdb20ab452b2c11cf8bfa2c0e88b5ad3/ogb/linkproppred/evaluate.py#L214
    '''

    if len(y_pred_neg) < K:
        print(len(y_pred_neg))
        print(f'[WARNING]: hits@{K} defaulted to 1')
        return 1.0 #{'hits@{}'.format(K): 1.0}

    kth_score_in_negative_edges = torch.topk(y_pred_neg, K,largest=True)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(
        y_pred_pos
    )
    return hitsK
    # return {'hits@{}'.format(K): hitsK}

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

# def train_decoder(device, Z, adj_label, weight_tensor, norm, train_mask):
#     num_nodes = Z.shape[0]
#     feat_dim = Z.shape[1]
#     decoder = Decoder(feat_dim, feat_dim).to(device)
#     opt = Adam(decoder.parameters(), lr = 0.01, weight_decay = 0.0)

#     for _ in range(100):
#         A_pred = decoder(Z)
#         loss = norm * F.binary_cross_entropy(A_pred.view(-1)[train_mask], adj_label.to_dense().view(-1)[train_mask], weight = weight_tensor)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
    
#     with torch.no_grad():
#         A_pred = decoder(Z).detach()
#     del decoder
#     del opt
#     return A_pred

def train_classifier(device, Z, labels, idx_train, idx_val, idx_test):
    hid_units = Z.shape[1]
    nb_classes = labels.shape[1]

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    
    Z = torch.FloatTensor(normalize(Z.cpu().numpy(), norm='l2')).to(device)
    
    train_embs = Z[idx_train].detach()
    val_embs = Z[idx_val].detach()
    test_embs = Z[idx_test].detach()

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    xent = nn.CrossEntropyLoss()
    tot = torch.zeros(1).to(device)
    accs = []
    for _ in range(50):
        log = LogReg(hid_units, nb_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr = 0.01, weight_decay = 0.0)

        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            
            loss.backward()
            opt.step()
        
        log.eval()
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        # print('acc:[{:.4f}]'.format(acc))
        tot += acc
    
    print('-' * 100)
    print('Average accuracy:[{:.4f}]'.format(tot.item() / 50))
    accs = torch.stack(accs)
    print('Mean:[{:.4f}]'.format(accs.mean().item()))
    print('Std :[{:.4f}]'.format(accs.std().item()))
    print('-' * 100)

def logist_regressor_classification(device, Z, labels, idx_train, idx_val, idx_test):
    hid_units = Z.shape[1]
    nb_classes = labels.shape[1]

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    
    Z = torch.FloatTensor(normalize(Z.cpu().numpy(), norm='l2')).to(device)
    
    train_embs = Z[idx_train].detach().data.cpu()
    val_embs = Z[idx_val].detach().data.cpu()
    test_embs = Z[idx_test].detach().data.cpu()

    train_lbls = torch.argmax(labels[0, idx_train], dim=1).detach().data.cpu()
    val_lbls = torch.argmax(labels[0, idx_val], dim=1).detach().data.cpu()
    test_lbls = torch.argmax(labels[0, idx_test], dim=1).detach().data.cpu()

    tot = torch.zeros(1)
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c), n_jobs=8, cv=5, verbose=0)

    clf.fit(train_embs, train_lbls)
    
    # val
    logits = clf.predict_proba(val_embs)
    preds = torch.argmax(torch.tensor(logits), dim=1)
    val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
    print('val_acc:[{:.4f}]'.format(val_acc))

    # test
    logits = clf.predict_proba(test_embs)
    preds = torch.argmax(torch.tensor(logits), dim=1)
    test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    print('test_acc:[{:.4f}]'.format(test_acc))

    return val_acc, test_acc

@torch.no_grad()
def _deg_excl_self(adj_dense: torch.Tensor) -> torch.Tensor:
    # degree excluding self-loops; adj_dense is 0/1 with diag=1
    # (faster & avoids creating a big diag tensor)
    deg = adj_dense.sum(dim=1)
    return (deg - torch.diag(adj_dense)).to(torch.long)

@torch.no_grad()
def _cosine_radii_to_centroid(Z: torch.Tensor, labels_np: np.ndarray) -> torch.Tensor:
    """
    Return per-node radii r[i] = 1 - cos( x_i, c_{label(i)} ), where x_i and the
    cluster centroid c_k are L2-normalized. Noise nodes (label=-1) get 0 by default.
    """
    device = Z.device
    X = F.normalize(Z, p=2, dim=1)                 # [N,d], unit vectors
    labels_t = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
    N, d = X.shape
    radii = torch.zeros(N, device=device, dtype=Z.dtype)

    valid = labels_t >= 0
    if not torch.any(valid):
        return radii  # all noise → zeros (harmless; caller can ignore via mask)

    # compute normalized centroid per cluster
    ks = torch.unique(labels_t[valid]).tolist()
    for k in ks:
        idx = (labels_t == k)
        xk = X[idx]
        if xk.numel() == 0:
            continue
        ck = F.normalize(xk.mean(dim=0, keepdim=True), p=2, dim=1)  # [1,d]
        # cosine similarity to centroid
        cos = (xk @ ck.t()).squeeze(1)                              # [nk]
        radii[idx] = 1.0 - cos                                      # [nk]
    return radii