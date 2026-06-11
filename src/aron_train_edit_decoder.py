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
from model import VGNAE_ENCODER, VGAE_ENCODER, MaskGAE_ENCODER, CIMAGELite_ENCODER, CIMAGEFull_ENCODER, dot_product_decode, MLP, LogReg
from loss import loss_function, inter_view_CL_loss, intra_view_CL_loss, symmetric_node_infonce_loss, Cluster
from utils import *
from input_data import CalN2V

from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Tuple

# ---- side experiment: offline C0p/CP "remove until done" curve ----
# 要跑這個實驗的時候把這個改成 True，就會在 train 結束後多跑一趟 sweep
ENABLE_C0P_SWEEP = True

# 每次評估的步長 & 最大刪除比例（以 candidate 邊的數量為基準）
# e.g. 0.05 → 每刪掉 5% 的候選邊評一次
C0P_SWEEP_STEP_FRAC = 0.05    # 你可以改成 0.1 / 0.01 等
C0P_SWEEP_MAX_FRAC  = 1.0     # 最多刪到 100% 的候選邊

# --- dense helper: works for both sparse and dense tensors
def _to_dense(A: torch.Tensor) -> torch.Tensor:
    return A.to_dense() if getattr(A, "is_sparse", False) else A


def _dense_graph_content_signature(graph_dense: torch.Tensor) -> tuple:
    graph = _to_dense(graph_dense).detach()
    shape = tuple(graph.shape)
    if len(shape) != 2 or shape[0] != shape[1]:
        return ("invalid", shape)
    binary = graph > 0
    edge_count = int(binary.sum().detach().cpu().item())
    if edge_count == 0:
        return ("content", shape, str(graph.device), 0, 0, 0, 0)
    rows, cols = binary.nonzero(as_tuple=True)
    rows_i = rows.to(torch.int64)
    cols_i = cols.to(torch.int64)
    n = int(shape[0])
    code = rows_i * int(n + 1) + cols_i
    sum_code = int(code.sum().detach().cpu().item())
    sum_sq = int((code * code).sum().detach().cpu().item())
    sum_mix = int(((rows_i + 1) * (cols_i + 1)).sum().detach().cpu().item())
    return ("content", shape, str(graph.device), edge_count, sum_code, sum_sq, sum_mix)


def _default_results_root() -> str:
    env_root = os.environ.get("ARON_EXPERIMENT_ROOT")
    if env_root:
        env_root = os.path.abspath(os.path.expanduser(env_root))
        if os.path.basename(env_root) == "results":
            return env_root
        return os.path.join(env_root, "results")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "experiments", "reversegnn-compactness", "results")


def _artifact_dir(*parts: str) -> str:
    path = os.path.join(_default_results_root(), *parts)
    os.makedirs(path, exist_ok=True)
    return path


def _is_cuda_oom_like(error: BaseException) -> bool:
    text = str(error)
    return (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "CUDA out of memory" in text
        or "CUBLAS_STATUS_ALLOC_FAILED" in text
    )


class BilinearGraphDecoder(nn.Module):
    """Trainable graph decoder for dense adjacency reconstruction."""
    def __init__(self, dim: int, normalize_input: bool = True):
        super().__init__()
        self.normalize_input = normalize_input
        self.weight = nn.Parameter(torch.eye(dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        X = F.normalize(Z, p=2, dim=1) if self.normalize_input else Z
        logits = X @ self.weight @ X.t() + self.bias
        probs = torch.sigmoid(logits)
        probs = 0.5 * (probs + probs.t())
        probs.fill_diagonal_(1.0)
        return probs

    def score_pairs(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        X = F.normalize(Z, p=2, dim=1) if self.normalize_input else Z
        src = src.to(device=Z.device, dtype=torch.long).view(-1)
        dst = dst.to(device=Z.device, dtype=torch.long).view(-1)
        xu = X.index_select(0, src)
        xv = X.index_select(0, dst)
        logits_uv = ((xu @ self.weight) * xv).sum(dim=1) + self.bias
        logits_vu = ((xv @ self.weight) * xu).sum(dim=1) + self.bias
        probs = 0.5 * (torch.sigmoid(logits_uv) + torch.sigmoid(logits_vu))
        return torch.where(src == dst, torch.ones_like(probs), probs)


class MLPPairGraphDecoder(nn.Module):
    """Chunked MLP scorer over pair features [zi, zj, |zi-zj|, zi*zj]."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        normalize_input: bool = True,
        max_pair_rows: int = 16,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        self.max_pair_rows = max(1, int(max_pair_rows))
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        X = F.normalize(Z, p=2, dim=1) if self.normalize_input else Z
        n_nodes = X.size(0)
        row_chunk = max(1, min(self.max_pair_rows, int(max(1, 131072 // max(1, n_nodes)))))
        row_probs = []
        for start in range(0, n_nodes, row_chunk):
            end = min(n_nodes, start + row_chunk)
            zi = X[start:end].unsqueeze(1).expand(-1, n_nodes, -1)
            zj = X.unsqueeze(0).expand(end - start, -1, -1)
            pair_feats = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj], dim=-1)
            logits = self.net(pair_feats.reshape(-1, pair_feats.size(-1))).view(end - start, n_nodes)
            row_probs.append(torch.sigmoid(logits))
        probs = torch.cat(row_probs, dim=0)
        probs = 0.5 * (probs + probs.t())
        probs.fill_diagonal_(1.0)
        return probs

    def _score_pairs_one_way(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        X = F.normalize(Z, p=2, dim=1) if self.normalize_input else Z
        src = src.to(device=Z.device, dtype=torch.long).view(-1)
        dst = dst.to(device=Z.device, dtype=torch.long).view(-1)
        if src.numel() == 0:
            return Z.new_empty((0,))
        step = max(1, int(batch_size or 32768))
        outs = []
        for start in range(0, src.numel(), step):
            end = min(src.numel(), start + step)
            u = src[start:end]
            v = dst[start:end]
            zi = X.index_select(0, u)
            zj = X.index_select(0, v)
            pair_feats = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj], dim=-1)
            outs.append(torch.sigmoid(self.net(pair_feats).view(-1)))
        return torch.cat(outs, dim=0)

    def score_pairs(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        src = src.to(device=Z.device, dtype=torch.long).view(-1)
        dst = dst.to(device=Z.device, dtype=torch.long).view(-1)
        probs = 0.5 * (
            self._score_pairs_one_way(Z, src, dst, batch_size=batch_size)
            + self._score_pairs_one_way(Z, dst, src, batch_size=batch_size)
        )
        return torch.where(src == dst, torch.ones_like(probs), probs)


def _adjacency_to_binary_csr(adj_like) -> sp.csr_matrix:
    """Return a loop-free binary CSR adjacency on CPU."""
    if isinstance(adj_like, torch.Tensor):
        dense = _to_dense(adj_like).detach().cpu().numpy()
        adj_csr = sp.csr_matrix((dense > 0).astype(np.float32))
    elif sp.issparse(adj_like):
        adj_csr = adj_like.tocsr().astype(np.float32)
    else:
        adj_csr = sp.csr_matrix((np.asarray(adj_like) > 0).astype(np.float32))
    adj_csr.setdiag(0.0)
    adj_csr.eliminate_zeros()
    if adj_csr.nnz > 0:
        adj_csr.data = np.ones_like(adj_csr.data, dtype=np.float32)
    return adj_csr


def _normalized_dense_feature(mat: sp.spmatrix, *, log_scale: bool = False) -> np.ndarray:
    arr = mat.toarray().astype(np.float32, copy=False)
    if log_scale:
        arr = np.log1p(np.maximum(arr, 0.0)).astype(np.float32, copy=False)
    max_val = float(np.nanmax(np.abs(arr))) if arr.size > 0 else 0.0
    if max_val > 0.0 and np.isfinite(max_val):
        arr = arr / max_val
    np.fill_diagonal(arr, 0.0)
    return arr.astype(np.float32, copy=False)


def _structural_pair_features_from_adj(adj_like) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute dense structural pair features from a sparse train graph."""
    adj_csr = _adjacency_to_binary_csr(adj_like)
    n_nodes = adj_csr.shape[0]
    deg = np.asarray(adj_csr.sum(axis=1)).reshape(-1).astype(np.float32)
    if deg.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )

    deg_feat = np.log1p(deg)
    deg_max = float(deg_feat.max()) if deg_feat.size > 0 else 0.0
    if deg_max > 0.0 and np.isfinite(deg_max):
        deg_feat = deg_feat / deg_max

    cn = (adj_csr @ adj_csr).astype(np.float32)
    cn.setdiag(0.0)
    cn.eliminate_zeros()

    inv_deg = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0.0)
    ra = (adj_csr @ sp.diags(inv_deg, offsets=0, shape=(n_nodes, n_nodes), dtype=np.float32) @ adj_csr).astype(np.float32)
    ra.setdiag(0.0)
    ra.eliminate_zeros()

    log_deg = np.zeros_like(deg)
    positive_log_mask = deg > 1.0
    log_deg[positive_log_mask] = np.log(deg[positive_log_mask])
    inv_log_deg = np.divide(1.0, log_deg, out=np.zeros_like(deg), where=log_deg > 0.0)
    aa = (adj_csr @ sp.diags(inv_log_deg, offsets=0, shape=(n_nodes, n_nodes), dtype=np.float32) @ adj_csr).astype(np.float32)
    aa.setdiag(0.0)
    aa.eliminate_zeros()

    return (
        deg_feat.astype(np.float32, copy=False),
        _normalized_dense_feature(cn, log_scale=True),
        _normalized_dense_feature(ra),
        _normalized_dense_feature(aa),
    )


def _dense_binary_adj_feature(adj_like) -> np.ndarray:
    adj_csr = _adjacency_to_binary_csr(adj_like)
    return adj_csr.toarray().astype(np.float32, copy=False)


def _torch_sparse_binary_adj(adj_like, *, device: torch.device) -> torch.Tensor:
    adj_csr = _adjacency_to_binary_csr(adj_like).tocoo()
    if adj_csr.nnz == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=torch.float32, device=device)
    else:
        indices_np = np.vstack([adj_csr.row, adj_csr.col]).astype(np.int64, copy=False)
        indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
        values = torch.as_tensor(adj_csr.data.astype(np.float32, copy=False), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=adj_csr.shape, device=device).coalesce()


class StructuralPairGraphDecoder(nn.Module):
    """Chunked pair MLP with embedding, graph-structural, and cluster features."""

    scalar_dim = 15

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        normalize_input: bool = True,
        max_pair_rows: int = 16,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        self.max_pair_rows = max(1, int(max_pair_rows))
        self.net = nn.Sequential(
            nn.Linear(dim * 4 + self.scalar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("node_degree_feat", torch.empty(0), persistent=False)
        self.register_buffer("cn_feat", torch.empty(0, 0), persistent=False)
        self.register_buffer("ra_feat", torch.empty(0, 0), persistent=False)
        self.register_buffer("aa_feat", torch.empty(0, 0), persistent=False)
        self.register_buffer("adj_feat", torch.empty(0, 0), persistent=False)
        self.register_buffer(
            "adj_sparse_feat",
            torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.float32),
                size=(0, 0),
            ).coalesce(),
            persistent=False,
        )
        self.register_buffer("label_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("core_mask", torch.empty(0, dtype=torch.bool), persistent=False)
        self._graph_context_key = None
        self._pair_feature_cache_key = None
        self._pair_feature_cache = {}

    def _graph_context_signature(self, adj_like) -> tuple:
        if isinstance(adj_like, torch.Tensor):
            dense = _to_dense(adj_like)
            shape = tuple(dense.shape)
            edge_count = 0
            if dense.numel() > 0:
                edge_count = int((dense.detach() > 0).sum().cpu().item())
            return (
                "tensor",
                shape,
                str(dense.device),
                int(dense.data_ptr()) if dense.device.type != "meta" else 0,
                int(getattr(dense, "_version", 0)),
                edge_count,
            )
        if sp.issparse(adj_like):
            csr = adj_like.tocsr()
            return ("sparse", tuple(csr.shape), int(csr.nnz), int(id(adj_like)))
        arr = np.asarray(adj_like)
        edge_count = int((arr > 0).sum()) if arr.size > 0 else 0
        return ("array", tuple(arr.shape), int(id(adj_like)), edge_count)

    def set_graph_context(self, adj_like) -> None:
        key = self._graph_context_signature(adj_like)
        if self._graph_context_key == key:
            return
        device = next(self.parameters()).device
        deg, cn, ra, aa = _structural_pair_features_from_adj(adj_like)
        self.node_degree_feat = torch.as_tensor(deg, dtype=torch.float32, device=device)
        self.cn_feat = torch.as_tensor(cn, dtype=torch.float32, device=device)
        self.ra_feat = torch.as_tensor(ra, dtype=torch.float32, device=device)
        self.aa_feat = torch.as_tensor(aa, dtype=torch.float32, device=device)
        self.adj_feat = torch.as_tensor(_dense_binary_adj_feature(adj_like), dtype=torch.float32, device=device)
        self.adj_sparse_feat = _torch_sparse_binary_adj(adj_like, device=device)
        self._graph_context_key = key
        self._clear_pair_feature_cache()

    def set_cluster_context(self, labels: np.ndarray | None, core_mask: torch.Tensor | None) -> None:
        device = next(self.parameters()).device
        if labels is None:
            n_nodes = int(self.node_degree_feat.numel())
            next_labels = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
        else:
            labels_np = np.asarray(labels, dtype=np.int64)
            next_labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
            n_nodes = int(labels_np.shape[0])
        if core_mask is None:
            next_core = torch.zeros((n_nodes,), dtype=torch.bool, device=device)
        else:
            next_core = core_mask.to(device=device).bool()
        if (
            tuple(self.label_ids.shape) == tuple(next_labels.shape)
            and tuple(self.core_mask.shape) == tuple(next_core.shape)
            and torch.equal(self.label_ids, next_labels)
            and torch.equal(self.core_mask, next_core)
        ):
            return
        self.label_ids = next_labels
        self.core_mask = next_core
        self._clear_pair_feature_cache()

    def _clear_pair_feature_cache(self) -> None:
        self._pair_feature_cache_key = None
        self._pair_feature_cache = {}

    def _node_vector(self, buf: torch.Tensor, n_nodes: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if buf.numel() != n_nodes:
            return torch.zeros((n_nodes,), dtype=dtype, device=device)
        return buf.to(device=device, dtype=dtype)

    def _pair_matrix(self, buf: torch.Tensor, n_nodes: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if buf.numel() == 0 or tuple(buf.shape) != (n_nodes, n_nodes):
            return torch.zeros((n_nodes, n_nodes), dtype=dtype, device=device)
        return buf.to(device=device, dtype=dtype)

    def _prototype_distances(self, X: torch.Tensor) -> torch.Tensor:
        n_nodes = X.size(0)
        if self.label_ids.numel() != n_nodes:
            return X.new_zeros((n_nodes,))
        labels = self.label_ids.to(device=X.device)
        out = X.new_zeros((n_nodes,))
        for cluster_id in torch.unique(labels).detach().cpu().tolist():
            if int(cluster_id) == -1:
                continue
            idx = (labels == int(cluster_id)).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            Xc = F.normalize(X.index_select(0, idx), p=2, dim=1)
            proto = F.normalize(Xc.mean(dim=0, keepdim=True), p=2, dim=1)
            out[idx] = 1.0 - (Xc @ proto.t()).squeeze(1)
        return out

    def _extra_pair_matrix_scalar_features(
        self,
        Z: torch.Tensor,
        start: int,
        end: int,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        return []

    def _extra_pair_scalar_features(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        return []

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        X = F.normalize(Z, p=2, dim=1) if self.normalize_input else Z
        X_cos = F.normalize(Z, p=2, dim=1)
        n_nodes = X.size(0)
        row_chunk = max(1, min(self.max_pair_rows, int(max(1, 131072 // max(1, n_nodes)))))

        deg = self._node_vector(self.node_degree_feat, n_nodes, X.dtype, X.device)
        cn = self._pair_matrix(self.cn_feat, n_nodes, X.dtype, X.device)
        ra = self._pair_matrix(self.ra_feat, n_nodes, X.dtype, X.device)
        aa = self._pair_matrix(self.aa_feat, n_nodes, X.dtype, X.device)
        if self.label_ids.numel() == n_nodes:
            labels = self.label_ids.to(device=X.device)
            non_noise = labels != -1
        else:
            labels = torch.full((n_nodes,), -1, dtype=torch.long, device=X.device)
            non_noise = torch.zeros((n_nodes,), dtype=torch.bool, device=X.device)
        core = self.core_mask.to(device=X.device).bool() if self.core_mask.numel() == n_nodes else torch.zeros((n_nodes,), dtype=torch.bool, device=X.device)
        proto_dist = self._prototype_distances(X_cos)

        row_probs = []
        for start in range(0, n_nodes, row_chunk):
            end = min(n_nodes, start + row_chunk)
            rows = end - start
            zi = X[start:end].unsqueeze(1).expand(-1, n_nodes, -1)
            zj = X.unsqueeze(0).expand(rows, -1, -1)
            raw_dot = (Z[start:end] @ Z.t()).unsqueeze(-1)
            cosine = (X_cos[start:end] @ X_cos.t()).unsqueeze(-1)
            deg_u = deg[start:end].view(-1, 1).expand(-1, n_nodes)
            deg_v = deg.view(1, -1).expand(rows, -1)
            label_u = labels[start:end].view(-1, 1)
            same_cluster = ((label_u == labels.view(1, -1)) & non_noise[start:end].view(-1, 1) & non_noise.view(1, -1)).to(X.dtype)
            core_u = core[start:end].to(X.dtype).view(-1, 1).expand(-1, n_nodes)
            core_v = core.to(X.dtype).view(1, -1).expand(rows, -1)
            cp_u = non_noise[start:end].to(X.dtype).view(-1, 1).expand(-1, n_nodes)
            cp_v = non_noise.to(X.dtype).view(1, -1).expand(rows, -1)
            proto_u = proto_dist[start:end].view(-1, 1).expand(-1, n_nodes)
            proto_v = proto_dist.view(1, -1).expand(rows, -1)
            scalar_list = [
                raw_dot.squeeze(-1),
                cosine.squeeze(-1),
                deg_u,
                deg_v,
                torch.abs(deg_u - deg_v),
                cn[start:end],
                ra[start:end],
                aa[start:end],
                same_cluster,
                core_u,
                core_v,
                cp_u,
                cp_v,
                proto_u,
                proto_v,
            ]
            scalar_list.extend(
                self._extra_pair_matrix_scalar_features(
                    Z,
                    start,
                    end,
                    n_nodes,
                    X.dtype,
                    X.device,
                )
            )
            scalar_feats = torch.stack(
                scalar_list,
                dim=-1,
            )
            pair_feats = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj, scalar_feats], dim=-1)
            logits = self.net(pair_feats.reshape(-1, pair_feats.size(-1))).view(rows, n_nodes)
            row_probs.append(torch.sigmoid(logits))
        probs = torch.cat(row_probs, dim=0)
        probs = 0.5 * (probs + probs.t())
        probs.fill_diagonal_(1.0)
        return probs

    def _score_pairs_one_way(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        X = F.normalize(Z, p=2, dim=1) if self.normalize_input else Z
        X_cos = F.normalize(Z, p=2, dim=1)
        n_nodes = X.size(0)
        src = src.to(device=Z.device, dtype=torch.long).view(-1)
        dst = dst.to(device=Z.device, dtype=torch.long).view(-1)
        if src.numel() == 0:
            return Z.new_empty((0,))

        deg = self._node_vector(self.node_degree_feat, n_nodes, X.dtype, X.device)
        cn = self._pair_matrix(self.cn_feat, n_nodes, X.dtype, X.device)
        ra = self._pair_matrix(self.ra_feat, n_nodes, X.dtype, X.device)
        aa = self._pair_matrix(self.aa_feat, n_nodes, X.dtype, X.device)
        if self.label_ids.numel() == n_nodes:
            labels = self.label_ids.to(device=X.device)
            non_noise = labels != -1
        else:
            labels = torch.full((n_nodes,), -1, dtype=torch.long, device=X.device)
            non_noise = torch.zeros((n_nodes,), dtype=torch.bool, device=X.device)
        core = self.core_mask.to(device=X.device).bool() if self.core_mask.numel() == n_nodes else torch.zeros((n_nodes,), dtype=torch.bool, device=X.device)
        proto_dist = self._prototype_distances(X_cos)

        step = max(1, int(batch_size or 32768))
        outs = []
        for start in range(0, src.numel(), step):
            end = min(src.numel(), start + step)
            u = src[start:end]
            v = dst[start:end]
            zi = X.index_select(0, u)
            zj = X.index_select(0, v)
            raw_dot = (Z.index_select(0, u) * Z.index_select(0, v)).sum(dim=1, keepdim=True)
            cosine = (X_cos.index_select(0, u) * X_cos.index_select(0, v)).sum(dim=1, keepdim=True)
            deg_u = deg.index_select(0, u).view(-1, 1)
            deg_v = deg.index_select(0, v).view(-1, 1)
            label_u = labels.index_select(0, u)
            label_v = labels.index_select(0, v)
            same_cluster = ((label_u == label_v) & non_noise.index_select(0, u) & non_noise.index_select(0, v)).to(X.dtype).view(-1, 1)
            core_u = core.index_select(0, u).to(X.dtype).view(-1, 1)
            core_v = core.index_select(0, v).to(X.dtype).view(-1, 1)
            cp_u = non_noise.index_select(0, u).to(X.dtype).view(-1, 1)
            cp_v = non_noise.index_select(0, v).to(X.dtype).view(-1, 1)
            proto_u = proto_dist.index_select(0, u).view(-1, 1)
            proto_v = proto_dist.index_select(0, v).view(-1, 1)
            scalar_parts = [
                raw_dot,
                cosine,
                deg_u,
                deg_v,
                torch.abs(deg_u - deg_v),
                cn[u, v].view(-1, 1),
                ra[u, v].view(-1, 1),
                aa[u, v].view(-1, 1),
                same_cluster,
                core_u,
                core_v,
                cp_u,
                cp_v,
                proto_u,
                proto_v,
            ]
            scalar_parts.extend(
                self._extra_pair_scalar_features(
                    Z,
                    u,
                    v,
                    n_nodes,
                    X.dtype,
                    X.device,
                )
            )
            scalar_feats = torch.cat(
                scalar_parts,
                dim=-1,
            )
            pair_feats = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj, scalar_feats], dim=-1)
            outs.append(torch.sigmoid(self.net(pair_feats).view(-1)))
        return torch.cat(outs, dim=0)

    def score_pairs(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        src = src.to(device=Z.device, dtype=torch.long).view(-1)
        dst = dst.to(device=Z.device, dtype=torch.long).view(-1)
        probs = 0.5 * (
            self._score_pairs_one_way(Z, src, dst, batch_size=batch_size)
            + self._score_pairs_one_way(Z, dst, src, batch_size=batch_size)
        )
        return torch.where(src == dst, torch.ones_like(probs), probs)


class ResidualStructuralPairPredictionDecoder(StructuralPairGraphDecoder):
    """Structural pair scorer for final prediction: raw dot logit plus learned residual."""

    def _combine_raw_dot_and_residual(self, raw_dot: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return raw_dot + residual

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        X = F.normalize(Z, p=2, dim=1) if self.normalize_input else Z
        X_cos = F.normalize(Z, p=2, dim=1)
        n_nodes = X.size(0)
        row_chunk = max(1, min(self.max_pair_rows, int(max(1, 131072 // max(1, n_nodes)))))

        deg = self._node_vector(self.node_degree_feat, n_nodes, X.dtype, X.device)
        cn = self._pair_matrix(self.cn_feat, n_nodes, X.dtype, X.device)
        ra = self._pair_matrix(self.ra_feat, n_nodes, X.dtype, X.device)
        aa = self._pair_matrix(self.aa_feat, n_nodes, X.dtype, X.device)
        if self.label_ids.numel() == n_nodes:
            labels = self.label_ids.to(device=X.device)
            non_noise = labels != -1
        else:
            labels = torch.full((n_nodes,), -1, dtype=torch.long, device=X.device)
            non_noise = torch.zeros((n_nodes,), dtype=torch.bool, device=X.device)
        core = self.core_mask.to(device=X.device).bool() if self.core_mask.numel() == n_nodes else torch.zeros((n_nodes,), dtype=torch.bool, device=X.device)
        proto_dist = self._prototype_distances(X_cos)

        row_logits = []
        for start in range(0, n_nodes, row_chunk):
            end = min(n_nodes, start + row_chunk)
            rows = end - start
            zi = X[start:end].unsqueeze(1).expand(-1, n_nodes, -1)
            zj = X.unsqueeze(0).expand(rows, -1, -1)
            raw_dot = Z[start:end] @ Z.t()
            cosine = (X_cos[start:end] @ X_cos.t()).unsqueeze(-1)
            deg_u = deg[start:end].view(-1, 1).expand(-1, n_nodes)
            deg_v = deg.view(1, -1).expand(rows, -1)
            label_u = labels[start:end].view(-1, 1)
            same_cluster = ((label_u == labels.view(1, -1)) & non_noise[start:end].view(-1, 1) & non_noise.view(1, -1)).to(X.dtype)
            core_u = core[start:end].to(X.dtype).view(-1, 1).expand(-1, n_nodes)
            core_v = core.to(X.dtype).view(1, -1).expand(rows, -1)
            cp_u = non_noise[start:end].to(X.dtype).view(-1, 1).expand(-1, n_nodes)
            cp_v = non_noise.to(X.dtype).view(1, -1).expand(rows, -1)
            proto_u = proto_dist[start:end].view(-1, 1).expand(-1, n_nodes)
            proto_v = proto_dist.view(1, -1).expand(rows, -1)
            scalar_list = [
                raw_dot,
                cosine.squeeze(-1),
                deg_u,
                deg_v,
                torch.abs(deg_u - deg_v),
                cn[start:end],
                ra[start:end],
                aa[start:end],
                same_cluster,
                core_u,
                core_v,
                cp_u,
                cp_v,
                proto_u,
                proto_v,
            ]
            scalar_list.extend(
                self._extra_pair_matrix_scalar_features(
                    Z,
                    start,
                    end,
                    n_nodes,
                    X.dtype,
                    X.device,
                )
            )
            scalar_feats = torch.stack(
                scalar_list,
                dim=-1,
            )
            pair_feats = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj, scalar_feats], dim=-1)
            residual = self.net(pair_feats.reshape(-1, pair_feats.size(-1))).view(rows, n_nodes)
            row_logits.append(self._combine_raw_dot_and_residual(raw_dot, residual))
        logits = torch.cat(row_logits, dim=0)
        logits = 0.5 * (logits + logits.t())
        logits.fill_diagonal_(0.0)
        return logits

    def _score_pairs_one_way(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        X = F.normalize(Z, p=2, dim=1) if self.normalize_input else Z
        X_cos = F.normalize(Z, p=2, dim=1)
        n_nodes = X.size(0)
        src = src.to(device=Z.device, dtype=torch.long).view(-1)
        dst = dst.to(device=Z.device, dtype=torch.long).view(-1)
        if src.numel() == 0:
            return Z.new_empty((0,))

        deg = self._node_vector(self.node_degree_feat, n_nodes, X.dtype, X.device)
        cn = self._pair_matrix(self.cn_feat, n_nodes, X.dtype, X.device)
        ra = self._pair_matrix(self.ra_feat, n_nodes, X.dtype, X.device)
        aa = self._pair_matrix(self.aa_feat, n_nodes, X.dtype, X.device)
        if self.label_ids.numel() == n_nodes:
            labels = self.label_ids.to(device=X.device)
            non_noise = labels != -1
        else:
            labels = torch.full((n_nodes,), -1, dtype=torch.long, device=X.device)
            non_noise = torch.zeros((n_nodes,), dtype=torch.bool, device=X.device)
        core = self.core_mask.to(device=X.device).bool() if self.core_mask.numel() == n_nodes else torch.zeros((n_nodes,), dtype=torch.bool, device=X.device)
        proto_dist = self._prototype_distances(X_cos)

        step = max(1, int(batch_size or 32768))
        outs = []
        for start in range(0, src.numel(), step):
            end = min(src.numel(), start + step)
            u = src[start:end]
            v = dst[start:end]
            zi = X.index_select(0, u)
            zj = X.index_select(0, v)
            raw_dot = (Z.index_select(0, u) * Z.index_select(0, v)).sum(dim=1, keepdim=True)
            cosine = (X_cos.index_select(0, u) * X_cos.index_select(0, v)).sum(dim=1, keepdim=True)
            deg_u = deg.index_select(0, u).view(-1, 1)
            deg_v = deg.index_select(0, v).view(-1, 1)
            label_u = labels.index_select(0, u)
            label_v = labels.index_select(0, v)
            same_cluster = ((label_u == label_v) & non_noise.index_select(0, u) & non_noise.index_select(0, v)).to(X.dtype).view(-1, 1)
            core_u = core.index_select(0, u).to(X.dtype).view(-1, 1)
            core_v = core.index_select(0, v).to(X.dtype).view(-1, 1)
            cp_u = non_noise.index_select(0, u).to(X.dtype).view(-1, 1)
            cp_v = non_noise.index_select(0, v).to(X.dtype).view(-1, 1)
            proto_u = proto_dist.index_select(0, u).view(-1, 1)
            proto_v = proto_dist.index_select(0, v).view(-1, 1)
            scalar_parts = [
                raw_dot,
                cosine,
                deg_u,
                deg_v,
                torch.abs(deg_u - deg_v),
                cn[u, v].view(-1, 1),
                ra[u, v].view(-1, 1),
                aa[u, v].view(-1, 1),
                same_cluster,
                core_u,
                core_v,
                cp_u,
                cp_v,
                proto_u,
                proto_v,
            ]
            scalar_parts.extend(
                self._extra_pair_scalar_features(
                    Z,
                    u,
                    v,
                    n_nodes,
                    X.dtype,
                    X.device,
                )
            )
            scalar_feats = torch.cat(
                scalar_parts,
                dim=-1,
            )
            pair_feats = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj, scalar_feats], dim=-1)
            residual = self.net(pair_feats).view(-1, 1)
            outs.append(self._combine_raw_dot_and_residual(raw_dot, residual).view(-1))
        return torch.cat(outs, dim=0)

    def score_pairs(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        src = src.to(device=Z.device, dtype=torch.long).view(-1)
        dst = dst.to(device=Z.device, dtype=torch.long).view(-1)
        logits = 0.5 * (
            self._score_pairs_one_way(Z, src, dst, batch_size=batch_size)
            + self._score_pairs_one_way(Z, dst, src, batch_size=batch_size)
        )
        return torch.where(src == dst, torch.zeros_like(logits), logits)


class NCNCResidualStructuralPairPredictionDecoder(ResidualStructuralPairPredictionDecoder):
    """Residual prediction decoder with an NCNC-style completed-CN feature."""

    scalar_dim = 16

    def _completion_cache_key(self, Z: torch.Tensor) -> tuple:
        return (
            int(Z.data_ptr()),
            int(getattr(Z, "_version", 0)),
            int(Z.size(0)),
            int(Z.size(1)),
            str(Z.device),
            str(Z.dtype),
        )

    def _ncnc_residual_cn_matrix(
        self,
        Z: torch.Tensor,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        adj_dense = self._pair_matrix(self.adj_feat, n_nodes, dtype, device)
        if adj_dense.numel() == 0 or tuple(adj_dense.shape) != (n_nodes, n_nodes):
            return torch.zeros((n_nodes, n_nodes), dtype=dtype, device=device)
        if self.adj_sparse_feat.numel() == 0 or tuple(self.adj_sparse_feat.shape) != (n_nodes, n_nodes):
            return torch.zeros((n_nodes, n_nodes), dtype=dtype, device=device)

        key = self._completion_cache_key(Z)
        if self._pair_feature_cache_key == key and "ncnc_residual_cn" in self._pair_feature_cache:
            return self._pair_feature_cache["ncnc_residual_cn"].to(device=device, dtype=dtype)

        with torch.no_grad():
            z_detached = Z.detach()
            raw_prob = torch.sigmoid(z_detached @ z_detached.t()).to(dtype=dtype, device=device)
            missing_mask = (1.0 - adj_dense).clamp_min(0.0)
            missing_mask.fill_diagonal_(0.0)
            prob_missing = raw_prob * missing_mask
            prob_missing.fill_diagonal_(0.0)

            adj_sparse = self.adj_sparse_feat.to(device=device, dtype=dtype).coalesce()
            one_side = torch.sparse.mm(adj_sparse, prob_missing)
            residual_cn = one_side + one_side.t()
            residual_cn.fill_diagonal_(0.0)
            residual_cn = torch.log1p(residual_cn.clamp_min(0.0))
            max_val = residual_cn.max()
            if bool(torch.isfinite(max_val)) and float(max_val.item()) > 0.0:
                residual_cn = residual_cn / max_val.clamp_min(1e-12)
            residual_cn = residual_cn.to(device=device, dtype=dtype)

        self._pair_feature_cache_key = key
        self._pair_feature_cache = {"ncnc_residual_cn": residual_cn}
        return residual_cn

    def _extra_pair_matrix_scalar_features(
        self,
        Z: torch.Tensor,
        start: int,
        end: int,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        residual_cn = self._ncnc_residual_cn_matrix(Z, n_nodes, dtype, device)
        return [residual_cn[start:end]]

    def _extra_pair_scalar_features(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        residual_cn = self._ncnc_residual_cn_matrix(Z, n_nodes, dtype, device)
        return [residual_cn[src, dst].view(-1, 1)]


class MultiOrderNCNCResidualStructuralPairPredictionDecoder(NCNCResidualStructuralPairPredictionDecoder):
    """NCNC residual decoder with gated higher-order completed-CN aggregation."""

    scalar_dim = 18

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        normalize_input: bool = True,
        max_pair_rows: int = 16,
    ):
        super().__init__(
            dim,
            hidden_dim,
            normalize_input=normalize_input,
            max_pair_rows=max_pair_rows,
        )
        self.multi_order_gate_logits = nn.Parameter(torch.tensor([-2.0, -3.0], dtype=torch.float32))

    @staticmethod
    def _positive_logmax_pair_feature(feature: torch.Tensor) -> torch.Tensor:
        out = torch.log1p(feature.clamp_min(0.0))
        out.fill_diagonal_(0.0)
        max_val = out.max()
        if bool(torch.isfinite(max_val)) and float(max_val.item()) > 0.0:
            out = out / max_val.clamp_min(1e-12)
        return out

    @staticmethod
    def _center_and_scale_pair_feature(feature: torch.Tensor) -> torch.Tensor:
        out = feature.clone()
        out.fill_diagonal_(0.0)
        out = out - out.mean()
        out.fill_diagonal_(0.0)
        max_abs = out.abs().max()
        if bool(torch.isfinite(max_abs)) and float(max_abs.item()) > 0.0:
            out = out / max_abs.clamp_min(1e-12)
        return out

    @staticmethod
    def _orthogonalize_pair_feature(feature: torch.Tensor, bases: list[torch.Tensor]) -> torch.Tensor:
        flat = feature.reshape(-1)
        for base in bases:
            base_flat = base.reshape(-1)
            denom = torch.dot(base_flat, base_flat).clamp_min(1e-12)
            flat = flat - torch.dot(flat, base_flat) / denom * base_flat
        return flat.view_as(feature)

    def _ncnc_multi_order_matrices(
        self,
        Z: torch.Tensor,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        adj_dense = self._pair_matrix(self.adj_feat, n_nodes, dtype, device)
        if adj_dense.numel() == 0 or tuple(adj_dense.shape) != (n_nodes, n_nodes):
            return [torch.zeros((n_nodes, n_nodes), dtype=dtype, device=device) for _ in range(3)]
        if self.adj_sparse_feat.numel() == 0 or tuple(self.adj_sparse_feat.shape) != (n_nodes, n_nodes):
            return [torch.zeros((n_nodes, n_nodes), dtype=dtype, device=device) for _ in range(3)]

        key = ("ncnc_multi",) + self._completion_cache_key(Z)
        if self._pair_feature_cache_key == key and "ncnc_multi_features" in self._pair_feature_cache:
            return [feat.to(device=device, dtype=dtype) for feat in self._pair_feature_cache["ncnc_multi_features"]]

        with torch.no_grad():
            z_detached = Z.detach()
            raw_prob = torch.sigmoid(z_detached @ z_detached.t()).to(dtype=dtype, device=device)
            missing_mask = (1.0 - adj_dense).clamp_min(0.0)
            missing_mask.fill_diagonal_(0.0)
            prob_missing = raw_prob * missing_mask
            prob_missing.fill_diagonal_(0.0)

            adj_sparse = self.adj_sparse_feat.to(device=device, dtype=dtype).coalesce()
            ap = torch.sparse.mm(adj_sparse, prob_missing)
            a2p = torch.sparse.mm(adj_sparse, ap)
            a3p = torch.sparse.mm(adj_sparse, a2p)

            order2 = ap + ap.t()
            order3 = a2p + a2p.t()
            order4 = a3p + a3p.t()
            for feature in (order2, order3, order4):
                feature.fill_diagonal_(0.0)

            order2_norm = self._positive_logmax_pair_feature(order2)
            order3_norm = self._positive_logmax_pair_feature(order3)
            order4_norm = self._positive_logmax_pair_feature(order4)

            order2_base = self._center_and_scale_pair_feature(order2_norm)
            order3_centered = self._center_and_scale_pair_feature(order3_norm)
            order3_orth = self._orthogonalize_pair_feature(order3_centered, [order2_base])
            order3_orth = self._center_and_scale_pair_feature(order3_orth)

            order4_centered = self._center_and_scale_pair_feature(order4_norm)
            order4_orth = self._orthogonalize_pair_feature(order4_centered, [order2_base, order3_orth])
            order4_orth = self._center_and_scale_pair_feature(order4_orth)

            features = [
                order2_norm.to(device=device, dtype=dtype),
                order3_orth.to(device=device, dtype=dtype),
                order4_orth.to(device=device, dtype=dtype),
            ]

        self._pair_feature_cache_key = key
        self._pair_feature_cache = {"ncnc_multi_features": features}
        return features

    def _gated_multi_order_features(
        self,
        Z: torch.Tensor,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        order2, order3, order4 = self._ncnc_multi_order_matrices(Z, n_nodes, dtype, device)
        gates = torch.sigmoid(self.multi_order_gate_logits).to(device=device, dtype=dtype)
        return [order2, order3 * gates[0], order4 * gates[1]]

    def _extra_pair_matrix_scalar_features(
        self,
        Z: torch.Tensor,
        start: int,
        end: int,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        return [feat[start:end] for feat in self._gated_multi_order_features(Z, n_nodes, dtype, device)]

    def _extra_pair_scalar_features(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        return [feat[src, dst].view(-1, 1) for feat in self._gated_multi_order_features(Z, n_nodes, dtype, device)]


class CompactMultiOrderResidualStructuralPairPredictionDecoder(MultiOrderNCNCResidualStructuralPairPredictionDecoder):
    """Multi-order residual decoder with explicit compactness-oriented pair features."""

    scalar_dim = 23

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        normalize_input: bool = True,
        max_pair_rows: int = 16,
    ):
        super().__init__(
            dim,
            hidden_dim,
            normalize_input=normalize_input,
            max_pair_rows=max_pair_rows,
        )
        self._compact_pair_feature_cache_key = None
        self._compact_pair_feature_cache = {}

    def _clear_pair_feature_cache(self) -> None:
        super()._clear_pair_feature_cache()
        self._compact_pair_feature_cache_key = None
        self._compact_pair_feature_cache = {}

    def _compact_feature_cache_key(self, Z: torch.Tensor) -> tuple:
        label_ptr = int(self.label_ids.data_ptr()) if self.label_ids.numel() else 0
        label_version = int(getattr(self.label_ids, "_version", 0)) if self.label_ids.numel() else 0
        core_ptr = int(self.core_mask.data_ptr()) if self.core_mask.numel() else 0
        core_version = int(getattr(self.core_mask, "_version", 0)) if self.core_mask.numel() else 0
        return (
            "compact_multi",
            label_ptr,
            label_version,
            core_ptr,
            core_version,
        ) + self._completion_cache_key(Z)

    def _compact_feature_matrices(
        self,
        Z: torch.Tensor,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        empty = [torch.zeros((n_nodes, n_nodes), dtype=dtype, device=device) for _ in range(5)]
        if self.label_ids.numel() != n_nodes:
            return empty

        key = self._compact_feature_cache_key(Z)
        if self._compact_pair_feature_cache_key == key and "compact_features" in self._compact_pair_feature_cache:
            return [feat.to(device=device, dtype=dtype) for feat in self._compact_pair_feature_cache["compact_features"]]

        with torch.no_grad():
            labels = self.label_ids.to(device=device)
            non_noise = labels != -1
            same_cluster = (labels[:, None] == labels[None, :]) & non_noise[:, None] & non_noise[None, :]
            core = self.core_mask.to(device=device).bool() if self.core_mask.numel() == n_nodes else torch.zeros((n_nodes,), dtype=torch.bool, device=device)
            noncompact = non_noise & (~core)

            X_cos = F.normalize(Z.detach(), p=2, dim=1)
            proto_dist = self._prototype_distances(X_cos).to(device=device, dtype=dtype)
            proto_u = proto_dist.view(-1, 1).expand(-1, n_nodes)
            proto_v = proto_dist.view(1, -1).expand(n_nodes, -1)

            same_f = same_cluster.to(dtype=dtype)
            core_bridge = same_cluster & (
                (core[:, None] & noncompact[None, :]) | (noncompact[:, None] & core[None, :])
            )
            noncompact_pair = same_cluster & noncompact[:, None] & noncompact[None, :]
            cross_cluster = (~same_cluster) & non_noise[:, None] & non_noise[None, :]

            compact_max = same_f * torch.maximum(proto_u, proto_v)
            compact_mean = same_f * (0.5 * (proto_u + proto_v))
            compact_gap = same_f * torch.abs(proto_u - proto_v)
            features = [
                compact_max,
                compact_mean,
                compact_gap,
                core_bridge.to(dtype=dtype),
                noncompact_pair.to(dtype=dtype) - cross_cluster.to(dtype=dtype),
            ]
            for feature in features:
                feature.fill_diagonal_(0.0)
            features = [feature.to(device=device, dtype=dtype) for feature in features]

        self._compact_pair_feature_cache_key = key
        self._compact_pair_feature_cache = {"compact_features": features}
        return features

    def _extra_pair_matrix_scalar_features(
        self,
        Z: torch.Tensor,
        start: int,
        end: int,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        features = super()._extra_pair_matrix_scalar_features(Z, start, end, n_nodes, dtype, device)
        features.extend(feat[start:end] for feat in self._compact_feature_matrices(Z, n_nodes, dtype, device))
        return features

    def _extra_pair_scalar_features(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        features = super()._extra_pair_scalar_features(Z, src, dst, n_nodes, dtype, device)
        features.extend(feat[src, dst].view(-1, 1) for feat in self._compact_feature_matrices(Z, n_nodes, dtype, device))
        return features


class GatedCompactMultiOrderResidualStructuralPairPredictionDecoder(CompactMultiOrderResidualStructuralPairPredictionDecoder):
    """Conservative compact-multi decoder: dot logit plus a gated bounded residual."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        normalize_input: bool = True,
        max_pair_rows: int = 16,
        gate_init_logit: float = -4.0,
        residual_scale: float = 1.0,
    ):
        super().__init__(
            dim,
            hidden_dim,
            normalize_input=normalize_input,
            max_pair_rows=max_pair_rows,
        )
        self.compact_residual_gate_logit = nn.Parameter(torch.tensor(float(gate_init_logit), dtype=torch.float32))
        self.compact_residual_scale = float(residual_scale)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _compact_residual_gate(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.sigmoid(self.compact_residual_gate_logit).to(device=device, dtype=dtype)

    def _combine_raw_dot_and_residual(self, raw_dot: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        gate = self._compact_residual_gate(raw_dot.dtype, raw_dot.device)
        bounded_residual = torch.tanh(residual) * float(self.compact_residual_scale)
        return raw_dot + gate * bounded_residual

    def extra_regularization_loss(self) -> torch.Tensor:
        return torch.sigmoid(self.compact_residual_gate_logit).abs()

    def extra_diagnostics(self) -> dict[str, float]:
        return {
            "prediction_compact_gate": float(torch.sigmoid(self.compact_residual_gate_logit).detach().cpu()),
            "prediction_compact_residual_scale": float(self.compact_residual_scale),
        }


class H3DeltaNCNCResidualStructuralPairPredictionDecoder(NCNCResidualStructuralPairPredictionDecoder):
    """NCNC decoder plus a zero-initialized gated h3 completed-CN delta branch."""

    scalar_dim = 16

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        normalize_input: bool = True,
        max_pair_rows: int = 16,
        gate_init_logit: float = -3.0,
    ):
        super().__init__(
            dim,
            hidden_dim,
            normalize_input=normalize_input,
            max_pair_rows=max_pair_rows,
        )
        self.h3_delta_gate_logit = nn.Parameter(torch.tensor(float(gate_init_logit), dtype=torch.float32))
        self.h3_delta_net = nn.Sequential(
            nn.Linear(dim * 4 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.h3_delta_net[-1].weight)
        nn.init.zeros_(self.h3_delta_net[-1].bias)

    def _h3_delta_cache_key(self, Z: torch.Tensor) -> tuple:
        return ("ncnc_h3_delta",) + self._completion_cache_key(Z)

    def _ncnc_h3_delta_matrices(
        self,
        Z: torch.Tensor,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adj_dense = self._pair_matrix(self.adj_feat, n_nodes, dtype, device)
        empty = torch.zeros((n_nodes, n_nodes), dtype=dtype, device=device)
        if adj_dense.numel() == 0 or tuple(adj_dense.shape) != (n_nodes, n_nodes):
            return empty, empty
        if self.adj_sparse_feat.numel() == 0 or tuple(self.adj_sparse_feat.shape) != (n_nodes, n_nodes):
            return empty, empty

        key = self._h3_delta_cache_key(Z)
        if self._pair_feature_cache_key == key and "ncnc_h3_delta_features" in self._pair_feature_cache:
            order2, h3 = self._pair_feature_cache["ncnc_h3_delta_features"]
            return order2.to(device=device, dtype=dtype), h3.to(device=device, dtype=dtype)

        with torch.no_grad():
            z_detached = Z.detach()
            raw_prob = torch.sigmoid(z_detached @ z_detached.t()).to(dtype=dtype, device=device)
            missing_mask = (1.0 - adj_dense).clamp_min(0.0)
            missing_mask.fill_diagonal_(0.0)
            prob_missing = raw_prob * missing_mask
            prob_missing.fill_diagonal_(0.0)

            adj_sparse = self.adj_sparse_feat.to(device=device, dtype=dtype).coalesce()
            ap = torch.sparse.mm(adj_sparse, prob_missing)
            a2p = torch.sparse.mm(adj_sparse, ap)

            order2 = ap + ap.t()
            h3 = a2p + a2p.t()
            order2.fill_diagonal_(0.0)
            h3.fill_diagonal_(0.0)

            order2_norm = MultiOrderNCNCResidualStructuralPairPredictionDecoder._positive_logmax_pair_feature(order2)
            h3_norm = MultiOrderNCNCResidualStructuralPairPredictionDecoder._positive_logmax_pair_feature(h3)
            order2_base = MultiOrderNCNCResidualStructuralPairPredictionDecoder._center_and_scale_pair_feature(order2_norm)
            h3_centered = MultiOrderNCNCResidualStructuralPairPredictionDecoder._center_and_scale_pair_feature(h3_norm)
            h3_orth = MultiOrderNCNCResidualStructuralPairPredictionDecoder._orthogonalize_pair_feature(
                h3_centered,
                [order2_base],
            )
            h3_orth = MultiOrderNCNCResidualStructuralPairPredictionDecoder._center_and_scale_pair_feature(h3_orth)

            features = (
                order2_norm.to(device=device, dtype=dtype),
                h3_orth.to(device=device, dtype=dtype),
            )

        self._pair_feature_cache_key = key
        self._pair_feature_cache = {"ncnc_h3_delta_features": features}
        return features

    def _h3_delta_gate(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.sigmoid(self.h3_delta_gate_logit).to(device=device, dtype=dtype)

    def extra_regularization_loss(self) -> torch.Tensor:
        return torch.sigmoid(self.h3_delta_gate_logit).abs()

    def extra_diagnostics(self) -> dict[str, float]:
        return {
            "prediction_h3_gate": float(torch.sigmoid(self.h3_delta_gate_logit).detach().cpu()),
        }

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        logits = super().forward(Z)
        X = F.normalize(Z, p=2, dim=1) if self.normalize_input else Z
        n_nodes = X.size(0)
        row_chunk = max(1, min(self.max_pair_rows, int(max(1, 131072 // max(1, n_nodes)))))
        order2, h3 = self._ncnc_h3_delta_matrices(Z, n_nodes, X.dtype, X.device)
        gate = self._h3_delta_gate(X.dtype, X.device)

        row_deltas = []
        for start in range(0, n_nodes, row_chunk):
            end = min(n_nodes, start + row_chunk)
            rows = end - start
            zi = X[start:end].unsqueeze(1).expand(-1, n_nodes, -1)
            zj = X.unsqueeze(0).expand(rows, -1, -1)
            raw_dot = (Z[start:end] @ Z.t()).unsqueeze(-1)
            scalar_feats = torch.stack(
                [
                    raw_dot.squeeze(-1),
                    order2[start:end],
                    h3[start:end],
                ],
                dim=-1,
            )
            pair_feats = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj, scalar_feats], dim=-1)
            row_deltas.append(self.h3_delta_net(pair_feats.reshape(-1, pair_feats.size(-1))).view(rows, n_nodes))
        delta = torch.cat(row_deltas, dim=0)
        delta = 0.5 * (delta + delta.t())
        delta.fill_diagonal_(0.0)
        logits = logits + gate * delta
        logits.fill_diagonal_(0.0)
        return logits

    def _h3_delta_score_pairs_one_way(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        X = F.normalize(Z, p=2, dim=1) if self.normalize_input else Z
        n_nodes = X.size(0)
        src = src.to(device=Z.device, dtype=torch.long).view(-1)
        dst = dst.to(device=Z.device, dtype=torch.long).view(-1)
        if src.numel() == 0:
            return Z.new_empty((0,))

        order2, h3 = self._ncnc_h3_delta_matrices(Z, n_nodes, X.dtype, X.device)
        step = max(1, int(batch_size or 32768))
        outs = []
        for start in range(0, src.numel(), step):
            end = min(src.numel(), start + step)
            u = src[start:end]
            v = dst[start:end]
            zi = X.index_select(0, u)
            zj = X.index_select(0, v)
            raw_dot = (Z.index_select(0, u) * Z.index_select(0, v)).sum(dim=1, keepdim=True)
            scalar_feats = torch.cat(
                [
                    raw_dot,
                    order2[u, v].view(-1, 1),
                    h3[u, v].view(-1, 1),
                ],
                dim=-1,
            )
            pair_feats = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj, scalar_feats], dim=-1)
            outs.append(self.h3_delta_net(pair_feats).view(-1))
        return torch.cat(outs, dim=0)

    def score_pairs(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        src = src.to(device=Z.device, dtype=torch.long).view(-1)
        dst = dst.to(device=Z.device, dtype=torch.long).view(-1)
        base_logits = super().score_pairs(Z, src, dst, batch_size=batch_size)
        delta_logits = 0.5 * (
            self._h3_delta_score_pairs_one_way(Z, src, dst, batch_size=batch_size)
            + self._h3_delta_score_pairs_one_way(Z, dst, src, batch_size=batch_size)
        )
        gate = self._h3_delta_gate(base_logits.dtype, base_logits.device)
        logits = base_logits + gate * delta_logits
        return torch.where(src == dst, torch.zeros_like(logits), logits)


class OCNResidualStructuralPairPredictionDecoder(ResidualStructuralPairPredictionDecoder):
    """Residual prediction decoder with OCN-style higher-order CN features."""

    scalar_dim = 18

    def _ocn_cache_key(self, n_nodes: int, dtype: torch.dtype, device: torch.device) -> tuple:
        adj_ptr = int(self.adj_feat.data_ptr()) if self.adj_feat.numel() else 0
        adj_version = int(getattr(self.adj_feat, "_version", 0)) if self.adj_feat.numel() else 0
        return (
            "ocn",
            adj_ptr,
            adj_version,
            int(n_nodes),
            str(device),
            str(dtype),
        )

    @staticmethod
    def _center_and_scale_pair_feature(feature: torch.Tensor) -> torch.Tensor:
        out = feature.clone()
        out.fill_diagonal_(0.0)
        out = out - out.mean()
        out.fill_diagonal_(0.0)
        max_abs = out.abs().max()
        if bool(torch.isfinite(max_abs)) and float(max_abs.item()) > 0.0:
            out = out / max_abs.clamp_min(1e-12)
        return out

    @staticmethod
    def _orthogonalize_pair_feature(feature: torch.Tensor, bases: list[torch.Tensor]) -> torch.Tensor:
        flat = feature.reshape(-1)
        for base in bases:
            base_flat = base.reshape(-1)
            denom = torch.dot(base_flat, base_flat).clamp_min(1e-12)
            flat = flat - torch.dot(flat, base_flat) / denom * base_flat
        return flat.view_as(feature)

    def _ocn_feature_matrices(
        self,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        adj_dense = self._pair_matrix(self.adj_feat, n_nodes, dtype, device)
        if adj_dense.numel() == 0 or tuple(adj_dense.shape) != (n_nodes, n_nodes):
            return [torch.zeros((n_nodes, n_nodes), dtype=dtype, device=device) for _ in range(3)]

        key = self._ocn_cache_key(n_nodes, dtype, device)
        if self._pair_feature_cache_key == key and "ocn_features" in self._pair_feature_cache:
            return [feat.to(device=device, dtype=dtype) for feat in self._pair_feature_cache["ocn_features"]]

        with torch.no_grad():
            adj_binary = (adj_dense > 0).to(device=device, dtype=dtype).clone()
            adj_binary.fill_diagonal_(0.0)
            deg = adj_binary.sum(dim=1).clamp_min(1.0)
            inv_sqrt_deg = deg.rsqrt()
            norm_adj = adj_binary * inv_sqrt_deg.view(-1, 1) * inv_sqrt_deg.view(1, -1)

            h2 = norm_adj @ norm_adj
            h3 = h2 @ norm_adj
            h4 = h3 @ norm_adj
            h5 = h4 @ norm_adj

            bases: list[torch.Tensor] = []
            h2_base = self._center_and_scale_pair_feature(torch.log1p(h2.clamp_min(0.0)))
            bases.append(h2_base)

            ocn_features: list[torch.Tensor] = []
            for raw_feature in (h3, h4, h5):
                scaled = self._center_and_scale_pair_feature(torch.log1p(raw_feature.clamp_min(0.0)))
                orthogonalized = self._orthogonalize_pair_feature(scaled, bases)
                normalized = self._center_and_scale_pair_feature(orthogonalized)
                bases.append(normalized)
                ocn_features.append(normalized.to(device=device, dtype=dtype))

        self._pair_feature_cache_key = key
        self._pair_feature_cache = {"ocn_features": ocn_features}
        return ocn_features

    def _extra_pair_matrix_scalar_features(
        self,
        Z: torch.Tensor,
        start: int,
        end: int,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        return [feat[start:end] for feat in self._ocn_feature_matrices(n_nodes, dtype, device)]

    def _extra_pair_scalar_features(
        self,
        Z: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        return [feat[src, dst].view(-1, 1) for feat in self._ocn_feature_matrices(n_nodes, dtype, device)]


def reconstruction_bce_loss(
    adj_pred: torch.Tensor,
    adj_label: torch.Tensor,
    norm: float,
    weight_tensor: torch.Tensor,
    train_mask: torch.Tensor,
    extra_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    flat_train_mask = train_mask.view(-1).bool()
    target = _to_dense(adj_label).view(-1)[flat_train_mask]
    pred = adj_pred.view(-1)[flat_train_mask]
    weights = weight_tensor
    if extra_mask is not None:
        keep = extra_mask.view(-1)[flat_train_mask].bool()
        if int(keep.sum().item()) == 0:
            return adj_pred.new_tensor(0.0)
        target = target[keep]
        pred = pred[keep]
        weights = weight_tensor[keep]
    if target.numel() == 0:
        return adj_pred.new_tensor(0.0)
    return norm * F.binary_cross_entropy(pred, target, weight=weights)




def _decoder_score_pairs(
    graph_decoder: nn.Module,
    Z: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    batch_size: int | None = None,
) -> torch.Tensor:
    src = src.to(device=Z.device, dtype=torch.long).view(-1)
    dst = dst.to(device=Z.device, dtype=torch.long).view(-1)
    if src.numel() == 0:
        return Z.new_empty((0,))
    if hasattr(graph_decoder, "score_pairs"):
        return graph_decoder.score_pairs(Z, src, dst, batch_size=batch_size)
    scores = graph_decoder(Z)
    return scores[src, dst]


def _sample_pairs_from_mask(pair_mask: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = pair_mask.device
    if count <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty
    ii, jj = pair_mask.triu(1).nonzero(as_tuple=True)
    if ii.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty
    take = torch.randint(0, ii.numel(), (int(count),), device=device)
    return ii.index_select(0, take), jj.index_select(0, take)


def _sampled_pair_bce_loss(
    graph_decoder: nn.Module,
    Z: torch.Tensor,
    pos_u: torch.Tensor,
    pos_v: torch.Tensor,
    neg_u: torch.Tensor,
    neg_v: torch.Tensor,
) -> torch.Tensor:
    scores = []
    labels = []
    if pos_u.numel() > 0:
        scores.append(_decoder_score_pairs(graph_decoder, Z, pos_u, pos_v))
        labels.append(torch.ones((pos_u.numel(),), dtype=Z.dtype, device=Z.device))
    if neg_u.numel() > 0:
        scores.append(_decoder_score_pairs(graph_decoder, Z, neg_u, neg_v))
        labels.append(torch.zeros((neg_u.numel(),), dtype=Z.dtype, device=Z.device))
    if not scores:
        return Z.sum() * 0.0
    return F.binary_cross_entropy(torch.cat(scores, dim=0), torch.cat(labels, dim=0))


def sampled_decoder_reconstruction_loss(
    graph_decoder: nn.Module,
    Z: torch.Tensor,
    train_edges_t: torch.Tensor,
    forbidden_mask: torch.Tensor,
    *,
    num_neg_per_pos: int = 1,
    max_pos_edges: int = 8192,
) -> torch.Tensor:
    if train_edges_t.numel() == 0:
        return Z.sum() * 0.0
    edges = train_edges_t.to(device=Z.device, dtype=torch.long)
    valid = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] < Z.size(0)) & (edges[:, 1] < Z.size(0)) & (edges[:, 0] != edges[:, 1])
    edges = edges[valid]
    if edges.numel() == 0:
        return Z.sum() * 0.0
    if edges.size(0) > int(max_pos_edges):
        order = torch.randperm(edges.size(0), device=Z.device)[: int(max_pos_edges)]
        edges = edges.index_select(0, order)
    neg_count = int(edges.size(0) * max(1, int(num_neg_per_pos)))
    allowed_neg = (~forbidden_mask.to(device=Z.device).bool()).clone()
    allowed_neg.fill_diagonal_(False)
    neg_u, neg_v = _sample_pairs_from_mask(allowed_neg, neg_count)
    return _sampled_pair_bce_loss(graph_decoder, Z, edges[:, 0], edges[:, 1], neg_u, neg_v)


def _sampled_pair_bce_logits_loss(
    scorer: nn.Module,
    Z: torch.Tensor,
    pos_u: torch.Tensor,
    pos_v: torch.Tensor,
    neg_u: torch.Tensor,
    neg_v: torch.Tensor,
) -> torch.Tensor:
    logits = []
    labels = []
    if pos_u.numel() > 0:
        logits.append(_decoder_score_pairs(scorer, Z, pos_u, pos_v))
        labels.append(torch.ones((pos_u.numel(),), dtype=Z.dtype, device=Z.device))
    if neg_u.numel() > 0:
        logits.append(_decoder_score_pairs(scorer, Z, neg_u, neg_v))
        labels.append(torch.zeros((neg_u.numel(),), dtype=Z.dtype, device=Z.device))
    if not logits:
        return Z.sum() * 0.0
    return F.binary_cross_entropy_with_logits(torch.cat(logits, dim=0), torch.cat(labels, dim=0))


def sampled_prediction_bce_logits_loss(
    scorer: nn.Module,
    Z: torch.Tensor,
    train_edges_t: torch.Tensor,
    forbidden_mask: torch.Tensor,
    *,
    num_neg_per_pos: int = 1,
    max_pos_edges: int = 8192,
) -> torch.Tensor:
    if train_edges_t.numel() == 0:
        return Z.sum() * 0.0
    edges = train_edges_t.to(device=Z.device, dtype=torch.long)
    valid = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] < Z.size(0)) & (edges[:, 1] < Z.size(0)) & (edges[:, 0] != edges[:, 1])
    edges = edges[valid]
    if edges.numel() == 0:
        return Z.sum() * 0.0
    if edges.size(0) > int(max_pos_edges):
        order = torch.randperm(edges.size(0), device=Z.device)[: int(max_pos_edges)]
        edges = edges.index_select(0, order)
    neg_count = int(edges.size(0) * max(1, int(num_neg_per_pos)))
    allowed_neg = (~forbidden_mask.to(device=Z.device).bool()).clone()
    allowed_neg.fill_diagonal_(False)
    neg_u, neg_v = _sample_pairs_from_mask(allowed_neg, neg_count)
    return _sampled_pair_bce_logits_loss(scorer, Z, edges[:, 0], edges[:, 1], neg_u, neg_v)


def _structural_endpoint_scores(
    graph_decoder: nn.Module,
    Z: torch.Tensor,
    endpoint: int,
    candidates: torch.Tensor,
) -> torch.Tensor | None:
    if not isinstance(graph_decoder, StructuralPairGraphDecoder) or candidates.numel() == 0:
        return None
    n_nodes = int(Z.size(0))
    dtype = Z.dtype
    device = Z.device
    endpoint_t = torch.tensor(int(endpoint), dtype=torch.long, device=device)
    try:
        score = Z.new_zeros((candidates.numel(),))
        used = False
        for weight, buf in (
            (1.0, graph_decoder.cn_feat),
            (0.5, graph_decoder.ra_feat),
            (0.5, graph_decoder.aa_feat),
        ):
            mat = graph_decoder._pair_matrix(buf, n_nodes, dtype, device)
            if mat.numel() == 0 or tuple(mat.shape) != (n_nodes, n_nodes):
                continue
            score = score + float(weight) * mat[endpoint_t, candidates]
            used = True
        return score if used else None
    except Exception:
        return None


def heart_train_margin_ranking_loss_pairs(
    graph_decoder: nn.Module,
    Z: torch.Tensor,
    train_edges_t: torch.Tensor,
    forbidden_mask: torch.Tensor,
    *,
    num_neg_per_pos: int,
    pool_factor: int,
    margin: float,
    neg_strategy: str = "random",
    struct_frac: float = 0.5,
    max_pos_edges: int = 8192,
    hard_only: bool = False,
    hard_margin: float = 0.2,
    dot_anchor_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    debug = {
        "heart_rank_pairs": 0,
        "heart_rank_pos": 0,
        "heart_rank_neg_pool": 0,
        "heart_rank_pos_mean": float("nan"),
        "heart_rank_neg_mean": float("nan"),
        "heart_rank_pairs_total": 0,
        "heart_rank_hard_pairs": 0,
        "heart_rank_easy_pairs": 0,
        "heart_rank_dot_gap_mean": float("nan"),
        "heart_rank_anchor_loss": float("nan"),
    }
    if train_edges_t.numel() == 0 or num_neg_per_pos <= 0:
        return Z.sum() * 0.0, debug

    edges = train_edges_t.to(device=Z.device, dtype=torch.long)
    valid = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] < Z.size(0)) & (edges[:, 1] < Z.size(0)) & (edges[:, 0] != edges[:, 1])
    edges = edges[valid]
    if edges.numel() == 0:
        return Z.sum() * 0.0, debug
    if edges.size(0) > int(max_pos_edges):
        order = torch.randperm(edges.size(0), device=Z.device)[: int(max_pos_edges)]
        edges = edges.index_select(0, order)

    forbidden = forbidden_mask.to(device=Z.device).bool()
    n_nodes = Z.size(0)
    pool_k = max(int(num_neg_per_pos), int(pool_factor) * max(1, int(num_neg_per_pos)))
    neg_strategy = str(neg_strategy or "random").lower()
    if neg_strategy not in {"random", "struct"}:
        neg_strategy = "random"
    struct_frac = max(0.0, min(1.0, float(struct_frac)))

    def _sample_endpoint_negatives(anchor: int, candidates: torch.Tensor, take: int) -> tuple[torch.Tensor, torch.Tensor]:
        take = int(take)
        if candidates.numel() == 0 or take <= 0:
            empty = torch.empty((0,), dtype=torch.long, device=Z.device)
            return empty, empty
        chosen_parts = []
        if neg_strategy == "struct" and struct_frac > 0.0:
            struct_scores = _structural_endpoint_scores(graph_decoder, Z, anchor, candidates)
            if struct_scores is not None and struct_scores.numel() > 0:
                struct_take = min(int(math.ceil(take * struct_frac)), int(candidates.numel()))
                if struct_take > 0:
                    top_idx = torch.topk(struct_scores, k=struct_take, largest=True).indices
                    chosen_parts.append(candidates.index_select(0, top_idx))
        chosen_count = sum(int(part.numel()) for part in chosen_parts)
        random_take = max(0, take - chosen_count)
        if random_take > 0:
            idx = torch.randint(0, candidates.numel(), (random_take,), device=Z.device)
            chosen_parts.append(candidates.index_select(0, idx))
        if not chosen_parts:
            empty = torch.empty((0,), dtype=torch.long, device=Z.device)
            return empty, empty
        chosen = torch.cat(chosen_parts, dim=0)[:take]
        anchors = torch.full((chosen.numel(),), int(anchor), dtype=torch.long, device=Z.device)
        return anchors, chosen

    pos_u = []
    pos_v = []
    neg_u = []
    neg_v = []
    for u_t, v_t in edges:
        u = int(u_t.item())
        v = int(v_t.item())
        cand_u = (~forbidden[u]).nonzero(as_tuple=True)[0]
        cand_v = (~forbidden[v]).nonzero(as_tuple=True)[0]
        if cand_u.numel() == 0 and cand_v.numel() == 0:
            continue
        cur_u = []
        cur_v = []
        need = int(pool_k)
        if cand_u.numel() > 0:
            take = max(1, need // 2)
            uu, vv = _sample_endpoint_negatives(u, cand_u, take)
            if uu.numel() > 0:
                cur_u.append(uu)
                cur_v.append(vv)
        if cand_v.numel() > 0:
            take = need - sum(x.numel() for x in cur_u)
            take = max(1, take)
            uu, vv = _sample_endpoint_negatives(v, cand_v, take)
            if uu.numel() > 0:
                cur_u.append(uu)
                cur_v.append(vv)
        if not cur_u:
            continue
        cu = torch.cat(cur_u, dim=0)[:need]
        cv = torch.cat(cur_v, dim=0)[:need]
        if cu.numel() < need:
            pad = need - cu.numel()
            cu = torch.cat([cu, cu[:1].expand(pad)], dim=0)
            cv = torch.cat([cv, cv[:1].expand(pad)], dim=0)
        pos_u.append(u_t.view(1))
        pos_v.append(v_t.view(1))
        neg_u.append(cu)
        neg_v.append(cv)

    if not pos_u:
        return Z.sum() * 0.0, debug

    pos_u_t = torch.cat(pos_u, dim=0)
    pos_v_t = torch.cat(pos_v, dim=0)
    neg_u_t = torch.cat(neg_u, dim=0)
    neg_v_t = torch.cat(neg_v, dim=0)
    valid_pos_count = pos_u_t.numel()
    hard_k = min(int(num_neg_per_pos), int(pool_k))

    neg_u_pool = neg_u_t.view(valid_pos_count, pool_k)
    neg_v_pool = neg_v_t.view(valid_pos_count, pool_k)
    with torch.no_grad():
        neg_scores_for_mining = _decoder_score_pairs(graph_decoder, Z, neg_u_t, neg_v_t).view(valid_pos_count, pool_k)
        hard_idx = torch.topk(neg_scores_for_mining, k=hard_k, dim=1, largest=True).indices

    hard_neg_u = neg_u_pool.gather(1, hard_idx).reshape(-1)
    hard_neg_v = neg_v_pool.gather(1, hard_idx).reshape(-1)
    hard_neg = _decoder_score_pairs(graph_decoder, Z, hard_neg_u, hard_neg_v).view(valid_pos_count, hard_k)
    pos_scores = _decoder_score_pairs(graph_decoder, Z, pos_u_t, pos_v_t).view(-1, 1).expand_as(hard_neg)
    pos_flat = pos_scores.reshape(-1)
    neg_flat = hard_neg.reshape(-1)
    target = torch.ones_like(pos_flat)

    with torch.no_grad():
        dot_pos = (Z.index_select(0, pos_u_t) * Z.index_select(0, pos_v_t)).sum(dim=1).view(-1, 1).expand_as(hard_neg)
        dot_neg = (Z.index_select(0, hard_neg_u) * Z.index_select(0, hard_neg_v)).sum(dim=1).view(valid_pos_count, hard_k)
        dot_pos_flat = dot_pos.reshape(-1)
        dot_neg_flat = dot_neg.reshape(-1)
        dot_gap = dot_pos_flat - dot_neg_flat
        hard_mask = dot_gap < float(hard_margin)

    rank_losses = F.margin_ranking_loss(pos_flat, neg_flat, target, margin=float(margin), reduction="none")
    hard_only = bool(hard_only)
    if hard_only:
        if bool(hard_mask.any().item()):
            loss = rank_losses[hard_mask].mean()
        else:
            loss = pos_flat.sum() * 0.0
        used_rank_pairs = int(hard_mask.sum().item())
    else:
        loss = rank_losses.mean()
        used_rank_pairs = int(rank_losses.numel())

    anchor_loss = pos_flat.new_tensor(0.0)
    dot_anchor_weight = max(0.0, float(dot_anchor_weight))
    if dot_anchor_weight > 0.0:
        anchor_mask = ~hard_mask if hard_only else torch.ones_like(hard_mask, dtype=torch.bool)
        if bool(anchor_mask.any().item()):
            pred_anchor = torch.cat([pos_flat[anchor_mask], neg_flat[anchor_mask]], dim=0)
            dot_anchor = torch.cat([dot_pos_flat[anchor_mask], dot_neg_flat[anchor_mask]], dim=0)
            anchor_loss = F.smooth_l1_loss(pred_anchor, dot_anchor.detach())
            loss = loss + dot_anchor_weight * anchor_loss

    debug.update(
        {
            "heart_rank_pairs": int(used_rank_pairs),
            "heart_rank_pos": int(valid_pos_count),
            "heart_rank_neg_pool": int(neg_u_t.numel()),
            "heart_rank_pos_mean": float(pos_scores.detach().mean().cpu()),
            "heart_rank_neg_mean": float(hard_neg.detach().mean().cpu()),
            "heart_rank_neg_strategy": neg_strategy,
            "heart_rank_pairs_total": int(pos_scores.numel()),
            "heart_rank_hard_pairs": int(hard_mask.sum().item()),
            "heart_rank_easy_pairs": int((~hard_mask).sum().item()),
            "heart_rank_dot_gap_mean": float(dot_gap.detach().mean().cpu()),
            "heart_rank_anchor_loss": float(anchor_loss.detach().cpu()) if dot_anchor_weight > 0.0 else float("nan"),
        }
    )
    return loss, debug


def hybrid_decoder_structure_losses_pairwise(
    graph_decoder: nn.Module,
    Z_edit: torch.Tensor,
    adj_label: torch.Tensor,
    train_edges_t: torch.Tensor,
    forbidden_mask: torch.Tensor,
    labels: np.ndarray | None,
    rewrite_mask: torch.Tensor | None,
    *,
    E0: int,
    add_ratio: float,
    remove_ratio: float,
    add_threshold: float | None,
    remove_threshold: float | None,
    add_quantile: float | None,
    remove_quantile: float | None,
    max_add: int | None,
    max_remove: int | None,
    degree_floor: int,
    same_cluster_only: bool,
    require_c0p_endpoint: bool,
    require_both_c0p: bool,
    require_c0p_noncompact_endpoint: bool,
    require_structural_support: bool,
    structural_support_mode: str,
    structural_min_cn: float,
    structural_min_ra: float,
    structural_min_aa: float,
    structural_support_mask: torch.Tensor | None,
    keep_weight: float,
    add_rank_weight: float,
    remove_rank_weight: float,
    rank_margin: float,
    rank_strategy: str,
    rank_neg_k: int,
    rank_pool_factor: int,
    pair_context: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    if pair_context is None:
        adj_dense = _to_dense(adj_label)
        ctx = _build_decoded_pair_context(
            adj_dense,
            labels,
            rewrite_mask,
            degree_floor=degree_floor,
            same_cluster_only=same_cluster_only,
            require_c0p_endpoint=require_c0p_endpoint,
            require_both_c0p=require_both_c0p,
            require_c0p_noncompact_endpoint=require_c0p_noncompact_endpoint,
            require_structural_support=require_structural_support,
            structural_support_mode=structural_support_mode,
            structural_min_cn=structural_min_cn,
            structural_min_ra=structural_min_ra,
            structural_min_aa=structural_min_aa,
            structural_support_mask=structural_support_mask,
        )
    else:
        ctx = pair_context
    debug_info = _empty_decoder_debug_info()
    debug_info.update(
        {
            "rewrite_nodes": int(ctx["core_mask"].sum().item()),
            "valid_pairs": int(ctx["valid_pairs"].triu(1).sum().item()),
            "add_pairs": int(ctx["add_pairs"].triu(1).sum().item()),
            "add_pairs_pre_struct": int(ctx["add_pairs_pre_struct"].triu(1).sum().item()),
            "struct_supported_add_pairs": int((ctx["add_pairs_pre_struct"] & ctx["structural_support"]).triu(1).sum().item()),
            "removable_pairs": int(ctx["removable_pairs"].triu(1).sum().item()),
        }
    )

    add_budget = max(0, int(round(float(add_ratio) * float(max(1, E0)))))
    if max_add is not None:
        add_budget = min(add_budget if add_budget > 0 else int(max_add), int(max_add))
    rem_budget = max(0, int(round(float(remove_ratio) * float(max(1, E0)))))
    if max_remove is not None:
        rem_budget = min(rem_budget if rem_budget > 0 else int(max_remove), int(max_remove))
    debug_info["add_budget"] = int(add_budget)
    debug_info["remove_budget"] = int(rem_budget)

    edges = train_edges_t.to(device=Z_edit.device, dtype=torch.long)
    keep_pos_mask = torch.zeros((0,), dtype=torch.bool, device=Z_edit.device)
    if edges.numel() > 0:
        valid = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] < Z_edit.size(0)) & (edges[:, 1] < Z_edit.size(0))
        edges = edges[valid]
        if edges.numel() > 0:
            keep_pos_mask = ~ctx["valid_pairs"][edges[:, 0], edges[:, 1]]
    keep_pos = edges[keep_pos_mask] if edges.numel() > 0 else edges
    if keep_pos.size(0) > 8192:
        order = torch.randperm(keep_pos.size(0), device=Z_edit.device)[:8192]
        keep_pos = keep_pos.index_select(0, order)
    keep_neg_mask = (~ctx["valid_pairs"]) & (~forbidden_mask.to(device=Z_edit.device).bool())
    keep_neg_mask.fill_diagonal_(False)
    neg_u, neg_v = _sample_pairs_from_mask(keep_neg_mask, int(max(keep_pos.size(0), 1)))
    keep_loss = _sampled_pair_bce_loss(
        graph_decoder,
        Z_edit,
        keep_pos[:, 0] if keep_pos.numel() > 0 else torch.empty((0,), dtype=torch.long, device=Z_edit.device),
        keep_pos[:, 1] if keep_pos.numel() > 0 else torch.empty((0,), dtype=torch.long, device=Z_edit.device),
        neg_u,
        neg_v,
    )

    add_ii, add_jj = ctx["add_pairs"].triu(1).nonzero(as_tuple=True)
    if add_ii.numel() > 500000:
        keep = torch.randperm(add_ii.numel(), device=Z_edit.device)[:500000]
        add_ii = add_ii.index_select(0, keep)
        add_jj = add_jj.index_select(0, keep)
    with torch.no_grad():
        add_scores_det = _decoder_score_pairs(graph_decoder, Z_edit.detach(), add_ii, add_jj).detach() if add_ii.numel() > 0 else Z_edit.new_empty((0,))
    add_sel = _select_candidate_indices(
        add_scores_det,
        prefer_high=True,
        threshold=add_threshold,
        quantile=add_quantile,
        budget=add_budget,
        max_count=max_add,
    )
    add_neg_pool = torch.ones_like(add_scores_det, dtype=torch.bool)
    if add_sel.numel() > 0:
        add_neg_pool[add_sel] = False
    add_neg_idx = add_neg_pool.nonzero(as_tuple=True)[0]
    if rank_strategy == "heart_like":
        add_neg_idx = _ordered_subset(
            add_scores_det,
            add_neg_idx,
            prefer_high=True,
            count=max(int(rank_neg_k) * max(1, add_sel.numel()), int(rank_pool_factor) * max(1, add_sel.numel())),
        )
    add_pos_scores = _decoder_score_pairs(graph_decoder, Z_edit, add_ii.index_select(0, add_sel), add_jj.index_select(0, add_sel)) if add_sel.numel() > 0 else Z_edit.new_empty((0,))
    add_neg_scores = _decoder_score_pairs(graph_decoder, Z_edit, add_ii.index_select(0, add_neg_idx), add_jj.index_select(0, add_neg_idx)) if add_neg_idx.numel() > 0 else Z_edit.new_empty((0,))
    add_rank_loss = _paired_margin_ranking_loss(add_pos_scores, add_neg_scores, rank_margin)

    rem_ii, rem_jj = ctx["removable_pairs"].triu(1).nonzero(as_tuple=True)
    with torch.no_grad():
        rem_scores_det = _decoder_score_pairs(graph_decoder, Z_edit.detach(), rem_ii, rem_jj).detach() if rem_ii.numel() > 0 else Z_edit.new_empty((0,))
    rem_sel = _select_candidate_indices(
        rem_scores_det,
        prefer_high=False,
        threshold=remove_threshold,
        quantile=remove_quantile,
        budget=rem_budget,
        max_count=max_remove,
    )
    rem_keep_pool = torch.ones_like(rem_scores_det, dtype=torch.bool)
    if rem_sel.numel() > 0:
        rem_keep_pool[rem_sel] = False
    rem_keep_idx = rem_keep_pool.nonzero(as_tuple=True)[0]
    if rank_strategy == "heart_like":
        rem_keep_idx = _ordered_subset(
            rem_scores_det,
            rem_keep_idx,
            prefer_high=True,
            count=max(int(rank_neg_k) * max(1, rem_sel.numel()), int(rank_pool_factor) * max(1, rem_sel.numel())),
        )
    rem_keep_scores = _decoder_score_pairs(graph_decoder, Z_edit, rem_ii.index_select(0, rem_keep_idx), rem_jj.index_select(0, rem_keep_idx)) if rem_keep_idx.numel() > 0 else Z_edit.new_empty((0,))
    rem_bad_scores = _decoder_score_pairs(graph_decoder, Z_edit, rem_ii.index_select(0, rem_sel), rem_jj.index_select(0, rem_sel)) if rem_sel.numel() > 0 else Z_edit.new_empty((0,))
    remove_rank_loss = _paired_margin_ranking_loss(rem_keep_scores, rem_bad_scores, rank_margin)

    debug_info.update(
        {
            "add_selected": int(add_sel.numel()),
            "add_negatives": int(add_neg_scores.numel()),
            "add_rank_pairs": int(min(add_pos_scores.numel(), add_neg_scores.numel())),
            "remove_selected": int(rem_sel.numel()),
            "remove_kept": int(rem_keep_scores.numel()),
            "remove_rank_pairs": int(min(rem_keep_scores.numel(), rem_bad_scores.numel())),
        }
    )
    total = keep_weight * keep_loss + add_rank_weight * add_rank_loss + remove_rank_weight * remove_rank_loss
    return total, keep_loss, add_rank_loss, remove_rank_loss, debug_info


@torch.no_grad()
def build_decoded_augmented_graph_from_decoder(
    graph_decoder: nn.Module,
    Z: torch.Tensor,
    adj_current_dense: torch.Tensor,
    labels: np.ndarray | None,
    node_mask: torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, int, int]:
    ctx = kwargs.get("pair_context", None)
    if ctx is None:
        ctx = _build_decoded_pair_context(
            adj_current_dense,
            labels,
            node_mask,
            degree_floor=int(kwargs.get("degree_floor", 0)),
            same_cluster_only=bool(kwargs.get("same_cluster_only", True)),
            require_c0p_endpoint=bool(kwargs.get("require_c0p_endpoint", True)),
            require_both_c0p=bool(kwargs.get("require_both_c0p", False)),
            require_c0p_noncompact_endpoint=bool(kwargs.get("require_c0p_noncompact_endpoint", False)),
            deg0_excl_self=kwargs.get("deg0_excl_self", None),
            require_structural_support=bool(kwargs.get("require_structural_support", False)),
            structural_support_mode=str(kwargs.get("structural_support_mode", "cn_or_ra")),
            structural_min_cn=float(kwargs.get("structural_min_cn", 1.0)),
            structural_min_ra=float(kwargs.get("structural_min_ra", 0.0)),
            structural_min_aa=float(kwargs.get("structural_min_aa", 0.0)),
            structural_support_mask=kwargs.get("structural_support_mask", None),
        )
    cand = ctx["add_pairs"] | ctx["removable_pairs"]
    ii, jj = cand.triu(1).nonzero(as_tuple=True)
    scores = adj_current_dense.new_zeros(adj_current_dense.shape)
    if ii.numel() > 0:
        vals = _decoder_score_pairs(graph_decoder, Z, ii, jj).detach()
        scores[ii, jj] = vals
        scores[jj, ii] = vals
    scores.fill_diagonal_(1.0)
    return build_decoded_augmented_graph(scores, adj_current_dense, labels, node_mask, **kwargs)


def _edge_score_values_from_dot(Z: torch.Tensor, edges) -> np.ndarray:
    edges_arr = np.asarray(edges)
    if edges_arr.size == 0:
        return np.asarray([], dtype=np.float64)
    out_shape = edges_arr.shape[:-1]
    flat = edges_arr.reshape(-1, 2)
    uu = torch.as_tensor(flat[:, 0], dtype=torch.long, device=Z.device)
    vv = torch.as_tensor(flat[:, 1], dtype=torch.long, device=Z.device)
    with torch.no_grad():
        vals = torch.sigmoid((Z.index_select(0, uu) * Z.index_select(0, vv)).sum(dim=1))
    return vals.detach().cpu().numpy().astype(np.float64).reshape(out_shape)


def _edge_score_values_from_decoder(graph_decoder: nn.Module, Z: torch.Tensor, edges) -> np.ndarray:
    edges_arr = np.asarray(edges)
    if edges_arr.size == 0:
        return np.asarray([], dtype=np.float64)
    out_shape = edges_arr.shape[:-1]
    flat = edges_arr.reshape(-1, 2)
    uu = torch.as_tensor(flat[:, 0], dtype=torch.long, device=Z.device)
    vv = torch.as_tensor(flat[:, 1], dtype=torch.long, device=Z.device)
    with torch.no_grad():
        vals = _decoder_score_pairs(graph_decoder, Z, uu, vv)
    return vals.detach().cpu().numpy().astype(np.float64).reshape(out_shape)


def get_scores_from_values(pos_scores, neg_scores):
    pos_scores = np.asarray(pos_scores, dtype=np.float64).reshape(-1)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    if neg_scores.ndim == 2:
        preds_all = np.concatenate([pos_scores, neg_scores.reshape(-1)])
        labels_all = np.concatenate([np.ones_like(pos_scores), np.zeros(neg_scores.size, dtype=np.float64)])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        pos_tensor = torch.tensor(pos_scores)
        neg_tensor = torch.tensor(neg_scores)
        hitk = [eval_hits_heart(pos_tensor, neg_tensor, k) for k in [1, 3, 10, 20, 50, 100]]
        return roc_score, ap_score, hitk
    neg_flat = neg_scores.reshape(-1)
    preds_all = np.hstack([pos_scores, neg_flat])
    labels_all = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_flat))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    pos_tensor = torch.tensor(pos_scores)
    neg_tensor = torch.tensor(neg_flat)
    hitk = [eval_hits(pos_tensor, neg_tensor, k) for k in [1, 3, 10, 20, 50, 100]]
    return roc_score, ap_score, hitk


def _score_source_diagnostics_from_values(dot_pos, dot_neg, decoder_pos, decoder_neg) -> dict[str, float]:
    diag = _score_source_diagnostics(None, None, [], [])
    dot_pos = np.asarray(dot_pos, dtype=np.float64).reshape(-1)
    dot_neg = np.asarray(dot_neg, dtype=np.float64).reshape(-1)
    dec_pos = np.asarray(decoder_pos, dtype=np.float64).reshape(-1)
    dec_neg = np.asarray(decoder_neg, dtype=np.float64).reshape(-1)
    if dot_pos.size > 0:
        diag["diag_dot_pos_mean"] = float(np.mean(dot_pos))
    if dot_neg.size > 0:
        diag["diag_dot_neg_mean"] = float(np.mean(dot_neg))
    if dec_pos.size > 0:
        diag["diag_decoder_pos_mean"] = float(np.mean(dec_pos))
    if dec_neg.size > 0:
        diag["diag_decoder_neg_mean"] = float(np.mean(dec_neg))
    dot_vals = np.concatenate([dot_pos, dot_neg])
    dec_vals = np.concatenate([dec_pos, dec_neg])
    if dot_vals.size > 1 and dec_vals.size > 1 and np.std(dot_vals) > 0 and np.std(dec_vals) > 0:
        diag["diag_dot_decoder_corr"] = float(np.corrcoef(dot_vals, dec_vals)[0, 1])
    return diag

def _resolve_node_mask(node_mask: torch.Tensor | None, num_nodes: int, device: torch.device) -> torch.Tensor:
    if node_mask is None:
        return torch.ones(num_nodes, dtype=torch.bool, device=device)
    return node_mask.to(device).bool()


def _decoded_structural_support_mask(
    graph_dense: torch.Tensor,
    *,
    mode: str = "cn_or_ra",
    min_cn: float = 1.0,
    min_ra: float = 0.0,
    min_aa: float = 0.0,
) -> torch.Tensor:
    device = graph_dense.device
    n_nodes = graph_dense.size(0)
    mode = str(mode or "cn_or_ra").lower()
    valid_modes = {"cn", "ra", "aa", "cn_or_ra", "cn_or_aa", "ra_or_aa", "any", "all"}
    if mode not in valid_modes:
        mode = "cn_or_ra"

    g0 = (graph_dense > 0).to(torch.float32).clone()
    g0.fill_diagonal_(0.0)
    deg = g0.sum(dim=1).clamp_min(1.0)

    def _passes(values: torch.Tensor, threshold: float) -> torch.Tensor:
        threshold = float(threshold)
        if threshold <= 0.0:
            return values > 0.0
        return values >= threshold

    terms: dict[str, torch.Tensor] = {}
    need_cn = mode in {"cn", "cn_or_ra", "cn_or_aa", "any", "all"}
    need_ra = mode in {"ra", "cn_or_ra", "ra_or_aa", "any", "all"}
    need_aa = mode in {"aa", "cn_or_aa", "ra_or_aa", "any", "all"}
    if need_cn:
        cn = g0 @ g0
        terms["cn"] = _passes(cn, min_cn)
    if need_ra:
        ra = (g0 * deg.reciprocal().view(1, -1)) @ g0
        terms["ra"] = _passes(ra, min_ra)
    if need_aa:
        log_deg = torch.log(deg.clamp_min(2.0))
        inv_log_deg = torch.where(log_deg > 0, log_deg.reciprocal(), torch.zeros_like(log_deg))
        aa = (g0 * inv_log_deg.view(1, -1)) @ g0
        terms["aa"] = _passes(aa, min_aa)

    if not terms:
        support = torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device)
    elif mode == "all":
        support = torch.ones((n_nodes, n_nodes), dtype=torch.bool, device=device)
        for term in terms.values():
            support = support & term
    elif mode == "cn_or_ra":
        support = terms.get("cn", torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device)) | terms.get("ra", torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device))
    elif mode == "cn_or_aa":
        support = terms.get("cn", torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device)) | terms.get("aa", torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device))
    elif mode == "ra_or_aa":
        support = terms.get("ra", torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device)) | terms.get("aa", torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device))
    elif mode == "any":
        support = torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device)
        for term in terms.values():
            support = support | term
    else:
        support = terms[mode]
    support.fill_diagonal_(False)
    return support


def _build_decoded_pair_context(
    adj_current_dense: torch.Tensor,
    labels: np.ndarray | None,
    node_mask: torch.Tensor | None,
    *,
    degree_floor: int = 0,
    same_cluster_only: bool = True,
    require_c0p_endpoint: bool = True,
    require_both_c0p: bool = False,
    require_c0p_noncompact_endpoint: bool = False,
    deg0_excl_self: torch.Tensor | None = None,
    require_structural_support: bool = False,
    structural_support_mode: str = "cn_or_ra",
    structural_min_cn: float = 1.0,
    structural_min_ra: float = 0.0,
    structural_min_aa: float = 0.0,
    structural_support_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    device = adj_current_dense.device
    n_nodes = adj_current_dense.size(0)
    g = (
        (adj_current_dense > 0).to(torch.float32)
        + (adj_current_dense.t() > 0).to(torch.float32)
        > 0
    ).to(torch.float32)
    g.fill_diagonal_(1.0)

    eye = torch.eye(n_nodes, dtype=torch.bool, device=device)
    existing = g > 0
    core_mask = _resolve_node_mask(node_mask, n_nodes, device)

    if labels is None:
        same_cluster = torch.ones((n_nodes, n_nodes), dtype=torch.bool, device=device)
        non_noise = torch.ones(n_nodes, dtype=torch.bool, device=device)
    else:
        labels_np = np.asarray(labels, dtype=np.int64)
        lbl_t = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
        non_noise = lbl_t != -1
        same_cluster = (lbl_t[:, None] == lbl_t[None, :]) & non_noise[:, None] & non_noise[None, :]

    if require_c0p_noncompact_endpoint:
        noncompact_cp = non_noise & (~core_mask)
        endpoint_ok = (core_mask[:, None] & noncompact_cp[None, :]) | (noncompact_cp[:, None] & core_mask[None, :])
    elif require_both_c0p:
        endpoint_ok = core_mask[:, None] & core_mask[None, :]
    elif require_c0p_endpoint:
        endpoint_ok = core_mask[:, None] | core_mask[None, :]
    else:
        endpoint_ok = torch.ones((n_nodes, n_nodes), dtype=torch.bool, device=device)

    valid_pairs = endpoint_ok & (~eye)
    if same_cluster_only:
        valid_pairs = valid_pairs & same_cluster
    else:
        valid_pairs = valid_pairs & non_noise[:, None] & non_noise[None, :]

    if deg0_excl_self is None:
        deg0 = _deg_excl_self(g)
    else:
        deg0 = deg0_excl_self.to(device)

    removable_pairs = valid_pairs & existing
    removable_pairs.fill_diagonal_(False)
    enough_degree = deg0 > int(degree_floor)
    removable_pairs = removable_pairs & enough_degree[:, None] & enough_degree[None, :]

    add_pairs = valid_pairs & (~existing)
    add_pairs_pre_struct = add_pairs.clone()
    structural_support = torch.ones((n_nodes, n_nodes), dtype=torch.bool, device=device)
    if require_structural_support:
        if structural_support_mask is not None:
            structural_support = structural_support_mask.to(device=device).bool()
            if tuple(structural_support.shape) != (n_nodes, n_nodes):
                structural_support = _decoded_structural_support_mask(
                    g,
                    mode=structural_support_mode,
                    min_cn=structural_min_cn,
                    min_ra=structural_min_ra,
                    min_aa=structural_min_aa,
                )
        else:
            structural_support = _decoded_structural_support_mask(
                g,
                mode=structural_support_mode,
                min_cn=structural_min_cn,
                min_ra=structural_min_ra,
                min_aa=structural_min_aa,
            )
        add_pairs = add_pairs & structural_support

    return {
        "graph_dense": g,
        "eye": eye,
        "existing": existing,
        "core_mask": core_mask,
        "same_cluster": same_cluster,
        "non_noise": non_noise,
        "valid_pairs": valid_pairs,
        "add_pairs": add_pairs,
        "add_pairs_pre_struct": add_pairs_pre_struct,
        "structural_support": structural_support,
        "removable_pairs": removable_pairs,
        "deg0": deg0,
    }


def _masked_pair_scores(score_matrix: torch.Tensor, pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ii, jj = pair_mask.triu(1).nonzero(as_tuple=True)
    if ii.numel() == 0:
        empty = score_matrix.new_empty((0,))
        return ii, jj, empty
    return ii, jj, score_matrix[ii, jj]


def _select_candidate_indices(
    scores: torch.Tensor,
    *,
    prefer_high: bool,
    threshold: float | None = None,
    quantile: float | None = None,
    budget: int = 0,
    max_count: int | None = None,
) -> torch.Tensor:
    if scores.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)

    if threshold is not None:
        keep = scores >= float(threshold) if prefer_high else scores <= float(threshold)
        selected = keep.nonzero(as_tuple=True)[0]
    elif quantile is not None and 0.0 < float(quantile) < 1.0:
        q = max(0.0, min(1.0, 1.0 - float(quantile))) if prefer_high else max(0.0, min(1.0, float(quantile)))
        thr = torch.quantile(scores, q)
        keep = scores >= thr if prefer_high else scores <= thr
        selected = keep.nonzero(as_tuple=True)[0]
    else:
        limit = max(0, int(budget))
        if max_count is not None:
            limit = min(limit if limit > 0 else int(max_count), int(max_count))
        if limit <= 0:
            return torch.empty((0,), dtype=torch.long, device=scores.device)
        order = torch.argsort(scores, descending=prefer_high)
        return order[:limit]

    if selected.numel() == 0:
        return selected
    order = torch.argsort(scores[selected], descending=prefer_high)
    selected = selected[order]
    if max_count is not None:
        selected = selected[: int(max_count)]
    return selected


def _paired_margin_ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        ref = pos_scores if pos_scores.numel() > 0 else neg_scores
        return ref.new_tensor(0.0)
    pair_count = min(pos_scores.numel(), neg_scores.numel())
    pos = pos_scores[:pair_count]
    neg = neg_scores[:pair_count]
    target = torch.ones(pair_count, device=pos.device)
    return F.margin_ranking_loss(pos, neg, target, margin=float(margin))


def _ordered_subset(
    scores: torch.Tensor,
    indices: torch.Tensor,
    *,
    prefer_high: bool,
    count: int,
) -> torch.Tensor:
    if indices.numel() == 0 or count <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    order = torch.argsort(scores.index_select(0, indices), descending=prefer_high)
    return indices.index_select(0, order[: min(int(count), order.numel())])


def _heart_like_margin_ranking_loss(
    scores: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    pos_anchor_idx: torch.Tensor,
    neg_pool_idx: torch.Tensor,
    *,
    num_neg_per_pos: int,
    margin: float,
) -> tuple[torch.Tensor, int, int]:
    if pos_anchor_idx.numel() == 0 or neg_pool_idx.numel() == 0 or num_neg_per_pos <= 0:
        ref = scores if scores.numel() > 0 else torch.zeros(1, device=ii.device)
        return ref.new_tensor(0.0), 0, 0

    neg_u = ii.index_select(0, neg_pool_idx)
    neg_v = jj.index_select(0, neg_pool_idx)
    pos_terms = []
    neg_terms = []
    used_neg = 0

    for pos_idx in pos_anchor_idx:
        pos_u = ii[pos_idx]
        pos_v = jj[pos_idx]
        shared_endpoint = (neg_u == pos_u) | (neg_v == pos_u) | (neg_u == pos_v) | (neg_v == pos_v)
        cand = neg_pool_idx[shared_endpoint]
        if cand.numel() == 0:
            cand = neg_pool_idx
        cand = cand[: int(num_neg_per_pos)]
        if cand.numel() == 0:
            continue
        pos_terms.append(scores[pos_idx].expand(cand.numel()))
        neg_terms.append(scores.index_select(0, cand))
        used_neg += int(cand.numel())

    if len(pos_terms) == 0:
        return scores.new_tensor(0.0), int(neg_pool_idx.numel()), 0

    pos_flat = torch.cat(pos_terms, dim=0)
    neg_flat = torch.cat(neg_terms, dim=0)
    target = torch.ones_like(pos_flat)
    return (
        F.margin_ranking_loss(pos_flat, neg_flat, target, margin=float(margin)),
        int(neg_pool_idx.numel()),
        int(pos_flat.numel()),
    )


def _build_forbidden_edge_mask(adj_like, num_nodes: int, device: torch.device) -> torch.Tensor:
    forbidden = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    adj_csr = _adjacency_to_binary_csr(adj_like)
    coo = adj_csr.tocoo()
    if coo.nnz > 0:
        rr = torch.as_tensor(coo.row, dtype=torch.long, device=device)
        cc = torch.as_tensor(coo.col, dtype=torch.long, device=device)
        forbidden[rr, cc] = True
        forbidden[cc, rr] = True
    forbidden.fill_diagonal_(True)
    return forbidden


def heart_train_margin_ranking_loss(
    adj_pred: torch.Tensor,
    train_edges_t: torch.Tensor,
    forbidden_mask: torch.Tensor,
    *,
    num_neg_per_pos: int,
    pool_factor: int,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    debug = {
        "heart_rank_pairs": 0,
        "heart_rank_pos": 0,
        "heart_rank_neg_pool": 0,
        "heart_rank_pos_mean": float("nan"),
        "heart_rank_neg_mean": float("nan"),
    }
    if train_edges_t.numel() == 0 or num_neg_per_pos <= 0:
        return adj_pred.sum() * 0.0, debug

    edges = train_edges_t.to(device=adj_pred.device, dtype=torch.long)
    forbidden = forbidden_mask.to(device=adj_pred.device).bool()
    pos_terms = []
    neg_terms = []
    neg_pool_total = 0
    n_nodes = adj_pred.size(0)
    pool_k_base = max(int(num_neg_per_pos), int(pool_factor) * max(1, int(num_neg_per_pos)))

    for u_t, v_t in edges:
        u = int(u_t.item())
        v = int(v_t.item())
        if u < 0 or v < 0 or u >= n_nodes or v >= n_nodes or u == v:
            continue
        cand_u = (~forbidden[u]).nonzero(as_tuple=True)[0]
        cand_v = (~forbidden[v]).nonzero(as_tuple=True)[0]
        if cand_u.numel() == 0 and cand_v.numel() == 0:
            continue
        scores = []
        if cand_u.numel() > 0:
            scores.append(adj_pred[u, cand_u])
        if cand_v.numel() > 0:
            scores.append(adj_pred[v, cand_v])
        neg_scores_all = torch.cat(scores, dim=0)
        if neg_scores_all.numel() == 0:
            continue
        neg_pool_total += int(neg_scores_all.numel())
        pool_k = min(int(pool_k_base), int(neg_scores_all.numel()))
        hard_pool = torch.topk(neg_scores_all, k=pool_k, largest=True).values
        hard_neg = hard_pool[: min(int(num_neg_per_pos), int(hard_pool.numel()))]
        if hard_neg.numel() == 0:
            continue
        pos_score = adj_pred[u, v].expand(hard_neg.numel())
        pos_terms.append(pos_score)
        neg_terms.append(hard_neg)

    if not pos_terms:
        return adj_pred.sum() * 0.0, debug

    pos_flat = torch.cat(pos_terms, dim=0)
    neg_flat = torch.cat(neg_terms, dim=0)
    target = torch.ones_like(pos_flat)
    loss = F.margin_ranking_loss(pos_flat, neg_flat, target, margin=float(margin))
    debug.update(
        {
            "heart_rank_pairs": int(pos_flat.numel()),
            "heart_rank_pos": int(len(pos_terms)),
            "heart_rank_neg_pool": int(neg_pool_total),
            "heart_rank_pos_mean": float(pos_flat.detach().mean().cpu()),
            "heart_rank_neg_mean": float(neg_flat.detach().mean().cpu()),
        }
    )
    return loss, debug


def _empty_decoder_debug_info() -> dict[str, int]:
    return {
        "rewrite_nodes": 0,
        "valid_pairs": 0,
        "add_pairs": 0,
        "add_pairs_pre_struct": 0,
        "struct_supported_add_pairs": 0,
        "add_budget": 0,
        "add_selected": 0,
        "add_negatives": 0,
        "add_rank_pairs": 0,
        "removable_pairs": 0,
        "remove_budget": 0,
        "remove_selected": 0,
        "remove_kept": 0,
        "remove_rank_pairs": 0,
        "heart_rank_pairs": 0,
        "heart_rank_pos": 0,
        "heart_rank_neg_pool": 0,
    }


def hybrid_decoder_structure_losses(
    adj_pred: torch.Tensor,
    adj_label: torch.Tensor,
    norm: float,
    weight_tensor: torch.Tensor,
    train_mask: torch.Tensor,
    labels: np.ndarray | None,
    rewrite_mask: torch.Tensor | None,
    *,
    E0: int,
    add_ratio: float,
    remove_ratio: float,
    add_threshold: float | None,
    remove_threshold: float | None,
    add_quantile: float | None,
    remove_quantile: float | None,
    max_add: int | None,
    max_remove: int | None,
    degree_floor: int,
    same_cluster_only: bool,
    require_c0p_endpoint: bool,
    require_both_c0p: bool,
    require_c0p_noncompact_endpoint: bool,
    require_structural_support: bool,
    structural_support_mode: str,
    structural_min_cn: float,
    structural_min_ra: float,
    structural_min_aa: float,
    structural_support_mask: torch.Tensor | None,
    keep_weight: float,
    add_rank_weight: float,
    remove_rank_weight: float,
    rank_margin: float,
    rank_strategy: str,
    rank_neg_k: int,
    rank_pool_factor: int,
    pair_context: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    if pair_context is None:
        adj_dense = _to_dense(adj_label)
        ctx = _build_decoded_pair_context(
            adj_dense,
            labels,
            rewrite_mask,
            degree_floor=degree_floor,
            same_cluster_only=same_cluster_only,
            require_c0p_endpoint=require_c0p_endpoint,
            require_both_c0p=require_both_c0p,
            require_c0p_noncompact_endpoint=require_c0p_noncompact_endpoint,
            require_structural_support=require_structural_support,
            structural_support_mode=structural_support_mode,
            structural_min_cn=structural_min_cn,
            structural_min_ra=structural_min_ra,
            structural_min_aa=structural_min_aa,
            structural_support_mask=structural_support_mask,
        )
    else:
        ctx = pair_context

    keep_mask = ~ctx["valid_pairs"]
    keep_loss = reconstruction_bce_loss(
        adj_pred,
        adj_label,
        norm,
        weight_tensor,
        train_mask,
        extra_mask=keep_mask,
    )

    add_budget = max(0, int(round(float(add_ratio) * float(max(1, E0)))))
    if max_add is not None:
        add_budget = min(add_budget if add_budget > 0 else int(max_add), int(max_add))
    rem_budget = max(0, int(round(float(remove_ratio) * float(max(1, E0)))))
    if max_remove is not None:
        rem_budget = min(rem_budget if rem_budget > 0 else int(max_remove), int(max_remove))

    add_ii, add_jj, add_scores = _masked_pair_scores(adj_pred, ctx["add_pairs"])
    add_sel = _select_candidate_indices(
        add_scores,
        prefer_high=True,
        threshold=add_threshold,
        quantile=add_quantile,
        budget=add_budget,
        max_count=max_add,
    )
    add_neg_pool = torch.ones_like(add_scores, dtype=torch.bool)
    if add_sel.numel() > 0:
        add_neg_pool[add_sel] = False
    add_neg_idx = add_neg_pool.nonzero(as_tuple=True)[0]
    add_neg_scores = add_scores.index_select(0, add_neg_idx) if add_neg_idx.numel() > 0 else add_scores.new_empty((0,))

    _, _, rem_scores = _masked_pair_scores(adj_pred, ctx["removable_pairs"])
    rem_sel = _select_candidate_indices(
        rem_scores,
        prefer_high=False,
        threshold=remove_threshold,
        quantile=remove_quantile,
        budget=rem_budget,
        max_count=max_remove,
    )
    rem_keep_pool = torch.ones_like(rem_scores, dtype=torch.bool)
    if rem_sel.numel() > 0:
        rem_keep_pool[rem_sel] = False
    rem_keep_idx = rem_keep_pool.nonzero(as_tuple=True)[0]
    rem_keep_scores = rem_scores.index_select(0, rem_keep_idx) if rem_keep_idx.numel() > 0 else rem_scores.new_empty((0,))

    add_rank_pairs = int(min(add_scores[add_sel].numel(), add_neg_scores.numel()))
    remove_rank_pairs = int(min(rem_keep_scores.numel(), rem_scores[rem_sel].numel()))
    if rank_strategy == "heart_like":
        add_boundary_pool = _ordered_subset(
            add_scores,
            add_neg_idx,
            prefer_high=True,
            count=max(int(rank_neg_k), int(rank_pool_factor) * max(1, int(add_sel.numel()))),
        )
        add_anchor_idx = _ordered_subset(
            add_scores,
            add_sel,
            prefer_high=False,
            count=max(1, min(int(add_sel.numel()), add_boundary_pool.numel())),
        )
        add_rank_loss, add_neg_count, add_rank_pairs = _heart_like_margin_ranking_loss(
            add_scores,
            add_ii,
            add_jj,
            add_anchor_idx,
            add_boundary_pool,
            num_neg_per_pos=rank_neg_k,
            margin=rank_margin,
        )

        rem_ii, rem_jj, _ = _masked_pair_scores(adj_pred, ctx["removable_pairs"])
        rem_boundary_pool = _ordered_subset(
            rem_scores,
            rem_sel,
            prefer_high=True,
            count=max(int(rank_neg_k), int(rank_pool_factor) * max(1, int(rem_sel.numel()))),
        )
        rem_anchor_idx = _ordered_subset(
            rem_scores,
            rem_keep_idx,
            prefer_high=False,
            count=max(1, min(int(rem_keep_idx.numel()), rem_boundary_pool.numel())),
        )
        remove_rank_loss, rem_neg_count, remove_rank_pairs = _heart_like_margin_ranking_loss(
            rem_scores,
            rem_ii,
            rem_jj,
            rem_anchor_idx,
            rem_boundary_pool,
            num_neg_per_pos=rank_neg_k,
            margin=rank_margin,
        )
        add_neg_scores = add_scores.index_select(0, add_boundary_pool) if add_boundary_pool.numel() > 0 else add_scores.new_empty((0,))
        rem_keep_scores = rem_scores.index_select(0, rem_anchor_idx) if rem_anchor_idx.numel() > 0 else rem_scores.new_empty((0,))
        rem_neg_scores = rem_scores.index_select(0, rem_boundary_pool) if rem_boundary_pool.numel() > 0 else rem_scores.new_empty((0,))
    else:
        if add_neg_scores.numel() > 0:
            add_neg_scores = add_neg_scores[torch.argsort(add_neg_scores, descending=False)]
        add_rank_loss = _paired_margin_ranking_loss(
            add_scores[add_sel],
            add_neg_scores,
            rank_margin,
        )

        if rem_keep_scores.numel() > 0:
            rem_keep_scores = rem_keep_scores[torch.argsort(rem_keep_scores, descending=True)]
        rem_neg_scores = rem_scores.index_select(0, rem_sel) if rem_sel.numel() > 0 else rem_scores.new_empty((0,))
        remove_rank_loss = _paired_margin_ranking_loss(
            rem_keep_scores,
            rem_neg_scores,
            rank_margin,
        )

    total = (
        float(keep_weight) * keep_loss
        + float(add_rank_weight) * add_rank_loss
        + float(remove_rank_weight) * remove_rank_loss
    )
    debug_info = _empty_decoder_debug_info()
    debug_info.update(
        {
            "rewrite_nodes": int(rewrite_mask.sum().item()) if rewrite_mask is not None else 0,
            "valid_pairs": int(add_scores.numel() + rem_scores.numel()),
            "add_pairs": int(add_scores.numel()),
            "add_pairs_pre_struct": int(ctx["add_pairs_pre_struct"].triu(1).sum().item()),
            "struct_supported_add_pairs": int((ctx["add_pairs_pre_struct"] & ctx["structural_support"]).triu(1).sum().item()),
            "add_budget": int(add_budget),
            "add_selected": int(add_sel.numel()),
            "add_negatives": int(add_neg_scores.numel()),
            "add_rank_pairs": int(add_rank_pairs),
            "removable_pairs": int(rem_scores.numel()),
            "remove_budget": int(rem_budget),
            "remove_selected": int(rem_sel.numel()),
            "remove_kept": int(rem_keep_scores.numel()),
            "remove_rank_pairs": int(remove_rank_pairs),
        }
    )
    return total, keep_loss, add_rank_loss, remove_rank_loss, debug_info


def prototype_compactness_loss(
    Z_edit: torch.Tensor,
    labels: np.ndarray | None,
    core_mask: torch.Tensor | None,
    *,
    temperature: float = 0.2,
    min_cluster_size: int = 2,
) -> torch.Tensor:
    if labels is None or core_mask is None:
        return Z_edit.new_tensor(0.0)
    labels_np = np.asarray(labels)
    core_mask = core_mask.to(Z_edit.device).bool()

    prototypes = []
    supervised_idx = []
    targets = []
    for cluster_pos, cluster_id in enumerate(sorted(set(labels_np.tolist()) - {-1})):
        idx_np = np.where(labels_np == cluster_id)[0]
        if idx_np.size == 0:
            continue
        idx_t = torch.as_tensor(idx_np, device=Z_edit.device, dtype=torch.long)
        sel_mask = core_mask.index_select(0, idx_t)
        if int(sel_mask.sum().item()) < int(min_cluster_size):
            continue
        sel_idx = idx_t[sel_mask]
        X = F.normalize(Z_edit.index_select(0, sel_idx), p=2, dim=1)
        proto = F.normalize(X.mean(dim=0, keepdim=True), p=2, dim=1)
        target_id = len(prototypes)
        prototypes.append(proto.squeeze(0))
        supervised_idx.append(sel_idx)
        targets.append(torch.full((sel_idx.numel(),), target_id, dtype=torch.long, device=Z_edit.device))

    if len(prototypes) <= 1:
        return Z_edit.new_tensor(0.0)

    proto_mat = torch.stack(prototypes, dim=0)
    all_idx = torch.cat(supervised_idx, dim=0)
    all_targets = torch.cat(targets, dim=0)
    X_sup = F.normalize(Z_edit.index_select(0, all_idx), p=2, dim=1)
    logits = (X_sup @ proto_mat.t()) / float(max(temperature, 1e-6))
    return F.cross_entropy(logits, all_targets)


def compactness_objective_loss(
    Z_edit: torch.Tensor,
    labels: np.ndarray | None,
    radius_mask: torch.Tensor | None,
    core_mask: torch.Tensor | None,
    *,
    compactness_objective: str,
    radius_metric: str = "cosine",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radius_loss = cluster_compactness_loss(Z_edit, labels, radius_mask, radius_metric=radius_metric)
    proto_loss = prototype_compactness_loss(Z_edit, labels, core_mask)
    if compactness_objective == "radius":
        total = radius_loss
    elif compactness_objective == "prototype":
        total = proto_loss
    elif compactness_objective == "hybrid":
        active = []
        if torch.isfinite(radius_loss):
            active.append(radius_loss)
        if torch.isfinite(proto_loss) and proto_loss.detach().abs().item() > 0:
            active.append(proto_loss)
        total = torch.stack(active).mean() if active else Z_edit.new_tensor(0.0)
    else:
        raise ValueError(f"Unsupported compactness_objective={compactness_objective}")
    return total, radius_loss, proto_loss


def dynamic_c0p_targets(
    Z_ref: torch.Tensor,
    adj_dense: torch.Tensor,
    gmm_k: int,
    gmm_tau: float,
    restrict_alpha: float,
    restrict_gamma: float,
) -> tuple[np.ndarray, torch.Tensor]:
    labels_epoch = gmm_labels(Z_ref.detach(), K=gmm_k, tau=gmm_tau, metric="cosine")
    deg_epoch = _deg_excl_self(adj_dense)
    c0p_mask_epoch, _, _, _ = select_gmm_cores(
        Z_ref.detach(),
        labels_epoch,
        degrees_excl_self=deg_epoch,
        alpha=restrict_alpha,
        gamma=restrict_gamma,
        B=Z_ref.size(1),
    )
    return labels_epoch, c0p_mask_epoch.bool().detach()


def _empty_pull_push_diagnostics(
    *,
    push_scope: str = "none",
    noncompact_push_strength: float = 0.0,
    noise_push_strength: float = 0.0,
    preserve_norm: bool = True,
) -> dict[str, float]:
    return {
        "editor_noncompact_push_strength": float(noncompact_push_strength),
        "editor_noise_push_strength": float(noise_push_strength),
        "editor_push_preserve_norm": float(int(bool(preserve_norm))),
        "push_noncompact_count": 0.0,
        "push_noise_count": 0.0,
        "push_noncompact_anchor_cosdist_before": float("nan"),
        "push_noncompact_anchor_cosdist_after": float("nan"),
        "push_noise_anchor_cosdist_before": float("nan"),
        "push_noise_anchor_cosdist_after": float("nan"),
    }


def _mean_cosine_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    if X.numel() == 0 or Y.numel() == 0:
        return float("nan")
    sim = F.cosine_similarity(X, Y, dim=1, eps=1e-12)
    return float((1.0 - sim).mean().detach().cpu())


def direct_pull_latent_per_cluster(
    Z: torch.Tensor,
    labels: np.ndarray | None,
    node_mask: torch.Tensor | None,
    pull_strength: float = 0.20,
    *,
    c0p_mask: torch.Tensor | None = None,
    pull_profile: str = "linear",
    pull_tau: float = 0.25,
    pull_deadzone: float = 0.0,
    pull_anchor: str = "selected",
    push_scope: str = "none",
    noncompact_push_strength: float = 0.0,
    noise_push_strength: float = 0.0,
    push_preserve_norm: bool = True,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    pull_profile_eff = str(pull_profile or "linear").lower()
    if pull_profile_eff not in {"linear", "log_distance"}:
        pull_profile_eff = "linear"
    pull_anchor_eff = str(pull_anchor or "selected").lower()
    if pull_anchor_eff not in {"selected", "core"}:
        pull_anchor_eff = "selected"
    pull_tau_eff = max(1e-6, float(pull_tau))
    pull_deadzone_eff = max(0.0, float(pull_deadzone))
    push_scope_eff = str(push_scope or "none").lower()
    if push_scope_eff not in {"none", "noncompact_cp", "noise", "noncompact_cp_and_noise"}:
        push_scope_eff = "none"
    diag = _empty_pull_push_diagnostics(
        push_scope=push_scope_eff,
        noncompact_push_strength=noncompact_push_strength,
        noise_push_strength=noise_push_strength,
        preserve_norm=push_preserve_norm,
    )
    if labels is None:
        return (Z, diag) if return_diagnostics else Z
    labels_np = np.asarray(labels, dtype=np.int64)
    Z_edit = Z.clone()
    if node_mask is None:
        node_mask = torch.ones(Z.size(0), dtype=torch.bool, device=Z.device)
    else:
        node_mask = node_mask.to(Z.device).bool()
    c0p_mask_t = None if c0p_mask is None else c0p_mask.to(Z.device).bool()

    for c in sorted(set(labels_np.tolist()) - {-1}):
        idx_np = np.where(labels_np == c)[0]
        if idx_np.size == 0:
            continue
        idx_t = torch.as_tensor(idx_np, device=Z.device, dtype=torch.long)
        sel_mask = node_mask.index_select(0, idx_t)
        if int(sel_mask.sum().item()) <= 1:
            continue
        sel_idx = idx_t[sel_mask]
        sel_Z = Z.index_select(0, sel_idx)
        anchor_idx = sel_idx
        if pull_anchor_eff == "core" and c0p_mask_t is not None:
            core_mask = c0p_mask_t.index_select(0, idx_t)
            if int(core_mask.sum().item()) > 0:
                anchor_idx = idx_t[core_mask]
        center = Z.index_select(0, anchor_idx).mean(dim=0, keepdim=True)
        if pull_profile_eff == "log_distance":
            center_expanded = center.expand_as(sel_Z)
            dist = (1.0 - F.cosine_similarity(sel_Z, center_expanded, dim=1, eps=1e-12)).clamp_min(0.0)
            dist_eff = (dist - pull_deadzone_eff).clamp_min(0.0)
            denom = math.log1p(1.0 / pull_tau_eff)
            weights = torch.log1p(dist_eff / pull_tau_eff) / max(denom, 1e-12)
            weights = float(pull_strength) * weights.clamp(0.0, 1.0)
            Z_edit[sel_idx] = sel_Z + weights.view(-1, 1) * (center - sel_Z)
        else:
            Z_edit[sel_idx] = sel_Z + pull_strength * (center - sel_Z)

    if push_scope_eff == "none" or (
        float(noncompact_push_strength) <= 0.0 and float(noise_push_strength) <= 0.0
    ):
        return (Z_edit, diag) if return_diagnostics else Z_edit

    device = Z.device
    lbl_t = torch.as_tensor(labels_np, device=device, dtype=torch.long)
    non_noise = lbl_t != -1
    if c0p_mask is None:
        c0p_mask_t = torch.zeros(Z.size(0), dtype=torch.bool, device=device)
    else:
        c0p_mask_t = c0p_mask.to(device).bool() & non_noise

    cluster_ids = sorted(set(labels_np.tolist()) - {-1})
    anchor_by_cluster: dict[int, torch.Tensor] = {}
    anchor_rows = []
    for c in cluster_ids:
        cluster_mask = lbl_t == int(c)
        cluster_idx = cluster_mask.nonzero(as_tuple=True)[0]
        if cluster_idx.numel() == 0:
            continue
        c0p_idx = (cluster_mask & c0p_mask_t).nonzero(as_tuple=True)[0]
        anchor_idx = c0p_idx if c0p_idx.numel() > 0 else cluster_idx
        anchor = Z_edit.index_select(0, anchor_idx).mean(dim=0)
        anchor_by_cluster[int(c)] = anchor
        anchor_rows.append(anchor)

    if not anchor_rows:
        return (Z_edit, diag) if return_diagnostics else Z_edit

    def _apply_push(sel_idx: torch.Tensor, anchors: torch.Tensor, strength: float) -> tuple[float, float]:
        if sel_idx.numel() == 0 or float(strength) <= 0.0:
            return float("nan"), float("nan")
        before = Z_edit.index_select(0, sel_idx)
        before_dist = _mean_cosine_distance(before, anchors)
        pushed = before + float(strength) * (before - anchors)
        if push_preserve_norm:
            orig_norm = Z.index_select(0, sel_idx).norm(dim=1, keepdim=True).clamp_min(1e-12)
            pushed_norm = pushed.norm(dim=1, keepdim=True).clamp_min(1e-12)
            pushed = pushed * (orig_norm / pushed_norm)
        Z_edit[sel_idx] = pushed
        after_dist = _mean_cosine_distance(Z_edit.index_select(0, sel_idx), anchors)
        return before_dist, after_dist

    if push_scope_eff in {"noncompact_cp", "noncompact_cp_and_noise"} and float(noncompact_push_strength) > 0.0:
        noncompact_mask = non_noise & (~c0p_mask_t)
        sel_parts = []
        anchor_parts = []
        for c, anchor in anchor_by_cluster.items():
            sel_idx = (noncompact_mask & (lbl_t == int(c))).nonzero(as_tuple=True)[0]
            if sel_idx.numel() == 0:
                continue
            sel_parts.append(sel_idx)
            anchor_parts.append(anchor.view(1, -1).expand(sel_idx.numel(), -1))
        if sel_parts:
            sel_idx_all = torch.cat(sel_parts, dim=0)
            anchors_all = torch.cat(anchor_parts, dim=0)
            before, after = _apply_push(sel_idx_all, anchors_all, float(noncompact_push_strength))
            diag["push_noncompact_count"] = float(sel_idx_all.numel())
            diag["push_noncompact_anchor_cosdist_before"] = before
            diag["push_noncompact_anchor_cosdist_after"] = after

    if push_scope_eff in {"noise", "noncompact_cp_and_noise"} and float(noise_push_strength) > 0.0:
        noise_idx = (lbl_t == -1).nonzero(as_tuple=True)[0]
        if noise_idx.numel() > 0:
            anchor_mat = torch.stack(anchor_rows, dim=0)
            sims = F.normalize(Z_edit.index_select(0, noise_idx), p=2, dim=1) @ F.normalize(anchor_mat, p=2, dim=1).t()
            nearest = sims.argmax(dim=1)
            anchors = anchor_mat.index_select(0, nearest)
            before, after = _apply_push(noise_idx, anchors, float(noise_push_strength))
            diag["push_noise_count"] = float(noise_idx.numel())
            diag["push_noise_anchor_cosdist_before"] = before
            diag["push_noise_anchor_cosdist_after"] = after

    return (Z_edit, diag) if return_diagnostics else Z_edit


def cluster_compactness_loss(
    Z_edit: torch.Tensor,
    labels: np.ndarray | None,
    node_mask: torch.Tensor | None,
    *,
    radius_metric: str = "cosine",
) -> torch.Tensor:
    values = cluster_radius_values(Z_edit, labels, node_mask, radius_metric=radius_metric)
    if values.numel() == 0:
        return Z_edit.new_tensor(0.0)
    return values.mean()


def cluster_radius_values(
    Z_edit: torch.Tensor,
    labels: np.ndarray | None,
    node_mask: torch.Tensor | None,
    *,
    radius_metric: str = "cosine",
) -> torch.Tensor:
    if labels is None:
        return Z_edit.new_empty((0,))
    labels_np = np.asarray(labels)
    if node_mask is None:
        node_mask = torch.ones(Z_edit.size(0), dtype=torch.bool, device=Z_edit.device)
    else:
        node_mask = node_mask.to(Z_edit.device).bool()

    metric = str(radius_metric or "cosine").lower()
    if metric not in {"cosine", "mahalanobis"}:
        raise ValueError(f"Unsupported compactness radius metric: {radius_metric}")

    X_cos = F.normalize(Z_edit, p=2, dim=1)
    values = []
    for c in sorted(set(labels_np.tolist()) - {-1}):
        idx_np = np.where(labels_np == c)[0]
        if idx_np.size == 0:
            continue
        idx_t = torch.as_tensor(idx_np, device=Z_edit.device, dtype=torch.long)
        sel_mask = node_mask.index_select(0, idx_t)
        sel_count = int(sel_mask.sum().item())
        if sel_count == 0 or (metric == "cosine" and sel_count <= 1):
            continue
        sel_idx = idx_t[sel_mask]
        if metric == "cosine":
            X = X_cos.index_select(0, sel_idx)
            center = F.normalize(X.mean(dim=0, keepdim=True), p=2, dim=1)
            values.append(1.0 - (X @ center.t()).squeeze(1))
        else:
            Xc = X_cos.index_select(0, idx_t)
            mu = Xc.mean(dim=0, keepdim=True)
            var = Xc.var(dim=0, unbiased=False, keepdim=True) + 1e-6
            X_sel = X_cos.index_select(0, sel_idx)
            values.append(torch.sqrt(((X_sel - mu) ** 2 / var).sum(dim=1)))
    if not values:
        return Z_edit.new_empty((0,))
    return torch.cat(values, dim=0)


def cluster_radius_summary(
    Z_edit: torch.Tensor,
    labels: np.ndarray | None,
    node_mask: torch.Tensor | None,
    *,
    radius_metric: str = "cosine",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = cluster_radius_values(Z_edit, labels, node_mask, radius_metric=radius_metric)
    if values.numel() == 0:
        zero = Z_edit.new_tensor(0.0)
        return zero, zero, zero
    mean = values.mean()
    p90 = torch.quantile(values, 0.90) if values.numel() > 1 else values[0]
    max_val = values.max()
    return mean, p90, max_val


def noncompact_node_mask(cp_mask: torch.Tensor | None, c0p_mask: torch.Tensor | None) -> torch.Tensor | None:
    if cp_mask is None or c0p_mask is None:
        return None
    return cp_mask.to(c0p_mask.device).bool() & (~c0p_mask.bool())


def _cluster_intra_degree(adj_dense: torch.Tensor, labels: np.ndarray | None) -> torch.Tensor:
    """Degree induced by each node's assigned non-noise cluster, excluding self-loops."""
    if labels is None:
        return _deg_excl_self(adj_dense)
    device = adj_dense.device
    labels_np = np.asarray(labels, dtype=np.int64)
    lbl_t = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
    non_noise = lbl_t != -1
    same_cluster = (lbl_t[:, None] == lbl_t[None, :]) & non_noise[:, None] & non_noise[None, :]
    g = ((adj_dense > 0) | (adj_dense.t() > 0)).to(torch.float32)
    g.fill_diagonal_(0.0)
    return (g * same_cluster.to(g.dtype)).sum(dim=1).to(torch.long)


def _cluster_min_degree_summary(
    adj_dense: torch.Tensor,
    labels: np.ndarray | None,
    node_mask: torch.Tensor | None,
    *,
    target: int = -1,
) -> dict[str, float]:
    """Summarize per-cluster minimum intra-cluster degree for the selected nodes."""
    empty = {
        "clusters": 0,
        "worst_min": float("nan"),
        "mean_min": float("nan"),
        "need_nodes": 0,
        "bad_clusters": 0,
    }
    if labels is None:
        return empty
    device = adj_dense.device
    labels_np = np.asarray(labels, dtype=np.int64)
    lbl_t = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
    non_noise = lbl_t != -1
    scope_mask = non_noise if node_mask is None else (node_mask.to(device).bool() & non_noise)
    if not bool(scope_mask.any().item()):
        return empty

    intra_deg = _cluster_intra_degree(adj_dense, labels)
    cluster_mins: list[float] = []
    need_nodes = 0
    bad_clusters = 0
    target_eff = int(target)
    for cluster_id in torch.unique(lbl_t[scope_mask]).detach().cpu().tolist():
        idx = scope_mask & (lbl_t == int(cluster_id))
        if not bool(idx.any().item()):
            continue
        vals = intra_deg[idx]
        min_val = float(vals.min().item())
        cluster_mins.append(min_val)
        if target_eff > 0:
            below = vals < target_eff
            need_nodes += int(below.sum().item())
            bad_clusters += int(bool(below.any().item()))
    if not cluster_mins:
        return empty
    return {
        "clusters": len(cluster_mins),
        "worst_min": float(min(cluster_mins)),
        "mean_min": float(np.mean(cluster_mins)),
        "need_nodes": need_nodes,
        "bad_clusters": bad_clusters,
    }


def _decoded_add_degree_target_mask(
    ctx: dict[str, torch.Tensor],
    target_deg0: torch.Tensor,
    target: int,
    mode: str,
) -> tuple[torch.Tensor, str]:
    """Choose which nodes the degree-target add pass should try to repair."""
    mode_eff = str(mode or "rewrite").lower()
    allowed = {"rewrite", "cp", "cluster_deficit", "rewrite_or_cluster_deficit"}
    if mode_eff not in allowed:
        mode_eff = "rewrite"

    rewrite_mask = ctx["core_mask"].bool()
    cp_mask = ctx["non_noise"].bool()
    deficit_mask = cp_mask & (target_deg0 < int(target))
    if mode_eff == "rewrite":
        target_mask = rewrite_mask
    elif mode_eff == "cp":
        target_mask = cp_mask
    elif mode_eff == "cluster_deficit":
        target_mask = deficit_mask
    else:
        target_mask = rewrite_mask | deficit_mask
    return target_mask.bool(), mode_eff


def _graph_diff_summary(
    base_dense: torch.Tensor,
    view_dense: torch.Tensor,
    labels: np.ndarray | None = None,
    node_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Undirected edge-set difference between two graph views, excluding self-loops."""
    device = view_dense.device
    n_nodes = view_dense.size(0)
    upper = torch.triu(torch.ones((n_nodes, n_nodes), dtype=torch.bool, device=device), diagonal=1)
    base = ((base_dense.to(device) > 0) | (base_dense.to(device).t() > 0)) & upper
    view = ((view_dense > 0) | (view_dense.t() > 0)) & upper
    added = view & (~base)
    removed = base & (~view)
    intersection = base & view
    union = base | view

    base_edges = int(base.sum().item())
    view_edges = int(view.sum().item())
    added_edges = int(added.sum().item())
    removed_edges = int(removed.sum().item())
    symdiff_edges = added_edges + removed_edges
    union_edges = int(union.sum().item())
    intersection_edges = int(intersection.sum().item())

    same_cluster_added = float("nan")
    same_cluster_removed = float("nan")
    cross_cluster_added = float("nan")
    cross_cluster_removed = float("nan")
    target_touch_added = float("nan")
    target_touch_removed = float("nan")
    if labels is not None:
        labels_np = np.asarray(labels, dtype=np.int64)
        lbl_t = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
        non_noise = lbl_t != -1
        same_cluster = (lbl_t[:, None] == lbl_t[None, :]) & non_noise[:, None] & non_noise[None, :]
        same_cluster = same_cluster & upper
        cp_pair = non_noise[:, None] & non_noise[None, :] & upper
        same_cluster_added = int((added & same_cluster).sum().item())
        same_cluster_removed = int((removed & same_cluster).sum().item())
        cross_cluster_added = int((added & cp_pair & (~same_cluster)).sum().item())
        cross_cluster_removed = int((removed & cp_pair & (~same_cluster)).sum().item())
    if node_mask is not None:
        target = node_mask.to(device).bool()
        target_touch = (target[:, None] | target[None, :]) & upper
        target_touch_added = int((added & target_touch).sum().item())
        target_touch_removed = int((removed & target_touch).sum().item())

    return {
        "base_edges": base_edges,
        "view_edges": view_edges,
        "added_vs_base": added_edges,
        "removed_vs_base": removed_edges,
        "symdiff_edges": symdiff_edges,
        "edge_jaccard": intersection_edges / float(max(1, union_edges)),
        "diff_frac_base": symdiff_edges / float(max(1, base_edges)),
        "add_frac_base": added_edges / float(max(1, base_edges)),
        "remove_frac_base": removed_edges / float(max(1, base_edges)),
        "same_cluster_added": same_cluster_added,
        "same_cluster_removed": same_cluster_removed,
        "cross_cluster_added": cross_cluster_added,
        "cross_cluster_removed": cross_cluster_removed,
        "target_touch_added": target_touch_added,
        "target_touch_removed": target_touch_removed,
    }


def non_target_preservation_loss(Z_edit: torch.Tensor, Z_base: torch.Tensor, node_mask: torch.Tensor | None) -> torch.Tensor:
    if node_mask is None:
        return Z_edit.new_tensor(0.0)
    keep_mask = (~node_mask.to(Z_edit.device).bool())
    if keep_mask.sum().item() == 0:
        return Z_edit.new_tensor(0.0)
    return F.mse_loss(Z_edit[keep_mask], Z_base.detach()[keep_mask])


def resolve_edit_targets(
    Z_ref: torch.Tensor,
    adj_dense: torch.Tensor,
    *,
    freeze_targets: bool,
    fixed_labels: np.ndarray | None,
    fixed_mask: torch.Tensor | None,
    gmm_k: int,
    gmm_tau: float,
    restrict_alpha: float,
    restrict_gamma: float,
) -> tuple[np.ndarray | None, torch.Tensor | None]:
    """Return either frozen edit targets or per-call dynamic targets."""
    if freeze_targets and fixed_labels is not None and fixed_mask is not None:
        return fixed_labels, fixed_mask.to(Z_ref.device).bool()
    return dynamic_c0p_targets(
        Z_ref.detach(),
        adj_dense,
        gmm_k=gmm_k,
        gmm_tau=gmm_tau,
        restrict_alpha=restrict_alpha,
        restrict_gamma=restrict_gamma,
    )


@torch.no_grad()
def build_decoded_augmented_graph(
    decoded_scores: torch.Tensor,
    adj_current_dense: torch.Tensor,
    labels: np.ndarray | None,
    node_mask: torch.Tensor | None,
    *,
    E0: int,
    add_ratio: float = 0.0,
    remove_ratio: float = 0.0,
    add_threshold: float | None = None,
    remove_threshold: float | None = None,
    add_quantile: float | None = None,
    remove_quantile: float | None = None,
    max_add: int | None = None,
    max_remove: int | None = None,
    per_node_cap_frac: float = 0.10,
    add_degree_target: int | None = None,
    add_degree_target_scope: str = "total",
    add_degree_target_nodes: str = "rewrite",
    guarantee_degree_target: bool = False,
    deg0_excl_self: torch.Tensor | None = None,
    degree_floor: int = 0,
    same_cluster_only: bool = True,
    require_c0p_endpoint: bool = True,
    require_both_c0p: bool = False,
    require_c0p_noncompact_endpoint: bool = False,
    require_structural_support: bool = False,
    structural_support_mode: str = "cn_or_ra",
    structural_min_cn: float = 1.0,
    structural_min_ra: float = 0.0,
    structural_min_aa: float = 0.0,
    structural_support_mask: torch.Tensor | None = None,
    pair_context: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, int, int]:
    """
    Rewrite the current graph using decoder scores from the pulled latent.

    Selection order:
      1) explicit score threshold, if provided
      2) score quantile threshold, if provided
      3) ratio-based budget fallback

    max_add/max_remove act as hard per-round safety caps on top of any threshold mode.
    """
    device = adj_current_dense.device
    N = adj_current_dense.size(0)

    score = 0.5 * (decoded_scores + decoded_scores.t())
    score = score.clone()
    score.fill_diagonal_(1.0)
    if pair_context is None:
        ctx = _build_decoded_pair_context(
            adj_current_dense,
            labels,
            node_mask,
            degree_floor=degree_floor,
            same_cluster_only=same_cluster_only,
            require_c0p_endpoint=require_c0p_endpoint,
            require_both_c0p=require_both_c0p,
            require_c0p_noncompact_endpoint=require_c0p_noncompact_endpoint,
            deg0_excl_self=deg0_excl_self,
            require_structural_support=require_structural_support,
            structural_support_mode=structural_support_mode,
            structural_min_cn=structural_min_cn,
            structural_min_ra=structural_min_ra,
            structural_min_aa=structural_min_aa,
            structural_support_mask=structural_support_mask,
        )
    else:
        ctx = pair_context
    g = ctx["graph_dense"].clone().to(score.dtype)
    eye = ctx["eye"]
    existing = ctx["existing"]
    valid_pairs = ctx["valid_pairs"]
    same_cluster = ctx["same_cluster"]
    deg0 = ctx["deg0"]
    deg_now = deg0.clone().to(torch.long)
    if per_node_cap_frac is None or float(per_node_cap_frac) <= 0:
        node_caps = torch.full((N,), N, dtype=torch.long, device=device)
    else:
        base_deg = torch.clamp(deg0, min=1)
        node_caps = torch.ceil(float(per_node_cap_frac) * base_deg.float()).to(torch.long)
        node_caps = torch.clamp(node_caps, min=1)
    add_degree_target_eff = -1 if add_degree_target is None else int(add_degree_target)
    guarantee_degree_target_eff = bool(guarantee_degree_target)
    add_degree_target_scope_eff = str(add_degree_target_scope or "total").lower()
    if add_degree_target_scope_eff not in {"total", "intra_cluster"}:
        add_degree_target_scope_eff = "total"
    target_deg0 = deg0
    if add_degree_target_scope_eff == "intra_cluster":
        target_deg0 = _cluster_intra_degree(g, labels).to(device=device, dtype=torch.long)
    target_deg_now = target_deg0.clone().to(torch.long)
    add_degree_target_mask, add_degree_target_nodes_eff = _decoded_add_degree_target_mask(
        ctx,
        target_deg0,
        add_degree_target_eff,
        add_degree_target_nodes,
    )
    add_degree_target_enabled = add_degree_target_eff > 0 and bool(add_degree_target_mask.any().item())
    if add_degree_target_enabled:
        target_deficit0 = torch.clamp(
            torch.full_like(target_deg0, add_degree_target_eff) - target_deg0,
            min=0,
        )
        target_deficit0 = torch.where(
            add_degree_target_mask,
            target_deficit0,
            torch.zeros_like(target_deficit0),
        )
        node_caps = torch.maximum(node_caps, target_deficit0)
    node_add_used = torch.zeros(N, dtype=torch.long, device=device)

    add_budget = max(0, int(round(float(add_ratio) * float(max(1, E0)))))
    remove_budget = max(0, int(round(float(remove_ratio) * float(max(1, E0)))))
    if max_add is not None:
        add_budget = min(add_budget if add_budget > 0 else int(max_add), int(max_add))
    if max_remove is not None:
        remove_budget = min(remove_budget if remove_budget > 0 else int(max_remove), int(max_remove))

    added = 0
    removed = 0

    cand_add = ctx["add_pairs"].clone()
    ii, jj = cand_add.triu(1).nonzero(as_tuple=True)
    if ii.numel() > 0:
        add_scores = score[ii, jj]
        if add_threshold is not None:
            keep = add_scores >= float(add_threshold)
            ii, jj, add_scores = ii[keep], jj[keep], add_scores[keep]
        elif add_quantile is not None and 0.0 < float(add_quantile) < 1.0 and add_scores.numel() > 0:
            q = max(0.0, min(1.0, 1.0 - float(add_quantile)))
            thr = torch.quantile(add_scores, q)
            keep = add_scores >= thr
            ii, jj, add_scores = ii[keep], jj[keep], add_scores[keep]
    if ii.numel() > 0:
        score_order = torch.argsort(add_scores, descending=True)
        limit = len(score_order)
        if add_threshold is None and add_quantile is None:
            limit = min(limit, add_budget)
        if max_add is not None:
            limit = min(limit, int(max_add))
        if add_degree_target_enabled:
            pair_deficit = torch.maximum(target_deficit0.index_select(0, ii), target_deficit0.index_select(0, jj))
            pair_deficit_sum = target_deficit0.index_select(0, ii) + target_deficit0.index_select(0, jj)
            degree_key = pair_deficit.float() * 1_000_000.0 + pair_deficit_sum.float() * 1_000.0 + add_scores
            degree_order = torch.argsort(degree_key, descending=True)
        else:
            degree_order = score_order.new_empty((0,))

        repair_added = 0
        repair_limit = limit

        def _try_add_candidates(
            order_idx: torch.Tensor,
            require_active_deficit: bool,
            *,
            limit_eff: int,
            enforce_node_caps: bool,
        ) -> None:
            nonlocal added
            for idx in order_idx.tolist():
                if added >= limit_eff:
                    break
                i = int(ii[idx].item())
                j = int(jj[idx].item())
                if require_active_deficit:
                    if add_degree_target_scope_eff == "intra_cluster" and not bool(same_cluster[i, j].item()):
                        continue
                    i_needs = bool(add_degree_target_mask[i].item()) and int(target_deg_now[i].item()) < add_degree_target_eff
                    j_needs = bool(add_degree_target_mask[j].item()) and int(target_deg_now[j].item()) < add_degree_target_eff
                    if not (i_needs or j_needs):
                        continue
                if enforce_node_caps and (node_add_used[i] >= node_caps[i] or node_add_used[j] >= node_caps[j]):
                    continue
                if g[i, j] > 0:
                    continue
                g[i, j] = 1.0
                g[j, i] = 1.0
                deg_now[i] += 1
                deg_now[j] += 1
                if add_degree_target_scope_eff == "total" or bool(same_cluster[i, j].item()):
                    target_deg_now[i] += 1
                    target_deg_now[j] += 1
                node_add_used[i] += 1
                node_add_used[j] += 1
                added += 1

        if add_degree_target_enabled:
            repair_start = int(added)
            repair_limit = len(degree_order) if guarantee_degree_target_eff else limit
            if max_add is not None:
                repair_limit = min(repair_limit, int(max_add))
            _try_add_candidates(
                degree_order,
                require_active_deficit=True,
                limit_eff=repair_limit,
                enforce_node_caps=not guarantee_degree_target_eff,
            )
            repair_added = int(added) - repair_start
        if added < limit:
            _try_add_candidates(
                score_order,
                require_active_deficit=False,
                limit_eff=limit,
                enforce_node_caps=True,
            )

        if add_degree_target_enabled:
            before_need = int(((target_deficit0 > 0) & add_degree_target_mask).sum().item())
            after_need = int(((target_deg_now < add_degree_target_eff) & add_degree_target_mask).sum().item())
            min_target_deg = int(target_deg_now[add_degree_target_mask].min().item()) if bool(add_degree_target_mask.any().item()) else -1
            target_mask_nodes = int(add_degree_target_mask.sum().item())
            cluster_before = _cluster_min_degree_summary(
                ctx["graph_dense"],
                labels,
                add_degree_target_mask,
                target=add_degree_target_eff,
            )
            cluster_after = _cluster_min_degree_summary(
                g,
                labels,
                add_degree_target_mask,
                target=add_degree_target_eff,
            )
            print(
                f"[DECODED-DEG] target={add_degree_target_eff} scope={add_degree_target_scope_eff} "
                f"target_nodes={add_degree_target_nodes_eff} target_mask_nodes={target_mask_nodes} "
                f"guarantee_target={int(guarantee_degree_target_eff)} repair_added={repair_added} "
                f"need_before={before_need} need_after={after_need} unrepaired={after_need} "
                f"min_target_deg={min_target_deg} add_limit={limit} repair_limit={repair_limit} "
                f"cluster_bad_before={int(cluster_before['bad_clusters'])} "
                f"cluster_bad_after={int(cluster_after['bad_clusters'])} "
                f"cluster_worst_min_before={cluster_before['worst_min']:.6f} "
                f"cluster_worst_min_after={cluster_after['worst_min']:.6f}"
            )

    cand_rem = ctx["removable_pairs"].clone()
    cand_rem.fill_diagonal_(False)
    ii, jj = cand_rem.triu(1).nonzero(as_tuple=True)
    if ii.numel() > 0:
        rem_scores = score[ii, jj]
        if remove_threshold is not None:
            keep = rem_scores <= float(remove_threshold)
            ii, jj, rem_scores = ii[keep], jj[keep], rem_scores[keep]
        elif remove_quantile is not None and 0.0 < float(remove_quantile) < 1.0 and rem_scores.numel() > 0:
            thr = torch.quantile(rem_scores, float(remove_quantile))
            keep = rem_scores <= thr
            ii, jj, rem_scores = ii[keep], jj[keep], rem_scores[keep]
    if ii.numel() > 0:
        order = torch.argsort(rem_scores, descending=False)
        limit = len(order)
        if remove_threshold is None and remove_quantile is None:
            limit = min(limit, remove_budget)
        if max_remove is not None:
            limit = min(limit, int(max_remove))
        for idx in order[:limit].tolist():
            i = int(ii[idx].item())
            j = int(jj[idx].item())
            if g[i, j] <= 0:
                continue
            if int(deg_now[i].item()) <= int(degree_floor) or int(deg_now[j].item()) <= int(degree_floor):
                continue
            g[i, j] = 0.0
            g[j, i] = 0.0
            deg_now[i] -= 1
            deg_now[j] -= 1
            removed += 1
            if max_remove is None and remove_threshold is None and remove_quantile is None and removed >= remove_budget:
                break
            if max_remove is not None and removed >= int(max_remove):
                break

    g.fill_diagonal_(1.0)
    return g, added, removed


def sample_constraint_preserving_two_view_graph(
    graph_dense: torch.Tensor,
    pair_context: dict[str, torch.Tensor] | None,
    *,
    E0: int,
    add_ratio: float,
    remove_ratio: float,
    degree_floor: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sample one stochastic CL graph view while preserving decoded rewrite constraints."""
    dense = _to_dense(graph_dense)
    g = ((dense > 0) | (dense.t() > 0)).to(torch.float32).clone()
    device = g.device
    n_nodes = g.size(0)
    eye = torch.eye(n_nodes, dtype=torch.bool, device=device)
    g.fill_diagonal_(1.0)

    e0_eff = max(1, int(E0))
    add_budget = max(0, int(round(float(add_ratio) * float(e0_eff))))
    remove_budget = max(0, int(round(float(remove_ratio) * float(e0_eff))))
    degree_floor_eff = max(0, int(degree_floor))
    deg_now = _deg_excl_self(g).to(torch.long)
    deg_start = deg_now.clone()

    if pair_context is None:
        existing = g > 0
        add_pairs = (~existing) & (~eye)
        removable_pairs = existing & (~eye)
    else:
        existing = g > 0
        add_pairs = pair_context.get("add_pairs", torch.zeros_like(g, dtype=torch.bool)).to(device=device).bool()
        removable_pairs = pair_context.get("removable_pairs", torch.zeros_like(g, dtype=torch.bool)).to(device=device).bool()
        add_pairs = add_pairs & (~existing) & (~eye)
        removable_pairs = removable_pairs & existing & (~eye)

    removed = 0
    rem_i, rem_j = removable_pairs.triu(1).nonzero(as_tuple=True)
    if remove_budget > 0 and rem_i.numel() > 0:
        order = torch.randperm(rem_i.numel(), device=device)
        for idx in order.tolist():
            if removed >= remove_budget:
                break
            i = int(rem_i[idx].item())
            j = int(rem_j[idx].item())
            if g[i, j] <= 0:
                continue
            if int(deg_now[i].item()) <= degree_floor_eff or int(deg_now[j].item()) <= degree_floor_eff:
                continue
            g[i, j] = 0.0
            g[j, i] = 0.0
            deg_now[i] -= 1
            deg_now[j] -= 1
            removed += 1

    added = 0
    add_i, add_j = add_pairs.triu(1).nonzero(as_tuple=True)
    if add_budget > 0 and add_i.numel() > 0:
        order = torch.randperm(add_i.numel(), device=device)
        for idx in order.tolist():
            if added >= add_budget:
                break
            i = int(add_i[idx].item())
            j = int(add_j[idx].item())
            if g[i, j] > 0:
                continue
            g[i, j] = 1.0
            g[j, i] = 1.0
            deg_now[i] += 1
            deg_now[j] += 1
            added += 1

    g.fill_diagonal_(1.0)
    deg_final = _deg_excl_self(g).to(torch.long)
    constraint_added = ((g > 0) & (~existing) & (~eye))
    constraint_add_violations = 0
    if pair_context is not None:
        constraint_add_violations = int((constraint_added & (~add_pairs)).triu(1).sum().item())
    degree_violations = 0
    if degree_floor_eff > 0:
        new_violation = (deg_start >= degree_floor_eff) & (deg_final < degree_floor_eff)
        degree_violations = int(new_violation.sum().item())
    edge_count = int((g.triu(1) > 0).sum().item())
    min_degree = int(deg_final.min().item()) if deg_final.numel() > 0 else 0
    return g, {
        "added": int(added),
        "removed": int(removed),
        "edge_count": int(edge_count),
        "min_degree": int(min_degree),
        "degree_violations": int(degree_violations),
        "constraint_add_violations": int(constraint_add_violations),
        "add_budget": int(add_budget),
        "remove_budget": int(remove_budget),
    }


def graph_edge_jaccard(g1: torch.Tensor, g2: torch.Tensor) -> float:
    a = (_to_dense(g1) > 0).triu(1)
    b = (_to_dense(g2) > 0).triu(1)
    union = int((a | b).sum().item())
    if union == 0:
        return 1.0
    inter = int((a & b).sum().item())
    return float(inter) / float(union)



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
    # c0p pruning (online removals that also consume the budget)
    c0p_prune_frac: float = 0.0,
    # === NEW: static pre-prune knobs (applied once before training) ===
    pre_prune_frac: float = 0.0,
    pre_prune_scope: str = "cp_all",
    # ===== NEW: online prune knobs (scope is encoded in `ver=prune_*`) =====
    prune_step_frac: float = 0.01,
    prune_max_frac:  float = 1.0,
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

    use_edited_decoder = bool(kwargs.get("use_edited_decoder", False))
    decoder_type = str(kwargs.get("decoder_type", "bilinear"))
    decoder_normalize_input = bool(kwargs.get("decoder_normalize_input", True))
    score_source = str(kwargs.get("score_source", "dot")).lower()
    if score_source not in {"dot", "decoder", "pred_decoder"}:
        raise ValueError(f"Unsupported score_source={score_source}; use dot, decoder, or pred_decoder.")
    decoder_objective = str(kwargs.get("decoder_objective", "hybrid"))
    decoder_recon_weight = float(kwargs.get("decoder_recon_weight", 1.0))
    decoder_keep_weight = float(kwargs.get("decoder_keep_weight", 1.0))
    decoder_add_rank_weight = float(kwargs.get("decoder_add_rank_weight", 1.0))
    decoder_remove_rank_weight = float(kwargs.get("decoder_remove_rank_weight", 1.0))
    decoder_rank_margin = float(kwargs.get("decoder_rank_margin", 0.2))
    decoder_rank_strategy = str(kwargs.get("decoder_rank_strategy", "easy"))
    decoder_rank_neg_k = int(kwargs.get("decoder_rank_neg_k", 8))
    decoder_rank_pool_factor = int(kwargs.get("decoder_rank_pool_factor", 4))
    heart_rank_weight = float(kwargs.get("heart_rank_weight", 0.0))
    heart_rank_margin = float(kwargs.get("heart_rank_margin", 0.2))
    heart_rank_neg_k = int(kwargs.get("heart_rank_neg_k", 8))
    heart_rank_pool_factor = int(kwargs.get("heart_rank_pool_factor", 4))
    prediction_decoder_type = str(kwargs.get("prediction_decoder_type", "none")).lower()
    prediction_rank_weight = float(kwargs.get("prediction_rank_weight", 1.0))
    prediction_bce_weight = float(kwargs.get("prediction_bce_weight", 0.1))
    prediction_rank_margin = float(kwargs.get("prediction_rank_margin", 0.2))
    prediction_rank_neg_k = int(kwargs.get("prediction_rank_neg_k", 16))
    prediction_rank_pool_factor = int(kwargs.get("prediction_rank_pool_factor", 8))
    prediction_rank_neg_strategy = str(kwargs.get("prediction_rank_neg_strategy", "random") or "random").lower()
    if prediction_rank_neg_strategy not in {"random", "struct"}:
        raise ValueError("prediction_rank_neg_strategy must be one of: random, struct")
    prediction_rank_struct_frac = float(kwargs.get("prediction_rank_struct_frac", 0.5))
    prediction_joint_start_epoch = int(kwargs.get("prediction_joint_start_epoch", -1))
    prediction_encoder_weight = float(kwargs.get("prediction_encoder_weight", 0.0))
    prediction_gate_l1_weight = float(kwargs.get("prediction_gate_l1_weight", 0.0))
    prediction_h3_gate_init = float(kwargs.get("prediction_h3_gate_init", -3.0))
    prediction_residual_gate_init = float(kwargs.get("prediction_residual_gate_init", -4.0))
    prediction_residual_scale = float(kwargs.get("prediction_residual_scale", 1.0))
    prediction_hard_residual_only = bool(kwargs.get("prediction_hard_residual_only", False))
    prediction_hard_margin = float(kwargs.get("prediction_hard_margin", 0.2))
    prediction_dot_anchor_weight = float(kwargs.get("prediction_dot_anchor_weight", 0.0))
    cl_mode = str(kwargs.get("cl_mode", "legacy") or "legacy").lower()
    if cl_mode not in {"legacy", "edit_two_aug"}:
        raise ValueError("cl_mode must be one of: legacy, edit_two_aug")
    prediction_graph = str(kwargs.get("prediction_graph", "train") or "train").lower()
    if prediction_graph not in {"train", "edit"}:
        raise ValueError("prediction_graph must be one of: train, edit")
    mlp_pair_max_rows = int(kwargs.get("mlp_pair_max_rows", 16))
    compactness_weight = float(kwargs.get("compactness_weight", 1.0))
    compactness_objective = str(kwargs.get("compactness_objective", "hybrid"))
    compactness_radius_metric = str(kwargs.get("compactness_radius_metric", "cosine")).lower()
    if compactness_radius_metric not in {"cosine", "mahalanobis"}:
        raise ValueError("compactness_radius_metric must be one of: cosine, mahalanobis")
    preserve_weight = float(kwargs.get("preserve_weight", 0.0))
    editor_hidden = int(kwargs.get("editor_hidden", hidden2 * 2))
    editor_pull_strength = float(kwargs.get("editor_pull_strength", 0.20))
    editor_pull_profile = str(kwargs.get("editor_pull_profile", "linear") or "linear").lower()
    if editor_pull_profile not in {"linear", "log_distance"}:
        raise ValueError("editor_pull_profile must be one of: linear, log_distance")
    editor_pull_tau = float(kwargs.get("editor_pull_tau", 0.25))
    editor_pull_deadzone = float(kwargs.get("editor_pull_deadzone", 0.0))
    editor_pull_anchor = str(kwargs.get("editor_pull_anchor", "selected") or "selected").lower()
    if editor_pull_anchor not in {"selected", "core"}:
        raise ValueError("editor_pull_anchor must be one of: selected, core")
    editor_push_scope = str(kwargs.get("editor_push_scope", "none") or "none").lower()
    if editor_push_scope not in {"none", "noncompact_cp", "noise", "noncompact_cp_and_noise"}:
        raise ValueError("editor_push_scope must be one of: none, noncompact_cp, noise, noncompact_cp_and_noise")
    editor_noncompact_push_strength = float(kwargs.get("editor_noncompact_push_strength", 0.0))
    editor_noise_push_strength = float(kwargs.get("editor_noise_push_strength", 0.0))
    editor_push_preserve_norm = bool(kwargs.get("editor_push_preserve_norm", True))
    editor_edit_scale = float(kwargs.get("editor_edit_scale", 0.10))
    edit_start_epoch = int(kwargs.get("edit_start_epoch", 0))
    edit_train_start_arg = int(kwargs.get("edit_train_start_epoch", -1))
    decoded_rewrite_start_arg = int(kwargs.get("decoded_rewrite_start_epoch", -1))
    edit_train_start_epoch = edit_start_epoch if edit_train_start_arg < 0 else edit_train_start_arg
    decoded_rewrite_start_epoch = edit_start_epoch if decoded_rewrite_start_arg < 0 else decoded_rewrite_start_arg
    decoded_rewrite_every = max(1, int(kwargs.get("decoded_rewrite_every", 1)))
    if prediction_joint_start_epoch < 0:
        prediction_joint_start_epoch = decoded_rewrite_start_epoch
    freeze_c0p_at_edit_start = bool(kwargs.get("freeze_c0p_at_edit_start", True))
    use_decoded_graph_augment = bool(kwargs.get("use_decoded_graph_augment", False))
    if (cl_mode == "edit_two_aug" or prediction_graph == "edit") and not use_decoded_graph_augment:
        raise ValueError("cl_mode=edit_two_aug and prediction_graph=edit require --use_decoded_graph_augment")
    decoded_add_ratio = float(kwargs.get("decoded_add_ratio", 0.0))
    decoded_remove_ratio = float(kwargs.get("decoded_remove_ratio", 0.0))
    decoded_add_threshold = kwargs.get("decoded_add_threshold", None)
    decoded_remove_threshold = kwargs.get("decoded_remove_threshold", None)
    decoded_add_quantile = kwargs.get("decoded_add_quantile", None)
    decoded_remove_quantile = kwargs.get("decoded_remove_quantile", None)
    decoded_max_add_per_round = kwargs.get("decoded_max_add_per_round", None)
    decoded_max_remove_per_round = kwargs.get("decoded_max_remove_per_round", None)
    decoded_same_cluster_only = bool(kwargs.get("decoded_same_cluster_only", False))
    decoded_require_c0p_endpoint = bool(kwargs.get("decoded_require_c0p_endpoint", False))
    decoded_require_both_c0p = bool(kwargs.get("decoded_require_both_c0p", False))
    decoded_require_c0p_noncompact_endpoint = bool(kwargs.get("decoded_require_c0p_noncompact_endpoint", False))
    decoded_require_structural_support = bool(kwargs.get("decoded_require_structural_support", False))
    decoded_struct_support = str(kwargs.get("decoded_struct_support", "cn_or_ra") or "cn_or_ra").lower()
    if decoded_struct_support not in {"cn", "ra", "aa", "cn_or_ra", "cn_or_aa", "ra_or_aa", "any", "all"}:
        raise ValueError("decoded_struct_support must be one of: cn, ra, aa, cn_or_ra, cn_or_aa, ra_or_aa, any, all")
    decoded_struct_min_cn = float(kwargs.get("decoded_struct_min_cn", 1.0))
    decoded_struct_min_ra = float(kwargs.get("decoded_struct_min_ra", 0.0))
    decoded_struct_min_aa = float(kwargs.get("decoded_struct_min_aa", 0.0))
    decoded_graph_aug_bound = kwargs.get("decoded_graph_aug_bound", None)
    decoded_add_degree_target = int(kwargs.get("decoded_add_degree_target", -1) or -1)
    decoded_add_degree_target_scope = str(kwargs.get("decoded_add_degree_target_scope", "total") or "total").lower()
    if decoded_add_degree_target_scope not in {"total", "intra_cluster"}:
        raise ValueError("decoded_add_degree_target_scope must be one of: total, intra_cluster")
    decoded_add_degree_target_nodes = str(kwargs.get("decoded_add_degree_target_nodes", "rewrite") or "rewrite").lower()
    if decoded_add_degree_target_nodes not in {"rewrite", "cp", "cluster_deficit", "rewrite_or_cluster_deficit"}:
        raise ValueError("decoded_add_degree_target_nodes must be one of: rewrite, cp, cluster_deficit, rewrite_or_cluster_deficit")
    decoded_guarantee_degree_target = bool(kwargs.get("decoded_guarantee_degree_target", False))
    decoded_degree_floor = kwargs.get("decoded_degree_floor", None)
    decoded_accumulate_into_base = bool(kwargs.get("decoded_accumulate_into_base", True))
    decoded_edit_end_epoch = kwargs.get("decoded_edit_end_epoch", -1)
    eval_log_every = int(kwargs.get("eval_log_every", 10))
    train_eval_every = max(1, int(kwargs.get("train_eval_every", 1)))
    skip_train_acc = bool(kwargs.get("skip_train_acc", False))
    decoder_diag_every = int(kwargs.get("decoder_diag_every", -1))
    edit_metric_every = int(kwargs.get("edit_metric_every", 1))
    decoded_audit_every = int(kwargs.get("decoded_audit_every", 0))
    decoded_audit_max_edges = max(0, int(kwargs.get("decoded_audit_max_edges", 4096)))
    edge_eval = bool(kwargs.get("edge_eval", True))
    separate_edit_training = bool(kwargs.get("separate_edit_training", False))
    edit_phase_retain_recon_weight = float(kwargs.get("edit_phase_retain_recon_weight", 0.0))
    edit_phase_retain_cl_weight = float(kwargs.get("edit_phase_retain_cl_weight", 0.0))
    phase2_freeze_encoder = bool(kwargs.get("phase2_freeze_encoder", True))
    edit_phase_encoder_lr_scale = float(kwargs.get("edit_phase_encoder_lr_scale", 0.0))
    phase2_task_main_loss = bool(kwargs.get("phase2_task_main_loss", False))
    edit_phase_edit_weight = float(kwargs.get("edit_phase_edit_weight", 0.10))
    phase2_decoder_inference_only = bool(kwargs.get("phase2_decoder_inference_only", True))
    decoder_warmup_in_phase1 = bool(kwargs.get("decoder_warmup_in_phase1", True))
    decoder_warmup_recon_weight = float(kwargs.get("decoder_warmup_recon_weight", 1.0))
    decoder_warmup_use_pulled_latent = bool(kwargs.get("decoder_warmup_use_pulled_latent", False))
    phase_cache_path = str(kwargs.get("phase_cache_path", "") or "")
    phase_cache_epoch_arg = int(kwargs.get("phase_cache_epoch", -1))
    phase_cache_load_mode = str(kwargs.get("phase_cache_load_mode", "full") or "full").lower()
    if phase_cache_load_mode not in {"full", "compatible", "encoder_only"}:
        raise ValueError("phase_cache_load_mode must be one of: full, compatible, encoder_only")
    skip_oom_epoch = bool(kwargs.get("skip_oom_epoch", False))
    pull_mask_scope = str(kwargs.get("pull_mask_scope", "cp"))
    compactness_mask_scope = str(kwargs.get("compactness_mask_scope", "cp"))
    rewrite_endpoint_scope = str(kwargs.get("rewrite_endpoint_scope", "c0p"))
    raw_ae_backbone = str(kwargs.get("ae_backbone", "vgnae")).lower()
    ae_backbone = "vgnae" if raw_ae_backbone == "vgae" else raw_ae_backbone
    ae_backbone = "cimage_full" if ae_backbone == "cimage" else ae_backbone
    if ae_backbone not in {"vgnae", "maskgae", "cimage_lite", "cimage_full"}:
        raise ValueError(f"Unsupported ae_backbone={raw_ae_backbone}; use vgnae, vgae, maskgae, cimage_lite, or cimage_full.")
    maskgae_mask_rate = float(kwargs.get("maskgae_mask_rate", 0.3))
    maskgae_feature_weight = float(kwargs.get("maskgae_feature_weight", 1.0))
    cimage_factor_weight = float(kwargs.get("cimage_factor_weight", 0.1))
    cimage_cluster_weight = float(kwargs.get("cimage_cluster_weight", 0.1))
    cimage_num_factors = int(kwargs.get("cimage_num_factors", 8))
    cimage_num_clusters = int(kwargs.get("cimage_num_clusters", 16))
    cimage_cluster_alpha = float(kwargs.get("cimage_cluster_alpha", 1.0))
    cimage_pseudo_label_threshold = float(kwargs.get("cimage_pseudo_label_threshold", 0.90))
    cimage_factor_select_ratio = float(kwargs.get("cimage_factor_select_ratio", 0.50))
    cimage_mrmr_redundancy_weight = float(kwargs.get("cimage_mrmr_redundancy_weight", 0.20))
    cimage_cluster_balance_weight = float(kwargs.get("cimage_cluster_balance_weight", 0.05))
    cimage_sce_power = float(kwargs.get("cimage_sce_power", 2.0))
    if maskgae_mask_rate < 0.0 or maskgae_mask_rate > 1.0:
        raise ValueError(f"maskgae_mask_rate must be in [0, 1], got {maskgae_mask_rate}")
    if maskgae_feature_weight < 0.0:
        raise ValueError(f"maskgae_feature_weight must be >= 0, got {maskgae_feature_weight}")
    if cimage_factor_weight < 0.0 or cimage_cluster_weight < 0.0:
        raise ValueError("cimage_factor_weight and cimage_cluster_weight must be >= 0")
    if cimage_num_factors <= 0 or cimage_num_clusters <= 0:
        raise ValueError("cimage_num_factors and cimage_num_clusters must be positive")
    if not (0.0 <= cimage_pseudo_label_threshold <= 1.0):
        raise ValueError("cimage_pseudo_label_threshold must be in [0, 1]")
    if not (0.0 < cimage_factor_select_ratio < 1.0):
        raise ValueError("cimage_factor_select_ratio must be in (0, 1)")
    if cimage_mrmr_redundancy_weight < 0.0 or cimage_cluster_balance_weight < 0.0:
        raise ValueError("cimage_mrmr_redundancy_weight and cimage_cluster_balance_weight must be >= 0")
    if cimage_sce_power <= 0.0:
        raise ValueError("cimage_sce_power must be positive")


    # ------------------------------------------------------------
    # Default HeaRT evaluation policy
    # No extra CLI flags needed.
    # ------------------------------------------------------------
    split_mode = str(kwargs.get("split_mode", "random")).lower()
    heart_data_dir = str(kwargs.get("heart_data_dir", "dataset"))
    heart_filename = str(kwargs.get("heart_filename", "samples.npy"))
    lp_train_graph = str(kwargs.get("lp_train_graph", "train")).lower()
    if lp_train_graph not in {"train", "full"}:
        raise ValueError(f"Unsupported lp_train_graph={lp_train_graph}; use train or full.")
    lp_full_graph_protocol = lp_train_graph == "full"

    is_heart = (split_mode == "heart")

    # HeaRT defaults. CLI kwargs can override these for paper-grade ranking eval.
    HEART_EVAL_EVERY_DEFAULT = 50
    HEART_VAL_FRAC_DEFAULT = 0.10
    HEART_TEST_ON_BEST_VAL_DEFAULT = False

    heart_eval_every_arg = kwargs.get("heart_eval_every", None)
    heart_val_frac_arg = kwargs.get("heart_val_frac", None)
    heart_eval_every = HEART_EVAL_EVERY_DEFAULT if heart_eval_every_arg is None else max(1, int(heart_eval_every_arg))
    heart_val_frac = HEART_VAL_FRAC_DEFAULT if heart_val_frac_arg is None else float(heart_val_frac_arg)
    if heart_val_frac <= 0.0 or heart_val_frac > 1.0:
        raise ValueError(f"heart_val_frac must be in (0, 1], got {heart_val_frac}")
    heart_test_on_best_val = bool(kwargs.get("heart_test_on_best_val", HEART_TEST_ON_BEST_VAL_DEFAULT))
    checkpoint_hit_index = {"hit1": 0, "hit3": 1, "hit10": 2, "hit20": 3, "hit50": 4, "hit100": 5}
    checkpoint_metric_choices = {"roc", "ap", *checkpoint_hit_index.keys()}
    heart_checkpoint_metric = str(kwargs.get("heart_checkpoint_metric", "roc")).lower()
    random_checkpoint_metric = str(kwargs.get("random_checkpoint_metric", "roc")).lower()
    if heart_checkpoint_metric not in checkpoint_metric_choices:
        raise ValueError(f"Unsupported heart_checkpoint_metric: {heart_checkpoint_metric}")
    if random_checkpoint_metric not in checkpoint_metric_choices:
        raise ValueError(f"Unsupported random_checkpoint_metric: {random_checkpoint_metric}")

    def _checkpoint_score(metric_name: str, val_roc_value, val_ap_value, val_hit_values):
        if metric_name == "roc":
            return float(val_roc_value)
        if metric_name == "ap":
            return float(val_ap_value)
        hit_idx = checkpoint_hit_index[metric_name]
        if val_hit_values is None or len(val_hit_values) <= hit_idx:
            return float("nan")
        return float(val_hit_values[hit_idx])

    def _heart_checkpoint_score(val_roc_value, val_ap_value, val_hit_values):
        return _checkpoint_score(heart_checkpoint_metric, val_roc_value, val_ap_value, val_hit_values)

    num_nodes = adj.shape[0]
    
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    if dataset_str in ['ogbl-ddi', 'ogbl-collab']:
        print("[SPLIT] ogbl")
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_ogbl(adj, dataset_str, idx_train, idx_val, idx_test)
    else:
        if split_mode == "heart":
            print(f"[SPLIT] HeaRT | root={heart_data_dir} | filename={heart_filename}")
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false =  mask_test_edges_heart(adj, dataset_str, heart_root=heart_data_dir, filename=heart_filename)
        elif split_mode == "cimage_paper":
            print("[SPLIT] CIMAGE paper RandomLinkSplit | num_val=0.1 num_test=0.05")
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_cimage_paper(
                adj,
                dataset_str,
                split_seed=seed,
            )
        else:
            print(f"[SPLIT] random old mask_test_edges | split_seed={seed}")
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(
                adj,
                dataset_str,
                split_seed=seed,
            )

    print(
        f"[SPLIT] types: {type(adj_train)}, {type(train_edges)}, {type(val_edges)}, "
        f"{type(val_edges_false)}, {type(test_edges)}, {type(test_edges_false)}"
    )
    print(
        f"[SPLIT] sizes: train={len(train_edges)} val={len(val_edges)} test={len(test_edges)} "
        f"| val_neg_shape={np.shape(val_edges_false)} test_neg_shape={np.shape(test_edges_false)}"
    )

    # ------------------------------------------------------------
    # HeaRT: fixed cheap validation subset per run
    # ------------------------------------------------------------
    val_edges_eval = val_edges
    val_edges_false_eval = val_edges_false
    val_subset_idx = None

    if is_heart and isinstance(val_edges_false, np.ndarray) and val_edges_false.ndim == 3:
        num_val_full = len(val_edges)
        keep_num = max(1, int(round(num_val_full * heart_val_frac)))
        keep_num = min(keep_num, num_val_full)

        rng = np.random.RandomState(seed if seed is not None else 0)
        val_subset_idx = np.sort(rng.choice(num_val_full, size=keep_num, replace=False))

        val_edges_eval = val_edges[val_subset_idx]
        val_edges_false_eval = val_edges_false[val_subset_idx]

        print(
            f"[HeaRT-EVAL] fixed val subset per run: {keep_num}/{num_val_full} "
            f"({keep_num / max(1, num_val_full):.3f}) | eval_every={heart_eval_every}"
        )
    elif is_heart:
        print("[HeaRT-EVAL] full validation used during training")
    else:
        print("[EVAL] random split full validation/test used during training")

    heart_full_val_during_training = bool(is_heart and len(val_edges_eval) == len(val_edges))
    if is_heart:
        print(f"[HeaRT-EVAL] checkpoint_metric=val_{heart_checkpoint_metric}")
    else:
        print(f"[EVAL] checkpoint_metric=val_{random_checkpoint_metric}")

    def _edge_key_array(edges) -> np.ndarray:
        arr = np.asarray(edges)
        if arr.size == 0:
            return np.empty((0,), dtype=np.int64)
        arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[1] < 2:
            return np.empty((0,), dtype=np.int64)
        u = np.minimum(arr[:, 0], arr[:, 1]).astype(np.int64, copy=False)
        v = np.maximum(arr[:, 0], arr[:, 1]).astype(np.int64, copy=False)
        return u * int(num_nodes) + v

    decoded_audit_edge_sets: dict[str, set[int]] = {}
    if decoded_audit_every > 0:
        decoded_audit_edge_sets = {
            "train_pos": set(_edge_key_array(train_edges).tolist()),
            "val_pos": set(_edge_key_array(val_edges).tolist()),
            "test_pos": set(_edge_key_array(test_edges).tolist()),
            "val_neg": set(_edge_key_array(val_edges_false).tolist()),
            "test_neg": set(_edge_key_array(test_edges_false).tolist()),
        }

    def _audit_decoded_rewrite_quality(
        epoch_value: int,
        pre_graph_dense: torch.Tensor,
        post_graph_dense: torch.Tensor,
        labels_np,
        c0p_mask_t: torch.Tensor | None,
        z_for_scores: torch.Tensor | None,
        scorer: nn.Module | None,
    ) -> None:
        if decoded_audit_every <= 0:
            return
        if epoch_value != decoded_rewrite_start_epoch and epoch_value != num_epoch - 1:
            if (epoch_value - decoded_rewrite_start_epoch) % max(1, decoded_audit_every) != 0:
                return
        try:
            pre_bool = (pre_graph_dense.detach() > 0)
            post_bool = (post_graph_dense.detach() > 0)
            upper = torch.triu(torch.ones_like(pre_bool, dtype=torch.bool), diagonal=1)
            added_idx = ((post_bool & (~pre_bool)) & upper).nonzero(as_tuple=False)
            removed_idx = ((pre_bool & (~post_bool)) & upper).nonzero(as_tuple=False)

            def _pair_stats(pairs_t: torch.Tensor, prefix: str) -> dict[str, float]:
                stats: dict[str, float] = {
                    f"{prefix}_count": float(pairs_t.size(0)),
                    f"{prefix}_train_pos": 0.0,
                    f"{prefix}_val_pos": 0.0,
                    f"{prefix}_test_pos": 0.0,
                    f"{prefix}_val_neg": 0.0,
                    f"{prefix}_test_neg": 0.0,
                    f"{prefix}_cn_mean": float("nan"),
                    f"{prefix}_cn_p90": float("nan"),
                    f"{prefix}_deg_mean": float("nan"),
                    f"{prefix}_same_cluster": float("nan"),
                    f"{prefix}_c0p_touch": float("nan"),
                    f"{prefix}_score_mean": float("nan"),
                    f"{prefix}_dot_mean": float("nan"),
                }
                if pairs_t.numel() == 0:
                    return stats

                pairs_np = pairs_t.detach().cpu().numpy().astype(np.int64, copy=False)
                keys = pairs_np[:, 0] * int(num_nodes) + pairs_np[:, 1]
                for name in ("train_pos", "val_pos", "test_pos", "val_neg", "test_neg"):
                    edge_set = decoded_audit_edge_sets.get(name, set())
                    stats[f"{prefix}_{name}"] = float(sum(int(k) in edge_set for k in keys))

                adj_np = pre_bool.detach().cpu().numpy().astype(np.float32, copy=False)
                np.fill_diagonal(adj_np, 0.0)
                u = pairs_np[:, 0]
                v = pairs_np[:, 1]
                deg = adj_np.sum(axis=1)
                cn_vals = (adj_np[u] * adj_np[v]).sum(axis=1)
                stats[f"{prefix}_cn_mean"] = float(np.mean(cn_vals)) if cn_vals.size else float("nan")
                stats[f"{prefix}_cn_p90"] = float(np.percentile(cn_vals, 90)) if cn_vals.size else float("nan")
                stats[f"{prefix}_deg_mean"] = float(np.mean((deg[u] + deg[v]) * 0.5)) if cn_vals.size else float("nan")

                if labels_np is not None:
                    labels_arr = np.asarray(labels_np, dtype=np.int64)
                    if labels_arr.size == int(num_nodes):
                        same = (labels_arr[u] == labels_arr[v]) & (labels_arr[u] != -1)
                        stats[f"{prefix}_same_cluster"] = float(np.mean(same)) if same.size else float("nan")
                if c0p_mask_t is not None and c0p_mask_t.numel() == int(num_nodes):
                    c0p_np = c0p_mask_t.detach().cpu().numpy().astype(bool, copy=False)
                    touch = c0p_np[u] | c0p_np[v]
                    stats[f"{prefix}_c0p_touch"] = float(np.mean(touch)) if touch.size else float("nan")

                if z_for_scores is not None and decoded_audit_max_edges > 0:
                    score_pairs_np = pairs_np
                    if score_pairs_np.shape[0] > decoded_audit_max_edges:
                        rng = np.random.RandomState((int(seed or 0) + 1) * 1000003 + int(epoch_value))
                        sel = rng.choice(score_pairs_np.shape[0], size=decoded_audit_max_edges, replace=False)
                        score_pairs_np = score_pairs_np[np.sort(sel)]
                    try:
                        dot_vals = _edge_score_values_from_dot(z_for_scores, score_pairs_np)
                        stats[f"{prefix}_dot_mean"] = float(np.mean(dot_vals.reshape(-1))) if dot_vals.size else float("nan")
                    except Exception:
                        pass
                    if scorer is not None:
                        try:
                            score_vals = _edge_score_values_from_decoder(scorer, z_for_scores, score_pairs_np)
                            stats[f"{prefix}_score_mean"] = float(np.mean(score_vals.reshape(-1))) if score_vals.size else float("nan")
                        except Exception:
                            pass
                return stats

            add_stats = _pair_stats(added_idx, "add")
            rem_stats = _pair_stats(removed_idx, "rem")
            print(
                f"[REWRITE-AUDIT][E{epoch_value:04d}] "
                f"add_count={add_stats['add_count']:.0f} rem_count={rem_stats['rem_count']:.0f} "
                f"add_val_pos={add_stats['add_val_pos']:.0f} add_test_pos={add_stats['add_test_pos']:.0f} "
                f"add_val_neg={add_stats['add_val_neg']:.0f} add_test_neg={add_stats['add_test_neg']:.0f} "
                f"add_cn_mean={add_stats['add_cn_mean']:.6f} add_cn_p90={add_stats['add_cn_p90']:.6f} "
                f"add_deg_mean={add_stats['add_deg_mean']:.6f} add_same_cluster={add_stats['add_same_cluster']:.6f} "
                f"add_c0p_touch={add_stats['add_c0p_touch']:.6f} add_score_mean={add_stats['add_score_mean']:.6f} "
                f"add_dot_mean={add_stats['add_dot_mean']:.6f} "
                f"rem_train_pos={rem_stats['rem_train_pos']:.0f} rem_val_pos={rem_stats['rem_val_pos']:.0f} "
                f"rem_test_pos={rem_stats['rem_test_pos']:.0f} rem_cn_mean={rem_stats['rem_cn_mean']:.6f} "
                f"rem_score_mean={rem_stats['rem_score_mean']:.6f} rem_dot_mean={rem_stats['rem_dot_mean']:.6f}"
            )
        except Exception as e:
            print(f"[REWRITE-AUDIT] failed at epoch {epoch_value}: {e}")

    if lp_full_graph_protocol:
        full_adj_train = adj_orig.copy().tocsr()
        full_edges_triu = sparse_to_tuple(sp.triu(full_adj_train))[0]
        print(
            f"[LP-PROTOCOL] full graph leakage enabled: encoder/reconstruction use all observed edges "
            f"| split_train_edges={len(train_edges)} full_train_edges={len(full_edges_triu)} "
            f"| heldout_val={len(val_edges)} heldout_test={len(test_edges)}"
        )
        adj_train = full_adj_train
        train_edges = np.asarray(full_edges_triu, dtype=np.int64)
    else:
        print("[LP-PROTOCOL] no-leak train graph: encoder/reconstruction use train split only")

    adj = adj_train

    def _make_reconstruction_train_mask() -> torch.Tensor:
        mask = torch.ones(num_nodes * num_nodes, dtype=torch.bool, requires_grad=False, device=device)
        if not lp_full_graph_protocol:
            for r, c in val_edges:
                mask[num_nodes * r + c] = False
            for r, c in test_edges:
                mask[num_nodes * r + c] = False
        return mask

    train_mask = _make_reconstruction_train_mask()
    training_instance_number = torch.sum(train_mask).item()

    train_edges_t = torch.as_tensor(np.asarray(train_edges, dtype=np.int64), dtype=torch.long, device=device)
    forbidden_edge_mask = _build_forbidden_edge_mask(adj_orig, num_nodes, device)

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

    def _rebuild_train_graph_state_from_dense(g_dense_with_loops: torch.Tensor):
        """Rebuild adjacency-dependent training tensors from a dense symmetric graph with self-loops."""
        g_dense = ((g_dense_with_loops > 0).to(torch.float32) + ((g_dense_with_loops > 0).to(torch.float32)).t() > 0).to(torch.float32)
        g_dense.fill_diagonal_(1.0)
        g_train_dense = g_dense.clone()
        g_train_dense.fill_diagonal_(0.0)
        adj_train_new = sp.csr_matrix(g_train_dense.detach().cpu().numpy())

        edge_index_new = g_dense.to(device).to_sparse().indices()

        adj_norm_tuple_new = preprocess_graph(adj_train_new)
        adj_label_tuple_new = sparse_to_tuple(adj_train_new + sp.eye(adj_train_new.shape[0]))

        adj_norm_new = torch.sparse.FloatTensor(
            torch.LongTensor(adj_norm_tuple_new[0].T),
            torch.FloatTensor(adj_norm_tuple_new[1]),
            torch.Size(adj_norm_tuple_new[2]),
        ).to(device)
        adj_label_new = torch.sparse.FloatTensor(
            torch.LongTensor(adj_label_tuple_new[0].T),
            torch.FloatTensor(adj_label_tuple_new[1]),
            torch.Size(adj_label_tuple_new[2]),
        ).coalesce().to(device)

        adj_sum = float(adj_train_new.sum())
        pos_weight_new = float(training_instance_number - adj_sum) / max(adj_sum, 1.0)
        norm_new = training_instance_number / float(max((training_instance_number - adj_sum) * 2, 1.0))
        weight_mask_new = adj_label_new.to_dense().view(-1)[train_mask] == 1
        weight_tensor_new = torch.ones(weight_mask_new.size(0), device=device)
        weight_tensor_new[weight_mask_new] = pos_weight_new
        return adj_train_new, edge_index_new, adj_norm_new, adj_label_new, pos_weight_new, norm_new, weight_tensor_new

    # init model and optimizer
    if ae_backbone == "maskgae":
        encoder = MaskGAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device, mask_rate=maskgae_mask_rate).to(device)
    elif ae_backbone == "cimage_lite":
        encoder = CIMAGELite_ENCODER(
            feat_dim,
            hidden1,
            hidden2,
            dropout,
            device,
            mask_rate=maskgae_mask_rate,
            num_factors=cimage_num_factors,
            num_clusters=cimage_num_clusters,
            cluster_alpha=cimage_cluster_alpha,
        ).to(device)
    elif ae_backbone == "cimage_full":
        encoder = CIMAGEFull_ENCODER(
            feat_dim,
            hidden1,
            hidden2,
            dropout,
            device,
            mask_rate=maskgae_mask_rate,
            num_factors=cimage_num_factors,
            num_clusters=cimage_num_clusters,
            pseudo_label_threshold=cimage_pseudo_label_threshold,
            factor_select_ratio=cimage_factor_select_ratio,
            mrmr_redundancy_weight=cimage_mrmr_redundancy_weight,
            cluster_balance_weight=cimage_cluster_balance_weight,
            sce_power=cimage_sce_power,
        ).to(device)
    else:
        encoder = VGNAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device) # encoder = VGAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device)

    graph_decoder = None
    if use_edited_decoder:
        if decoder_type == "bilinear":
            graph_decoder = BilinearGraphDecoder(hidden2, normalize_input=decoder_normalize_input).to(device)
        elif decoder_type == "mlp_pair":
            graph_decoder = MLPPairGraphDecoder(
                hidden2,
                hidden_dim=max(hidden2, editor_hidden),
                normalize_input=decoder_normalize_input,
                max_pair_rows=mlp_pair_max_rows,
            ).to(device)
        elif decoder_type == "pair_mlp_struct":
            graph_decoder = StructuralPairGraphDecoder(
                hidden2,
                hidden_dim=max(hidden2, editor_hidden),
                normalize_input=decoder_normalize_input,
                max_pair_rows=mlp_pair_max_rows,
            ).to(device)
        else:
            raise ValueError(f"Unsupported decoder_type={decoder_type}; use 'bilinear', 'mlp_pair', or 'pair_mlp_struct'.")

    prediction_decoder = None
    if prediction_decoder_type in {"", "none", "off"}:
        prediction_decoder_type = "none"
    elif prediction_decoder_type == "pair_residual_struct":
        prediction_decoder = ResidualStructuralPairPredictionDecoder(
            hidden2,
            hidden_dim=max(hidden2, editor_hidden),
            normalize_input=decoder_normalize_input,
            max_pair_rows=mlp_pair_max_rows,
        ).to(device)
    elif prediction_decoder_type == "pair_residual_struct_ncnc":
        prediction_decoder = NCNCResidualStructuralPairPredictionDecoder(
            hidden2,
            hidden_dim=max(hidden2, editor_hidden),
            normalize_input=decoder_normalize_input,
            max_pair_rows=mlp_pair_max_rows,
        ).to(device)
    elif prediction_decoder_type == "pair_residual_struct_ncnc_h3_delta":
        prediction_decoder = H3DeltaNCNCResidualStructuralPairPredictionDecoder(
            hidden2,
            hidden_dim=max(hidden2, editor_hidden),
            normalize_input=decoder_normalize_input,
            max_pair_rows=mlp_pair_max_rows,
            gate_init_logit=prediction_h3_gate_init,
        ).to(device)
    elif prediction_decoder_type == "pair_residual_struct_ncnc_multi":
        prediction_decoder = MultiOrderNCNCResidualStructuralPairPredictionDecoder(
            hidden2,
            hidden_dim=max(hidden2, editor_hidden),
            normalize_input=decoder_normalize_input,
            max_pair_rows=mlp_pair_max_rows,
        ).to(device)
    elif prediction_decoder_type == "pair_residual_struct_compact_multi":
        prediction_decoder = CompactMultiOrderResidualStructuralPairPredictionDecoder(
            hidden2,
            hidden_dim=max(hidden2, editor_hidden),
            normalize_input=decoder_normalize_input,
            max_pair_rows=mlp_pair_max_rows,
        ).to(device)
    elif prediction_decoder_type == "pair_residual_struct_compact_multi_gated":
        prediction_decoder = GatedCompactMultiOrderResidualStructuralPairPredictionDecoder(
            hidden2,
            hidden_dim=max(hidden2, editor_hidden),
            normalize_input=decoder_normalize_input,
            max_pair_rows=mlp_pair_max_rows,
            gate_init_logit=prediction_residual_gate_init,
            residual_scale=prediction_residual_scale,
        ).to(device)
    elif prediction_decoder_type == "pair_residual_struct_ocn":
        prediction_decoder = OCNResidualStructuralPairPredictionDecoder(
            hidden2,
            hidden_dim=max(hidden2, editor_hidden),
            normalize_input=decoder_normalize_input,
            max_pair_rows=mlp_pair_max_rows,
        ).to(device)
    else:
        raise ValueError(
            f"Unsupported prediction_decoder_type={prediction_decoder_type}; use 'none', 'pair_residual_struct', 'pair_residual_struct_ncnc', 'pair_residual_struct_ncnc_h3_delta', 'pair_residual_struct_ncnc_multi', 'pair_residual_struct_compact_multi', 'pair_residual_struct_compact_multi_gated', or 'pair_residual_struct_ocn'."
        )

    def _set_struct_decoder_context(
        graph_dense=None,
        labels_np: np.ndarray | None = None,
        core_mask_t: torch.Tensor | None = None,
    ) -> None:
        if isinstance(graph_decoder, StructuralPairGraphDecoder):
            if graph_dense is not None:
                graph_decoder.set_graph_context(graph_dense)
            graph_decoder.set_cluster_context(labels_np, core_mask_t)

    def _set_prediction_decoder_context(
        graph_dense=None,
        labels_np: np.ndarray | None = None,
        core_mask_t: torch.Tensor | None = None,
    ) -> None:
        if isinstance(prediction_decoder, StructuralPairGraphDecoder):
            if graph_dense is not None:
                prediction_decoder.set_graph_context(graph_dense)
            prediction_decoder.set_cluster_context(labels_np, core_mask_t)

    def _prediction_decoder_extra_diagnostics() -> dict[str, float]:
        if prediction_decoder is None or not hasattr(prediction_decoder, "extra_diagnostics"):
            return {}
        try:
            return prediction_decoder.extra_diagnostics()
        except Exception:
            return {}

    def _decoded_graph_scorer():
        return graph_decoder if graph_decoder is not None else prediction_decoder

    def _set_decoded_graph_scorer_context(
        graph_dense=None,
        labels_np: np.ndarray | None = None,
        core_mask_t: torch.Tensor | None = None,
    ) -> None:
        if graph_decoder is not None:
            _set_struct_decoder_context(graph_dense, labels_np, core_mask_t)
        elif prediction_decoder is not None:
            _set_prediction_decoder_context(graph_dense, labels_np, core_mask_t)

    if isinstance(graph_decoder, StructuralPairGraphDecoder):
        _set_struct_decoder_context(adj_train, None, None)
        print("[DECODER] pair_mlp_struct edit graph context initialized from train graph")
    if isinstance(prediction_decoder, StructuralPairGraphDecoder):
        _set_prediction_decoder_context(adj_train, None, None)
        print(f"[PRED-DECODER] {prediction_decoder_type} graph context initialized from train graph")

    if score_source == "decoder" and graph_decoder is None:
        raise ValueError("score_source=decoder requires --use_edited_decoder so an edit decoder is available.")
    if score_source == "pred_decoder" and prediction_decoder is None:
        raise ValueError("score_source=pred_decoder requires a structural prediction decoder.")

    def _score_adjacency(z: torch.Tensor) -> torch.Tensor:
        if score_source == "decoder":
            return graph_decoder(z)
        if score_source == "pred_decoder":
            return prediction_decoder(z)
        return dot_product_decode(z)

    def _edge_score_values_for_source(z: torch.Tensor, edges) -> np.ndarray:
        if score_source == "decoder":
            return _edge_score_values_from_decoder(graph_decoder, z, edges)
        if score_source == "pred_decoder":
            return _edge_score_values_from_decoder(prediction_decoder, z, edges)
        return _edge_score_values_from_dot(z, edges)

    def _evaluate_edges_for_source(
        z: torch.Tensor,
        edges_pos,
        edges_neg,
        score_matrix_np: np.ndarray | None = None,
    ):
        if score_matrix_np is not None:
            return get_scores(dataset_str, edges_pos, edges_neg, score_matrix_np, adj_orig)
        pos_scores = _edge_score_values_for_source(z, edges_pos)
        neg_scores = _edge_score_values_for_source(z, edges_neg)
        return get_scores_from_values(pos_scores, neg_scores)

    def _set_module_requires_grad(module: nn.Module | None, flag: bool):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad_(flag)

    def _module_params(module: nn.Module | None) -> list[torch.nn.Parameter]:
        if module is None:
            return []
        return [p for p in module.parameters() if p.requires_grad]

    def _build_main_optimizer():
        _set_module_requires_grad(encoder, True)
        _set_module_requires_grad(graph_decoder, True)
        _set_module_requires_grad(prediction_decoder, True)
        params = list(encoder.parameters())
        params += list(graph_decoder.parameters()) if graph_decoder is not None else []
        params += list(prediction_decoder.parameters()) if prediction_decoder is not None else []
        return Adam(params, lr=learning_rate, weight_decay=weight_decay)

    def _build_phase2_optimizer():
        if graph_decoder is None and prediction_decoder is None:
            return _build_main_optimizer()
        if phase2_freeze_encoder or edit_phase_encoder_lr_scale <= 0.0:
            _set_module_requires_grad(encoder, False)
            _set_module_requires_grad(graph_decoder, True)
            _set_module_requires_grad(prediction_decoder, True)
            params = _module_params(graph_decoder) + _module_params(prediction_decoder)
            return Adam(params, lr=learning_rate, weight_decay=weight_decay) if params else _build_main_optimizer()
        _set_module_requires_grad(encoder, True)
        _set_module_requires_grad(graph_decoder, True)
        _set_module_requires_grad(prediction_decoder, True)
        decoder_params = _module_params(graph_decoder) + _module_params(prediction_decoder)
        groups = [
            {"params": list(encoder.parameters()), "lr": learning_rate * edit_phase_encoder_lr_scale, "weight_decay": weight_decay},
        ]
        if decoder_params:
            groups.append({"params": decoder_params, "lr": learning_rate, "weight_decay": weight_decay})
        return Adam(groups)

    optimizer = _build_main_optimizer()
    phase2_optimizer_activated = False
    cimage_backbones = {"cimage_lite", "cimage_full"}

    def _encoder_reconstruction_loss(A_pred_cur: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        beta_eff = 0.0 if ae_backbone in {"maskgae"} | cimage_backbones else beta
        recon_cur = loss_function(
            A_pred_cur,
            adj_label,
            encoder.mean,
            encoder.logstd,
            norm,
            weight_tensor,
            alpha,
            beta_eff,
            train_mask,
        )
        feature_loss = A_pred_cur.new_tensor(0.0)
        if ae_backbone in {"maskgae", "cimage_lite"} and maskgae_feature_weight != 0.0:
            feature_loss = encoder.masked_feature_loss()
            recon_cur = recon_cur + maskgae_feature_weight * feature_loss
        cimage_factor_loss = A_pred_cur.new_tensor(0.0)
        cimage_cluster_loss = A_pred_cur.new_tensor(0.0)
        if ae_backbone in cimage_backbones:
            cimage_factor_loss, cimage_cluster_loss = encoder.cimage_losses()
            recon_cur = (
                recon_cur
                + cimage_factor_weight * cimage_factor_loss
                + cimage_cluster_weight * cimage_cluster_loss
            )
        return recon_cur, feature_loss, cimage_factor_loss, cimage_cluster_loss

    if use_edited_decoder:
        print(
            f"[PATH] edited-decoder path active | decoder={decoder_type} | normalize_input={decoder_normalize_input} | "
            f"ae_backbone={ae_backbone}"
            f"{' (vgae alias)' if raw_ae_backbone == 'vgae' else ''}"
            f"{' (cimage alias)' if raw_ae_backbone == 'cimage' else ''} | "
            f"maskgae_mask_rate={maskgae_mask_rate} maskgae_feature_w={maskgae_feature_weight} | "
            f"cimage_factor_w={cimage_factor_weight} cimage_cluster_w={cimage_cluster_weight} "
            f"cimage_factors={cimage_num_factors} cimage_clusters={cimage_num_clusters} "
            f"cimage_pseudo_thr={cimage_pseudo_label_threshold} cimage_factor_select={cimage_factor_select_ratio} "
            f"cimage_mrmr_red={cimage_mrmr_redundancy_weight} cimage_balance={cimage_cluster_balance_weight} "
            f"cimage_sce_power={cimage_sce_power} | "
            f"freeze_c0p={freeze_c0p_at_edit_start} | decoded_graph_augment={use_decoded_graph_augment} | "
            f"accumulate_base={decoded_accumulate_into_base} | edit_start_epoch={edit_start_epoch} | "
            f"edit_train_start={edit_train_start_epoch} | rewrite_start={decoded_rewrite_start_epoch} | rewrite_every={decoded_rewrite_every} | edit_end_epoch={decoded_edit_end_epoch} | "
            f"separate_edit_training={separate_edit_training} | retain_recon={edit_phase_retain_recon_weight} | retain_cl={edit_phase_retain_cl_weight} | "
            f"add_ratio={decoded_add_ratio} remove_ratio={decoded_remove_ratio} | "
            f"add_thr={decoded_add_threshold} remove_thr={decoded_remove_threshold} | "
            f"add_q={decoded_add_quantile} remove_q={decoded_remove_quantile} | "
            f"max_add={decoded_max_add_per_round} max_remove={decoded_max_remove_per_round} | "
            f"decoder_objective={decoder_objective} | compactness_objective={compactness_objective} | "
            f"compactness_radius_metric={compactness_radius_metric} | "
            f"score_source={score_source} | mlp_pair_max_rows={mlp_pair_max_rows} | skip_oom_epoch={int(skip_oom_epoch)} | "
            f"recon_w={decoder_recon_weight} keep_w={decoder_keep_weight} add_rank_w={decoder_add_rank_weight} "
            f"remove_rank_w={decoder_remove_rank_weight} rank_margin={decoder_rank_margin} "
            f"rank_strategy={decoder_rank_strategy} rank_neg_k={decoder_rank_neg_k} rank_pool_factor={decoder_rank_pool_factor} | "
            f"heart_rank_w={heart_rank_weight} heart_margin={heart_rank_margin} "
            f"heart_neg_k={heart_rank_neg_k} heart_pool_factor={heart_rank_pool_factor} | "
            f"cl_mode={cl_mode} prediction_graph={prediction_graph} feat_mask_ratio={feat_maske_ratio} | "
            f"compactness_weight={compactness_weight} | preserve_weight={preserve_weight} | "
            f"c0p_noncompact_endpoint={int(decoded_require_c0p_noncompact_endpoint)} | "
            f"decoded_struct_support={int(decoded_require_structural_support)}:{decoded_struct_support} "
            f"min_cn={decoded_struct_min_cn} min_ra={decoded_struct_min_ra} min_aa={decoded_struct_min_aa} | "
            f"decoded_add_degree_target={decoded_add_degree_target} | "
            f"decoded_add_degree_target_scope={decoded_add_degree_target_scope} | "
            f"decoded_add_degree_target_nodes={decoded_add_degree_target_nodes} | "
            f"decoded_guarantee_degree_target={decoded_guarantee_degree_target} | "
            f"phase2_freeze_encoder={phase2_freeze_encoder} | edit_phase_encoder_lr_scale={edit_phase_encoder_lr_scale} | "
            f"decoder_warmup_in_phase1={decoder_warmup_in_phase1} | decoder_warmup_recon_weight={decoder_warmup_recon_weight} | "
            f"decoder_warmup_use_pulled_latent={decoder_warmup_use_pulled_latent} | "
            f"pull_strength={editor_pull_strength} | pull_profile={editor_pull_profile} "
            f"pull_tau={editor_pull_tau} pull_deadzone={editor_pull_deadzone} pull_anchor={editor_pull_anchor} | "
            f"compactness_weight={compactness_weight} | "
            f"push_scope={editor_push_scope} | noncompact_push={editor_noncompact_push_strength} | "
            f"noise_push={editor_noise_push_strength} | push_preserve_norm={int(editor_push_preserve_norm)} | "
            f"pull_scope={pull_mask_scope} | compactness_scope={compactness_mask_scope} | rewrite_scope={rewrite_endpoint_scope}"
        )
    if prediction_decoder is not None:
        print(
            f"[PRED-DECODER] active | type={prediction_decoder_type} | score_source={score_source} | "
            f"rank_w={prediction_rank_weight} bce_w={prediction_bce_weight} margin={prediction_rank_margin} "
            f"neg_k={prediction_rank_neg_k} pool_factor={prediction_rank_pool_factor} "
            f"neg_strategy={prediction_rank_neg_strategy} struct_frac={prediction_rank_struct_frac} | "
            f"hard_only={int(prediction_hard_residual_only)} hard_margin={prediction_hard_margin} "
            f"dot_anchor_w={prediction_dot_anchor_weight} | "
            f"joint_start={prediction_joint_start_epoch} encoder_w={prediction_encoder_weight} "
            f"gate_l1_w={prediction_gate_l1_weight} h3_gate_init={prediction_h3_gate_init} "
            f"residual_gate_init={prediction_residual_gate_init} residual_scale={prediction_residual_scale}"
        )
    if not use_edited_decoder and prediction_decoder is None:
        print(
            f"[PATH] baseline dot-product path active | ae_backbone={ae_backbone}"
            f"{' (vgae alias)' if raw_ae_backbone == 'vgae' else ''}"
            f"{' (cimage alias)' if raw_ae_backbone == 'cimage' else ''} | "
            f"maskgae_mask_rate={maskgae_mask_rate} maskgae_feature_w={maskgae_feature_weight}"
            f" cimage_factor_w={cimage_factor_weight} cimage_cluster_w={cimage_cluster_weight}"
            f" cimage_factors={cimage_num_factors} cimage_clusters={cimage_num_clusters}"
            f" cimage_pseudo_thr={cimage_pseudo_label_threshold} cimage_factor_select={cimage_factor_select_ratio}"
        )
    print(
        f"[EVAL-POLICY] eval_log_every={eval_log_every} train_eval_every={train_eval_every} "
        f"skip_train_acc={int(skip_train_acc)} decoder_diag_every={decoder_diag_every} "
        f"edit_metric_every={edit_metric_every} decoded_audit_every={decoded_audit_every} "
        f"decoded_audit_max_edges={decoded_audit_max_edges} edge_eval={int(edge_eval)} heart_eval_every={heart_eval_every} "
        f"heart_val_frac={heart_val_frac} random_checkpoint_metric={random_checkpoint_metric}"
    )

    def _edit_allowed_before_end(ep: int) -> bool:
        try:
            end_ep = int(decoded_edit_end_epoch) if decoded_edit_end_epoch is not None else -1
        except Exception:
            end_ep = -1
        return not (end_ep >= 0 and ep > end_ep)

    def _in_edit_phase(ep: int) -> bool:
        if not (use_edited_decoder and (graph_decoder is not None)):
            return False
        return ep >= edit_train_start_epoch and _edit_allowed_before_end(ep)

    def _decoded_edit_active(ep: int) -> bool:
        if _decoded_graph_scorer() is None:
            return False
        return ep >= decoded_rewrite_start_epoch and _edit_allowed_before_end(ep)

    def _decoded_rewrite_due(ep: int) -> bool:
        if not _decoded_edit_active(ep):
            return False
        if ep == decoded_rewrite_start_epoch or ep == num_epoch - 1:
            return True
        return ((ep - decoded_rewrite_start_epoch) % decoded_rewrite_every) == 0

    def _decoder_warmup_active(ep: int) -> bool:
        return bool(
            use_edited_decoder
            and (graph_decoder is not None)
            and decoder_warmup_in_phase1
            and ep < edit_train_start_epoch
        )

    data_augmenter = MLP(hidden2, hidden2).to(device)
    # data_augmenter = VGAE_ENCODER(feat_dim, hidden1, hidden2, dropout, device).to(device)
    data_augmenter_optimizer = Adam(data_augmenter.parameters(), lr = 0.01, weight_decay = weight_decay)

    best_acc = 0.0
    best_test_roc_seen = -float("inf")
    best_test_ap_at_best_test_roc = 0.0
    best_val_roc_at_best_test_roc = 0.0
    best_val_ap_at_best_test_roc = 0.0
    best_test_epoch = -1
    best_hit1 = 0.0;best_hit3 = 0.0;best_hit10 = 0.0; best_hit20 = 0.0; best_hit50 = 0.0; best_hit100 = 0.0
    best_hit1_test_roc = 0.0;best_hit3_test_roc = 0.0;best_hit10_test_roc = 0.0; best_hit20_test_roc = 0.0; best_hit50_test_roc = 0.0; best_hit100_test_roc = 0.0
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
    edit_recon_hist = []
    edit_heart_rank_hist = []
    edit_compact_hist = []
    edit_preserve_hist = []
    radius_before_hist = []
    radius_after_hist = []
    c0p_radius_before_hist = []
    c0p_radius_after_hist = []
    cp_radius_before_hist = []
    cp_radius_after_hist = []
    radius_anchor_hist = []
    delta_from_anchor_hist = []
    delta_from_prev_rewrite_hist = []
    rewrite_applied_hist = []
    rewrite_epoch_hist = []
    rewrite_radius_before_hist = []
    rewrite_radius_after_hist = []
    rewrite_delta_hist = []
    rewrite_delta_anchor_hist = []
    rewrite_delta_prev_hist = []
    radius_anchor_value = None
    prev_rewrite_radius_after_value = None
    last_valid_delta_prev_rewrite = float('nan')
    
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
                recon0, maskgae_feat0, cimage_factor0, cimage_cluster0 = _encoder_reconstruction_loss(A_pred0)
                # only intra-view CL during pretrain
                if loss_ver == "nei":
                    cl0 = inter_view_CL_loss(device, Z0, Z0, adj_label, gamma, temperature)
                else:
                    cl0 = intra_view_CL_loss(device, Z0, adj_label, gamma, temperature)
                loss0 = recon0 + cl0
                loss0.backward()
                optimizer.step()
                if (pe+1) % 20 == 0 or pe == 0:
                    msg = f"[pretrain] epoch {pe+1}/{pretrain_epochs} loss={float(loss0):.4f}"
                    if ae_backbone in {"maskgae", "cimage_lite"}:
                        msg += f" maskgae_feat={float(maskgae_feat0.detach().cpu()):.4f}"
                    if ae_backbone in cimage_backbones:
                        msg += (
                            f" cimage_factor={float(cimage_factor0.detach().cpu()):.4f}"
                            f" cimage_cluster={float(cimage_cluster0.detach().cpu()):.4f}"
                        )
                    print(msg)

            with torch.no_grad():
                Z0 = encoder(features, edge_index)
                frozen_scores = dot_product_decode(Z0).detach()

            # --------------------------------------------------
            # Build frozen (per-dataset) GMM clusters on Z0
            # --------------------------------------------------
            # build frozen clusters on Z0 for this dataset/run
            try:
                Z0_det = Z0.detach()
                labels0 = gmm_labels(
                    Z0_det, K=gmm_k, tau=gmm_tau, metric="cosine"
                )
                deg_base = _deg_excl_self(adj_label.to_dense())
                core0, _, _, _ = select_gmm_cores(
                    Z0_det,
                    labels0,
                    degrees_excl_self=deg_base,
                    alpha=restrict_alpha,
                    gamma=restrict_gamma,
                    B=Z0_det.size(1),
                )
                # store as numpy so TSNE can reuse
                frozen_gmm_labels = labels0
                frozen_core_mask  = core0.bool().detach().cpu().numpy()
                print(f"[GMM-FROZEN] built frozen GMM clusters on Z0 "
                    f"(K={gmm_k}, tau={gmm_tau})")
                # ---- TSNE cache: save Z0 + frozen clusters/core for later TSNE ----
                try:
                    # radii on Z0 (per-node), non-noise only is handled later in analysis scripts
                    radii0, _, _ = per_cluster_stats_diag(Z0_det, labels0, normalize_cosine=True)

                    # output dir keyed by dataset/ver/seed/pre_prune_frac
                    out_dir = os.path.join(
                        tsne_cache_root,
                        str(dataset_str), str(ver),
                        f"seed{seed}",
                        f"preprune_{pre_prune_frac:.2f}",
                    )

                    y_np = None
                    try:
                        # node labels (ground-truth classes)
                        y_np = labels.detach().cpu().numpy() if hasattr(labels, "detach") else None
                    except Exception:
                        y_np = None

                    meta = dict(
                        dataset=str(dataset_str),
                        ver=str(ver),
                        seed=int(seed),
                        pre_prune_frac=float(pre_prune_frac),
                        pre_prune_scope=str(pre_prune_scope),
                        gmm_k=int(gmm_k),
                        gmm_tau=float(gmm_tau),
                        alpha=float(restrict_alpha),
                        gamma=float(restrict_gamma),
                        stage="Z0_frozen",
                    )

                    save_tsne_cache(
                        out_dir,
                        Z=Z0_det,
                        gmm_labels=np.asarray(labels0, dtype=np.int64),
                        core_mask=np.asarray(frozen_core_mask, dtype=bool),
                        y=y_np,
                        radii=radii0.detach().cpu().numpy(),
                        meta=meta,
                        prefix="Z0",
                    )
                except Exception as e:
                    print(f"[TSNE|CACHE] save Z0 failed: {e}")
            except Exception as e:
                print(f"[GMM-FROZEN] failed to build frozen clusters: {e}")
                frozen_gmm_labels = None
                frozen_core_mask  = None

            if frozen_scores_path:
                try:
                    torch.save(frozen_scores.cpu(), frozen_scores_path)
                    print(f"[pretrain] Saved frozen scores -> {frozen_scores_path}")
                except Exception as e:
                    print(f"[pretrain] Save frozen scores failed: {e}")
            # only enforce shape if needed
            if needs_scores:
                if frozen_scores is None or frozen_scores.shape != (num_nodes, num_nodes):
                    print("[WARN] score-based adds disabled (no valid frozen_scores).")
                    frozen_scores = None
            if pretrained_ckpt_path:
                try:
                    torch.save(encoder.state_dict(), pretrained_ckpt_path)
                    print(f"[pretrain] Saved encoder checkpoint -> {pretrained_ckpt_path}")
                except Exception as e:
                    print(f"[pretrain] Save encoder checkpoint failed: {e}")

        # --- STATIC PRE-PRUNE (C0p / CP) BEFORE MAIN TRAINING ---
        if pre_prune_frac > 0.0:
            print(f"[static-preprune] frac={pre_prune_frac:.2f}, scope={pre_prune_scope}")

            # Base graph: original TRAIN split (no val/test edges)
            g_np = adj_train.toarray().astype(np.float32)
            np.fill_diagonal(g_np, 0.0)
            N = g_np.shape[0]

            # 1) GMM on PRETRAINED embeddings Z0
            Z0_det = Z0.detach()
            labels_cluster = gmm_labels(
                Z0_det, K=gmm_k, tau=gmm_tau, metric="cosine"
            )
            uniq, counts = np.unique(labels_cluster[labels_cluster >= 0], return_counts=True)
            print(f"[static-preprune] GMM clusters (non-noise): {len(uniq)} | sizes={counts.tolist()}")

            # 2) Core mask (C0p) using α,γ gating and base degrees
            deg_base_np = g_np.sum(axis=1).astype(np.int64)
            deg_base = torch.from_numpy(deg_base_np).to(Z0_det.device)
            core_mask, _, _, _ = select_gmm_cores(
                Z0_det, labels_cluster,
                degrees_excl_self=deg_base,
                alpha=restrict_alpha,
                gamma=restrict_gamma,
                B=Z0_det.size(1),
            )
            core_mask_np = core_mask.bool().detach().cpu().numpy()

            # 3) Candidate edges according to pre_prune_scope
            g_torch = torch.from_numpy(g_np)
            lbl_t = torch.from_numpy(labels_cluster.astype(np.int64))
            same_cluster = (lbl_t[:, None] == lbl_t[None, :]) & (lbl_t[:, None] != -1)
            existing = (g_torch > 0)
            core_t = torch.from_numpy(core_mask_np)

            if pre_prune_scope == "cp_all":
                cand = existing.clone()
            elif pre_prune_scope == "c0p_only":
                both_core = (core_t[:, None] & core_t[None, :])
                cand = existing & same_cluster & both_core
            elif pre_prune_scope == "cp_minus_c0p":
                both_core = (core_t[:, None] & core_t[None, :])
                cand = existing & same_cluster & (~both_core)
            else:
                raise ValueError(f"Unknown pre_prune_scope: {pre_prune_scope}")

            cand.fill_diagonal_(False)
            idx_i, idx_j = cand.triu(1).nonzero(as_tuple=True)
            num_cand = int(idx_i.numel())
            print(f"[static-preprune] candidate edges ({pre_prune_scope}) = {num_cand}")

            if num_cand == 0:
                print("[static-preprune] no candidates; skip pre-prune.")
            else:
                # 4) Rank by BASE cosine similarity (lowest first)
                X0 = F.normalize(Z0_det, p=2, dim=1)
                S0 = X0 @ X0.t()                  # [N, N], on GPU if Z0_det is on GPU

                # move indices to same device as S0 for indexing
                idx_i_dev = idx_i.to(S0.device)
                idx_j_dev = idx_j.to(S0.device)

                sim = S0[idx_i_dev, idx_j_dev]    # [num_cand]
                order = torch.argsort(sim, descending=False)   # worst (lowest sim) first

                # bring sorted indices back to CPU / numpy for the pruning loop
                order_cpu = order.cpu()
                idx_i_sorted = idx_i[order_cpu].cpu().numpy()
                idx_j_sorted = idx_j[order_cpu].cpu().numpy()
                pairs = np.stack([idx_i_sorted, idx_j_sorted], axis=1)

                target_remove = int(round(pre_prune_frac * num_cand))
                target_remove = max(0, min(target_remove, num_cand))
                print(f"[static-preprune] target_remove = {target_remove} edges")

                # 5) NO DEGREE FLOOR: just keep a degree array for logging if you want
                deg = g_np.sum(axis=1).astype(np.int64).reshape(-1)

                removed = 0
                for i, j in pairs:
                    if removed >= target_remove:
                        break
                    i = int(i); j = int(j)
                    if g_np[i, j] == 0.0:
                        continue
                    # remove undirected edge (no degree_floor constraint)
                    g_np[i, j] = 0.0
                    g_np[j, i] = 0.0
                    deg[i] -= 1
                    deg[j] -= 1
                    removed += 1

                print(f"[static-preprune] actually removed {removed} edges "
                    f"({removed / max(1, num_cand):.3f} of candidates)")

                # 6) Replace TRAIN graph with pruned graph
                adj_train = sp.csr_matrix(g_np)
                adj = adj_train
                
                # ---- rebuild all training tensors on the pruned train graph ----
                # 1) train_mask & training_instance_number
                train_mask = _make_reconstruction_train_mask()
                training_instance_number = torch.sum(train_mask).item()

                # 2) APPNP / edge_index on pruned graph
                edge_index = from_scipy_sparse_matrix(adj)[0].to(device)

                # 3) adj_label & adj_norm from pruned adj_train
                adj_norm_tuple = preprocess_graph(adj)                        # Laplacian on pruned graph
                adj_label_sp = adj_train + sp.eye(adj_train.shape[0])
                adj_label_tuple = sparse_to_tuple(adj_label_sp)

                adj_norm = torch.sparse.FloatTensor(
                    torch.LongTensor(adj_norm_tuple[0].T),
                    torch.FloatTensor(adj_norm_tuple[1]),
                    torch.Size(adj_norm_tuple[2]),
                ).to(device)

                adj_label = torch.sparse.FloatTensor(
                    torch.LongTensor(adj_label_tuple[0].T),
                    torch.FloatTensor(adj_label_tuple[1]),
                    torch.Size(adj_label_tuple[2]),
                ).to(device)

                # 4) pos_weight, norm, weight_mask, weight_tensor on pruned graph
                pos_weight = float(training_instance_number - adj.sum()) / adj.sum()
                norm = training_instance_number / float((training_instance_number - adj.sum()) * 2)

                weight_mask = adj_label.to_dense().view(-1)[train_mask] == 1
                weight_tensor = torch.ones(weight_mask.size(0)).to(device)
                weight_tensor[weight_mask] = pos_weight

                # ---- IMPORTANT ----
                # Rebuild all training tensors that depend on adj / adj_train.
                # You ALREADY have this code a few lines above:
                #   - train_mask, training_instance_number
                #   - pos_weight, norm
                #   - adj_label ( = adj_train + I ), sparse_to_tuple(...)
                #   - adj_norm = preprocess_graph(adj)
                #   - adj_norm / adj_label / features sparse tensors
                #   - weight_mask, weight_tensor
                #   - edge_index = from_scipy_sparse_matrix(adj)[0].to(device)
                #
                # Move that block into a small helper or re-run it here
                # so that from here on, the main training loop uses the
                # STATICALLY PRUNED train graph.
        
        # If we loaded only the scores (not weights), that’s fine.
        # If we loaded weights too, great. Either way, re-init optimizer so both DESC/ASC start fresh.
        optimizer = _build_main_optimizer()
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

    # GLOBAL (per-run) frozen GMM cluster & core mask.
    # We will *always* define these based on the pre-trained encoder Z0 on the ORIGINAL graph
    # and then re-use them for TSNE / analysis, regardless of prune_fraction.

    # cumulative trackers (reset for this dataset/run)
    global_used_adds = 0
    global_used_removals = 0             # count removals across all epochs
    node_used_adds   = torch.zeros(N, dtype=torch.long, device=g0.device)

    fixed_c0p_labels = None
    fixed_c0p_mask = None
    stage1_anchor_Z = None
    stage1_anchor_graph_dense = None
    stage1_phase_boundary_logged = False
    frozen_gmm_labels = locals().get('frozen_gmm_labels', None)
    frozen_core_mask = locals().get('frozen_core_mask', None)
    radii_f = None
    lbl_np = None
    core_np = None
    tsne_cache_root = _artifact_dir("artifacts", "tsne_cache")
    cluster_plot_root = _artifact_dir("plots", "cluster")
    radius_history_root = _artifact_dir("radius")

    # sanity checks (helpful when switching datasets)
    print(f"[AUG-INIT] N={N} | E0={E0} | aug_ratio(global)={aug_ratio} | aug_bound(per-node)={aug_bound}")
    if frozen_scores is not None and frozen_scores.shape != (N, N):
        raise ValueError(f"frozen_scores shape {tuple(frozen_scores.shape)} != {(N,N)} for current dataset")
    # -------------------- REMOVE-ONLY INIT (for ver: remove_only_*) --------------------
    remove_state = None
    if isinstance(ver, str) and ver.startswith("remove_only_"):
        import re

        # parse keep% from ver: ..._keep100/_keep90/_keep75/_keep50/_keep25
        m_keep = re.search(r"_keep(\d+)", ver)
        keep_pct = int(m_keep.group(1)) if m_keep else 100
        keep_frac = max(0.0, min(1.0, keep_pct / 100.0))

        # parse scope from ver name (use same naming family as your prune_*):
        # remove_only_cp_all_keepXX
        # remove_only_c0p_only_keepXX
        # remove_only_cp_minus_c0p_keepXX
        if "cp_minus_c0p" in ver:
            scope_name = "cp_minus_c0p"
        elif "c0p_only" in ver:
            scope_name = "c0p_only"
        else:
            scope_name = "cp_all"   # default

        print(f"[REMOVE-ONLY-INIT] scope={scope_name} keep={keep_pct}%")

        # base embedding on ORIGINAL train graph for a FIXED scope definition + fixed ranking
        with torch.no_grad():
            Z_base = encoder(features, edge_index)

        # fixed labels + fixed core mask (same as PRUNE INIT logic)
        labels_fixed = gmm_labels(Z_base.detach(), K=gmm_k, tau=gmm_tau, metric="cosine")
        deg_base = _deg_excl_self(adj_label.to_dense())
        core_mask_fixed, _, _, _ = select_gmm_cores(
            Z_base, labels_fixed, degrees_excl_self=deg_base,
            alpha=restrict_alpha, gamma=restrict_gamma, B=Z_base.size(1)
        )

        core_t = core_mask_fixed.bool()
        lbl_t  = torch.from_numpy(labels_fixed.astype(np.int64)).to(core_t.device)

        g0_dense  = adj_label.to_dense()
        existing  = (g0_dense > 0)
        same_cl   = (lbl_t[:, None] == lbl_t[None, :]) & (lbl_t[:, None] != -1)
        both_core = (core_t[:, None] & core_t[None, :])

        # FIXED scope candidates (on ORIGINAL graph)
        if scope_name == "cp_all":
            cand = existing.clone()
        elif scope_name == "c0p_only":
            cand = existing & same_cl & both_core
        elif scope_name == "cp_minus_c0p":
            cand = existing & same_cl & (~both_core)
        else:
            raise ValueError(f"[REMOVE-ONLY-INIT] unknown scope_name={scope_name}")

        cand.fill_diagonal_(False)

        # candidate edge list (upper triangle)
        ii, jj = cand.triu(1).nonzero(as_tuple=True)
        E_scope0 = int(ii.numel())
        if E_scope0 == 0:
            print("[REMOVE-ONLY-INIT] no scope edges; disabling remove-only mode.")
            remove_state = {"disabled": True}
        else:
            # cosine similarity ranking from base embedding (remove LOW sim first)
            Zb = torch.nn.functional.normalize(Z_base.detach(), p=2, dim=1)
            sim = (Zb[ii] * Zb[jj]).sum(dim=1)  # [-1,1]
            order = torch.argsort(sim, descending=False)  # low-sim first

            ii = ii[order].to(torch.long)
            jj = jj[order].to(torch.long)

            target_remove_total = int(round((1.0 - keep_frac) * E_scope0))
            print(f"[REMOVE-ONLY-INIT] E_scope0={E_scope0} target_remove_total={target_remove_total}")

            remove_state = {
                "disabled": False,
                "labels_fixed": labels_fixed,              # numpy
                "core_mask_fixed": core_mask_fixed,        # torch bool
                "scope_name": scope_name,
                "keep_pct": keep_pct,
                "E_scope0": E_scope0,
                "target_remove_total": target_remove_total,
                "ii_rank": ii,                             # torch long
                "jj_rank": jj,                             # torch long
                "rank_ptr": 0,
                "removed_scope_so_far": 0,
            }

            print(f"[PRUNE] candidates={num_cand} step={step_edges} max={max_remove}")
    
    best_val_roc = -float("inf")
    best_epoch = -1
    best_state_cpu = None   # store on CPU to save GPU mem
    best_graph_dense_cpu = None
    best_meta_cpu = None    # best-by-val edit/compactness metadata for analysis
    best_val_ap = 0.0
    best_val_roc_subset = -float("inf")
    best_val_ap_subset = 0.0
    best_subset_epoch = -1
    best_checkpoint_score = -float("inf")
    best_subset_checkpoint_score = -float("inf")

    decoded_static_view_enabled = bool(
        use_decoded_graph_augment
        and (not decoded_accumulate_into_base)
        and freeze_c0p_at_edit_start
        and phase2_freeze_encoder
        and phase2_decoder_inference_only
    )
    static_decoded_aug_graph_dense = None
    static_decoded_aug_edge_index = None
    static_decoded_labels = None
    static_decoded_mask = None
    static_decoded_graph_added = 0
    static_decoded_graph_removed = 0
    static_decoded_built_epoch = None
    active_decoded_aug_graph_dense = None
    active_decoded_aug_edge_index = None
    active_decoded_labels = None
    active_decoded_mask = None
    active_decoded_rewrite_mask = None
    active_decoded_pair_context = None
    active_decoded_graph_added = 0
    active_decoded_graph_removed = 0
    active_decoded_built_epoch = None
    if decoded_static_view_enabled:
        print("[EDIT-GRAPH] static decoded view cache enabled (freeze targets + freeze encoder + decoder inference-only + temporary view)")

    decoded_structural_support_cache: dict[tuple, torch.Tensor] = {}
    decoded_structural_support_cache_logged: set[tuple] = set()
    decoded_structural_support_cache_limit = 8

    def _decoded_structural_support_cached(adj_current_dense: torch.Tensor) -> torch.Tensor | None:
        if not decoded_require_structural_support:
            return None
        shape = tuple(adj_current_dense.shape)
        if len(shape) != 2 or shape[0] != shape[1]:
            return None
        mode_key = (
            str(decoded_struct_support),
            float(decoded_struct_min_cn),
            float(decoded_struct_min_ra),
            float(decoded_struct_min_aa),
            str(adj_current_dense.device),
            int(shape[0]),
        )
        graph_key = _dense_graph_content_signature(adj_current_dense)
        key = mode_key + graph_key
        cached = decoded_structural_support_cache.get(key)
        if cached is not None:
            decoded_structural_support_cache.pop(key, None)
            decoded_structural_support_cache[key] = cached
        else:
            cached = _decoded_structural_support_mask(
                adj_current_dense,
                mode=decoded_struct_support,
                min_cn=decoded_struct_min_cn,
                min_ra=decoded_struct_min_ra,
                min_aa=decoded_struct_min_aa,
            ).detach()
            while len(decoded_structural_support_cache) >= decoded_structural_support_cache_limit:
                decoded_structural_support_cache.pop(next(iter(decoded_structural_support_cache)))
            decoded_structural_support_cache[key] = cached
            if key not in decoded_structural_support_cache_logged:
                decoded_structural_support_cache_logged.add(key)
                print(
                    f"[DECODED-STRUCT] cached support mode={decoded_struct_support} "
                    f"min_cn={decoded_struct_min_cn:.6f} min_ra={decoded_struct_min_ra:.6f} "
                    f"min_aa={decoded_struct_min_aa:.6f} graph_key={graph_key[0]} "
                    f"supported_pairs={int(cached.triu(1).sum().item())}"
                )
        return cached

    def _decoded_pair_context_for_graph(
        graph_dense: torch.Tensor,
        labels_np: np.ndarray | None,
        rewrite_mask_t: torch.Tensor | None,
        *,
        degree_floor: int,
        structural_support_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return _build_decoded_pair_context(
            graph_dense,
            labels_np,
            rewrite_mask_t,
            degree_floor=degree_floor,
            same_cluster_only=decoded_same_cluster_only,
            require_c0p_endpoint=decoded_require_c0p_endpoint,
            require_both_c0p=decoded_require_both_c0p,
            require_c0p_noncompact_endpoint=decoded_require_c0p_noncompact_endpoint,
            require_structural_support=decoded_require_structural_support,
            structural_support_mode=decoded_struct_support,
            structural_min_cn=decoded_struct_min_cn,
            structural_min_ra=decoded_struct_min_ra,
            structural_min_aa=decoded_struct_min_aa,
            structural_support_mask=structural_support_mask,
        )

    # optional: a convenient on-disk checkpoint path (safe default)
    ckpt_dir = os.path.join("checkpoints", dataset_str)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path_runtime = os.path.join(ckpt_dir, f"{ver}_{run_tag or 'run'}_best.pt")

    start_epoch = 0
    phase_cache_loaded = False
    phase_cache_saved = False
    if phase_cache_path:
        if phase_cache_epoch_arg >= 0:
            phase_cache_save_epoch = min(max(0, phase_cache_epoch_arg), max(0, num_epoch - 1))
        else:
            phase_cache_save_epoch = min(max(0, decoded_rewrite_start_epoch - 1), max(0, num_epoch - 1))
        if os.path.exists(phase_cache_path):
            try:
                print(f"[PHASE-CACHE] Loading phase cache from {phase_cache_path} (mode={phase_cache_load_mode})")
                phase_state = torch.load(phase_cache_path, map_location=device)

                def _module_state_compatible(module: nn.Module | None, state: dict | None) -> bool:
                    if module is None or state is None:
                        return False
                    try:
                        current = module.state_dict()
                        if set(current.keys()) != set(state.keys()):
                            return False
                        for key, value in current.items():
                            cached_value = state[key]
                            if tuple(value.shape) != tuple(cached_value.shape):
                                return False
                        return True
                    except Exception:
                        return False

                def _load_compatible_module(module: nn.Module | None, state: dict | None, name: str) -> bool:
                    if module is None or state is None:
                        return False
                    if phase_cache_load_mode == "full":
                        module.load_state_dict(state, strict=True)
                        return True
                    if _module_state_compatible(module, state):
                        module.load_state_dict(state, strict=True)
                        print(f"[PHASE-CACHE] Loaded compatible {name} weights")
                        return True
                    print(f"[PHASE-CACHE] Skipped incompatible {name} weights")
                    return False

                encoder.load_state_dict(phase_state["encoder"], strict=True)
                if phase_cache_load_mode == "encoder_only":
                    if phase_state.get("graph_decoder") is not None or phase_state.get("prediction_decoder") is not None:
                        print("[PHASE-CACHE] Reusing encoder/phase anchors only; decoder, optimizer, and best-state are reset")
                else:
                    _load_compatible_module(graph_decoder, phase_state.get("graph_decoder"), "graph_decoder")
                    _load_compatible_module(prediction_decoder, phase_state.get("prediction_decoder"), "prediction_decoder")
                    if phase_cache_load_mode == "full" and phase_state.get("optimizer") is not None:
                        try:
                            optimizer.load_state_dict(phase_state["optimizer"])
                        except Exception as e:
                            print(f"[PHASE-CACHE] optimizer load failed; rebuilding optimizer: {e}")
                            optimizer = _build_main_optimizer()
                    elif phase_cache_load_mode == "compatible":
                        print("[PHASE-CACHE] Compatible load keeps a fresh optimizer and resets best-state selection")

                if phase_cache_load_mode == "full":
                    if phase_state.get("best_state") is not None:
                        best_state_cpu = phase_state.get("best_state")
                    if phase_state.get("best_meta") is not None:
                        best_meta_cpu = phase_state.get("best_meta")
                    best_val_roc = float(phase_state.get("best_val_roc", best_val_roc))
                    best_val_ap = float(phase_state.get("best_val_ap", best_val_ap))
                    best_epoch = int(phase_state.get("best_epoch", best_epoch))
                    best_checkpoint_score = float(phase_state.get("best_checkpoint_score", best_checkpoint_score))
                    best_val_roc_subset = float(phase_state.get("best_val_roc_subset", best_val_roc_subset))
                    best_val_ap_subset = float(phase_state.get("best_val_ap_subset", best_val_ap_subset))
                    best_subset_epoch = int(phase_state.get("best_subset_epoch", best_subset_epoch))
                    best_subset_checkpoint_score = float(
                        phase_state.get("best_subset_checkpoint_score", best_subset_checkpoint_score)
                    )
                if phase_state.get("stage1_anchor_Z") is not None:
                    stage1_anchor_Z = phase_state["stage1_anchor_Z"].to(device=device)
                if phase_state.get("stage1_anchor_graph_dense") is not None:
                    stage1_anchor_graph_dense = phase_state["stage1_anchor_graph_dense"].to(device=device)
                if phase_state.get("fixed_c0p_labels") is not None:
                    fixed_c0p_labels = np.asarray(phase_state["fixed_c0p_labels"], dtype=np.int64)
                if phase_state.get("fixed_c0p_mask") is not None:
                    fixed_c0p_mask = phase_state["fixed_c0p_mask"].to(device=device).bool()
                if phase_state.get("frozen_gmm_labels") is not None:
                    frozen_gmm_labels = np.asarray(phase_state["frozen_gmm_labels"], dtype=np.int64)
                if phase_state.get("frozen_core_mask") is not None:
                    frozen_core_mask = np.asarray(phase_state["frozen_core_mask"], dtype=bool)
                stage1_phase_boundary_logged = bool(phase_state.get("stage1_phase_boundary_logged", False))
                loaded_epoch = int(phase_state.get("epoch", phase_cache_save_epoch))
                start_epoch = min(max(0, loaded_epoch + 1), num_epoch)
                phase_cache_loaded = True
                phase_cache_saved = True
                print(
                    f"[PHASE-CACHE] Loaded epoch={loaded_epoch}; resuming from epoch {start_epoch} "
                    f"| cached_best_epoch={best_epoch} cached_best_score={best_checkpoint_score:.6f}"
                )
            except Exception as e:
                print(f"[PHASE-CACHE] Load failed; training from epoch 0: {e}")
                start_epoch = 0
                phase_cache_loaded = False
                phase_cache_saved = False
        else:
            print(f"[PHASE-CACHE] No cache at {phase_cache_path}; will save after epoch {phase_cache_save_epoch}")
    else:
        phase_cache_save_epoch = -1

    aug_edge_index = edge_index
    aug_feat = features

    for epoch in tqdm(range(start_epoch, num_epoch)):
        t = time.time()
        t1 = time.time()
        added_this_epoch = 0
        removed_this_epoch = 0
        modification_ratio = 0.0
        decoded_rewrite_applied_this_epoch = False
        decoded_pre_graph_dense_epoch = None
        decoded_pre_rewrite_graph_dense_epoch = None
        decoded_metric_labels_epoch = None
        decoded_metric_mask_epoch = None
        decoded_rewrite_mask_epoch = None
        decoded_reuse_active_view_this_epoch = False
        edit_two_aug_cl_loss = None
        edit_two_aug_view_stats = {
            "active": 0,
            "jaccard": float("nan"),
            "v1_added": 0,
            "v1_removed": 0,
            "v2_added": 0,
            "v2_removed": 0,
            "v1_min_degree": float("nan"),
            "v2_min_degree": float("nan"),
            "degree_violations": 0,
            "constraint_add_violations": 0,
        }
        pull_push_diag_epoch = _empty_pull_push_diagnostics(
            push_scope=editor_push_scope,
            noncompact_push_strength=editor_noncompact_push_strength,
            noise_push_strength=editor_noise_push_strength,
            preserve_norm=editor_push_preserve_norm,
        )
        encoder.train()
        if graph_decoder is not None:
            graph_decoder.train()
        if prediction_decoder is not None:
            prediction_decoder.train()
        #print(f"trn time1 {time.time()-t1:.2f} s", flush=True)
        optimizer.zero_grad()
        
        Z = encoder(features, edge_index) # Z = encoder(features, adj_norm)
        hidden_repr = encoder.Z
        in_edit_phase = _in_edit_phase(epoch)

        if use_edited_decoder and _in_edit_phase(epoch) and stage1_anchor_Z is None:
            stage1_anchor_Z = Z.detach().clone()
            stage1_anchor_graph_dense = _to_dense(adj_label).detach().clone()
            if use_edited_decoder and freeze_c0p_at_edit_start and fixed_c0p_mask is None:
                try:
                    fixed_c0p_labels = gmm_labels(Z.detach(), K=gmm_k, tau=gmm_tau, metric="cosine")
                    deg_for_c0p = _deg_excl_self(adj_label.to_dense())
                    fixed_c0p_mask, _, _, _ = select_gmm_cores(
                        Z.detach(),
                        fixed_c0p_labels,
                        degrees_excl_self=deg_for_c0p,
                        alpha=restrict_alpha,
                        gamma=restrict_gamma,
                        B=Z.size(1),
                    )
                    fixed_c0p_mask = fixed_c0p_mask.bool().detach()

                    # Persist the exact edit-start frozen targets for downstream analysis / TSNE.
                    # This keeps final visualization aligned with the targets actually used in phase 2.
                    frozen_gmm_labels = np.asarray(fixed_c0p_labels, dtype=np.int64).copy()
                    frozen_core_mask = fixed_c0p_mask.detach().cpu().numpy().astype(bool)

                    print(f"[EDIT] fixed c0p size = {int(fixed_c0p_mask.sum().item())}")
                except Exception as e:
                    print(f"[EDIT] failed to initialize c0p mask: {e}")
                    fixed_c0p_labels = None
                    fixed_c0p_mask = torch.zeros(Z.size(0), dtype=torch.bool, device=device)
            if separate_edit_training and not phase2_optimizer_activated:
                if phase2_decoder_inference_only:
                    _set_module_requires_grad(graph_decoder, False)
                    _set_module_requires_grad(prediction_decoder, True)
                    phase2_params = _module_params(prediction_decoder)
                    if (not phase2_freeze_encoder) and edit_phase_encoder_lr_scale > 0.0:
                        _set_module_requires_grad(encoder, True)
                        groups = [
                            {"params": list(encoder.parameters()), "lr": learning_rate * edit_phase_encoder_lr_scale, "weight_decay": weight_decay},
                        ]
                        if phase2_params:
                            groups.append({"params": phase2_params, "lr": learning_rate, "weight_decay": weight_decay})
                        optimizer = Adam(groups)
                    elif phase2_params:
                        _set_module_requires_grad(encoder, False)
                        optimizer = Adam(phase2_params, lr=learning_rate, weight_decay=weight_decay)
                else:
                    optimizer = _build_phase2_optimizer()
                phase2_optimizer_activated = True
                if not stage1_phase_boundary_logged:
                    print(
                        f"[EDIT-PHASE] entering phase-2 at epoch {epoch} | "
                        f"retain_recon={edit_phase_retain_recon_weight} retain_cl={edit_phase_retain_cl_weight} | "
                        f"freeze_encoder={phase2_freeze_encoder} encoder_lr_scale={edit_phase_encoder_lr_scale} | "
                        f"task_main={phase2_task_main_loss} edit_w={edit_phase_edit_weight} | "
                        f"decoder_inference_only={phase2_decoder_inference_only}"
                    )
                    stage1_phase_boundary_logged = True

        # original loss
        A_pred = dot_product_decode(Z)

                # adjusted_weight_tensor = weight_tensor * torch.abs(A_pred.view(-1)[train_mask] - adj_label.to_dense().view(-1)[train_mask]).detach()
                # recon_loss = loss_function(A_pred, adj_label, encoder.mean, encoder.logstd, norm, adjusted_weight_tensor, alpha, beta, train_mask)
        (
            recon_loss,
            maskgae_feat_loss,
            cimage_factor_loss,
            cimage_cluster_loss,
        ) = _encoder_reconstruction_loss(A_pred)
        if cl_mode == "edit_two_aug":
            ori_intra_CL = Z.new_tensor(0.0)
        elif(loss_ver=="nei"):
            ori_intra_CL = inter_view_CL_loss(device, Z, Z, adj_label, gamma, temperature)
        else:
            ori_intra_CL = intra_view_CL_loss(device, Z, adj_label, gamma, temperature)
        loss = recon_loss + ori_intra_CL

        decoded_labels_epoch = None
        decoded_mask_epoch = None
        
        # Generate K graphs
        graph_refresh_due = (epoch % 10 == 0) or (use_decoded_graph_augment and _decoded_rewrite_due(epoch))
        if graph_refresh_due:
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
                added_this_epoch = 0
                removed_this_epoch = 0
                aug_feat = drop_feature(features.to_dense(), feat_maske_ratio)     
            # ***** NEW: frozen-score A/B variants *****
            elif ver in ("aron_desc", "aron_asc"):
                degree_floor = max(0, int(degree_threshold) - 1)   # align with excl-self
                order = "desc" if ver == "aron_desc" else "asc"
                
                removed_this_epoch = 0 # dummy

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

                score_thr = robust_score_quantile_from_scores(scores_for_rank, base_block, q=late_score_q, max_elems=2_000_000)


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
                # Parse from `ver`
                # We keep your "kind" parsing: remove_only_(intra|inter|both)
                # Scope parsing is optional; actual scope is enforced by remove_state init.
                #
                # Accept examples like:
                #   remove_only_intra_c0p_keep90
                #   remove_only_both_cp_all_keep75
                #   remove_only_inter_keep50
                # ------------------------------------------------------------
                m = re.match(r"^remove_only_(intra|inter|both)(?:_([a-zA-Z0-9_]+))?(?:_keep(\d+))?$", ver)
                if not m:
                    print(f"[REMOVE-ONLY] Unrecognized ver={ver}; skipping this epoch.")
                    g = _to_dense(adj_label)
                    aug_feat = features if feat_maske_ratio <= 0 else drop_feature(features, feat_maske_ratio)
                else:
                    kind = m.group(1)                     # "intra" | "inter" | "both"
                    scope_tag = m.group(2) or "scope"     # only for printing/debug
                    keep_pct = int(m.group(3)) if m.group(3) else None

                    # ---- FIXED cluster labels (GMM only) ----
                    if (remove_state is None) or remove_state.get("disabled", False):
                        labels_cluster = np.full((num_nodes,), -1, dtype=np.int64)
                    else:
                        labels_cluster = remove_state["labels_fixed"]

                    # ---- compactness BEFORE ----
                    radii, _, _ = per_cluster_stats_diag(Z.detach(), labels_cluster, normalize_cosine=True)
                    if (labels_cluster != -1).sum() > 0:
                        mean_radius = float(radii[torch.from_numpy(labels_cluster) != -1].mean().item())
                    else:
                        mean_radius = float(radii.mean().item())
                    print(f"[RADIUS] epoch={epoch} mean={mean_radius:.6f}")
                    radius_hist.append(mean_radius)
                    radius_epoch_hist.append(epoch)

                    # ---- EXACT keep% quota schedule ----
                    # (linear schedule to reach target_remove_total by final epoch)
                    if (remove_state is None) or remove_state.get("disabled", False):
                        epoch_quota_edges = 0
                        target_total = 0
                        removed_so_far = 0
                        E_scope0 = 0
                    else:
                        target_total = int(remove_state["target_remove_total"])
                        removed_so_far = int(remove_state["removed_scope_so_far"])
                        E_scope0 = int(remove_state["E_scope0"])

                        target_cum = int(round(target_total * float(epoch + 1) / max(1, num_epoch)))
                        epoch_quota_edges = max(0, target_cum - removed_so_far)

                    # helpful print
                    if keep_pct is None and (remove_state is not None) and (not remove_state.get("disabled", False)):
                        keep_pct = int(remove_state.get("keep_pct", -1))
                    print(
                        f"[REMOVE-ONLY] kind={kind} scope_tag={scope_tag} keep={keep_pct}% "
                        f"| epoch_quota_remove={epoch_quota_edges} "
                        f"| removed_so_far={removed_so_far}/{target_total} "
                        f"| E_scope0={E_scope0}"
                    )

                    dropped_intra = 0
                    dropped_inter = 0

                    # ---- Execute removals (exact count if possible) ----
                    removed_this_epoch = 0
                    if epoch_quota_edges > 0 and (remove_state is not None) and (not remove_state.get("disabled", False)):

                        g_dense = _to_dense(adj_label)
                        degree_floor = max(0, int(degree_threshold) - 1)
                        deg_now = _deg_excl_self(g_dense)

                        lbl_t = torch.from_numpy(labels_cluster).to(g_dense.device)
                        # intra if same label and label != -1
                        def _is_intra(i: int, j: int) -> bool:
                            li = int(lbl_t[i].item())
                            lj = int(lbl_t[j].item())
                            return (li != -1) and (li == lj)

                        ii_rank = remove_state["ii_rank"]
                        jj_rank = remove_state["jj_rank"]
                        ptr = int(remove_state["rank_ptr"])

                        removed = 0
                        while removed < epoch_quota_edges and ptr < ii_rank.numel():
                            i = int(ii_rank[ptr].item())
                            j = int(jj_rank[ptr].item())
                            ptr += 1

                            # still exists?
                            if g_dense[i, j].item() <= 0:
                                continue

                            # kind filter
                            intra_flag = _is_intra(i, j)
                            if kind == "intra" and (not intra_flag):
                                continue
                            if kind == "inter" and intra_flag:
                                continue

                            # degree-floor safety
                            if (deg_now[i].item() - 1) < degree_floor:
                                continue
                            if (deg_now[j].item() - 1) < degree_floor:
                                continue

                            # remove undirected edge
                            g_dense[i, j] = 0
                            g_dense[j, i] = 0
                            deg_now[i] -= 1
                            deg_now[j] -= 1

                            removed += 1
                            if intra_flag:
                                dropped_intra += 1
                            else:
                                dropped_inter += 1

                        # write back
                        adj_label = g_dense

                        # update remove-only state
                        remove_state["rank_ptr"] = ptr
                        remove_state["removed_scope_so_far"] = int(remove_state["removed_scope_so_far"] + removed)
                        removed_this_epoch = removed

                        # remaining scope edges
                        E_scope0 = int(remove_state["E_scope0"])
                        remain = max(0, E_scope0 - int(remove_state["removed_scope_so_far"]))
                        remain_pct = 100.0 * remain / max(1, E_scope0)
                        print(
                            f"[REMOVE-ONLY] removed_this_epoch={removed_this_epoch} "
                            f"(intra={dropped_intra}, inter={dropped_inter}) | "
                            f"removed_scope_so_far={remove_state['removed_scope_so_far']}/{remove_state['target_remove_total']} | "
                            f"remaining_scope={remain} ({remain_pct:.2f}%)"
                        )

                    # ---- bookkeeping: adds=0 in remove-only ----
                    added_this_epoch = 0
                    global_used_adds     += 0
                    global_used_removals += removed_this_epoch

                    # per-epoch modification ratio (relative to ORIGINAL undirected |E| = E0)
                    modification_ratio = (added_this_epoch + removed_this_epoch) / float(max(1, E0))

                    print(
                        f"[BUDGET] add/remove this epoch = {added_this_epoch}/{removed_this_epoch} "
                        f"(intra={dropped_intra}, inter={dropped_inter}) | "
                        f"global_used adds/removes = {global_used_adds}/{global_used_removals}"
                    )

                    # keep feature aug behavior consistent with others
                    aug_feat = features if feat_maske_ratio <= 0 else drop_feature(features, feat_maske_ratio)

                    # maintain `g` mirror like other branches
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
                            added_this_epoch = 0
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
            decoded_labels_epoch = None
            decoded_mask_epoch = None
            decoded_graph_added = 0
            decoded_graph_removed = 0
            if (
                use_decoded_graph_augment
                and _decoded_edit_active(epoch)
                and (not _decoded_rewrite_due(epoch))
                and active_decoded_aug_graph_dense is not None
            ):
                g = active_decoded_aug_graph_dense
                aug_edge_index = (
                    active_decoded_aug_edge_index
                    if active_decoded_aug_edge_index is not None
                    else g.to_sparse().indices()
                )
                decoded_labels_epoch = active_decoded_labels
                decoded_mask_epoch = active_decoded_mask
                decoded_rewrite_mask_epoch = active_decoded_rewrite_mask
                decoded_reuse_active_view_this_epoch = True
                if epoch % max(1, eval_log_every) == 0:
                    print(
                        f"[EDIT-GRAPH][E{epoch:04d}] reuse_active=1 built_epoch={active_decoded_built_epoch} "
                        f"rewrite_every={decoded_rewrite_every} add={active_decoded_graph_added} "
                        f"remove={active_decoded_graph_removed}"
                    )
            else:
                aug_edge_index = g.to_sparse().indices()

            # ★★ 只有 remove_only_* 和 prune_* 才更新 base view graph
            if (ver.startswith("remove_only_")) or (ver.startswith("prune_")):
                edge_index = aug_edge_index

            if use_decoded_graph_augment and _decoded_rewrite_due(epoch):
                try:
                    decoded_pre_rewrite_graph_dense_epoch = g.detach().clone()
                    decoded_degree_floor_eff = max(0, int(degree_threshold) - 1) if decoded_degree_floor is None else int(decoded_degree_floor)
                    decoded_bound_eff = None if (decoded_graph_aug_bound is None or float(decoded_graph_aug_bound) <= 0) else float(decoded_graph_aug_bound)

                    if decoded_static_view_enabled and static_decoded_aug_edge_index is not None:
                        g = static_decoded_aug_graph_dense.clone()
                        aug_edge_index = static_decoded_aug_edge_index
                        decoded_labels_epoch = static_decoded_labels
                        decoded_mask_epoch = static_decoded_mask
                        decoded_graph_added = static_decoded_graph_added
                        decoded_graph_removed = static_decoded_graph_removed
                        decoded_rewrite_applied_this_epoch = True
                        print(
                            f"[EDIT-GRAPH][E{epoch:04d}] reuse_static=1 built_epoch={static_decoded_built_epoch} "
                            f"add={decoded_graph_added} remove={decoded_graph_removed} "
                            f"same_cluster_only={decoded_same_cluster_only} c0p_endpoint={decoded_require_c0p_endpoint} "
                            f"c0p_noncompact_endpoint={decoded_require_c0p_noncompact_endpoint} "
                            f"per_node_cap={decoded_bound_eff} add_degree_target={decoded_add_degree_target} "
                            f"add_degree_target_scope={decoded_add_degree_target_scope} "
                            f"add_degree_target_nodes={decoded_add_degree_target_nodes} "
                            f"guarantee_degree_target={decoded_guarantee_degree_target} accumulate_base={decoded_accumulate_into_base}"
                        )
                    else:
                        if decoded_accumulate_into_base:
                            decoded_pre_graph_dense_epoch = g.clone().detach()
                        decoded_labels_epoch, decoded_mask_epoch = resolve_edit_targets(
                            Z,
                            _to_dense(adj_label),
                            freeze_targets=freeze_c0p_at_edit_start,
                            fixed_labels=fixed_c0p_labels,
                            fixed_mask=fixed_c0p_mask,
                            gmm_k=gmm_k,
                            gmm_tau=gmm_tau,
                            restrict_alpha=restrict_alpha,
                            restrict_gamma=restrict_gamma,
                        )
                        decoded_c0p_mask_epoch = decoded_mask_epoch
                        decoded_cp_mask_epoch = torch.tensor((decoded_labels_epoch != -1), dtype=torch.bool, device=Z.device) if decoded_labels_epoch is not None else None
                        decoded_pull_mask_epoch = decoded_c0p_mask_epoch if pull_mask_scope == "c0p" else decoded_cp_mask_epoch
                        decoded_rewrite_mask_epoch = decoded_c0p_mask_epoch if rewrite_endpoint_scope == "c0p" else decoded_cp_mask_epoch
                        z_pull_seed, pull_push_diag_epoch = direct_pull_latent_per_cluster(
                            Z.detach(),
                            decoded_labels_epoch,
                            decoded_pull_mask_epoch,
                            pull_strength=editor_pull_strength,
                            c0p_mask=decoded_c0p_mask_epoch,
                            pull_profile=editor_pull_profile,
                            pull_tau=editor_pull_tau,
                            pull_deadzone=editor_pull_deadzone,
                            pull_anchor=editor_pull_anchor,
                            push_scope=editor_push_scope,
                            noncompact_push_strength=editor_noncompact_push_strength,
                            noise_push_strength=editor_noise_push_strength,
                            push_preserve_norm=editor_push_preserve_norm,
                            return_diagnostics=True,
                        )
                        print(
                            f"[PULL-PUSH][E{epoch:04d}] scope={editor_push_scope} "
                            f"pull_scope={pull_mask_scope} pull_strength={editor_pull_strength:.6f} "
                            f"pull_profile={editor_pull_profile} pull_tau={editor_pull_tau:.6f} "
                            f"pull_deadzone={editor_pull_deadzone:.6f} pull_anchor={editor_pull_anchor} "
                            f"noncompact_push={editor_noncompact_push_strength:.6f} noise_push={editor_noise_push_strength:.6f} "
                            f"preserve_norm={int(editor_push_preserve_norm)} "
                            f"noncompact_count={int(pull_push_diag_epoch['push_noncompact_count'])} "
                            f"noise_count={int(pull_push_diag_epoch['push_noise_count'])} "
                            f"noncompact_cosdist_before={pull_push_diag_epoch['push_noncompact_anchor_cosdist_before']:.6f} "
                            f"noncompact_cosdist_after={pull_push_diag_epoch['push_noncompact_anchor_cosdist_after']:.6f} "
                            f"noise_cosdist_before={pull_push_diag_epoch['push_noise_anchor_cosdist_before']:.6f} "
                            f"noise_cosdist_after={pull_push_diag_epoch['push_noise_anchor_cosdist_after']:.6f}"
                        )
                        decoder_graph_context = g
                        decoded_scorer = _decoded_graph_scorer()
                        if decoded_scorer is None:
                            raise RuntimeError("decoded graph augment requires an edit decoder or prediction decoder scorer")
                        _set_decoded_graph_scorer_context(decoder_graph_context, decoded_labels_epoch, decoded_c0p_mask_epoch)
                        decoded_structural_support_epoch = _decoded_structural_support_cached(g)
                        decoded_pair_context_epoch = _decoded_pair_context_for_graph(
                            g,
                            decoded_labels_epoch,
                            decoded_rewrite_mask_epoch,
                            degree_floor=decoded_degree_floor_eff,
                            structural_support_mask=decoded_structural_support_epoch,
                        )
                        if isinstance(decoded_scorer, (MLPPairGraphDecoder, StructuralPairGraphDecoder)):
                            g_decoded, decoded_graph_added, decoded_graph_removed = build_decoded_augmented_graph_from_decoder(
                                decoded_scorer,
                                z_pull_seed,
                                g,
                                decoded_labels_epoch,
                                decoded_rewrite_mask_epoch,
                                E0=E0,
                                add_ratio=decoded_add_ratio,
                                remove_ratio=decoded_remove_ratio,
                                add_threshold=decoded_add_threshold,
                                remove_threshold=decoded_remove_threshold,
                                add_quantile=decoded_add_quantile,
                                remove_quantile=decoded_remove_quantile,
                                max_add=decoded_max_add_per_round,
                                max_remove=decoded_max_remove_per_round,
                                per_node_cap_frac=decoded_bound_eff,
                                add_degree_target=decoded_add_degree_target,
                                add_degree_target_scope=decoded_add_degree_target_scope,
                                add_degree_target_nodes=decoded_add_degree_target_nodes,
                                guarantee_degree_target=decoded_guarantee_degree_target,
                                deg0_excl_self=None,
                                degree_floor=decoded_degree_floor_eff,
                                same_cluster_only=decoded_same_cluster_only,
                                require_c0p_endpoint=decoded_require_c0p_endpoint,
                                require_both_c0p=decoded_require_both_c0p,
                                require_c0p_noncompact_endpoint=decoded_require_c0p_noncompact_endpoint,
                                require_structural_support=decoded_require_structural_support,
                                structural_support_mode=decoded_struct_support,
                                structural_min_cn=decoded_struct_min_cn,
                                structural_min_ra=decoded_struct_min_ra,
                                structural_min_aa=decoded_struct_min_aa,
                                structural_support_mask=decoded_structural_support_epoch,
                                pair_context=decoded_pair_context_epoch,
                            )
                        else:
                            with torch.no_grad():
                                decoded_scores = decoded_scorer(z_pull_seed).detach()
                            g_decoded, decoded_graph_added, decoded_graph_removed = build_decoded_augmented_graph(
                                decoded_scores,
                                g,
                                decoded_labels_epoch,
                                decoded_rewrite_mask_epoch,
                                E0=E0,
                                add_ratio=decoded_add_ratio,
                                remove_ratio=decoded_remove_ratio,
                                add_threshold=decoded_add_threshold,
                                remove_threshold=decoded_remove_threshold,
                                add_quantile=decoded_add_quantile,
                                remove_quantile=decoded_remove_quantile,
                                max_add=decoded_max_add_per_round,
                                max_remove=decoded_max_remove_per_round,
                                per_node_cap_frac=decoded_bound_eff,
                                add_degree_target=decoded_add_degree_target,
                                add_degree_target_scope=decoded_add_degree_target_scope,
                                add_degree_target_nodes=decoded_add_degree_target_nodes,
                                guarantee_degree_target=decoded_guarantee_degree_target,
                                deg0_excl_self=None,
                                degree_floor=decoded_degree_floor_eff,
                                same_cluster_only=decoded_same_cluster_only,
                                require_c0p_endpoint=decoded_require_c0p_endpoint,
                                require_both_c0p=decoded_require_both_c0p,
                                require_c0p_noncompact_endpoint=decoded_require_c0p_noncompact_endpoint,
                                require_structural_support=decoded_require_structural_support,
                                structural_support_mode=decoded_struct_support,
                                structural_min_cn=decoded_struct_min_cn,
                                structural_min_ra=decoded_struct_min_ra,
                                structural_min_aa=decoded_struct_min_aa,
                                structural_support_mask=decoded_structural_support_epoch,
                                pair_context=decoded_pair_context_epoch,
                            )
                        g = g_decoded
                        aug_edge_index = g.to_sparse().indices()
                        active_decoded_aug_graph_dense = g.detach()
                        active_decoded_aug_edge_index = aug_edge_index.detach().clone()
                        active_decoded_labels = None if decoded_labels_epoch is None else np.asarray(decoded_labels_epoch, dtype=np.int64).copy()
                        active_decoded_mask = None if decoded_c0p_mask_epoch is None else decoded_c0p_mask_epoch.detach().clone()
                        active_decoded_rewrite_mask = None if decoded_rewrite_mask_epoch is None else decoded_rewrite_mask_epoch.detach().clone()
                        active_decoded_graph_added = int(decoded_graph_added)
                        active_decoded_graph_removed = int(decoded_graph_removed)
                        active_decoded_built_epoch = int(epoch)
                        active_structural_support_epoch = _decoded_structural_support_cached(active_decoded_aug_graph_dense)
                        active_decoded_pair_context = _decoded_pair_context_for_graph(
                            active_decoded_aug_graph_dense,
                            active_decoded_labels,
                            active_decoded_rewrite_mask,
                            degree_floor=decoded_degree_floor_eff,
                            structural_support_mask=active_structural_support_epoch,
                        )
                        decoded_rewrite_applied_this_epoch = True
                        _audit_decoded_rewrite_quality(
                            epoch,
                            decoded_pre_rewrite_graph_dense_epoch,
                            g,
                            decoded_labels_epoch,
                            decoded_c0p_mask_epoch,
                            z_pull_seed,
                            decoded_scorer,
                        )

                        if decoded_static_view_enabled and not decoded_accumulate_into_base:
                            static_decoded_aug_graph_dense = g.detach().clone()
                            static_decoded_aug_edge_index = aug_edge_index.detach().clone()
                            static_decoded_labels = None if decoded_labels_epoch is None else np.asarray(decoded_labels_epoch, dtype=np.int64).copy()
                            static_decoded_mask = None if decoded_mask_epoch is None else decoded_mask_epoch.detach().clone()
                            static_decoded_graph_added = int(decoded_graph_added)
                            static_decoded_graph_removed = int(decoded_graph_removed)
                            static_decoded_built_epoch = int(epoch)
                            print(
                                f"[EDIT-GRAPH][E{epoch:04d}] cached_static_view=1 add={decoded_graph_added} remove={decoded_graph_removed} "
                                f"same_cluster_only={decoded_same_cluster_only} c0p_endpoint={decoded_require_c0p_endpoint} "
                                f"c0p_noncompact_endpoint={decoded_require_c0p_noncompact_endpoint} "
                                f"per_node_cap={decoded_bound_eff} add_degree_target={decoded_add_degree_target} "
                                f"add_degree_target_scope={decoded_add_degree_target_scope} "
                                f"add_degree_target_nodes={decoded_add_degree_target_nodes} "
                                f"guarantee_degree_target={decoded_guarantee_degree_target} accumulate_base={decoded_accumulate_into_base}"
                            )

                        if decoded_accumulate_into_base:
                            adj_train, edge_index, adj_norm, adj_label, pos_weight, norm, weight_tensor = _rebuild_train_graph_state_from_dense(g)
                            _set_struct_decoder_context(adj_train, decoded_labels_epoch, decoded_c0p_mask_epoch)
                            decoded_metric_labels_epoch = decoded_labels_epoch
                            decoded_metric_mask_epoch = decoded_c0p_mask_epoch
                        added_this_epoch += int(decoded_graph_added)
                        removed_this_epoch += int(decoded_graph_removed)
                        modification_ratio = (added_this_epoch + removed_this_epoch) / float(max(1, E0))
                        print(
                            f"[EDIT-GRAPH][E{epoch:04d}] add={decoded_graph_added} remove={decoded_graph_removed} "
                            f"same_cluster_only={decoded_same_cluster_only} c0p_endpoint={decoded_require_c0p_endpoint} "
                            f"c0p_noncompact_endpoint={decoded_require_c0p_noncompact_endpoint} "
                            f"per_node_cap={decoded_bound_eff} add_degree_target={decoded_add_degree_target} "
                            f"add_degree_target_scope={decoded_add_degree_target_scope} "
                            f"add_degree_target_nodes={decoded_add_degree_target_nodes} "
                            f"guarantee_degree_target={decoded_guarantee_degree_target} accumulate_base={decoded_accumulate_into_base} "
                            f"add_thr={decoded_add_threshold} remove_thr={decoded_remove_threshold} "
                            f"add_q={decoded_add_quantile} remove_q={decoded_remove_quantile} "
                            f"max_add={decoded_max_add_per_round} max_remove={decoded_max_remove_per_round}"
                        )
                except Exception as e:
                    print(f"[EDIT-GRAPH][TRAIN] decoded graph rewrite failed at epoch {epoch}: {e}")

            if decoded_rewrite_applied_this_epoch and decoded_pre_rewrite_graph_dense_epoch is not None:
                try:
                    diff_pre = _graph_diff_summary(
                        decoded_pre_rewrite_graph_dense_epoch,
                        g,
                        decoded_labels_epoch,
                        decoded_mask_epoch,
                    )
                    print(
                        f"[GRAPH-DIFF][E{epoch:04d}] scope=pre_rewrite "
                        f"base_edges={int(diff_pre['base_edges'])} view_edges={int(diff_pre['view_edges'])} "
                        f"added_vs_base={int(diff_pre['added_vs_base'])} removed_vs_base={int(diff_pre['removed_vs_base'])} "
                        f"symdiff_edges={int(diff_pre['symdiff_edges'])} "
                        f"edge_jaccard={diff_pre['edge_jaccard']:.6f} diff_frac_base={diff_pre['diff_frac_base']:.6f} "
                        f"add_frac_base={diff_pre['add_frac_base']:.6f} remove_frac_base={diff_pre['remove_frac_base']:.6f} "
                        f"same_cluster_added={diff_pre['same_cluster_added']:.6f} same_cluster_removed={diff_pre['same_cluster_removed']:.6f} "
                        f"cross_cluster_added={diff_pre['cross_cluster_added']:.6f} cross_cluster_removed={diff_pre['cross_cluster_removed']:.6f} "
                        f"target_touch_added={diff_pre['target_touch_added']:.6f} target_touch_removed={diff_pre['target_touch_removed']:.6f} "
                        f"add_budget_ratio={decoded_add_ratio:.6f} remove_budget_ratio={decoded_remove_ratio:.6f} "
                        f"accumulate_base={decoded_accumulate_into_base}"
                    )
                    if decoded_accumulate_into_base and stage1_anchor_graph_dense is not None:
                        diff_start = _graph_diff_summary(
                            stage1_anchor_graph_dense,
                            g,
                            decoded_labels_epoch,
                            decoded_mask_epoch,
                        )
                        print(
                            f"[GRAPH-DIFF][E{epoch:04d}] scope=edit_start "
                            f"base_edges={int(diff_start['base_edges'])} view_edges={int(diff_start['view_edges'])} "
                            f"added_vs_base={int(diff_start['added_vs_base'])} removed_vs_base={int(diff_start['removed_vs_base'])} "
                            f"symdiff_edges={int(diff_start['symdiff_edges'])} "
                            f"edge_jaccard={diff_start['edge_jaccard']:.6f} diff_frac_base={diff_start['diff_frac_base']:.6f} "
                            f"add_frac_base={diff_start['add_frac_base']:.6f} remove_frac_base={diff_start['remove_frac_base']:.6f} "
                            f"same_cluster_added={diff_start['same_cluster_added']:.6f} same_cluster_removed={diff_start['same_cluster_removed']:.6f} "
                            f"cross_cluster_added={diff_start['cross_cluster_added']:.6f} cross_cluster_removed={diff_start['cross_cluster_removed']:.6f} "
                            f"target_touch_added={diff_start['target_touch_added']:.6f} target_touch_removed={diff_start['target_touch_removed']:.6f} "
                            f"add_budget_ratio={decoded_add_ratio:.6f} remove_budget_ratio={decoded_remove_ratio:.6f} "
                            f"accumulate_base={decoded_accumulate_into_base}"
                        )
                except Exception as e:
                    print(f"[GRAPH-DIFF][E{epoch:04d}] graph-diff summary failed: {e}")
            
            # Aron (for minimum node degree)
            # Node degree calculation (ignoring self-loops)
            node_degrees = torch.sum(g, dim=1) - torch.diag(g)

            # Calculate minimum and average node degrees
            min_degree = torch.min(node_degrees)
            mean_degree = torch.mean(node_degrees.float())
            minimum_node_degree_history.append(min_degree.item())
            degree_cp_min = float("nan")
            degree_cp_mean = float("nan")
            degree_c0p_min = float("nan")
            degree_c0p_mean = float("nan")
            degree_cluster_line = None
            try:
                if decoded_labels_epoch is not None:
                    degree_cp_mask = torch.as_tensor(
                        np.asarray(decoded_labels_epoch) != -1,
                        dtype=torch.bool,
                        device=node_degrees.device,
                    )
                    if bool(degree_cp_mask.any().item()):
                        cp_degrees = node_degrees[degree_cp_mask]
                        degree_cp_min = float(cp_degrees.min().item())
                        degree_cp_mean = float(cp_degrees.float().mean().item())
                if decoded_mask_epoch is not None:
                    degree_c0p_mask = decoded_mask_epoch.to(node_degrees.device).bool()
                    if bool(degree_c0p_mask.any().item()):
                        c0p_degrees = node_degrees[degree_c0p_mask]
                        degree_c0p_min = float(c0p_degrees.min().item())
                        degree_c0p_mean = float(c0p_degrees.float().mean().item())
                if decoded_labels_epoch is not None:
                    degree_target_for_summary = int(decoded_add_degree_target) if int(decoded_add_degree_target) > 0 else -1
                    cp_cluster_summary = _cluster_min_degree_summary(
                        g,
                        decoded_labels_epoch,
                        torch.as_tensor(
                            np.asarray(decoded_labels_epoch) != -1,
                            dtype=torch.bool,
                            device=node_degrees.device,
                        ),
                        target=degree_target_for_summary,
                    )
                    c0p_cluster_summary = _cluster_min_degree_summary(
                        g,
                        decoded_labels_epoch,
                        decoded_mask_epoch,
                        target=degree_target_for_summary,
                    )
                    degree_cluster_line = (
                        f"[DEGREE-CLUSTER][E{epoch:04d}] target={degree_target_for_summary} "
                        f"cp_clusters={int(cp_cluster_summary['clusters'])} "
                        f"cp_worst_min={cp_cluster_summary['worst_min']:.6f} "
                        f"cp_mean_min={cp_cluster_summary['mean_min']:.6f} "
                        f"cp_need_nodes={int(cp_cluster_summary['need_nodes'])} "
                        f"cp_bad_clusters={int(cp_cluster_summary['bad_clusters'])} "
                        f"c0p_clusters={int(c0p_cluster_summary['clusters'])} "
                        f"c0p_worst_min={c0p_cluster_summary['worst_min']:.6f} "
                        f"c0p_mean_min={c0p_cluster_summary['mean_min']:.6f} "
                        f"c0p_need_nodes={int(c0p_cluster_summary['need_nodes'])} "
                        f"c0p_bad_clusters={int(c0p_cluster_summary['bad_clusters'])} "
                        f"added_this_epoch={int(added_this_epoch)} removed_this_epoch={int(removed_this_epoch)}"
                    )
            except Exception as e:
                print(f"[DEGREE][E{epoch:04d}] degree-scope summary failed: {e}")
            print(
                f"[DEGREE][E{epoch:04d}] "
                f"min_degree={float(min_degree.item()):.6f} mean_degree={float(mean_degree.item()):.6f} "
                f"c0p_min_degree={degree_c0p_min:.6f} c0p_mean_degree={degree_c0p_mean:.6f} "
                f"cp_min_degree={degree_cp_min:.6f} cp_mean_degree={degree_cp_mean:.6f} "
                f"added_this_epoch={int(added_this_epoch)} removed_this_epoch={int(removed_this_epoch)}"
            )
            if degree_cluster_line is not None:
                print(degree_cluster_line)

        modification_ratio_history.append(modification_ratio)
        add_hist.append(int(added_this_epoch))
        remove_hist.append(int(removed_this_epoch))

        if (
            use_decoded_graph_augment
            and _decoded_edit_active(epoch)
            and active_decoded_aug_graph_dense is not None
            and decoded_labels_epoch is None
        ):
            decoded_labels_epoch = active_decoded_labels
            decoded_mask_epoch = active_decoded_mask
            decoded_rewrite_mask_epoch = active_decoded_rewrite_mask

        # Calcualte Augment View
        # bias_Z = encoder(features, aug_edge_index)
        try:
            bias_Z = encoder(aug_feat, aug_edge_index)
        except RuntimeError as e:
            if not (skip_oom_epoch and _is_cuda_oom_like(e)):
                raise
            print(f"[OOM-FALLBACK] epoch={epoch} bias_Z=reuse_Z: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Reuse the main-view latent when the augmented pass cannot fit.
            # This keeps the epoch alive for stability-focused reruns.
            bias_Z = Z

        edit_two_aug_cl_loss = bias_Z.new_tensor(0.0)
        cross_view_loss = bias_Z.new_tensor(0.0)
        if cl_mode == "edit_two_aug":
            if (
                use_decoded_graph_augment
                and _decoded_edit_active(epoch)
                and active_decoded_aug_graph_dense is not None
            ):
                try:
                    cl_graph_dense = active_decoded_aug_graph_dense
                    cl_degree_floor = max(0, int(degree_threshold) - 1) if decoded_degree_floor is None else int(decoded_degree_floor)
                    cl_pair_context = active_decoded_pair_context
                    if cl_pair_context is None:
                        cl_struct_support = _decoded_structural_support_cached(cl_graph_dense)
                        cl_pair_context = _decoded_pair_context_for_graph(
                            cl_graph_dense,
                            decoded_labels_epoch,
                            decoded_rewrite_mask_epoch,
                            degree_floor=cl_degree_floor,
                            structural_support_mask=cl_struct_support,
                        )
                    view_add_ratio = max(0.0, float(decoded_add_ratio) * 0.5)
                    view_remove_ratio = max(0.0, float(decoded_remove_ratio) * 0.5)
                    cl_g1, cl_stats1 = sample_constraint_preserving_two_view_graph(
                        cl_graph_dense,
                        cl_pair_context,
                        E0=E0,
                        add_ratio=view_add_ratio,
                        remove_ratio=view_remove_ratio,
                        degree_floor=cl_degree_floor,
                    )
                    cl_g2, cl_stats2 = sample_constraint_preserving_two_view_graph(
                        cl_graph_dense,
                        cl_pair_context,
                        E0=E0,
                        add_ratio=view_add_ratio,
                        remove_ratio=view_remove_ratio,
                        degree_floor=cl_degree_floor,
                    )
                    cl_feat1 = features if feat_maske_ratio <= 0 else drop_feature(features, feat_maske_ratio)
                    cl_feat2 = features if feat_maske_ratio <= 0 else drop_feature(features, feat_maske_ratio)
                    z_view1 = encoder(cl_feat1, cl_g1.to_sparse().indices())
                    z_view2 = encoder(cl_feat2, cl_g2.to_sparse().indices())
                    edit_two_aug_cl_loss = symmetric_node_infonce_loss(z_view1, z_view2, gamma, temperature)
                    edit_two_aug_view_stats = {
                        "active": 1,
                        "jaccard": graph_edge_jaccard(cl_g1, cl_g2),
                        "v1_added": int(cl_stats1["added"]),
                        "v1_removed": int(cl_stats1["removed"]),
                        "v2_added": int(cl_stats2["added"]),
                        "v2_removed": int(cl_stats2["removed"]),
                        "v1_min_degree": int(cl_stats1["min_degree"]),
                        "v2_min_degree": int(cl_stats2["min_degree"]),
                        "degree_violations": int(cl_stats1["degree_violations"]) + int(cl_stats2["degree_violations"]),
                        "constraint_add_violations": int(cl_stats1["constraint_add_violations"]) + int(cl_stats2["constraint_add_violations"]),
                    }
                    if epoch % max(1, eval_log_every) == 0:
                        print(
                            f"[CL-VIEW][E{epoch:04d}] mode=edit_two_aug "
                            f"g_edit_add={active_decoded_graph_added} g_edit_remove={active_decoded_graph_removed} "
                            f"v1_add={edit_two_aug_view_stats['v1_added']} v1_remove={edit_two_aug_view_stats['v1_removed']} "
                            f"v2_add={edit_two_aug_view_stats['v2_added']} v2_remove={edit_two_aug_view_stats['v2_removed']} "
                            f"jaccard={edit_two_aug_view_stats['jaccard']:.6f} "
                            f"min_deg_v1={edit_two_aug_view_stats['v1_min_degree']} min_deg_v2={edit_two_aug_view_stats['v2_min_degree']} "
                            f"degree_viol={edit_two_aug_view_stats['degree_violations']} "
                            f"constraint_add_viol={edit_two_aug_view_stats['constraint_add_violations']} "
                            f"cl_loss={float(edit_two_aug_cl_loss.detach().cpu()):.6f}"
                        )
                except RuntimeError as e:
                    if not (skip_oom_epoch and _is_cuda_oom_like(e)):
                        raise
                    print(f"[OOM-FALLBACK] epoch={epoch} edit_two_aug_cl_loss=0: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            elif epoch % max(1, eval_log_every) == 0:
                print(f"[CL-VIEW][E{epoch:04d}] mode=edit_two_aug waiting_for_active_g_edit=1")
        else:
            try:
                cross_view_loss = inter_view_CL_loss(device, hidden_repr, encoder.Z.detach(), adj_label, delta, temperature)
            except RuntimeError as e:
                if not (skip_oom_epoch and _is_cuda_oom_like(e)):
                    raise
                print(f"[OOM-FALLBACK] epoch={epoch} cross_view_loss=0: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        aug_loss = bias_Z.new_tensor(0.0)
        maskgae_aug_feat_loss = bias_Z.new_tensor(0.0)
        cimage_aug_factor_loss = bias_Z.new_tensor(0.0)
        cimage_aug_cluster_loss = bias_Z.new_tensor(0.0)
        if cl_mode != "edit_two_aug":
            try:
                (
                    aug_loss,
                    maskgae_aug_feat_loss,
                    cimage_aug_factor_loss,
                    cimage_aug_cluster_loss,
                ) = _encoder_reconstruction_loss(dot_product_decode(bias_Z)) # aug_loss = loss_function(dot_product_decode(bias_Z), aug_adj_labels[i], encoder.mean, encoder.logstd, aug_norms[i], aug_weight_tensors[i], alpha, beta, train_mask)
            except RuntimeError as e:
                if not (skip_oom_epoch and _is_cuda_oom_like(e)):
                    raise
                print(f"[OOM-FALLBACK] epoch={epoch} aug_loss=0: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        edit_recon_loss = bias_Z.new_tensor(0.0)
        edit_keep_loss = bias_Z.new_tensor(0.0)
        edit_add_rank_loss = bias_Z.new_tensor(0.0)
        edit_remove_rank_loss = bias_Z.new_tensor(0.0)
        edit_heart_rank_loss = bias_Z.new_tensor(0.0)
        edit_heart_rank_debug = {
            "heart_rank_pairs": 0,
            "heart_rank_pos": 0,
            "heart_rank_neg_pool": 0,
            "heart_rank_pos_mean": float("nan"),
            "heart_rank_neg_mean": float("nan"),
        }
        edit_compact_loss = bias_Z.new_tensor(0.0)
        edit_compact_radius_loss = bias_Z.new_tensor(0.0)
        edit_compact_proto_loss = bias_Z.new_tensor(0.0)
        edit_preserve_loss = bias_Z.new_tensor(0.0)
        edit_decoder_debug = _empty_decoder_debug_info()
        edit_total_loss = bias_Z.sum() * 0.0
        decoder_warmup_recon_loss = bias_Z.new_tensor(0.0)
        decoder_warmup_total_loss = bias_Z.sum() * 0.0
        if _decoder_warmup_active(epoch):
            try:
                warm_seed = Z.detach()
                if decoder_warmup_use_pulled_latent:
                    warm_labels, warm_mask = resolve_edit_targets(
                        warm_seed,
                        _to_dense(adj_label),
                        freeze_targets=freeze_c0p_at_edit_start,
                        fixed_labels=fixed_c0p_labels,
                        fixed_mask=fixed_c0p_mask,
                        gmm_k=gmm_k,
                        gmm_tau=gmm_tau,
                        restrict_alpha=restrict_alpha,
                        restrict_gamma=restrict_gamma,
                    )
                    warm_c0p_mask = warm_mask
                    warm_cp_mask = torch.tensor((warm_labels != -1), dtype=torch.bool, device=Z.device) if warm_labels is not None else None
                    warm_pull_mask = warm_c0p_mask if pull_mask_scope == "c0p" else warm_cp_mask
                    warm_seed = direct_pull_latent_per_cluster(
                        warm_seed,
                        warm_labels,
                        warm_pull_mask,
                        pull_strength=editor_pull_strength,
                        c0p_mask=warm_c0p_mask,
                        pull_profile=editor_pull_profile,
                        pull_tau=editor_pull_tau,
                        pull_deadzone=editor_pull_deadzone,
                        pull_anchor=editor_pull_anchor,
                        push_scope=editor_push_scope,
                        noncompact_push_strength=editor_noncompact_push_strength,
                        noise_push_strength=editor_noise_push_strength,
                        push_preserve_norm=editor_push_preserve_norm,
                    ).detach()
                if isinstance(graph_decoder, (MLPPairGraphDecoder, StructuralPairGraphDecoder)):
                    decoder_warmup_recon_loss = sampled_decoder_reconstruction_loss(
                        graph_decoder,
                        warm_seed,
                        train_edges_t,
                        forbidden_edge_mask,
                        num_neg_per_pos=max(1, min(int(heart_rank_neg_k), 4)),
                    )
                else:
                    warm_pred = graph_decoder(warm_seed)
                    decoder_warmup_recon_loss = reconstruction_bce_loss(
                        warm_pred, adj_label, norm, weight_tensor, train_mask
                    )
                decoder_warmup_total_loss = decoder_warmup_recon_weight * decoder_warmup_recon_loss
            except Exception as e:
                print(f"[EDIT][WARMUP] decoder warmup failed at epoch {epoch}: {e}")
        elif use_edited_decoder and (graph_decoder is not None) and _in_edit_phase(epoch):
            try:
                preserve_anchor = stage1_anchor_Z.detach() if stage1_anchor_Z is not None else Z.detach()
                edit_seed = Z if (separate_edit_training and in_edit_phase) else bias_Z
                if use_decoded_graph_augment and decoded_labels_epoch is not None and decoded_mask_epoch is not None:
                    edit_labels_epoch, c0p_mask_epoch = decoded_labels_epoch, decoded_mask_epoch
                    cp_mask_epoch = torch.tensor((edit_labels_epoch != -1), dtype=torch.bool, device=Z.device) if edit_labels_epoch is not None else None
                else:
                    edit_labels_epoch, c0p_mask_epoch = resolve_edit_targets(
                        edit_seed,
                        _to_dense(adj_label),
                        freeze_targets=freeze_c0p_at_edit_start,
                        fixed_labels=fixed_c0p_labels,
                        fixed_mask=fixed_c0p_mask,
                        gmm_k=gmm_k,
                        gmm_tau=gmm_tau,
                        restrict_alpha=restrict_alpha,
                        restrict_gamma=restrict_gamma,
                    )
                    cp_mask_epoch = torch.tensor((edit_labels_epoch != -1), dtype=torch.bool, device=Z.device) if edit_labels_epoch is not None else None

                pull_mask = c0p_mask_epoch if pull_mask_scope == "c0p" else cp_mask_epoch
                compactness_mask = c0p_mask_epoch if compactness_mask_scope == "c0p" else cp_mask_epoch
                rewrite_mask = c0p_mask_epoch if rewrite_endpoint_scope == "c0p" else cp_mask_epoch
                edit_active_graph_context = (
                    active_decoded_aug_graph_dense
                    if use_decoded_graph_augment and active_decoded_aug_graph_dense is not None
                    else None
                )
                edit_pair_context = (
                    active_decoded_pair_context
                    if edit_active_graph_context is not None and active_decoded_pair_context is not None
                    else None
                )
                edit_structural_support_mask = None
                if edit_pair_context is None:
                    edit_structural_support_mask = _decoded_structural_support_cached(_to_dense(adj_label))

                z_edit, pull_push_diag_epoch = direct_pull_latent_per_cluster(
                    edit_seed,
                    edit_labels_epoch,
                    pull_mask,
                    pull_strength=editor_pull_strength,
                    c0p_mask=c0p_mask_epoch,
                    pull_profile=editor_pull_profile,
                    pull_tau=editor_pull_tau,
                    pull_deadzone=editor_pull_deadzone,
                    pull_anchor=editor_pull_anchor,
                    push_scope=editor_push_scope,
                    noncompact_push_strength=editor_noncompact_push_strength,
                    noise_push_strength=editor_noise_push_strength,
                    push_preserve_norm=editor_push_preserve_norm,
                    return_diagnostics=True,
                )
                _set_struct_decoder_context(edit_active_graph_context, edit_labels_epoch, c0p_mask_epoch)
                pairwise_decoder_training = isinstance(graph_decoder, (MLPPairGraphDecoder, StructuralPairGraphDecoder))
                if decoder_objective == "recon":
                    if pairwise_decoder_training:
                        edit_recon_loss = sampled_decoder_reconstruction_loss(
                            graph_decoder,
                            z_edit,
                            train_edges_t,
                            forbidden_edge_mask,
                            num_neg_per_pos=max(1, min(int(heart_rank_neg_k), 4)),
                        )
                    else:
                        A_edit_pred = graph_decoder(z_edit)
                        edit_recon_loss = reconstruction_bce_loss(A_edit_pred, adj_label, norm, weight_tensor, train_mask)
                elif decoder_objective == "hybrid":
                    decoded_degree_floor_eff = max(0, int(degree_threshold) - 1) if decoded_degree_floor is None else int(decoded_degree_floor)
                    if pairwise_decoder_training:
                        (
                            edit_recon_loss,
                            edit_keep_loss,
                            edit_add_rank_loss,
                            edit_remove_rank_loss,
                            edit_decoder_debug,
                        ) = hybrid_decoder_structure_losses_pairwise(
                            graph_decoder,
                            z_edit,
                            adj_label,
                            train_edges_t,
                            forbidden_edge_mask,
                            edit_labels_epoch,
                            rewrite_mask,
                            E0=E0,
                            add_ratio=decoded_add_ratio,
                            remove_ratio=decoded_remove_ratio,
                            add_threshold=decoded_add_threshold,
                            remove_threshold=decoded_remove_threshold,
                            add_quantile=decoded_add_quantile,
                            remove_quantile=decoded_remove_quantile,
                            max_add=decoded_max_add_per_round,
                            max_remove=decoded_max_remove_per_round,
                            degree_floor=decoded_degree_floor_eff,
                            same_cluster_only=decoded_same_cluster_only,
                            require_c0p_endpoint=decoded_require_c0p_endpoint,
                            require_both_c0p=decoded_require_both_c0p,
                            require_c0p_noncompact_endpoint=decoded_require_c0p_noncompact_endpoint,
                            require_structural_support=decoded_require_structural_support,
                            structural_support_mode=decoded_struct_support,
                            structural_min_cn=decoded_struct_min_cn,
                            structural_min_ra=decoded_struct_min_ra,
                            structural_min_aa=decoded_struct_min_aa,
                            structural_support_mask=edit_structural_support_mask,
                            keep_weight=decoder_keep_weight,
                            add_rank_weight=decoder_add_rank_weight,
                            remove_rank_weight=decoder_remove_rank_weight,
                            rank_margin=decoder_rank_margin,
                            rank_strategy=decoder_rank_strategy,
                            rank_neg_k=decoder_rank_neg_k,
                            rank_pool_factor=decoder_rank_pool_factor,
                            pair_context=edit_pair_context,
                        )
                    else:
                        A_edit_pred = graph_decoder(z_edit)
                        (
                            edit_recon_loss,
                            edit_keep_loss,
                            edit_add_rank_loss,
                            edit_remove_rank_loss,
                            edit_decoder_debug,
                        ) = hybrid_decoder_structure_losses(
                            A_edit_pred,
                            adj_label,
                            norm,
                            weight_tensor,
                            train_mask,
                            edit_labels_epoch,
                            rewrite_mask,
                            E0=E0,
                            add_ratio=decoded_add_ratio,
                            remove_ratio=decoded_remove_ratio,
                            add_threshold=decoded_add_threshold,
                            remove_threshold=decoded_remove_threshold,
                            add_quantile=decoded_add_quantile,
                            remove_quantile=decoded_remove_quantile,
                            max_add=decoded_max_add_per_round,
                            max_remove=decoded_max_remove_per_round,
                            degree_floor=decoded_degree_floor_eff,
                            same_cluster_only=decoded_same_cluster_only,
                            require_c0p_endpoint=decoded_require_c0p_endpoint,
                            require_both_c0p=decoded_require_both_c0p,
                            require_c0p_noncompact_endpoint=decoded_require_c0p_noncompact_endpoint,
                            require_structural_support=decoded_require_structural_support,
                            structural_support_mode=decoded_struct_support,
                            structural_min_cn=decoded_struct_min_cn,
                            structural_min_ra=decoded_struct_min_ra,
                            structural_min_aa=decoded_struct_min_aa,
                            structural_support_mask=edit_structural_support_mask,
                            keep_weight=decoder_keep_weight,
                            add_rank_weight=decoder_add_rank_weight,
                            remove_rank_weight=decoder_remove_rank_weight,
                            rank_margin=decoder_rank_margin,
                            rank_strategy=decoder_rank_strategy,
                            rank_neg_k=decoder_rank_neg_k,
                            rank_pool_factor=decoder_rank_pool_factor,
                            pair_context=edit_pair_context,
                        )
                else:
                    raise ValueError(f"Unsupported decoder_objective={decoder_objective}")
                if heart_rank_weight != 0.0:
                    if pairwise_decoder_training:
                        edit_heart_rank_loss, edit_heart_rank_debug = heart_train_margin_ranking_loss_pairs(
                            graph_decoder,
                            z_edit,
                            train_edges_t,
                            forbidden_edge_mask,
                            num_neg_per_pos=heart_rank_neg_k,
                            pool_factor=heart_rank_pool_factor,
                            margin=heart_rank_margin,
                        )
                    else:
                        edit_heart_rank_loss, edit_heart_rank_debug = heart_train_margin_ranking_loss(
                            A_edit_pred,
                            train_edges_t,
                            forbidden_edge_mask,
                            num_neg_per_pos=heart_rank_neg_k,
                            pool_factor=heart_rank_pool_factor,
                            margin=heart_rank_margin,
                        )
                    edit_decoder_debug.update(
                        {
                            "heart_rank_pairs": int(edit_heart_rank_debug.get("heart_rank_pairs", 0)),
                            "heart_rank_pos": int(edit_heart_rank_debug.get("heart_rank_pos", 0)),
                            "heart_rank_neg_pool": int(edit_heart_rank_debug.get("heart_rank_neg_pool", 0)),
                        }
                    )
                (
                    edit_compact_loss,
                    edit_compact_radius_loss,
                    edit_compact_proto_loss,
                ) = compactness_objective_loss(
                    z_edit,
                    edit_labels_epoch,
                    compactness_mask,
                    c0p_mask_epoch,
                    compactness_objective=compactness_objective,
                    radius_metric=compactness_radius_metric,
                )
                edit_preserve_loss = non_target_preservation_loss(
                    z_edit,
                    preserve_anchor,
                    pull_mask,
                )
                edit_structure_loss = edit_recon_loss if decoder_objective == "hybrid" else (decoder_recon_weight * edit_recon_loss)
                edit_total_loss = (
                    edit_structure_loss
                    + compactness_weight * edit_compact_loss
                    + preserve_weight * edit_preserve_loss
                    + heart_rank_weight * edit_heart_rank_loss
                )
            except Exception as e:
                print(f"[EDIT][TRAIN] edited branch failed at epoch {epoch}: {e}")

        prediction_train_z = Z
        prediction_context_dense = _to_dense(adj_label)
        prediction_using_edit_graph_this_epoch = 0
        if (
            prediction_graph == "edit"
            and use_decoded_graph_augment
            and _decoded_edit_active(epoch)
            and active_decoded_aug_graph_dense is not None
        ):
            prediction_train_z = bias_Z
            prediction_context_dense = active_decoded_aug_graph_dense
            prediction_using_edit_graph_this_epoch = 1

        prediction_rank_loss = bias_Z.new_tensor(0.0)
        prediction_bce_loss = bias_Z.new_tensor(0.0)
        prediction_joint_rank_loss = bias_Z.new_tensor(0.0)
        prediction_joint_bce_loss = bias_Z.new_tensor(0.0)
        prediction_extra_reg_loss = bias_Z.new_tensor(0.0)
        prediction_total_loss = bias_Z.sum() * 0.0
        prediction_debug = {
            "heart_rank_pairs": 0,
            "heart_rank_pos": 0,
            "heart_rank_neg_pool": 0,
            "heart_rank_pos_mean": float("nan"),
            "heart_rank_neg_mean": float("nan"),
            "heart_rank_pairs_total": 0,
            "heart_rank_hard_pairs": 0,
            "heart_rank_easy_pairs": 0,
            "heart_rank_dot_gap_mean": float("nan"),
            "heart_rank_anchor_loss": float("nan"),
        }

        def _prediction_objective(pred_z: torch.Tensor):
            pred_rank = pred_z.sum() * 0.0
            pred_bce = pred_z.sum() * 0.0
            pred_debug = dict(prediction_debug)
            try:
                pred_labels, pred_c0p_mask = resolve_edit_targets(
                    pred_z.detach(),
                    prediction_context_dense,
                    freeze_targets=freeze_c0p_at_edit_start,
                    fixed_labels=fixed_c0p_labels,
                    fixed_mask=fixed_c0p_mask,
                    gmm_k=gmm_k,
                    gmm_tau=gmm_tau,
                    restrict_alpha=restrict_alpha,
                    restrict_gamma=restrict_gamma,
                )
                _set_prediction_decoder_context(prediction_context_dense, pred_labels, pred_c0p_mask)
                if prediction_rank_weight != 0.0:
                    pred_rank, pred_debug = heart_train_margin_ranking_loss_pairs(
                        prediction_decoder,
                        pred_z,
                        train_edges_t,
                        forbidden_edge_mask,
                        num_neg_per_pos=prediction_rank_neg_k,
                        pool_factor=prediction_rank_pool_factor,
                        margin=prediction_rank_margin,
                        neg_strategy=prediction_rank_neg_strategy,
                        struct_frac=prediction_rank_struct_frac,
                        hard_only=prediction_hard_residual_only,
                        hard_margin=prediction_hard_margin,
                        dot_anchor_weight=prediction_dot_anchor_weight,
                    )
                if prediction_bce_weight != 0.0:
                    pred_bce = sampled_prediction_bce_logits_loss(
                        prediction_decoder,
                        pred_z,
                        train_edges_t,
                        forbidden_edge_mask,
                        num_neg_per_pos=max(1, min(int(prediction_rank_neg_k), 4)),
                    )
            except Exception as e:
                print(f"[PRED-DECODER][TRAIN] prediction objective failed at epoch {epoch}: {e}")
            pred_total = prediction_rank_weight * pred_rank + prediction_bce_weight * pred_bce
            return pred_total, pred_rank, pred_bce, pred_debug

        if prediction_decoder is not None and (prediction_rank_weight != 0.0 or prediction_bce_weight != 0.0):
            (
                prediction_total_loss,
                prediction_rank_loss,
                prediction_bce_loss,
                prediction_debug,
            ) = _prediction_objective(prediction_train_z.detach())
            if prediction_encoder_weight != 0.0 and epoch >= prediction_joint_start_epoch:
                (
                    prediction_joint_total_loss,
                    prediction_joint_rank_loss,
                    prediction_joint_bce_loss,
                    _,
                ) = _prediction_objective(prediction_train_z)
                prediction_total_loss = prediction_total_loss + prediction_encoder_weight * prediction_joint_total_loss

        if (
            prediction_decoder is not None
            and prediction_gate_l1_weight != 0.0
            and hasattr(prediction_decoder, "extra_regularization_loss")
        ):
            try:
                prediction_extra_reg_loss = prediction_decoder.extra_regularization_loss()
                prediction_extra_reg_loss = prediction_extra_reg_loss.to(device=bias_Z.device, dtype=bias_Z.dtype)
                if prediction_extra_reg_loss.dim() > 0:
                    prediction_extra_reg_loss = prediction_extra_reg_loss.sum()
                prediction_total_loss = prediction_total_loss + prediction_gate_l1_weight * prediction_extra_reg_loss
            except Exception as e:
                print(f"[PRED-DECODER][TRAIN] extra regularization failed at epoch {epoch}: {e}")

        # if(loss_ver=="nei"):
            # intra_CL = inter_view_CL_loss(device, bias_Z, bias_Z, adj_label, gamma, temperature)
        # else:
        intra_CL = bias_Z.new_tensor(0.0)
        if cl_mode != "edit_two_aug":
            try:
                intra_CL = intra_view_CL_loss(device, bias_Z, adj_label, gamma, temperature)
            except RuntimeError as e:
                if not (skip_oom_epoch and _is_cuda_oom_like(e)):
                    raise
                print(f"[OOM-FALLBACK] epoch={epoch} intra_CL=0: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        aug_losses = aug_loss + intra_CL

        if cl_mode == "edit_two_aug":
            base_task_loss = loss + edit_two_aug_cl_loss * aug_graph_weight
        else:
            base_task_loss = loss + cross_view_loss + aug_losses * aug_graph_weight

        if separate_edit_training and in_edit_phase and use_edited_decoder and (graph_decoder is not None):
            if phase2_task_main_loss:
                loss = base_task_loss
                if edit_phase_retain_recon_weight != 0.0:
                    loss = loss + edit_phase_retain_recon_weight * (recon_loss + aug_loss)
                if edit_phase_retain_cl_weight != 0.0:
                    loss = loss + edit_phase_retain_cl_weight * (ori_intra_CL + cross_view_loss + intra_CL + edit_two_aug_cl_loss)
                if edit_phase_edit_weight != 0.0:
                    loss = loss + edit_phase_edit_weight * edit_total_loss
            else:
                loss = bias_Z.new_tensor(0.0)
                loss = loss + edit_total_loss
                if edit_phase_retain_recon_weight != 0.0:
                    loss = loss + edit_phase_retain_recon_weight * (recon_loss + aug_loss)
                if edit_phase_retain_cl_weight != 0.0:
                    loss = loss + edit_phase_retain_cl_weight * (ori_intra_CL + cross_view_loss + intra_CL + edit_two_aug_cl_loss)
        else:
            loss = base_task_loss
            if _decoder_warmup_active(epoch):
                loss = loss + decoder_warmup_total_loss
            elif edit_total_loss is not None:
                loss = loss + edit_total_loss * aug_graph_weight
        if prediction_decoder is not None:
            loss = loss + prediction_total_loss
        #print(f'aug_loss: {aug_loss}, intra_CL: {intra_CL}')

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}: total={float(loss.detach().cpu())}, "
                f"recon={float(recon_loss.detach().cpu())}, aug={float(aug_loss.detach().cpu())}, intra={float(intra_CL.detach().cpu())}, "
                f"maskgae_feat={float(maskgae_feat_loss.detach().cpu())}, maskgae_aug_feat={float(maskgae_aug_feat_loss.detach().cpu())}, "
                f"cimage_factor={float(cimage_factor_loss.detach().cpu())}, cimage_cluster={float(cimage_cluster_loss.detach().cpu())}, "
                f"cimage_aug_factor={float(cimage_aug_factor_loss.detach().cpu())}, cimage_aug_cluster={float(cimage_aug_cluster_loss.detach().cpu())}, "
                f"edit_recon={float(edit_recon_loss.detach().cpu())}, keep={float(edit_keep_loss.detach().cpu())}, "
                f"add_rank={float(edit_add_rank_loss.detach().cpu())}, remove_rank={float(edit_remove_rank_loss.detach().cpu())}, "
                f"heart_rank={float(edit_heart_rank_loss.detach().cpu())}, "
                f"pred_rank={float(prediction_rank_loss.detach().cpu())}, pred_bce={float(prediction_bce_loss.detach().cpu())}, "
                f"pred_joint_rank={float(prediction_joint_rank_loss.detach().cpu())}, pred_joint_bce={float(prediction_joint_bce_loss.detach().cpu())}, "
                f"pred_extra_reg={float(prediction_extra_reg_loss.detach().cpu())}, "
                f"edit_compact={float(edit_compact_loss.detach().cpu())}, compact_radius={float(edit_compact_radius_loss.detach().cpu())}, "
                f"compact_proto={float(edit_compact_proto_loss.detach().cpu())}, edit_preserve={float(edit_preserve_loss.detach().cpu())}, "
                f"rewrite_nodes={edit_decoder_debug['rewrite_nodes']}, valid_pairs={edit_decoder_debug['valid_pairs']}, "
                f"add_pairs={edit_decoder_debug['add_pairs']}, add_selected={edit_decoder_debug['add_selected']}, "
                f"add_rank_pairs={edit_decoder_debug['add_rank_pairs']}, removable_pairs={edit_decoder_debug['removable_pairs']}, "
                f"remove_selected={edit_decoder_debug['remove_selected']}, remove_rank_pairs={edit_decoder_debug['remove_rank_pairs']}"
            )

        # Update Model
        skip_epoch_due_to_oom = False
        try:
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            if not (skip_oom_epoch and _is_cuda_oom_like(e)):
                raise
            print(f"[OOM-SKIP] epoch={epoch} during backward/step: {e}")
            optimizer.zero_grad(set_to_none=True)
            skip_epoch_due_to_oom = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        del bias_Z
        del encoder.Z
        del encoder.mean
        del encoder.logstd
        for _cache_attr in (
            "masked_feature_pred",
            "masked_feature_target",
            "masked_feature_mask",
            "cimage_factor_mask",
            "cimage_factor_scores",
            "cimage_cluster_q",
        ):
            if hasattr(encoder, _cache_attr):
                delattr(encoder, _cache_attr)
        del Z
        del hidden_repr
        torch.cuda.empty_cache()
        if skip_epoch_due_to_oom:
            continue
        #print(f"trn time2 {time.time()-t1:.2f} s", flush=True)
        ########################################################
        # Evaluate edge prediction
        t1 = time.time()
        encoder.eval()
        if graph_decoder is not None:
            graph_decoder.eval()
        if prediction_decoder is not None:
            prediction_decoder.eval()
        # print(f"test time {time.time()-t1:.2f} s")
        with torch.no_grad():
            inference_time_start = time.time()
            eval_edge_index = edge_index
            eval_adj_dense = _to_dense(adj_label)
            eval_prediction_graph_active = 0
            if prediction_graph == "edit":
                if active_decoded_aug_edge_index is not None and active_decoded_aug_graph_dense is not None:
                    eval_edge_index = active_decoded_aug_edge_index
                    eval_adj_dense = active_decoded_aug_graph_dense
                    eval_prediction_graph_active = 1
                elif decoded_static_view_enabled and static_decoded_aug_edge_index is not None and static_decoded_aug_graph_dense is not None:
                    eval_edge_index = static_decoded_aug_edge_index
                    eval_adj_dense = static_decoded_aug_graph_dense
                    eval_prediction_graph_active = 1
            Z = encoder(features, eval_edge_index) # Z = encoder(features, adj_norm)
            if is_heart:
                should_eval = (epoch % max(1, heart_eval_every) == 0) or (epoch == num_epoch - 1)
            else:
                should_eval = (epoch % train_eval_every == 0) or (epoch == num_epoch - 1)
            need_full_score = (should_eval and not edge_eval) or (not skip_train_acc)
            score_needs_decoder_context = need_full_score and (
                (score_source == "decoder" and use_edited_decoder and (graph_decoder is not None))
                or (score_source == "pred_decoder" and prediction_decoder is not None)
            )
            score_needs_decoder_context = score_needs_decoder_context or (
                should_eval
                and edge_eval
                and (
                    (score_source == "decoder" and use_edited_decoder and (graph_decoder is not None))
                    or (score_source == "pred_decoder" and prediction_decoder is not None)
                )
            )
            if score_needs_decoder_context:
                try:
                    score_labels, score_c0p_mask = resolve_edit_targets(
                        Z,
                        eval_adj_dense,
                        freeze_targets=freeze_c0p_at_edit_start,
                        fixed_labels=fixed_c0p_labels,
                        fixed_mask=fixed_c0p_mask,
                        gmm_k=gmm_k,
                        gmm_tau=gmm_tau,
                        restrict_alpha=restrict_alpha,
                        restrict_gamma=restrict_gamma,
                    )
                    if score_source == "decoder" and use_edited_decoder and (graph_decoder is not None):
                        _set_struct_decoder_context(eval_adj_dense, score_labels, score_c0p_mask)
                    if prediction_decoder is not None:
                        _set_prediction_decoder_context(eval_adj_dense, score_labels, score_c0p_mask)
                except Exception as e:
                    print(f"[DECODER-DIAG] score context failed at epoch {epoch}: {e}")
            A_pred = None
            A_pred_np = None
            train_acc = float("nan")
            if need_full_score:
                A_pred = _score_adjacency(Z)
                if not skip_train_acc:
                    train_acc = get_acc(A_pred.data.cpu(), adj_label.data.cpu())
                if should_eval:
                    A_pred_np = A_pred.data.cpu().numpy()
            radius_before = Z.new_tensor(0.0)
            radius_after = Z.new_tensor(0.0)
            c0p_radius_before = Z.new_tensor(0.0)
            c0p_radius_after = Z.new_tensor(0.0)
            cp_radius_before = Z.new_tensor(0.0)
            cp_radius_after = Z.new_tensor(0.0)
            noncompact_radius_before = Z.new_tensor(0.0)
            noncompact_radius_after = Z.new_tensor(0.0)
            noncompact_radius_p90_before = Z.new_tensor(0.0)
            noncompact_radius_p90_after = Z.new_tensor(0.0)
            noncompact_radius_max_before = Z.new_tensor(0.0)
            noncompact_radius_max_after = Z.new_tensor(0.0)
            edit_metrics_active = use_edited_decoder and (graph_decoder is not None) and _decoded_edit_active(epoch)
            run_edit_metrics = (
                edit_metrics_active
                and edit_metric_every > 0
                and ((epoch % max(1, edit_metric_every) == 0) or (epoch == num_epoch - 1))
            )
            if edit_metrics_active and not run_edit_metrics:
                nan_metric = Z.new_tensor(float("nan"))
                radius_before = nan_metric
                radius_after = nan_metric
                c0p_radius_before = nan_metric
                c0p_radius_after = nan_metric
                cp_radius_before = nan_metric
                cp_radius_after = nan_metric
                noncompact_radius_before = nan_metric
                noncompact_radius_after = nan_metric
                noncompact_radius_p90_before = nan_metric
                noncompact_radius_p90_after = nan_metric
                noncompact_radius_max_before = nan_metric
                noncompact_radius_max_after = nan_metric
            if run_edit_metrics:
                try:
                    eval_labels, eval_c0p_mask = resolve_edit_targets(
                        Z,
                        _to_dense(adj_label),
                        freeze_targets=freeze_c0p_at_edit_start,
                        fixed_labels=fixed_c0p_labels,
                        fixed_mask=fixed_c0p_mask,
                        gmm_k=gmm_k,
                        gmm_tau=gmm_tau,
                        restrict_alpha=restrict_alpha,
                        restrict_gamma=restrict_gamma,
                    )
                    eval_cp_mask = torch.tensor((eval_labels != -1), dtype=torch.bool, device=Z.device) if eval_labels is not None else None
                    eval_pull_mask = eval_c0p_mask if pull_mask_scope == "c0p" else eval_cp_mask
                    eval_compactness_mask = eval_c0p_mask if compactness_mask_scope == "c0p" else eval_cp_mask
                    eval_rewrite_mask = eval_c0p_mask if rewrite_endpoint_scope == "c0p" else eval_cp_mask
                    eval_noncompact_mask = noncompact_node_mask(eval_cp_mask, eval_c0p_mask)
                    _set_decoded_graph_scorer_context(None, eval_labels, eval_c0p_mask)

                    radius_before = cluster_compactness_loss(Z, eval_labels, eval_compactness_mask, radius_metric=compactness_radius_metric)
                    c0p_radius_before = cluster_compactness_loss(Z, eval_labels, eval_c0p_mask, radius_metric=compactness_radius_metric)
                    cp_radius_before = cluster_compactness_loss(Z, eval_labels, eval_cp_mask, radius_metric=compactness_radius_metric)
                    (
                        noncompact_radius_before,
                        noncompact_radius_p90_before,
                        noncompact_radius_max_before,
                    ) = cluster_radius_summary(Z, eval_labels, eval_noncompact_mask, radius_metric=compactness_radius_metric)

                    if use_decoded_graph_augment:
                        if decoded_accumulate_into_base:
                            if decoded_rewrite_applied_this_epoch and decoded_pre_graph_dense_epoch is not None:
                                metric_labels = decoded_metric_labels_epoch if decoded_metric_labels_epoch is not None else eval_labels
                                metric_c0p_mask = decoded_metric_mask_epoch if decoded_metric_mask_epoch is not None else eval_c0p_mask
                                metric_cp_mask = torch.tensor((metric_labels != -1), dtype=torch.bool, device=Z.device) if metric_labels is not None else None
                                metric_compactness_mask = metric_c0p_mask if compactness_mask_scope == "c0p" else metric_cp_mask
                                metric_noncompact_mask = noncompact_node_mask(metric_cp_mask, metric_c0p_mask)

                                Z_before_graph = encoder(features, decoded_pre_graph_dense_epoch.to_sparse().indices())
                                radius_before = cluster_compactness_loss(Z_before_graph, metric_labels, metric_compactness_mask, radius_metric=compactness_radius_metric)
                                radius_after = cluster_compactness_loss(Z, metric_labels, metric_compactness_mask, radius_metric=compactness_radius_metric)
                                c0p_radius_before = cluster_compactness_loss(Z_before_graph, metric_labels, metric_c0p_mask, radius_metric=compactness_radius_metric)
                                c0p_radius_after = cluster_compactness_loss(Z, metric_labels, metric_c0p_mask, radius_metric=compactness_radius_metric)
                                cp_radius_before = cluster_compactness_loss(Z_before_graph, metric_labels, metric_cp_mask, radius_metric=compactness_radius_metric)
                                cp_radius_after = cluster_compactness_loss(Z, metric_labels, metric_cp_mask, radius_metric=compactness_radius_metric)
                                (
                                    noncompact_radius_before,
                                    noncompact_radius_p90_before,
                                    noncompact_radius_max_before,
                                ) = cluster_radius_summary(Z_before_graph, metric_labels, metric_noncompact_mask, radius_metric=compactness_radius_metric)
                                (
                                    noncompact_radius_after,
                                    noncompact_radius_p90_after,
                                    noncompact_radius_max_after,
                                ) = cluster_radius_summary(Z, metric_labels, metric_noncompact_mask, radius_metric=compactness_radius_metric)
                            else:
                                radius_after = radius_before
                                c0p_radius_after = c0p_radius_before
                                cp_radius_after = cp_radius_before
                                noncompact_radius_after = noncompact_radius_before
                                noncompact_radius_p90_after = noncompact_radius_p90_before
                                noncompact_radius_max_after = noncompact_radius_max_before
                        else:
                            if active_decoded_aug_edge_index is not None:
                                Z_eval = encoder(features, active_decoded_aug_edge_index)
                            elif decoded_static_view_enabled and static_decoded_aug_edge_index is not None:
                                Z_eval = encoder(features, static_decoded_aug_edge_index)
                            else:
                                Z_pull_eval = direct_pull_latent_per_cluster(
                                    Z,
                                    eval_labels,
                                    eval_pull_mask,
                                    pull_strength=editor_pull_strength,
                                    c0p_mask=eval_c0p_mask,
                                    pull_profile=editor_pull_profile,
                                    pull_tau=editor_pull_tau,
                                    pull_deadzone=editor_pull_deadzone,
                                    pull_anchor=editor_pull_anchor,
                                    push_scope=editor_push_scope,
                                    noncompact_push_strength=editor_noncompact_push_strength,
                                    noise_push_strength=editor_noise_push_strength,
                                    push_preserve_norm=editor_push_preserve_norm,
                                )
                                decoded_degree_floor_eff = max(0, int(degree_threshold) - 1) if decoded_degree_floor is None else int(decoded_degree_floor)
                                decoded_bound_eff = None if (decoded_graph_aug_bound is None or float(decoded_graph_aug_bound) <= 0) else float(decoded_graph_aug_bound)
                                decoded_eval_scorer = _decoded_graph_scorer()
                                if decoded_eval_scorer is None:
                                    raise RuntimeError("decoded graph augment eval requires an edit decoder or prediction decoder scorer")
                                if isinstance(decoded_eval_scorer, (MLPPairGraphDecoder, StructuralPairGraphDecoder)):
                                    g_eval, _, _ = build_decoded_augmented_graph_from_decoder(
                                        decoded_eval_scorer,
                                        Z_pull_eval,
                                        _to_dense(adj_label),
                                        eval_labels,
                                        eval_rewrite_mask,
                                        E0=E0,
                                        add_ratio=decoded_add_ratio,
                                        remove_ratio=decoded_remove_ratio,
                                        add_threshold=decoded_add_threshold,
                                        remove_threshold=decoded_remove_threshold,
                                        add_quantile=decoded_add_quantile,
                                        remove_quantile=decoded_remove_quantile,
                                        max_add=decoded_max_add_per_round,
                                        max_remove=decoded_max_remove_per_round,
                                        per_node_cap_frac=decoded_bound_eff,
                                        add_degree_target=decoded_add_degree_target,
                                        add_degree_target_scope=decoded_add_degree_target_scope,
                                        add_degree_target_nodes=decoded_add_degree_target_nodes,
                                        guarantee_degree_target=decoded_guarantee_degree_target,
                                        deg0_excl_self=None,
                                        degree_floor=decoded_degree_floor_eff,
                                        same_cluster_only=decoded_same_cluster_only,
                                        require_c0p_endpoint=decoded_require_c0p_endpoint,
                                        require_both_c0p=decoded_require_both_c0p,
                                        require_c0p_noncompact_endpoint=decoded_require_c0p_noncompact_endpoint,
                                        require_structural_support=decoded_require_structural_support,
                                        structural_support_mode=decoded_struct_support,
                                        structural_min_cn=decoded_struct_min_cn,
                                        structural_min_ra=decoded_struct_min_ra,
                                        structural_min_aa=decoded_struct_min_aa,
                                        structural_support_mask=_decoded_structural_support_cached(_to_dense(adj_label)),
                                    )
                                else:
                                    with torch.no_grad():
                                        decoded_scores_eval = decoded_eval_scorer(Z_pull_eval)
                                    g_eval, _, _ = build_decoded_augmented_graph(
                                        decoded_scores_eval,
                                        _to_dense(adj_label),
                                        eval_labels,
                                        eval_rewrite_mask,
                                        E0=E0,
                                        add_ratio=decoded_add_ratio,
                                        remove_ratio=decoded_remove_ratio,
                                        add_threshold=decoded_add_threshold,
                                        remove_threshold=decoded_remove_threshold,
                                        add_quantile=decoded_add_quantile,
                                        remove_quantile=decoded_remove_quantile,
                                        max_add=decoded_max_add_per_round,
                                        max_remove=decoded_max_remove_per_round,
                                        per_node_cap_frac=decoded_bound_eff,
                                        add_degree_target=decoded_add_degree_target,
                                        add_degree_target_scope=decoded_add_degree_target_scope,
                                        add_degree_target_nodes=decoded_add_degree_target_nodes,
                                        guarantee_degree_target=decoded_guarantee_degree_target,
                                        deg0_excl_self=None,
                                        degree_floor=decoded_degree_floor_eff,
                                        same_cluster_only=decoded_same_cluster_only,
                                        require_c0p_endpoint=decoded_require_c0p_endpoint,
                                        require_both_c0p=decoded_require_both_c0p,
                                        require_c0p_noncompact_endpoint=decoded_require_c0p_noncompact_endpoint,
                                        require_structural_support=decoded_require_structural_support,
                                        structural_support_mode=decoded_struct_support,
                                        structural_min_cn=decoded_struct_min_cn,
                                        structural_min_ra=decoded_struct_min_ra,
                                        structural_min_aa=decoded_struct_min_aa,
                                        structural_support_mask=_decoded_structural_support_cached(_to_dense(adj_label)),
                                    )
                                Z_eval = encoder(features, g_eval.to_sparse().indices())
                            radius_after = cluster_compactness_loss(Z_eval, eval_labels, eval_compactness_mask, radius_metric=compactness_radius_metric)
                            c0p_radius_after = cluster_compactness_loss(Z_eval, eval_labels, eval_c0p_mask, radius_metric=compactness_radius_metric)
                            cp_radius_after = cluster_compactness_loss(Z_eval, eval_labels, eval_cp_mask, radius_metric=compactness_radius_metric)
                            (
                                noncompact_radius_after,
                                noncompact_radius_p90_after,
                                noncompact_radius_max_after,
                            ) = cluster_radius_summary(Z_eval, eval_labels, eval_noncompact_mask, radius_metric=compactness_radius_metric)
                    else:
                        Z_eval = direct_pull_latent_per_cluster(
                            Z,
                            eval_labels,
                            eval_pull_mask,
                            pull_strength=editor_pull_strength,
                            c0p_mask=eval_c0p_mask,
                            pull_profile=editor_pull_profile,
                            pull_tau=editor_pull_tau,
                            pull_deadzone=editor_pull_deadzone,
                            pull_anchor=editor_pull_anchor,
                            push_scope=editor_push_scope,
                            noncompact_push_strength=editor_noncompact_push_strength,
                            noise_push_strength=editor_noise_push_strength,
                            push_preserve_norm=editor_push_preserve_norm,
                        )
                        radius_after = cluster_compactness_loss(Z_eval, eval_labels, eval_compactness_mask, radius_metric=compactness_radius_metric)
                        c0p_radius_after = cluster_compactness_loss(Z_eval, eval_labels, eval_c0p_mask, radius_metric=compactness_radius_metric)
                        cp_radius_after = cluster_compactness_loss(Z_eval, eval_labels, eval_cp_mask, radius_metric=compactness_radius_metric)
                        (
                            noncompact_radius_after,
                            noncompact_radius_p90_after,
                            noncompact_radius_max_after,
                        ) = cluster_radius_summary(Z_eval, eval_labels, eval_noncompact_mask, radius_metric=compactness_radius_metric)
                except Exception as e:
                    print(f"[EDIT][EVAL] edited eval failed at epoch {epoch}: {e}")
        
        # A_pred = train_decoder(device, encoder.Z.clone().detach(), adj_label, weight_tensor, norm, train_mask)
        # print(A_pred.shape)
        if should_eval and (not edge_eval) and A_pred_np is None:
            raise RuntimeError(f"Missing adjacency scores for evaluation at epoch {epoch}")

        if is_heart:
            if should_eval:
                # cheap fixed-subset validation
                val_roc_subset, val_ap_subset, val_hit_subset = _evaluate_edges_for_source(
                    Z,
                    val_edges_eval,
                    val_edges_false_eval,
                    score_matrix_np=A_pred_np,
                )

                val_roc = val_roc_subset
                val_ap = val_ap_subset
                val_hit = val_hit_subset

                test_roc = float("nan")
                test_ap = float("nan")
                test_hit = [float("nan")] * 6
                ran_full_val = False

                val_subset_checkpoint_score = _heart_checkpoint_score(
                    val_roc_subset,
                    val_ap_subset,
                    val_hit_subset,
                )

                if heart_full_val_during_training:
                    best_val_roc_subset = val_roc_subset
                    best_val_ap_subset = val_ap_subset
                    best_subset_epoch = epoch
                    ran_full_val = True

                    if heart_test_on_best_val:
                        test_roc, test_ap, test_hit = _evaluate_edges_for_source(
                            Z,
                            test_edges,
                            test_edges_false,
                            score_matrix_np=A_pred_np,
                        )
                # only pay for full validation if the cheap subset improves on the selection metric
                elif val_subset_checkpoint_score > best_subset_checkpoint_score:
                    best_subset_checkpoint_score = val_subset_checkpoint_score
                    best_val_roc_subset = val_roc_subset
                    best_val_ap_subset = val_ap_subset
                    best_subset_epoch = epoch

                    val_roc, val_ap, val_hit = _evaluate_edges_for_source(
                        Z,
                        val_edges,
                        val_edges_false,
                        score_matrix_np=A_pred_np,
                    )
                    ran_full_val = True

                    if heart_test_on_best_val:
                        test_roc, test_ap, test_hit = _evaluate_edges_for_source(
                            Z,
                            test_edges,
                            test_edges_false,
                            score_matrix_np=A_pred_np,
                        )
            else:
                val_roc = float("nan")
                val_ap = float("nan")
                val_hit = [float("nan")] * 6
                test_roc = float("nan")
                test_ap = float("nan")
                test_hit = [float("nan")] * 6
                ran_full_val = False

        else:
            if should_eval:
                val_roc, val_ap, val_hit = _evaluate_edges_for_source(
                    Z, val_edges, val_edges_false, score_matrix_np=A_pred_np
                )
                test_roc, test_ap, test_hit = _evaluate_edges_for_source(
                    Z, test_edges, test_edges_false, score_matrix_np=A_pred_np
                )
                ran_full_val = True
            else:
                val_roc = float("nan")
                val_ap = float("nan")
                val_hit = [float("nan")] * 6
                test_roc = float("nan")
                test_ap = float("nan")
                test_hit = [float("nan")] * 6
                ran_full_val = False

        score_diag = _score_source_diagnostics(None, None, [], [])
        score_diag.update(
            {
                "diag_pred_val_hit10": float("nan"),
                "diag_pred_pos_mean": float("nan"),
                "diag_pred_neg_mean": float("nan"),
                "diag_dot_pred_corr": float("nan"),
                "diag_decoder_pred_corr": float("nan"),
            }
        )
        run_decoder_diag = (
            ((use_edited_decoder and (graph_decoder is not None)) or (prediction_decoder is not None))
            and np.isfinite(val_roc)
            and decoder_diag_every != 0
            and (
                decoder_diag_every < 0
                or (epoch % max(1, decoder_diag_every) == 0)
                or (epoch == num_epoch - 1)
            )
        )
        if run_decoder_diag:
            try:
                diag_labels, diag_c0p_mask = resolve_edit_targets(
                    Z,
                    _to_dense(adj_label),
                    freeze_targets=freeze_c0p_at_edit_start,
                    fixed_labels=fixed_c0p_labels,
                    fixed_mask=fixed_c0p_mask,
                    gmm_k=gmm_k,
                    gmm_tau=gmm_tau,
                    restrict_alpha=restrict_alpha,
                    restrict_gamma=restrict_gamma,
                )
                if use_edited_decoder and (graph_decoder is not None):
                    _set_struct_decoder_context(None, diag_labels, diag_c0p_mask)
                if prediction_decoder is not None:
                    _set_prediction_decoder_context(None, diag_labels, diag_c0p_mask)
                diag_edges_pos = val_edges if ran_full_val else val_edges_eval
                diag_edges_neg = val_edges_false if ran_full_val else val_edges_false_eval
                dot_pos_diag = _edge_score_values_from_dot(Z, diag_edges_pos)
                dot_neg_diag = _edge_score_values_from_dot(Z, diag_edges_neg)
                _, _, dot_hit_diag = get_scores_from_values(dot_pos_diag, dot_neg_diag)
                score_diag["diag_dot_val_hit10"] = float(dot_hit_diag[2]) if len(dot_hit_diag) > 2 else float("nan")

                decoder_pos_diag = np.asarray([], dtype=np.float64)
                decoder_neg_diag = np.asarray([], dtype=np.float64)
                if use_edited_decoder and (graph_decoder is not None):
                    decoder_pos_diag = _edge_score_values_from_decoder(graph_decoder, Z, diag_edges_pos)
                    decoder_neg_diag = _edge_score_values_from_decoder(graph_decoder, Z, diag_edges_neg)
                    _, _, decoder_hit_diag = get_scores_from_values(decoder_pos_diag, decoder_neg_diag)
                    decoder_diag = _score_source_diagnostics_from_values(dot_pos_diag, dot_neg_diag, decoder_pos_diag, decoder_neg_diag)
                    score_diag.update(decoder_diag)
                    score_diag["diag_dot_val_hit10"] = float(dot_hit_diag[2]) if len(dot_hit_diag) > 2 else float("nan")
                    score_diag["diag_decoder_val_hit10"] = float(decoder_hit_diag[2]) if len(decoder_hit_diag) > 2 else float("nan")

                pred_pos_diag = np.asarray([], dtype=np.float64)
                pred_neg_diag = np.asarray([], dtype=np.float64)
                if prediction_decoder is not None:
                    pred_pos_diag = _edge_score_values_from_decoder(prediction_decoder, Z, diag_edges_pos)
                    pred_neg_diag = _edge_score_values_from_decoder(prediction_decoder, Z, diag_edges_neg)
                    _, _, pred_hit_diag = get_scores_from_values(pred_pos_diag, pred_neg_diag)
                    score_diag["diag_pred_val_hit10"] = float(pred_hit_diag[2]) if len(pred_hit_diag) > 2 else float("nan")
                    if pred_pos_diag.size > 0:
                        score_diag["diag_pred_pos_mean"] = float(np.mean(pred_pos_diag.reshape(-1)))
                    if pred_neg_diag.size > 0:
                        score_diag["diag_pred_neg_mean"] = float(np.mean(pred_neg_diag.reshape(-1)))
                    dot_vals = np.concatenate([np.asarray(dot_pos_diag).reshape(-1), np.asarray(dot_neg_diag).reshape(-1)])
                    pred_vals = np.concatenate([pred_pos_diag.reshape(-1), pred_neg_diag.reshape(-1)])
                    if dot_vals.size > 1 and pred_vals.size > 1 and np.std(dot_vals) > 0 and np.std(pred_vals) > 0:
                        score_diag["diag_dot_pred_corr"] = float(np.corrcoef(dot_vals, pred_vals)[0, 1])
                    if decoder_pos_diag.size > 0 or decoder_neg_diag.size > 0:
                        dec_vals = np.concatenate([decoder_pos_diag.reshape(-1), decoder_neg_diag.reshape(-1)])
                        if dec_vals.size > 1 and pred_vals.size > 1 and np.std(dec_vals) > 0 and np.std(pred_vals) > 0:
                            score_diag["diag_decoder_pred_corr"] = float(np.corrcoef(dec_vals, pred_vals)[0, 1])

                prediction_extra_diag = _prediction_decoder_extra_diagnostics()
                print(
                    f"[DECODER-DIAG][E{epoch:04d}] "
                    f"decoder_type={decoder_type} pred_decoder_type={prediction_decoder_type} normalize_input={int(decoder_normalize_input)} score_source={score_source} "
                    f"heart_rank_weight={heart_rank_weight:.6f} prediction_rank_weight={prediction_rank_weight:.6f} "
                    f"prediction_h3_gate={prediction_extra_diag.get('prediction_h3_gate', float('nan')):.6f} "
                    f"prediction_compact_gate={prediction_extra_diag.get('prediction_compact_gate', float('nan')):.6f} "
                    f"dot_val_hit10={score_diag['diag_dot_val_hit10']:.6f} "
                    f"decoder_val_hit10={score_diag['diag_decoder_val_hit10']:.6f} "
                    f"pred_val_hit10={score_diag['diag_pred_val_hit10']:.6f} "
                    f"dot_pos_mean={score_diag['diag_dot_pos_mean']:.6f} dot_neg_mean={score_diag['diag_dot_neg_mean']:.6f} "
                    f"decoder_pos_mean={score_diag['diag_decoder_pos_mean']:.6f} decoder_neg_mean={score_diag['diag_decoder_neg_mean']:.6f} "
                    f"pred_pos_mean={score_diag['diag_pred_pos_mean']:.6f} pred_neg_mean={score_diag['diag_pred_neg_mean']:.6f} "
                    f"dot_decoder_corr={score_diag['diag_dot_decoder_corr']:.6f} "
                    f"dot_pred_corr={score_diag['diag_dot_pred_corr']:.6f} decoder_pred_corr={score_diag['diag_decoder_pred_corr']:.6f}"
                )
            except Exception as e:
                print(f"[DECODER-DIAG] failed at epoch {epoch}: {e}")

        radius_before_val = float(radius_before.detach().cpu())
        radius_after_val = float(radius_after.detach().cpu())
        radius_delta_val = radius_after_val - radius_before_val

        c0p_radius_before_val = float(c0p_radius_before.detach().cpu())
        c0p_radius_after_val = float(c0p_radius_after.detach().cpu())
        cp_radius_before_val = float(cp_radius_before.detach().cpu())
        cp_radius_after_val = float(cp_radius_after.detach().cpu())
        noncompact_radius_before_val = float(noncompact_radius_before.detach().cpu())
        noncompact_radius_after_val = float(noncompact_radius_after.detach().cpu())
        noncompact_radius_p90_before_val = float(noncompact_radius_p90_before.detach().cpu())
        noncompact_radius_p90_after_val = float(noncompact_radius_p90_after.detach().cpu())
        noncompact_radius_max_before_val = float(noncompact_radius_max_before.detach().cpu())
        noncompact_radius_max_after_val = float(noncompact_radius_max_after.detach().cpu())
        rewrite_applied_now = bool(use_edited_decoder and epoch >= decoded_rewrite_start_epoch and decoded_rewrite_applied_this_epoch)
        if use_edited_decoder and _decoded_edit_active(epoch):
            if radius_anchor_value is None and np.isfinite(radius_before_val):
                radius_anchor_value = radius_before_val
            delta_from_anchor = (radius_after_val - float(radius_anchor_value)) if radius_anchor_value is not None else float('nan')

            # Always track compactness history during edit phase, even when no decoded graph rewrite happens.
            # The old logic only appended when rewrite_applied_now=True, which made no-rewrite runs look like
            # they had "no radius history" even though radius_before/radius_after were being computed correctly.
            radius_hist.append(radius_after_val)
            radius_epoch_hist.append(epoch)
            c0p_radius_before_hist.append(c0p_radius_before_val)
            c0p_radius_after_hist.append(c0p_radius_after_val)
            cp_radius_before_hist.append(cp_radius_before_val)
            cp_radius_after_hist.append(cp_radius_after_val)

            if rewrite_applied_now:
                if prev_rewrite_radius_after_value is None:
                    delta_from_prev_rewrite = 0.0
                else:
                    delta_from_prev_rewrite = radius_after_val - float(prev_rewrite_radius_after_value)
                prev_rewrite_radius_after_value = radius_after_val
                last_valid_delta_prev_rewrite = delta_from_prev_rewrite
                rewrite_epoch_hist.append(epoch)
                rewrite_radius_before_hist.append(radius_before_val)
                rewrite_radius_after_hist.append(radius_after_val)
                rewrite_delta_hist.append(radius_delta_val)
                rewrite_delta_anchor_hist.append(delta_from_anchor)
                rewrite_delta_prev_hist.append(delta_from_prev_rewrite)
            else:
                delta_from_prev_rewrite = last_valid_delta_prev_rewrite
        else:
            delta_from_anchor = float('nan')
            delta_from_prev_rewrite = last_valid_delta_prev_rewrite

        # track validation curve instead of test
        if (not is_heart) or ran_full_val:
            roc_history.append(val_roc)              # per-epoch full-val ROC
            val_hit_history.append([float(h) for h in val_hit])
        else:
            roc_history.append(float("nan"))
            val_hit_history.append([float("nan")] * 6)
        edit_recon_hist.append(float(edit_recon_loss.detach().cpu()))
        edit_heart_rank_hist.append(float(edit_heart_rank_loss.detach().cpu()))
        edit_compact_hist.append(float(edit_compact_loss.detach().cpu()))
        edit_preserve_hist.append(float(edit_preserve_loss.detach().cpu()))
        radius_before_hist.append(radius_before_val)
        radius_after_hist.append(radius_after_val)
        radius_anchor_hist.append(float(radius_anchor_value) if radius_anchor_value is not None else float('nan'))
        delta_from_anchor_hist.append(delta_from_anchor)
        delta_from_prev_rewrite_hist.append(delta_from_prev_rewrite)
        rewrite_applied_hist.append(1 if rewrite_applied_now else 0)

        if use_edited_decoder and should_eval and (epoch % max(1, eval_log_every) == 0 or epoch == num_epoch - 1):
            if is_heart:
                if ran_full_val:
                    eval_tag = "heart_full_val"
                elif np.isfinite(val_roc):
                    eval_tag = "heart_subset_val"
                else:
                    eval_tag = "heart_skip"
            else:
                eval_tag = "standard"

            print(
                f"[EDIT][E{epoch:04d}][{eval_tag}] loss={float(loss.detach().cpu()):.6f} "
                f"recon={float(recon_loss.detach().cpu()):.6f} aug={float(aug_loss.detach().cpu()):.6f} "
                f"maskgae_feat={float(maskgae_feat_loss.detach().cpu()):.6f} maskgae_aug_feat={float(maskgae_aug_feat_loss.detach().cpu()):.6f} "
                f"cimage_factor={float(cimage_factor_loss.detach().cpu()):.6f} cimage_cluster={float(cimage_cluster_loss.detach().cpu()):.6f} "
                f"cimage_aug_factor={float(cimage_aug_factor_loss.detach().cpu()):.6f} cimage_aug_cluster={float(cimage_aug_cluster_loss.detach().cpu()):.6f} "
                f"warm_recon={float(decoder_warmup_recon_loss.detach().cpu()):.6f} "
                f"edit_recon={float(edit_recon_loss.detach().cpu()):.6f} keep={float(edit_keep_loss.detach().cpu()):.6f} "
                f"add_rank={float(edit_add_rank_loss.detach().cpu()):.6f} remove_rank={float(edit_remove_rank_loss.detach().cpu()):.6f} "
                f"heart_rank={float(edit_heart_rank_loss.detach().cpu()):.6f} "
                f"pred_rank={float(prediction_rank_loss.detach().cpu()):.6f} pred_bce={float(prediction_bce_loss.detach().cpu()):.6f} "
                f"pred_joint_rank={float(prediction_joint_rank_loss.detach().cpu()):.6f} pred_joint_bce={float(prediction_joint_bce_loss.detach().cpu()):.6f} "
                f"pred_extra_reg={float(prediction_extra_reg_loss.detach().cpu()):.6f} "
                f"cl_mode={cl_mode} edit_two_aug_cl={float(edit_two_aug_cl_loss.detach().cpu()):.6f} "
                f"cl_view_jaccard={float(edit_two_aug_view_stats['jaccard']):.6f} "
                f"pred_graph={prediction_graph} pred_graph_train_edit_active={int(prediction_using_edit_graph_this_epoch)} "
                f"pred_graph_eval_edit_active={int(eval_prediction_graph_active)} "
                f"compact={float(edit_compact_loss.detach().cpu()):.6f} compact_radius={float(edit_compact_radius_loss.detach().cpu()):.6f} "
                f"compact_proto={float(edit_compact_proto_loss.detach().cpu()):.6f} "
                f"preserve={float(edit_preserve_loss.detach().cpu()):.6f} "
                f"radius_metric={compactness_radius_metric} radius_before={radius_before_val:.6f} radius_after={radius_after_val:.6f} "
                f"delta={radius_delta_val:.6f} radius_anchor={(float(radius_anchor_value) if radius_anchor_value is not None else float('nan')):.6f} "
                f"c0p_radius_before={c0p_radius_before_val:.6f} c0p_radius_after={c0p_radius_after_val:.6f} "
                f"cp_radius_before={cp_radius_before_val:.6f} cp_radius_after={cp_radius_after_val:.6f} "
                f"noncompact_radius_before={noncompact_radius_before_val:.6f} noncompact_radius_after={noncompact_radius_after_val:.6f} "
                f"noncompact_radius_p90_before={noncompact_radius_p90_before_val:.6f} noncompact_radius_p90_after={noncompact_radius_p90_after_val:.6f} "
                f"noncompact_radius_max_before={noncompact_radius_max_before_val:.6f} noncompact_radius_max_after={noncompact_radius_max_after_val:.6f} "
                f"delta_anchor={delta_from_anchor:.6f} delta_prev_rewrite={delta_from_prev_rewrite:.6f} "
                f"rewrite_applied={int(rewrite_applied_now)} "
                f"push_noncompact_count={int(pull_push_diag_epoch.get('push_noncompact_count', 0.0))} "
                f"push_noise_count={int(pull_push_diag_epoch.get('push_noise_count', 0.0))} "
                f"push_noncompact_cosdist_before={float(pull_push_diag_epoch.get('push_noncompact_anchor_cosdist_before', float('nan'))):.6f} "
                f"push_noncompact_cosdist_after={float(pull_push_diag_epoch.get('push_noncompact_anchor_cosdist_after', float('nan'))):.6f} "
                f"push_noise_cosdist_before={float(pull_push_diag_epoch.get('push_noise_anchor_cosdist_before', float('nan'))):.6f} "
                f"push_noise_cosdist_after={float(pull_push_diag_epoch.get('push_noise_anchor_cosdist_after', float('nan'))):.6f} "
                f"rewrite_nodes={edit_decoder_debug['rewrite_nodes']} "
                f"valid_pairs={edit_decoder_debug['valid_pairs']} "
                f"add_pairs={edit_decoder_debug['add_pairs']} "
                f"add_budget={edit_decoder_debug['add_budget']} "
                f"add_selected={edit_decoder_debug['add_selected']} "
                f"add_negatives={edit_decoder_debug['add_negatives']} "
                f"add_rank_pairs={edit_decoder_debug['add_rank_pairs']} "
                f"removable_pairs={edit_decoder_debug['removable_pairs']} "
                f"remove_budget={edit_decoder_debug['remove_budget']} "
                f"remove_selected={edit_decoder_debug['remove_selected']} "
                f"remove_kept={edit_decoder_debug['remove_kept']} "
                f"remove_rank_pairs={edit_decoder_debug['remove_rank_pairs']} "
                f"heart_rank_pairs={edit_decoder_debug.get('heart_rank_pairs', 0)} "
                f"heart_rank_pos={edit_decoder_debug.get('heart_rank_pos', 0)} "
                f"heart_rank_neg_pool={edit_decoder_debug.get('heart_rank_neg_pool', 0)} "
                f"val_roc={val_roc:.6f} val_hit10={float(val_hit[2]):.6f} test_hit10={float(test_hit[2]):.6f}"
            )
        
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
        if np.isfinite(test_roc):
            if test_hit[0] > best_hit1:
                best_hit1 = test_hit[0]
                best_hit1_test_roc = test_roc
                best_hit1_ep = epoch 
            if test_hit[1] > best_hit3:
                best_hit3 = test_hit[1]
                best_hit3_test_roc = test_roc
                best_hit3_ep = epoch 
            if test_hit[2] > best_hit10:
                best_hit10 = test_hit[2]
                best_hit10_test_roc = test_roc
                best_hit10_ep = epoch
            if test_hit[3] > best_hit20:
                best_hit20 = test_hit[3]
                best_hit20_test_roc = test_roc
                best_hit20_ep = epoch 
            if test_hit[4] > best_hit50:
                best_hit50 = test_hit[4]
                best_hit50_test_roc = test_roc
                best_hit50_ep = epoch 
            if test_hit[5] > best_hit100:
                best_hit100 = test_hit[5]
                best_hit100_test_roc = test_roc
                best_hit100_ep = epoch    
            if test_roc > best_test_roc_seen:
                best_test_roc_seen = test_roc
                best_test_ap_at_best_test_roc = test_ap
                best_val_roc_at_best_test_roc = val_roc
                best_val_ap_at_best_test_roc = val_ap
                best_test_epoch = epoch
                #print(f'Update Best Test ROC, Epoch = {epoch+1}, val_roc = {best_val_roc_at_best_test_roc:.3f}, val_ap = {best_val_ap_at_best_test_roc:.3f}, test_roc = {best_test_roc_seen:.3f}, test_ap = {best_test_ap_at_best_test_roc:.3f}')
        #print('-' * 100)
        
        # --------- SELECT BEST BY VALIDATION METRIC (NO TEST LEAKAGE) ---------
        checkpoint_metric_label = heart_checkpoint_metric if is_heart else random_checkpoint_metric
        checkpoint_score = _checkpoint_score(checkpoint_metric_label, val_roc, val_ap, val_hit)
        checkpoint_graph_ready = (prediction_graph != "edit") or bool(eval_prediction_graph_active)
        if (
            checkpoint_graph_ready
            and ((not is_heart) or ran_full_val)
            and np.isfinite(checkpoint_score)
            and (checkpoint_score > best_checkpoint_score)
        ):
            best_checkpoint_score = checkpoint_score
            best_val_roc = val_roc
            best_val_ap = val_ap
            best_epoch = epoch

            # store CPU copy of the best weights in a structured checkpoint dict
            best_state_cpu = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
            }
            if use_edited_decoder and (graph_decoder is not None):
                best_state_cpu["graph_decoder"] = {
                    k: v.detach().cpu().clone() for k, v in graph_decoder.state_dict().items()
                }
            if prediction_decoder is not None:
                best_state_cpu["prediction_decoder"] = {
                    k: v.detach().cpu().clone() for k, v in prediction_decoder.state_dict().items()
                }
            if prediction_graph == "edit" and active_decoded_aug_graph_dense is not None:
                best_graph_dense_cpu = active_decoded_aug_graph_dense.detach().cpu().clone()
                best_state_cpu["graph_dense"] = best_graph_dense_cpu
            elif decoded_accumulate_into_base:
                best_graph_dense_cpu = _to_dense(adj_label).detach().cpu().clone()
                best_state_cpu["graph_dense"] = best_graph_dense_cpu

            # Keep the best-by-val edit-side metadata together with the checkpoint.
            # Without this, no-rewrite runs can end up printing stale / default compactness stats.
            prediction_extra_diag = _prediction_decoder_extra_diagnostics()
            best_meta_cpu = {
                "epoch": int(epoch),
                "val_roc": float(val_roc),
                "val_ap": float(val_ap),
                "val_hit1": float(val_hit[0]) if len(val_hit) > 0 else float("nan"),
                "val_hit3": float(val_hit[1]) if len(val_hit) > 1 else float("nan"),
                "val_hit10": float(val_hit[2]) if len(val_hit) > 2 else float("nan"),
                "val_hit20": float(val_hit[3]) if len(val_hit) > 3 else float("nan"),
                "val_hit50": float(val_hit[4]) if len(val_hit) > 4 else float("nan"),
                "val_hit100": float(val_hit[5]) if len(val_hit) > 5 else float("nan"),
                "radius_before": float(radius_before_val),
                "radius_after": float(radius_after_val),
                "radius_delta": float(radius_delta_val),
                "compactness_radius_metric": compactness_radius_metric,
                "c0p_radius_before": float(c0p_radius_before_val),
                "c0p_radius_after": float(c0p_radius_after_val),
                "cp_radius_before": float(cp_radius_before_val),
                "cp_radius_after": float(cp_radius_after_val),
                "noncompact_radius_before": float(noncompact_radius_before_val),
                "noncompact_radius_after": float(noncompact_radius_after_val),
                "noncompact_radius_p90_before": float(noncompact_radius_p90_before_val),
                "noncompact_radius_p90_after": float(noncompact_radius_p90_after_val),
                "noncompact_radius_max_before": float(noncompact_radius_max_before_val),
                "noncompact_radius_max_after": float(noncompact_radius_max_after_val),
                "editor_noncompact_push_strength": float(pull_push_diag_epoch.get("editor_noncompact_push_strength", float("nan"))),
                "editor_noise_push_strength": float(pull_push_diag_epoch.get("editor_noise_push_strength", float("nan"))),
                "editor_push_preserve_norm": float(pull_push_diag_epoch.get("editor_push_preserve_norm", float("nan"))),
                "push_noncompact_count": float(pull_push_diag_epoch.get("push_noncompact_count", float("nan"))),
                "push_noise_count": float(pull_push_diag_epoch.get("push_noise_count", float("nan"))),
                "push_noncompact_anchor_cosdist_before": float(pull_push_diag_epoch.get("push_noncompact_anchor_cosdist_before", float("nan"))),
                "push_noncompact_anchor_cosdist_after": float(pull_push_diag_epoch.get("push_noncompact_anchor_cosdist_after", float("nan"))),
                "push_noise_anchor_cosdist_before": float(pull_push_diag_epoch.get("push_noise_anchor_cosdist_before", float("nan"))),
                "push_noise_anchor_cosdist_after": float(pull_push_diag_epoch.get("push_noise_anchor_cosdist_after", float("nan"))),
                "radius_anchor": float(radius_anchor_value) if radius_anchor_value is not None else float("nan"),
                "delta_anchor": float(delta_from_anchor),
                "delta_prev_rewrite": float(delta_from_prev_rewrite),
                "rewrite_applied": int(rewrite_applied_now),
                "edit_recon": float(edit_recon_loss.detach().cpu()),
                "edit_keep": float(edit_keep_loss.detach().cpu()),
                "edit_add_rank": float(edit_add_rank_loss.detach().cpu()),
                "edit_remove_rank": float(edit_remove_rank_loss.detach().cpu()),
                "edit_heart_rank": float(edit_heart_rank_loss.detach().cpu()),
                "lp_full_graph": int(lp_full_graph_protocol),
                "ae_backbone": ae_backbone,
                "maskgae_mask_rate": float(maskgae_mask_rate),
                "maskgae_feature_weight": float(maskgae_feature_weight),
                "maskgae_feature_loss": float(maskgae_feat_loss.detach().cpu()),
                "maskgae_aug_feature_loss": float(maskgae_aug_feat_loss.detach().cpu()),
                "cimage_factor_weight": float(cimage_factor_weight),
                "cimage_cluster_weight": float(cimage_cluster_weight),
                "cimage_num_factors": int(cimage_num_factors),
                "cimage_num_clusters": int(cimage_num_clusters),
                "cimage_pseudo_label_threshold": float(cimage_pseudo_label_threshold),
                "cimage_factor_select_ratio": float(cimage_factor_select_ratio),
                "cimage_mrmr_redundancy_weight": float(cimage_mrmr_redundancy_weight),
                "cimage_cluster_balance_weight": float(cimage_cluster_balance_weight),
                "cimage_sce_power": float(cimage_sce_power),
                "cimage_factor_loss": float(cimage_factor_loss.detach().cpu()),
                "cimage_cluster_loss": float(cimage_cluster_loss.detach().cpu()),
                "cimage_aug_factor_loss": float(cimage_aug_factor_loss.detach().cpu()),
                "cimage_aug_cluster_loss": float(cimage_aug_cluster_loss.detach().cpu()),
                "prediction_rank": float(prediction_rank_loss.detach().cpu()),
                "prediction_bce": float(prediction_bce_loss.detach().cpu()),
                "prediction_joint_rank": float(prediction_joint_rank_loss.detach().cpu()),
                "prediction_joint_bce": float(prediction_joint_bce_loss.detach().cpu()),
                "prediction_extra_reg": float(prediction_extra_reg_loss.detach().cpu()),
                "cl_mode": cl_mode,
                "edit_two_aug_cl": float(edit_two_aug_cl_loss.detach().cpu()),
                "cl_view_jaccard": float(edit_two_aug_view_stats["jaccard"]),
                "cl_view1_added": int(edit_two_aug_view_stats["v1_added"]),
                "cl_view1_removed": int(edit_two_aug_view_stats["v1_removed"]),
                "cl_view2_added": int(edit_two_aug_view_stats["v2_added"]),
                "cl_view2_removed": int(edit_two_aug_view_stats["v2_removed"]),
                "cl_view_degree_violations": int(edit_two_aug_view_stats["degree_violations"]),
                "cl_view_constraint_add_violations": int(edit_two_aug_view_stats["constraint_add_violations"]),
                "prediction_graph": prediction_graph,
                "prediction_graph_train_edit_active": int(prediction_using_edit_graph_this_epoch),
                "prediction_graph_eval_edit_active": int(eval_prediction_graph_active),
                "prediction_rank_pairs": int(prediction_debug.get("heart_rank_pairs", 0)),
                "prediction_rank_pos": int(prediction_debug.get("heart_rank_pos", 0)),
                "prediction_rank_neg_pool": int(prediction_debug.get("heart_rank_neg_pool", 0)),
                "prediction_rank_pairs_total": int(prediction_debug.get("heart_rank_pairs_total", 0)),
                "prediction_rank_hard_pairs": int(prediction_debug.get("heart_rank_hard_pairs", 0)),
                "prediction_rank_easy_pairs": int(prediction_debug.get("heart_rank_easy_pairs", 0)),
                "prediction_rank_dot_gap_mean": float(prediction_debug.get("heart_rank_dot_gap_mean", float("nan"))),
                "prediction_rank_anchor_loss": float(prediction_debug.get("heart_rank_anchor_loss", float("nan"))),
                "heart_rank_pairs": int(edit_decoder_debug.get("heart_rank_pairs", 0)),
                "heart_rank_pos": int(edit_decoder_debug.get("heart_rank_pos", 0)),
                "heart_rank_neg_pool": int(edit_decoder_debug.get("heart_rank_neg_pool", 0)),
                "edit_add_pairs_pre_struct": int(edit_decoder_debug.get("add_pairs_pre_struct", 0)),
                "edit_struct_supported_add_pairs": int(edit_decoder_debug.get("struct_supported_add_pairs", 0)),
                "heart_rank_pos_mean": float(edit_heart_rank_debug.get("heart_rank_pos_mean", float("nan"))),
                "heart_rank_neg_mean": float(edit_heart_rank_debug.get("heart_rank_neg_mean", float("nan"))),
                "decoder_normalize_input": int(decoder_normalize_input),
                "heart_rank_weight": float(heart_rank_weight),
                "heart_rank_margin": float(heart_rank_margin),
                "heart_rank_neg_k": int(heart_rank_neg_k),
                "prediction_decoder_type": prediction_decoder_type,
                "prediction_rank_weight": float(prediction_rank_weight),
                "prediction_bce_weight": float(prediction_bce_weight),
                "prediction_encoder_weight": float(prediction_encoder_weight),
                "prediction_rank_neg_strategy": prediction_rank_neg_strategy,
                "prediction_rank_struct_frac": float(prediction_rank_struct_frac),
                "prediction_hard_residual_only": int(prediction_hard_residual_only),
                "prediction_hard_margin": float(prediction_hard_margin),
                "prediction_dot_anchor_weight": float(prediction_dot_anchor_weight),
                "prediction_gate_l1_weight": float(prediction_gate_l1_weight),
                "prediction_h3_gate_init": float(prediction_h3_gate_init),
                "prediction_residual_gate_init": float(prediction_residual_gate_init),
                "prediction_residual_scale": float(prediction_residual_scale),
                "prediction_joint_start_epoch": int(prediction_joint_start_epoch),
                "decoded_require_structural_support": int(decoded_require_structural_support),
                "decoded_struct_min_cn": float(decoded_struct_min_cn),
                "decoded_struct_min_ra": float(decoded_struct_min_ra),
                "decoded_struct_min_aa": float(decoded_struct_min_aa),
                "prediction_h3_gate": float(prediction_extra_diag.get("prediction_h3_gate", float("nan"))),
                "prediction_compact_gate": float(prediction_extra_diag.get("prediction_compact_gate", float("nan"))),
                "prediction_compact_residual_scale": float(prediction_extra_diag.get("prediction_compact_residual_scale", float("nan"))),
                "diag_dot_val_hit10": float(score_diag.get("diag_dot_val_hit10", float("nan"))),
                "diag_decoder_val_hit10": float(score_diag.get("diag_decoder_val_hit10", float("nan"))),
                "diag_pred_val_hit10": float(score_diag.get("diag_pred_val_hit10", float("nan"))),
                "diag_dot_pos_mean": float(score_diag.get("diag_dot_pos_mean", float("nan"))),
                "diag_dot_neg_mean": float(score_diag.get("diag_dot_neg_mean", float("nan"))),
                "diag_decoder_pos_mean": float(score_diag.get("diag_decoder_pos_mean", float("nan"))),
                "diag_decoder_neg_mean": float(score_diag.get("diag_decoder_neg_mean", float("nan"))),
                "diag_pred_pos_mean": float(score_diag.get("diag_pred_pos_mean", float("nan"))),
                "diag_pred_neg_mean": float(score_diag.get("diag_pred_neg_mean", float("nan"))),
                "diag_dot_decoder_corr": float(score_diag.get("diag_dot_decoder_corr", float("nan"))),
                "diag_dot_pred_corr": float(score_diag.get("diag_dot_pred_corr", float("nan"))),
                "diag_decoder_pred_corr": float(score_diag.get("diag_decoder_pred_corr", float("nan"))),
                "edit_compact": float(edit_compact_loss.detach().cpu()),
                "edit_compact_radius": float(edit_compact_radius_loss.detach().cpu()),
                "edit_compact_proto": float(edit_compact_proto_loss.detach().cpu()),
                "edit_preserve": float(edit_preserve_loss.detach().cpu()),
                "selection_metric": checkpoint_metric_label,
                "selection_score": float(checkpoint_score),
            }

            # also save to disk (optional, helpful for crashes / later reuse)
            try:
                state_to_save = dict(best_state_cpu)
                state_to_save["best_meta"] = best_meta_cpu
                torch.save(state_to_save, best_ckpt_path_runtime)
                print(f"[CKPT] Saved best-by-val {checkpoint_metric_label} at epoch {epoch} -> {best_ckpt_path_runtime}")
            except Exception as e:
                print(f"[CKPT] Warning: failed to save best checkpoint: {e}")

        if phase_cache_path and (not phase_cache_saved) and epoch == phase_cache_save_epoch:
            try:
                phase_dir = os.path.dirname(phase_cache_path)
                if phase_dir:
                    os.makedirs(phase_dir, exist_ok=True)
                phase_state = {
                    "epoch": int(epoch),
                    "dataset": str(dataset_str),
                    "seed": int(seed) if seed is not None else None,
                    "run_tag": str(run_tag),
                    "ver": str(ver),
                    "encoder": encoder.state_dict(),
                    "graph_decoder": graph_decoder.state_dict() if graph_decoder is not None else None,
                    "prediction_decoder": prediction_decoder.state_dict() if prediction_decoder is not None else None,
                    "optimizer": optimizer.state_dict(),
                    "best_state": best_state_cpu,
                    "best_meta": best_meta_cpu,
                    "best_val_roc": float(best_val_roc),
                    "best_val_ap": float(best_val_ap),
                    "best_epoch": int(best_epoch),
                    "best_checkpoint_score": float(best_checkpoint_score),
                    "best_val_roc_subset": float(best_val_roc_subset),
                    "best_val_ap_subset": float(best_val_ap_subset),
                    "best_subset_epoch": int(best_subset_epoch),
                    "best_subset_checkpoint_score": float(best_subset_checkpoint_score),
                    "stage1_anchor_Z": stage1_anchor_Z.detach().cpu() if stage1_anchor_Z is not None else None,
                    "stage1_anchor_graph_dense": (
                        stage1_anchor_graph_dense.detach().cpu()
                        if stage1_anchor_graph_dense is not None
                        else None
                    ),
                    "fixed_c0p_labels": (
                        np.asarray(fixed_c0p_labels, dtype=np.int64)
                        if fixed_c0p_labels is not None
                        else None
                    ),
                    "fixed_c0p_mask": fixed_c0p_mask.detach().cpu() if fixed_c0p_mask is not None else None,
                    "frozen_gmm_labels": (
                        np.asarray(frozen_gmm_labels, dtype=np.int64)
                        if frozen_gmm_labels is not None
                        else None
                    ),
                    "frozen_core_mask": (
                        np.asarray(frozen_core_mask, dtype=bool)
                        if frozen_core_mask is not None
                        else None
                    ),
                    "stage1_phase_boundary_logged": bool(stage1_phase_boundary_logged),
                    "phase_cache_load_mode": str(phase_cache_load_mode),
                    "edit_train_start_epoch": int(edit_train_start_epoch),
                    "decoded_rewrite_start_epoch": int(decoded_rewrite_start_epoch),
                    "prediction_joint_start_epoch": int(prediction_joint_start_epoch),
                    "decoder_warmup_in_phase1": int(decoder_warmup_in_phase1),
                    "decoder_warmup_recon_weight": float(decoder_warmup_recon_weight),
                    "editor_pull_strength": float(editor_pull_strength),
                    "editor_pull_profile": str(editor_pull_profile),
                    "editor_pull_tau": float(editor_pull_tau),
                    "editor_pull_deadzone": float(editor_pull_deadzone),
                    "editor_pull_anchor": str(editor_pull_anchor),
                    "prediction_decoder_type": str(prediction_decoder_type),
                    "decoder_type": str(decoder_type),
                    "hidden1": int(hidden1),
                    "hidden2": int(hidden2),
                    "dropout": float(dropout),
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "feat_mask_ratio": float(feat_maske_ratio),
                    "cl_mode": str(cl_mode),
                    "prediction_graph": str(prediction_graph),
                }
                torch.save(phase_state, phase_cache_path)
                phase_cache_saved = True
                print(f"[PHASE-CACHE] Saved phase cache at epoch {epoch} -> {phase_cache_path}")
            except Exception as e:
                print(f"[PHASE-CACHE] Warning: failed to save phase cache: {e}")
                
        # -------- offline C0p sweep with fixed encoder & fixed GMM ---------
    # side experiment：只在你開 ENABLE_C0P_SWEEP 時啟用
    if ENABLE_C0P_SWEEP and is_remove_only and (cluster_method == "gmm"):
        # For offline sweep we don't enforce min-degree; we want to see the full curve
        sweep_degree_floor = 0
        print(f"[CP-SWEEP] start | ver={ver}, degree_floor={sweep_degree_floor} (no floor for sweep)")

        _ = run_c0p_sweep_static(
            dataset_str=dataset_str,
            ver=ver,
            seed=seed,
            run_tag=str(run_tag),
            device=device,
            encoder=encoder,
            features=features,
            adj_train=adj_train,
            adj_orig=adj_orig,
            val_edges=val_edges,
            val_edges_false=val_edges_false,
            test_edges=test_edges,
            test_edges_false=test_edges_false,
            gmm_k=gmm_k,
            gmm_tau=gmm_tau,
            alpha_c0p=restrict_alpha,
            gamma_c0p=restrict_gamma,
            degree_floor=sweep_degree_floor,  # = 0 → enforce_floor=False
            step_frac=C0P_SWEEP_STEP_FRAC,
            max_frac=C0P_SWEEP_MAX_FRAC,
            sweep_scope=sweep_scope,    # cp_all / c0p_only / cp_minus_c0p
            out_dir=None,
        )

    # print(f'best classification epoch = {best_classi_epoch+1}, val_acc = {best_acc:.3f}, test_acc = {best_test_acc:.3f}')
    if best_epoch >= 0:
        print(
            f'[BEST VALIDATION] epoch = {best_epoch+1}, '
            f'val_roc = {best_val_roc:.6f}, val_ap = {best_val_ap:.6f}'
        )
        if best_meta_cpu is not None:
            print(
                f'[BEST VALIDATION HIT@K] '
                f'1={best_meta_cpu.get("val_hit1", float("nan")):.6f}, '
                f'3={best_meta_cpu.get("val_hit3", float("nan")):.6f}, '
                f'10={best_meta_cpu.get("val_hit10", float("nan")):.6f}, '
                f'20={best_meta_cpu.get("val_hit20", float("nan")):.6f}, '
                f'50={best_meta_cpu.get("val_hit50", float("nan")):.6f}, '
                f'100={best_meta_cpu.get("val_hit100", float("nan")):.6f}'
            )
            print(
                f'[BEST VALIDATION META] '
                f'radius_before = {best_meta_cpu.get("radius_before", float("nan")):.6f}, '
                f'radius_after = {best_meta_cpu.get("radius_after", float("nan")):.6f}, '
                f'delta = {best_meta_cpu.get("radius_delta", float("nan")):.6f}, '
                f'radius_metric = {best_meta_cpu.get("compactness_radius_metric", compactness_radius_metric)}, '
                f'c0p_radius_before = {best_meta_cpu.get("c0p_radius_before", float("nan")):.6f}, '
                f'c0p_radius_after = {best_meta_cpu.get("c0p_radius_after", float("nan")):.6f}, '
                f'cp_radius_before = {best_meta_cpu.get("cp_radius_before", float("nan")):.6f}, '
                f'cp_radius_after = {best_meta_cpu.get("cp_radius_after", float("nan")):.6f}, '
                f'noncompact_radius_before = {best_meta_cpu.get("noncompact_radius_before", float("nan")):.6f}, '
                f'noncompact_radius_after = {best_meta_cpu.get("noncompact_radius_after", float("nan")):.6f}, '
                f'noncompact_radius_p90_before = {best_meta_cpu.get("noncompact_radius_p90_before", float("nan")):.6f}, '
                f'noncompact_radius_p90_after = {best_meta_cpu.get("noncompact_radius_p90_after", float("nan")):.6f}, '
                f'noncompact_radius_max_before = {best_meta_cpu.get("noncompact_radius_max_before", float("nan")):.6f}, '
                f'noncompact_radius_max_after = {best_meta_cpu.get("noncompact_radius_max_after", float("nan")):.6f}, '
                f'radius_anchor = {best_meta_cpu.get("radius_anchor", float("nan")):.6f}, '
                f'delta_anchor = {best_meta_cpu.get("delta_anchor", float("nan")):.6f}, '
                f'delta_prev_rewrite = {best_meta_cpu.get("delta_prev_rewrite", float("nan")):.6f}, '
                f'rewrite_applied = {best_meta_cpu.get("rewrite_applied", -1)}, '
                f'edit_recon = {best_meta_cpu.get("edit_recon", float("nan")):.6f}, '
                f'edit_keep = {best_meta_cpu.get("edit_keep", float("nan")):.6f}, '
                f'edit_add_rank = {best_meta_cpu.get("edit_add_rank", float("nan")):.6f}, '
                f'edit_remove_rank = {best_meta_cpu.get("edit_remove_rank", float("nan")):.6f}, '
                f'edit_heart_rank = {best_meta_cpu.get("edit_heart_rank", float("nan")):.6f}, '
                f'maskgae_feature_loss = {best_meta_cpu.get("maskgae_feature_loss", float("nan")):.6f}, '
                f'maskgae_aug_feature_loss = {best_meta_cpu.get("maskgae_aug_feature_loss", float("nan")):.6f}, '
                f'cimage_factor_loss = {best_meta_cpu.get("cimage_factor_loss", float("nan")):.6f}, '
                f'cimage_cluster_loss = {best_meta_cpu.get("cimage_cluster_loss", float("nan")):.6f}, '
                f'cimage_aug_factor_loss = {best_meta_cpu.get("cimage_aug_factor_loss", float("nan")):.6f}, '
                f'cimage_aug_cluster_loss = {best_meta_cpu.get("cimage_aug_cluster_loss", float("nan")):.6f}, '
                f'heart_rank_pairs = {best_meta_cpu.get("heart_rank_pairs", float("nan")):.0f}, '
                f'cl_mode = {best_meta_cpu.get("cl_mode", cl_mode)}, '
                f'edit_two_aug_cl = {best_meta_cpu.get("edit_two_aug_cl", float("nan")):.6f}, '
                f'cl_view_jaccard = {best_meta_cpu.get("cl_view_jaccard", float("nan")):.6f}, '
                f'prediction_graph = {best_meta_cpu.get("prediction_graph", prediction_graph)}, '
                f'prediction_graph_eval_edit_active = {best_meta_cpu.get("prediction_graph_eval_edit_active", float("nan")):.0f}, '
                f'prediction_rank = {best_meta_cpu.get("prediction_rank", float("nan")):.6f}, '
                f'prediction_bce = {best_meta_cpu.get("prediction_bce", float("nan")):.6f}, '
                f'prediction_extra_reg = {best_meta_cpu.get("prediction_extra_reg", float("nan")):.6f}, '
                f'prediction_h3_gate = {best_meta_cpu.get("prediction_h3_gate", float("nan")):.6f}, '
                f'prediction_compact_gate = {best_meta_cpu.get("prediction_compact_gate", float("nan")):.6f}, '
                f'prediction_rank_pairs = {best_meta_cpu.get("prediction_rank_pairs", float("nan")):.0f}, '
                f'prediction_rank_hard_pairs = {best_meta_cpu.get("prediction_rank_hard_pairs", float("nan")):.0f}, '
                f'prediction_rank_easy_pairs = {best_meta_cpu.get("prediction_rank_easy_pairs", float("nan")):.0f}, '
                f'prediction_rank_anchor_loss = {best_meta_cpu.get("prediction_rank_anchor_loss", float("nan")):.6f}, '
                f'dot_val_hit10 = {best_meta_cpu.get("diag_dot_val_hit10", float("nan")):.6f}, '
                f'decoder_val_hit10 = {best_meta_cpu.get("diag_decoder_val_hit10", float("nan")):.6f}, '
                f'pred_val_hit10 = {best_meta_cpu.get("diag_pred_val_hit10", float("nan")):.6f}, '
                f'decoder_pos_mean = {best_meta_cpu.get("diag_decoder_pos_mean", float("nan")):.6f}, '
                f'decoder_neg_mean = {best_meta_cpu.get("diag_decoder_neg_mean", float("nan")):.6f}, '
                f'dot_decoder_corr = {best_meta_cpu.get("diag_dot_decoder_corr", float("nan")):.6f}, '
                f'pred_pos_mean = {best_meta_cpu.get("diag_pred_pos_mean", float("nan")):.6f}, '
                f'pred_neg_mean = {best_meta_cpu.get("diag_pred_neg_mean", float("nan")):.6f}, '
                f'dot_pred_corr = {best_meta_cpu.get("diag_dot_pred_corr", float("nan")):.6f}, '
                f'edit_compact = {best_meta_cpu.get("edit_compact", float("nan")):.6f}, '
                f'edit_compact_radius = {best_meta_cpu.get("edit_compact_radius", float("nan")):.6f}, '
                f'edit_compact_proto = {best_meta_cpu.get("edit_compact_proto", float("nan")):.6f}, '
                f'edit_preserve = {best_meta_cpu.get("edit_preserve", float("nan")):.6f}'
            )
    else:
        print('[BEST VALIDATION] unavailable')

    # --------- RELOAD BEST MODEL BEFORE FINAL TEST ---------
    if best_state_cpu is not None:
        if isinstance(best_state_cpu, dict) and ("encoder" in best_state_cpu):
            encoder.load_state_dict(best_state_cpu["encoder"], strict=True)
            if (graph_decoder is not None) and ("graph_decoder" in best_state_cpu):
                graph_decoder.load_state_dict(best_state_cpu["graph_decoder"], strict=True)
            if (prediction_decoder is not None) and ("prediction_decoder" in best_state_cpu):
                prediction_decoder.load_state_dict(best_state_cpu["prediction_decoder"], strict=True)
            if "graph_dense" in best_state_cpu:
                best_graph_dense_cpu = best_state_cpu["graph_dense"].detach().cpu().clone()
            if "best_meta" in best_state_cpu:
                best_meta_cpu = best_state_cpu["best_meta"]
        else:
            # backward compatibility for old plain state_dict checkpoints
            if isinstance(best_state_cpu, dict):
                enc_state = {k: v for k, v in best_state_cpu.items() if k not in ("graph_dense", "best_meta", "graph_decoder", "prediction_decoder")}
            else:
                enc_state = best_state_cpu
            encoder.load_state_dict(enc_state, strict=True)
        ckpt_metric_label = heart_checkpoint_metric if is_heart else random_checkpoint_metric
        print(f"[CKPT] Reloaded best weights (by val {ckpt_metric_label}) from epoch {best_epoch+1}")
    elif os.path.exists(best_ckpt_path_runtime):
        # fallback if only on-disk exists
        try:
            state = torch.load(best_ckpt_path_runtime, map_location=device)
            if isinstance(state, dict) and ("encoder" in state):
                encoder.load_state_dict(state["encoder"], strict=True)
                if (graph_decoder is not None) and ("graph_decoder" in state):
                    graph_decoder.load_state_dict(state["graph_decoder"], strict=True)
                if (prediction_decoder is not None) and ("prediction_decoder" in state):
                    prediction_decoder.load_state_dict(state["prediction_decoder"], strict=True)
                if "graph_dense" in state:
                    best_graph_dense_cpu = state["graph_dense"].detach().cpu().clone()
                if "best_meta" in state:
                    best_meta_cpu = state["best_meta"]
            else:
                if isinstance(state, dict):
                    enc_state = {k: v for k, v in state.items() if k not in ("graph_dense", "best_meta", "graph_decoder", "prediction_decoder")}
                else:
                    enc_state = state
                encoder.load_state_dict(enc_state, strict=True)
            print(f"[CKPT] Reloaded best weights from disk: {best_ckpt_path_runtime}")
        except Exception as e:
            print(f"[CKPT] ERROR loading best checkpoint; using last epoch weights: {e}")
    else:
        print("[CKPT] No best checkpoint found; using last epoch weights.")

    if best_graph_dense_cpu is not None:
        best_graph_dense = best_graph_dense_cpu.to(device)
        best_edge_index = best_graph_dense.to_sparse().indices()
        best_adj_label = best_graph_dense.to_sparse().coalesce()
    else:
        best_edge_index = edge_index
        best_adj_label = adj_label

    # --------- FINAL TEST EVAL WITH BEST MODEL ---------
    encoder.eval()
    if graph_decoder is not None:
        graph_decoder.eval()
    if prediction_decoder is not None:
        prediction_decoder.eval()
    final_score_diag = _score_source_diagnostics(None, None, [], [])
    with torch.no_grad():
        Z_best = encoder(features, best_edge_index)
        if (use_edited_decoder and (graph_decoder is not None)) or (prediction_decoder is not None):
            try:
                final_labels, final_c0p_mask = resolve_edit_targets(
                    Z_best,
                    _to_dense(best_adj_label),
                    freeze_targets=freeze_c0p_at_edit_start,
                    fixed_labels=fixed_c0p_labels,
                    fixed_mask=fixed_c0p_mask,
                    gmm_k=gmm_k,
                    gmm_tau=gmm_tau,
                    restrict_alpha=restrict_alpha,
                    restrict_gamma=restrict_gamma,
                )
                if use_edited_decoder and (graph_decoder is not None):
                    _set_struct_decoder_context(best_adj_label, final_labels, final_c0p_mask)
                if prediction_decoder is not None:
                    _set_prediction_decoder_context(best_adj_label, final_labels, final_c0p_mask)
            except Exception as e:
                print(f"[DECODER-DIAG] final score context failed: {e}")
        if (use_edited_decoder and (graph_decoder is not None)) or (prediction_decoder is not None):
            dot_pos_final = _edge_score_values_from_dot(Z_best, test_edges)
            dot_neg_final = _edge_score_values_from_dot(Z_best, test_edges_false)
            _, _, dot_final_hit = get_scores_from_values(dot_pos_final, dot_neg_final)
            if dot_pos_final.size > 0:
                final_score_diag["diag_dot_pos_mean"] = float(np.mean(dot_pos_final.reshape(-1)))
            if dot_neg_final.size > 0:
                final_score_diag["diag_dot_neg_mean"] = float(np.mean(dot_neg_final.reshape(-1)))
            final_score_diag["diag_dot_test_hit10"] = float(dot_final_hit[2]) if len(dot_final_hit) > 2 else float("nan")
            if use_edited_decoder and (graph_decoder is not None):
                decoder_pos_final = _edge_score_values_from_decoder(graph_decoder, Z_best, test_edges)
                decoder_neg_final = _edge_score_values_from_decoder(graph_decoder, Z_best, test_edges_false)
                _, _, decoder_final_hit = get_scores_from_values(decoder_pos_final, decoder_neg_final)
                final_score_diag.update(
                    _score_source_diagnostics_from_values(
                        dot_pos_final,
                        dot_neg_final,
                        decoder_pos_final,
                        decoder_neg_final,
                    )
                )
                final_score_diag["diag_dot_test_hit10"] = float(dot_final_hit[2]) if len(dot_final_hit) > 2 else float("nan")
                final_score_diag["diag_decoder_test_hit10"] = float(decoder_final_hit[2]) if len(decoder_final_hit) > 2 else float("nan")
            if prediction_decoder is not None:
                pred_pos_final = _edge_score_values_from_decoder(prediction_decoder, Z_best, test_edges)
                pred_neg_final = _edge_score_values_from_decoder(prediction_decoder, Z_best, test_edges_false)
                _, _, pred_decoder_final_hit = get_scores_from_values(pred_pos_final, pred_neg_final)
                pred_final_diag = _score_source_diagnostics_from_values(
                    dot_pos_final,
                    dot_neg_final,
                    pred_pos_final,
                    pred_neg_final,
                )
                final_score_diag["diag_pred_test_hit10"] = float(pred_decoder_final_hit[2]) if len(pred_decoder_final_hit) > 2 else float("nan")
                final_score_diag["diag_pred_pos_mean"] = pred_final_diag.get("diag_decoder_pos_mean", float("nan"))
                final_score_diag["diag_pred_neg_mean"] = pred_final_diag.get("diag_decoder_neg_mean", float("nan"))
                final_score_diag["diag_dot_pred_corr"] = pred_final_diag.get("diag_dot_decoder_corr", float("nan"))
                if use_edited_decoder and (graph_decoder is not None):
                    dec_vals = np.concatenate(
                        [
                            np.asarray(decoder_pos_final).reshape(-1),
                            np.asarray(decoder_neg_final).reshape(-1),
                        ]
                    )
                    pred_vals = np.concatenate(
                        [
                            np.asarray(pred_pos_final).reshape(-1),
                            np.asarray(pred_neg_final).reshape(-1),
                        ]
                    )
                    if dec_vals.size > 1 and pred_vals.size > 1 and np.std(dec_vals) > 0 and np.std(pred_vals) > 0:
                        final_score_diag["diag_decoder_pred_corr"] = float(np.corrcoef(dec_vals, pred_vals)[0, 1])

    print(f"[SCORE] validation/test score_source={score_source} prediction_graph={prediction_graph} best_graph_saved={int(best_graph_dense_cpu is not None)}")
    final_test_roc, final_test_ap, final_test_hit = _evaluate_edges_for_source(
        Z_best,
        test_edges,
        test_edges_false,
        score_matrix_np=None,
    )
    final_ckpt_metric_label = heart_checkpoint_metric if is_heart else random_checkpoint_metric
    print(f"[BEST CHECKPOINT BY VAL {final_ckpt_metric_label.upper()}] epoch = {best_epoch+1}, "
        f"val_roc = {best_val_roc:.3f}, val_ap = {best_val_ap:.3f}, selection_score = {best_checkpoint_score:.6f}")
    print(f"[FINAL TEST] test_roc = {final_test_roc:.5f}, test_ap = {final_test_ap:.5f}")
    print(f"[FINAL TEST] Hit@K: 1={final_test_hit[0]}, 3={final_test_hit[1]}, 10={final_test_hit[2]}, "
        f"20={final_test_hit[3]}, 50={final_test_hit[4]}, 100={final_test_hit[5]}")
    if (use_edited_decoder and (graph_decoder is not None)) or (prediction_decoder is not None):
        final_prediction_extra_diag = _prediction_decoder_extra_diagnostics()
        print(
            f"[DECODER-DIAG][FINAL] "
            f"decoder_type={decoder_type} pred_decoder_type={prediction_decoder_type} normalize_input={int(decoder_normalize_input)} score_source={score_source} "
            f"heart_rank_weight={heart_rank_weight:.6f} prediction_rank_weight={prediction_rank_weight:.6f} "
            f"prediction_h3_gate={final_prediction_extra_diag.get('prediction_h3_gate', float('nan')):.6f} "
            f"prediction_compact_gate={final_prediction_extra_diag.get('prediction_compact_gate', float('nan')):.6f} "
            f"dot_test_hit10={final_score_diag.get('diag_dot_test_hit10', float('nan')):.6f} "
            f"decoder_test_hit10={final_score_diag.get('diag_decoder_test_hit10', float('nan')):.6f} "
            f"pred_test_hit10={final_score_diag.get('diag_pred_test_hit10', float('nan')):.6f} "
            f"dot_pos_mean={final_score_diag.get('diag_dot_pos_mean', float('nan')):.6f} "
            f"dot_neg_mean={final_score_diag.get('diag_dot_neg_mean', float('nan')):.6f} "
            f"decoder_pos_mean={final_score_diag.get('diag_decoder_pos_mean', float('nan')):.6f} "
            f"decoder_neg_mean={final_score_diag.get('diag_decoder_neg_mean', float('nan')):.6f} "
            f"pred_pos_mean={final_score_diag.get('diag_pred_pos_mean', float('nan')):.6f} "
            f"pred_neg_mean={final_score_diag.get('diag_pred_neg_mean', float('nan')):.6f} "
            f"dot_decoder_corr={final_score_diag.get('diag_dot_decoder_corr', float('nan')):.6f} "
            f"dot_pred_corr={final_score_diag.get('diag_dot_pred_corr', float('nan')):.6f} "
            f"decoder_pred_corr={final_score_diag.get('diag_decoder_pred_corr', float('nan')):.6f}"
        )

    if best_meta_cpu is not None:
        print(
            f"[SANITY SUMMARY] best_val_epoch={int(best_meta_cpu['epoch'])+1} val_roc={float(best_meta_cpu['val_roc']):.6f} "
            f"radius_metric={best_meta_cpu.get('compactness_radius_metric', compactness_radius_metric)} "
            f"radius_before={float(best_meta_cpu['radius_before']):.6f} radius_after={float(best_meta_cpu['radius_after']):.6f} "
            f"delta={float(best_meta_cpu['radius_delta']):.6f} "
            f"c0p_radius_before={float(best_meta_cpu.get('c0p_radius_before', float('nan'))):.6f} "
            f"c0p_radius_after={float(best_meta_cpu.get('c0p_radius_after', float('nan'))):.6f} "
            f"cp_radius_before={float(best_meta_cpu.get('cp_radius_before', float('nan'))):.6f} "
            f"cp_radius_after={float(best_meta_cpu.get('cp_radius_after', float('nan'))):.6f} "
            f"noncompact_radius_before={float(best_meta_cpu.get('noncompact_radius_before', float('nan'))):.6f} "
            f"noncompact_radius_after={float(best_meta_cpu.get('noncompact_radius_after', float('nan'))):.6f} "
            f"noncompact_radius_p90_before={float(best_meta_cpu.get('noncompact_radius_p90_before', float('nan'))):.6f} "
            f"noncompact_radius_p90_after={float(best_meta_cpu.get('noncompact_radius_p90_after', float('nan'))):.6f} "
            f"noncompact_radius_max_before={float(best_meta_cpu.get('noncompact_radius_max_before', float('nan'))):.6f} "
            f"noncompact_radius_max_after={float(best_meta_cpu.get('noncompact_radius_max_after', float('nan'))):.6f} "
            f"editor_noncompact_push_strength={float(best_meta_cpu.get('editor_noncompact_push_strength', float('nan'))):.6f} "
            f"editor_noise_push_strength={float(best_meta_cpu.get('editor_noise_push_strength', float('nan'))):.6f} "
            f"editor_push_preserve_norm={float(best_meta_cpu.get('editor_push_preserve_norm', float('nan'))):.0f} "
            f"push_noncompact_count={float(best_meta_cpu.get('push_noncompact_count', float('nan'))):.0f} "
            f"push_noise_count={float(best_meta_cpu.get('push_noise_count', float('nan'))):.0f} "
            f"push_noncompact_anchor_cosdist_before={float(best_meta_cpu.get('push_noncompact_anchor_cosdist_before', float('nan'))):.6f} "
            f"push_noncompact_anchor_cosdist_after={float(best_meta_cpu.get('push_noncompact_anchor_cosdist_after', float('nan'))):.6f} "
            f"push_noise_anchor_cosdist_before={float(best_meta_cpu.get('push_noise_anchor_cosdist_before', float('nan'))):.6f} "
            f"push_noise_anchor_cosdist_after={float(best_meta_cpu.get('push_noise_anchor_cosdist_after', float('nan'))):.6f} "
            f"radius_anchor={float(best_meta_cpu['radius_anchor']):.6f} "
            f"delta_anchor={float(best_meta_cpu['delta_anchor']):.6f} "
            f"delta_prev_rewrite={float(best_meta_cpu['delta_prev_rewrite']):.6f} "
            f"rewrite_applied={int(best_meta_cpu['rewrite_applied'])} "
            f"edit_recon={float(best_meta_cpu['edit_recon']):.6f} "
            f"edit_keep={float(best_meta_cpu.get('edit_keep', float('nan'))):.6f} "
            f"edit_add_rank={float(best_meta_cpu.get('edit_add_rank', float('nan'))):.6f} "
            f"edit_remove_rank={float(best_meta_cpu.get('edit_remove_rank', float('nan'))):.6f} "
            f"edit_heart_rank={float(best_meta_cpu.get('edit_heart_rank', float('nan'))):.6f} "
            f"lp_full_graph={float(best_meta_cpu.get('lp_full_graph', float('nan'))):.0f} "
            f"maskgae_feature_loss={float(best_meta_cpu.get('maskgae_feature_loss', float('nan'))):.6f} "
            f"maskgae_aug_feature_loss={float(best_meta_cpu.get('maskgae_aug_feature_loss', float('nan'))):.6f} "
            f"cimage_factor_loss={float(best_meta_cpu.get('cimage_factor_loss', float('nan'))):.6f} "
            f"cimage_cluster_loss={float(best_meta_cpu.get('cimage_cluster_loss', float('nan'))):.6f} "
            f"cimage_aug_factor_loss={float(best_meta_cpu.get('cimage_aug_factor_loss', float('nan'))):.6f} "
            f"cimage_aug_cluster_loss={float(best_meta_cpu.get('cimage_aug_cluster_loss', float('nan'))):.6f} "
            f"cimage_factor_weight={float(best_meta_cpu.get('cimage_factor_weight', float('nan'))):.6f} "
            f"cimage_cluster_weight={float(best_meta_cpu.get('cimage_cluster_weight', float('nan'))):.6f} "
            f"cimage_num_factors={float(best_meta_cpu.get('cimage_num_factors', float('nan'))):.0f} "
            f"cimage_num_clusters={float(best_meta_cpu.get('cimage_num_clusters', float('nan'))):.0f} "
            f"cimage_pseudo_label_threshold={float(best_meta_cpu.get('cimage_pseudo_label_threshold', float('nan'))):.6f} "
            f"cimage_factor_select_ratio={float(best_meta_cpu.get('cimage_factor_select_ratio', float('nan'))):.6f} "
            f"cimage_mrmr_redundancy_weight={float(best_meta_cpu.get('cimage_mrmr_redundancy_weight', float('nan'))):.6f} "
            f"cimage_cluster_balance_weight={float(best_meta_cpu.get('cimage_cluster_balance_weight', float('nan'))):.6f} "
            f"cimage_sce_power={float(best_meta_cpu.get('cimage_sce_power', float('nan'))):.6f} "
            f"heart_rank_pairs={float(best_meta_cpu.get('heart_rank_pairs', float('nan'))):.0f} "
            f"prediction_rank={float(best_meta_cpu.get('prediction_rank', float('nan'))):.6f} "
            f"prediction_bce={float(best_meta_cpu.get('prediction_bce', float('nan'))):.6f} "
            f"prediction_joint_rank={float(best_meta_cpu.get('prediction_joint_rank', float('nan'))):.6f} "
            f"prediction_joint_bce={float(best_meta_cpu.get('prediction_joint_bce', float('nan'))):.6f} "
            f"prediction_extra_reg={float(best_meta_cpu.get('prediction_extra_reg', float('nan'))):.6f} "
            f"prediction_rank_pairs={float(best_meta_cpu.get('prediction_rank_pairs', float('nan'))):.0f} "
            f"prediction_rank_pairs_total={float(best_meta_cpu.get('prediction_rank_pairs_total', float('nan'))):.0f} "
            f"prediction_rank_hard_pairs={float(best_meta_cpu.get('prediction_rank_hard_pairs', float('nan'))):.0f} "
            f"prediction_rank_easy_pairs={float(best_meta_cpu.get('prediction_rank_easy_pairs', float('nan'))):.0f} "
            f"prediction_rank_dot_gap_mean={float(best_meta_cpu.get('prediction_rank_dot_gap_mean', float('nan'))):.6f} "
            f"prediction_rank_anchor_loss={float(best_meta_cpu.get('prediction_rank_anchor_loss', float('nan'))):.6f} "
            f"prediction_rank_weight={float(best_meta_cpu.get('prediction_rank_weight', float('nan'))):.6f} "
            f"prediction_bce_weight={float(best_meta_cpu.get('prediction_bce_weight', float('nan'))):.6f} "
            f"prediction_encoder_weight={float(best_meta_cpu.get('prediction_encoder_weight', float('nan'))):.6f} "
            f"prediction_rank_struct_frac={float(best_meta_cpu.get('prediction_rank_struct_frac', float('nan'))):.6f} "
            f"prediction_hard_residual_only={float(best_meta_cpu.get('prediction_hard_residual_only', float('nan'))):.0f} "
            f"prediction_hard_margin={float(best_meta_cpu.get('prediction_hard_margin', float('nan'))):.6f} "
            f"prediction_dot_anchor_weight={float(best_meta_cpu.get('prediction_dot_anchor_weight', float('nan'))):.6f} "
            f"prediction_gate_l1_weight={float(best_meta_cpu.get('prediction_gate_l1_weight', float('nan'))):.6f} "
            f"prediction_h3_gate_init={float(best_meta_cpu.get('prediction_h3_gate_init', float('nan'))):.6f} "
            f"prediction_h3_gate={float(best_meta_cpu.get('prediction_h3_gate', float('nan'))):.6f} "
            f"prediction_residual_gate_init={float(best_meta_cpu.get('prediction_residual_gate_init', float('nan'))):.6f} "
            f"prediction_residual_scale={float(best_meta_cpu.get('prediction_residual_scale', float('nan'))):.6f} "
            f"prediction_compact_gate={float(best_meta_cpu.get('prediction_compact_gate', float('nan'))):.6f} "
            f"prediction_compact_residual_scale={float(best_meta_cpu.get('prediction_compact_residual_scale', float('nan'))):.6f} "
            f"decoded_require_structural_support={float(best_meta_cpu.get('decoded_require_structural_support', float('nan'))):.0f} "
            f"decoded_struct_min_cn={float(best_meta_cpu.get('decoded_struct_min_cn', float('nan'))):.6f} "
            f"decoded_struct_min_ra={float(best_meta_cpu.get('decoded_struct_min_ra', float('nan'))):.6f} "
            f"decoded_struct_min_aa={float(best_meta_cpu.get('decoded_struct_min_aa', float('nan'))):.6f} "
            f"edit_add_pairs_pre_struct={float(best_meta_cpu.get('edit_add_pairs_pre_struct', float('nan'))):.0f} "
            f"edit_struct_supported_add_pairs={float(best_meta_cpu.get('edit_struct_supported_add_pairs', float('nan'))):.0f} "
            f"decoder_normalize_input={float(best_meta_cpu.get('decoder_normalize_input', float('nan'))):.0f} "
            f"heart_rank_weight={float(best_meta_cpu.get('heart_rank_weight', float('nan'))):.6f} "
            f"heart_rank_margin={float(best_meta_cpu.get('heart_rank_margin', float('nan'))):.6f} "
            f"heart_rank_neg_k={float(best_meta_cpu.get('heart_rank_neg_k', float('nan'))):.0f} "
            f"selection_score={float(best_meta_cpu.get('selection_score', float('nan'))):.6f} "
            f"diag_dot_val_hit10={float(best_meta_cpu.get('diag_dot_val_hit10', float('nan'))):.6f} "
            f"diag_decoder_val_hit10={float(best_meta_cpu.get('diag_decoder_val_hit10', float('nan'))):.6f} "
            f"diag_pred_val_hit10={float(best_meta_cpu.get('diag_pred_val_hit10', float('nan'))):.6f} "
            f"diag_dot_pos_mean={float(best_meta_cpu.get('diag_dot_pos_mean', float('nan'))):.6f} "
            f"diag_dot_neg_mean={float(best_meta_cpu.get('diag_dot_neg_mean', float('nan'))):.6f} "
            f"diag_decoder_pos_mean={float(best_meta_cpu.get('diag_decoder_pos_mean', float('nan'))):.6f} "
            f"diag_decoder_neg_mean={float(best_meta_cpu.get('diag_decoder_neg_mean', float('nan'))):.6f} "
            f"diag_dot_decoder_corr={float(best_meta_cpu.get('diag_dot_decoder_corr', float('nan'))):.6f} "
            f"diag_pred_pos_mean={float(best_meta_cpu.get('diag_pred_pos_mean', float('nan'))):.6f} "
            f"diag_pred_neg_mean={float(best_meta_cpu.get('diag_pred_neg_mean', float('nan'))):.6f} "
            f"diag_dot_pred_corr={float(best_meta_cpu.get('diag_dot_pred_corr', float('nan'))):.6f} "
            f"diag_decoder_pred_corr={float(best_meta_cpu.get('diag_decoder_pred_corr', float('nan'))):.6f} "
            f"edit_compact={float(best_meta_cpu['edit_compact']):.6f} "
            f"edit_compact_radius={float(best_meta_cpu.get('edit_compact_radius', float('nan'))):.6f} "
            f"edit_compact_proto={float(best_meta_cpu.get('edit_compact_proto', float('nan'))):.6f}"
        )

        # Extra visibility: best validation checkpoint after edit starts.
        post_edit_epochs = [e for e in radius_epoch_hist if e >= edit_start_epoch and e < len(roc_history)]
        if len(post_edit_epochs) > 0 and len(roc_history) > 0:
            best_post_epoch = max(post_edit_epochs, key=lambda e: roc_history[e])
            best_post_idx = radius_epoch_hist.index(best_post_epoch)
            print(
                f"[SANITY SUMMARY POST-EDIT] epoch={best_post_epoch+1} val_roc={roc_history[best_post_epoch]:.6f} "
                f"radius_before={radius_before_hist[best_post_idx]:.6f} radius_after={radius_after_hist[best_post_idx]:.6f} "
                f"delta={radius_after_hist[best_post_idx] - radius_before_hist[best_post_idx]:.6f} "
                f"c0p_radius_before={c0p_radius_before_hist[best_post_idx]:.6f} "
                f"c0p_radius_after={c0p_radius_after_hist[best_post_idx]:.6f} "
                f"cp_radius_before={cp_radius_before_hist[best_post_idx]:.6f} "
                f"cp_radius_after={cp_radius_after_hist[best_post_idx]:.6f} "
                f"radius_anchor={radius_anchor_hist[best_post_idx]:.6f} "
                f"delta_anchor={delta_from_anchor_hist[best_post_idx]:.6f} "
                f"delta_prev_rewrite={delta_from_prev_rewrite_hist[best_post_idx]:.6f} "
                f"rewrite_applied={rewrite_applied_hist[best_post_idx]} "
                f"edit_recon={edit_recon_hist[best_post_idx]:.6f} edit_compact={edit_compact_hist[best_post_idx]:.6f}"
            )

        if len(rewrite_epoch_hist) > 0 and len(roc_history) > 0:
            ref_epoch0 = int(best_meta_cpu['epoch'])
            prior = [i for i, e in enumerate(rewrite_epoch_hist) if e <= ref_epoch0]
            chosen = prior[-1] if len(prior) > 0 else len(rewrite_epoch_hist) - 1
            print(
                f"[SANITY REWRITE SUMMARY] ref_epoch={ref_epoch0+1} rewrite_epoch={rewrite_epoch_hist[chosen]+1} "
                f"radius_before={rewrite_radius_before_hist[chosen]:.6f} radius_after={rewrite_radius_after_hist[chosen]:.6f} "
                f"delta={rewrite_delta_hist[chosen]:.6f} radius_anchor={(float(radius_anchor_value) if radius_anchor_value is not None else float('nan')):.6f} "
                f"delta_anchor={rewrite_delta_anchor_hist[chosen]:.6f} delta_prev_rewrite={rewrite_delta_prev_hist[chosen]:.6f}"
            )
    elif len(radius_before_hist) > 0:
        best_idx = max(range(len(roc_history)), key=lambda i: roc_history[i])
        print(
            f"[SANITY SUMMARY] best_val_epoch={best_idx+1} val_roc={roc_history[best_idx]:.6f} "
            f"radius_before={radius_before_hist[best_idx]:.6f} radius_after={radius_after_hist[best_idx]:.6f} "
            f"delta={radius_after_hist[best_idx] - radius_before_hist[best_idx]:.6f} "
            f"c0p_radius_before={c0p_radius_before_hist[best_idx]:.6f} "
            f"c0p_radius_after={c0p_radius_after_hist[best_idx]:.6f} "
            f"cp_radius_before={cp_radius_before_hist[best_idx]:.6f} "
            f"cp_radius_after={cp_radius_after_hist[best_idx]:.6f} "
            f"radius_anchor={radius_anchor_hist[best_idx]:.6f} "
            f"delta_anchor={delta_from_anchor_hist[best_idx]:.6f} "
            f"delta_prev_rewrite={delta_from_prev_rewrite_hist[best_idx]:.6f} "
            f"rewrite_applied={rewrite_applied_hist[best_idx]} "
            f"edit_recon={edit_recon_hist[best_idx]:.6f} edit_compact={edit_compact_hist[best_idx]:.6f}"
        )
        if len(rewrite_epoch_hist) > 0:
            ref_epoch0 = best_idx
            prior = [i for i, e in enumerate(rewrite_epoch_hist) if e <= ref_epoch0]
            chosen = prior[-1] if len(prior) > 0 else len(rewrite_epoch_hist) - 1
            print(
                f"[SANITY REWRITE SUMMARY] ref_epoch={best_idx+1} rewrite_epoch={rewrite_epoch_hist[chosen]+1} "
                f"radius_before={rewrite_radius_before_hist[chosen]:.6f} radius_after={rewrite_radius_after_hist[chosen]:.6f} "
                f"delta={rewrite_delta_hist[chosen]:.6f} radius_anchor={(float(radius_anchor_value) if radius_anchor_value is not None else float('nan')):.6f} "
                f"delta_anchor={rewrite_delta_anchor_hist[chosen]:.6f} delta_prev_rewrite={rewrite_delta_prev_hist[chosen]:.6f}"
            )

    print(f"Total training time {time.time() - training_time_start}")
    print(f"Average minimum node degree {sum(minimum_node_degree_history) / len(minimum_node_degree_history)}")
    print(minimum_node_degree_history)

    # GMM clusters on final embedding
    # 1) cluster labels on final embedding
    labels_cluster = gmm_labels(Z_best.detach(), K=gmm_k, tau=gmm_tau, metric="cosine")

    # 2) degree for alpha-gamma gating
    adj_dense = adj_label.to_dense()
    deg_excl_self = adj_dense.sum(dim=1) - torch.diag(adj_dense)

    # 3) core (C0p) mask
    core_mask, thr_per_node, dmin_hat, radii = select_gmm_cores(
        Z=Z_best.detach(),
        labels=labels_cluster,
        degrees_excl_self=deg_excl_self,
        alpha=restrict_alpha,
        gamma=restrict_gamma,
        B=Z_best.size(1),
        normalize_cosine=True,
    )

    # 4) TSNE cluster visualization with scope
    scope_str = None
    if pre_prune_scope is not None:
        scope_str = pre_prune_scope

    if Z_best is not None and (cluster_method in ["gmm", "louvain"]):
        try:
            print("[TSNE] plotting latent clusters for final encoder ...")
            Z_vis = Z_best    # or Z_best, whichever you’re using as “final”

            if frozen_gmm_labels is not None:
                labels_cluster = frozen_gmm_labels
                print("[TSNE] using frozen GMM labels from Z0")
            else:
                print("[TSNE] WARNING: frozen GMM labels missing, re-fitting GMM on final Z")
                labels_cluster = gmm_labels(
                    Z_vis.detach(), K=gmm_k, tau=gmm_tau, metric="cosine"
                )

            if frozen_core_mask is not None:
                core_mask = torch.from_numpy(frozen_core_mask).bool()
            else:
                deg_final = _deg_excl_self(adj_label.to_dense())
                core_mask, _, _, _ = select_gmm_cores(
                    Z_vis,
                    labels_cluster,
                    degrees_excl_self=deg_final,
                    alpha=restrict_alpha,
                    gamma=restrict_gamma,
                    B=Z_vis.size(1),
                )
            # ---- TSNE cache: save FINAL embedding + the exact labels/core we visualize ----
            try:
                out_dir = os.path.join(
                    tsne_cache_root,
                    str(dataset_str), str(ver),
                    f"seed{seed}",
                    f"preprune_{pre_prune_frac:.2f}",
                )

                # core_mask might be on CPU; ensure numpy
                core_np = core_mask.detach().cpu().numpy() if hasattr(core_mask, "detach") else np.asarray(core_mask, dtype=bool)
                lbl_np = np.asarray(labels_cluster, dtype=np.int64)

                # radii on final Z (per-node)
                radii_f, _, _ = per_cluster_stats_diag(Z_vis.detach(), lbl_np, normalize_cosine=True)

                y_np = None
                try:
                    y_np = labels.detach().cpu().numpy() if hasattr(labels, "detach") else None
                except Exception:
                    y_np = None

                meta = dict(
                    dataset=str(dataset_str),
                    ver=str(ver),
                    seed=int(seed),
                    pre_prune_frac=float(pre_prune_frac),
                    pre_prune_scope=str(pre_prune_scope),
                    gmm_k=int(gmm_k),
                    gmm_tau=float(gmm_tau),
                    alpha=float(restrict_alpha),
                    gamma=float(restrict_gamma),
                    stage="final",
                    best_epoch=int(best_epoch + 1) if "best_epoch" in locals() else None,
                    val_roc=float(best_val_roc) if "best_val_roc" in locals() else None,
                    val_ap=float(best_val_ap) if "best_val_ap" in locals() else None,
                    test_hit1=float(final_test_hit[0]) if "final_test_hit" in locals() else None,
                    test_hit3=float(final_test_hit[1]) if "final_test_hit" in locals() else None,
                    test_hit10=float(final_test_hit[2]) if "final_test_hit" in locals() else None,
                )

                save_tsne_cache(
                    out_dir,
                    Z=Z_vis.detach(),
                    gmm_labels=lbl_np,
                    core_mask=core_np,
                    y=y_np,
                    radii=radii_f.detach().cpu().numpy(),
                    meta=meta,
                    prefix="final",
                )
            except Exception as e:
                print(f"[TSNE|CACHE] save final failed: {e}")
                
            VisualizeCluster(
                dataset_str,
                Z_vis,
                labels_cluster,
                core_mask=core_mask,
                scope=pre_prune_scope if "pre_prune_scope" in locals() else scope,
                suffix=f"_ver{ver}_{pre_prune_frac:.2f}" if "pre_prune_frac" in locals() else "",
                out_dir=os.path.join(
                    cluster_plot_root,
                    str(dataset_str),
                    str(ver),
                    f"seed{seed}",
                    f"preprune_{pre_prune_frac:.2f}",
                ),
            )
        except Exception as e:
            print(f"[TSNE] skipped due to error: {e}")
            
        # ---- print final radius summary at end of run ----
        try:
            if radii_f is None or lbl_np is None or core_np is None:
                raise RuntimeError("final radius artifacts not available")
            r = radii_f.detach().cpu().numpy().astype(np.float32)  # per-node radii (final)
            is_noise = (lbl_np == -1)
            is_core = core_np & (~is_noise)
            is_noncore = (~core_np) & (~is_noise)

            def _summ(arr, name):
                if arr.size == 0:
                    print(f"[FINAL-RADIUS] {name}: empty")
                    return
                print(f"[FINAL-RADIUS] {name}: mean={arr.mean():.6f} std={arr.std():.6f} "
                    f"p50={np.percentile(arr,50):.6f} p90={np.percentile(arr,90):.6f} max={arr.max():.6f} n={arr.size}")

            _summ(r[~is_noise], "all_non_noise")
            _summ(r[is_core], "core(c0p)")
            _summ(r[is_noncore], "noncore")
        except Exception as e:
            print(f"[FINAL-RADIUS] print failed: {e}")
    
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
            out_dir = radius_history_root
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
            use_rewrite_series = (len(rewrite_epoch_hist) > 0 and len(rewrite_epoch_hist) == len(radius_epoch_hist[:len(rewrite_epoch_hist)]))
            if len(rewrite_epoch_hist) > 0:
                epochs = rewrite_epoch_hist[:len(rewrite_epoch_hist)]
                radius_vals = [float(x) for x in rewrite_radius_after_hist]

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

            df_dict = {
                "epoch": epochs,
                "radius_mean": radius_vals,
                "c0p_radius_before": [c0p_radius_before_hist[radius_epoch_hist.index(e)] if e in radius_epoch_hist else float("nan") for e in epochs],
                "c0p_radius_after": [c0p_radius_after_hist[radius_epoch_hist.index(e)] if e in radius_epoch_hist else float("nan") for e in epochs],
                "cp_radius_before": [cp_radius_before_hist[radius_epoch_hist.index(e)] if e in radius_epoch_hist else float("nan") for e in epochs],
                "cp_radius_after": [cp_radius_after_hist[radius_epoch_hist.index(e)] if e in radius_epoch_hist else float("nan") for e in epochs],
                "val_hit@1": val_hit1,
                "val_hit@3": val_hit3,
                "val_hit@10": val_hit10,
                "val_hit@20": val_hit20,
                "val_hit@50": val_hit50,
                "val_hit@100": val_hit100,
                "added": added,
                "removed": removed,
                "mod_ratio": mod,
            }
            if len(rewrite_epoch_hist) > 0:
                df_dict["radius_before"] = rewrite_radius_before_hist[:len(epochs)]
                df_dict["radius_after"] = rewrite_radius_after_hist[:len(epochs)]
                df_dict["delta"] = rewrite_delta_hist[:len(epochs)]
                df_dict["delta_anchor"] = rewrite_delta_anchor_hist[:len(epochs)]
                df_dict["delta_prev_rewrite"] = rewrite_delta_prev_hist[:len(epochs)]
            df = pd.DataFrame(df_dict)
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



def _edge_score_values_np(adj_rec: np.ndarray, edges) -> np.ndarray:
    edges_arr = np.asarray(edges)
    if edges_arr.size == 0:
        return np.asarray([], dtype=np.float64)
    edges_arr = edges_arr.reshape(-1, 2)
    uu = edges_arr[:, 0].astype(np.int64)
    vv = edges_arr[:, 1].astype(np.int64)
    return np.asarray(adj_rec[uu, vv], dtype=np.float64)


def _score_source_diagnostics(
    dot_scores: np.ndarray | None,
    decoder_scores: np.ndarray | None,
    edges_pos,
    edges_neg,
) -> dict[str, float]:
    diag = {
        "diag_dot_val_hit10": float("nan"),
        "diag_decoder_val_hit10": float("nan"),
        "diag_dot_pos_mean": float("nan"),
        "diag_dot_neg_mean": float("nan"),
        "diag_decoder_pos_mean": float("nan"),
        "diag_decoder_neg_mean": float("nan"),
        "diag_dot_decoder_corr": float("nan"),
    }
    if dot_scores is not None:
        dot_pos = _edge_score_values_np(dot_scores, edges_pos)
        dot_neg = _edge_score_values_np(dot_scores, edges_neg)
        if dot_pos.size > 0:
            diag["diag_dot_pos_mean"] = float(np.mean(dot_pos))
        if dot_neg.size > 0:
            diag["diag_dot_neg_mean"] = float(np.mean(dot_neg))
    if decoder_scores is not None:
        dec_pos = _edge_score_values_np(decoder_scores, edges_pos)
        dec_neg = _edge_score_values_np(decoder_scores, edges_neg)
        if dec_pos.size > 0:
            diag["diag_decoder_pos_mean"] = float(np.mean(dec_pos))
        if dec_neg.size > 0:
            diag["diag_decoder_neg_mean"] = float(np.mean(dec_neg))
    if dot_scores is not None and decoder_scores is not None:
        all_edges = np.concatenate([np.asarray(edges_pos).reshape(-1, 2), np.asarray(edges_neg).reshape(-1, 2)], axis=0)
        if all_edges.shape[0] > 200000:
            all_edges = all_edges[:200000]
        dot_vals = _edge_score_values_np(dot_scores, all_edges)
        dec_vals = _edge_score_values_np(decoder_scores, all_edges)
        if dot_vals.size > 1 and dec_vals.size > 1 and np.std(dot_vals) > 0 and np.std(dec_vals) > 0:
            diag["diag_dot_decoder_corr"] = float(np.corrcoef(dot_vals, dec_vals)[0, 1])
    return diag


def get_scores(dataset_str, edges_pos, edges_neg, adj_rec, adj_orig):
    """
    Supports both:
      1) old flat negatives: [num_neg, 2]
      2) HeaRT negatives:    [num_pos, K, 2]
    """
    def _score_edge(u, v):
        val = adj_rec[u, v]
        try:
            return float(val.item())
        except Exception:
            return float(val)

    edges_pos = np.asarray(edges_pos)
    pos_scores = np.asarray([_score_edge(int(u), int(v)) for u, v in edges_pos], dtype=np.float64)

    edges_neg = np.asarray(edges_neg)

    # HeaRT mode: per-positive negatives [num_pos, K, 2]
    if edges_neg.ndim == 3 and edges_neg.shape[-1] == 2:
        neg_scores = np.zeros((edges_neg.shape[0], edges_neg.shape[1]), dtype=np.float64)
        for i in range(edges_neg.shape[0]):
            for j in range(edges_neg.shape[1]):
                u = int(edges_neg[i, j, 0])
                v = int(edges_neg[i, j, 1])
                neg_scores[i, j] = _score_edge(u, v)

        preds_all = np.concatenate([pos_scores, neg_scores.reshape(-1)])
        labels_all = np.concatenate([
            np.ones_like(pos_scores, dtype=np.float64),
            np.zeros(neg_scores.size, dtype=np.float64),
        ])

        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        hitk = []
        pos_tensor = torch.tensor(pos_scores)
        neg_tensor = torch.tensor(neg_scores)
        for k in [1, 3, 10, 20, 50, 100]:
            hitk.append(eval_hits_heart(pos_tensor, neg_tensor, k))

        return roc_score, ap_score, hitk

    # old mode: shared flat negatives [num_neg, 2]
    neg_scores = np.asarray([_score_edge(int(u), int(v)) for u, v in edges_neg], dtype=np.float64)

    preds_all = np.hstack([pos_scores, neg_scores])
    labels_all = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    hitk = []
    pos_tensor = torch.tensor(pos_scores)
    neg_tensor = torch.tensor(neg_scores)
    for k in [1, 3, 10, 20, 50, 100]:
        hitk.append(eval_hits(pos_tensor, neg_tensor, k))
    return roc_score, ap_score, hitk


def eval_hits(y_pred_pos, y_pred_neg, K):
    """
    compute Hits@K
    For each positive target node, the negative target nodes are the same.
    y_pred_neg is an array.
    rank y_pred_pos[i] against y_pred_neg for each i
    From:
    https://github.com/snap-stanford/ogb/blob/1c875697fdb20ab452b2c11cf8bfa2c0e88b5ad3/ogb/linkproppred/evaluate.py#L214
    """

    if len(y_pred_neg) < K:
        print(len(y_pred_neg))
        print(f'[WARNING]: hits@{K} defaulted to 1')
        return 1.0

    kth_score_in_negative_edges = torch.topk(y_pred_neg, K, largest=True)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
    return hitsK


def eval_hits_heart(y_pred_pos, y_pred_neg, K):
    """
    HeaRT-style personalized Hits@K.
    y_pred_pos: [num_pos]
    y_pred_neg: [num_pos, num_neg_per_pos]
    """
    if y_pred_neg.ndim != 2:
        raise ValueError(f"Expected y_pred_neg shape [num_pos, Kneg], got {tuple(y_pred_neg.shape)}")

    num_pos, num_neg = y_pred_neg.shape
    if num_neg < K:
        print(f'[WARNING]: hits@{K} defaulted to 1 because num_neg={num_neg} < K')
        return 1.0

    kth_scores = torch.topk(y_pred_neg, K, dim=1, largest=True)[0][:, -1]
    hits = (y_pred_pos > kth_scores).float().mean().item()
    return hits
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
