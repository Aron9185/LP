'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import torch

import os
import pickle

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def mask_test_edges(adj, dataset_str, split_seed=None):
    # Function to build test set with 10% positive links
    rng = np.random.RandomState(0 if split_seed is None else int(split_seed))

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    rng.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    if split_seed is None:
        filename = f'/home/retro/ARON/mask_edge/{dataset_str}_mask_edge.pkl'
    else:
        filename = f'/home/retro/ARON/mask_edge/{dataset_str}_splitseed{int(split_seed)}_mask_edge.pkl'
    if os.path.exists(filename):
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = load_data(filename)
        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
        
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = rng.randint(0, adj.shape[0])
        idx_j = rng.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = rng.randint(0, adj.shape[0])
        idx_j = rng.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_data(filename, (adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false))
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

import numpy as np
import scipy.sparse as sp
import os

def mask_test_edges_ogbl(adj, dataset_str, idx_train=None, idx_val=None, idx_test=None):
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    
    # Non-OGB dataset: Manually split edges and generate negative edges
    num_test = int(np.floor(edges.shape[0] / 5.))
    num_val = int(np.floor(edges.shape[0] / 5.))

    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    all_edge_set = set(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    
    train_edge_idx = all_edge_set - set(val_edge_idx) - set(test_edge_idx)
    
    train_edges = edges[list(train_edge_idx)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    #train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    print(f"len(edges): {len(edges)}")

    filename = f'/home/retro/SECRET/mask_edge/{dataset_str}_mask_edge.pkl'
    if os.path.exists(filename):
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = load_data(filename)
        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # Rebuild the adjacency matrix with the training edges
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # Save the mask data for non-OGB datasets
    if idx_train is None:
        save_data(filename, (adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false))

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def _read_heart_pos_edges(path):
    edges = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad edge line in {path}: {line}")
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            edges.append([u, v])
    return np.asarray(edges, dtype=np.int64)


def _resolve_heart_neg_files(dataset_dir, filename="samples.npy"):
    valid_path = os.path.join(dataset_dir, f"heart_valid_{filename}")
    test_path = os.path.join(dataset_dir, f"heart_test_{filename}")

    if os.path.exists(valid_path) and os.path.exists(test_path):
        return valid_path, test_path

    valid_path = os.path.join(dataset_dir, "heart_valid_samples.npy")
    test_path = os.path.join(dataset_dir, "heart_test_samples.npy")

    if os.path.exists(valid_path) and os.path.exists(test_path):
        return valid_path, test_path

    raise FileNotFoundError(
        f"Cannot find HeaRT negative files in {dataset_dir}. "
        f"Tried heart_valid_{filename}, heart_test_{filename}, "
        f"heart_valid_samples.npy, heart_test_samples.npy"
    )


def mask_test_edges_heart(adj, dataset_str, heart_root="dataset", filename="samples.npy"):
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    dataset_dir = os.path.join(heart_root, dataset_str)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"HeaRT dataset directory not found: {dataset_dir}")

    train_pos_path = os.path.join(dataset_dir, "train_pos.txt")
    valid_pos_path = os.path.join(dataset_dir, "valid_pos.txt")
    test_pos_path = os.path.join(dataset_dir, "test_pos.txt")

    train_edges = _read_heart_pos_edges(train_pos_path)
    val_edges = _read_heart_pos_edges(valid_pos_path)
    test_edges = _read_heart_pos_edges(test_pos_path)

    valid_neg_path, test_neg_path = _resolve_heart_neg_files(dataset_dir, filename=filename)
    val_edges_false = np.load(valid_neg_path)
    test_edges_false = np.load(test_neg_path)

    if val_edges_false.ndim != 3 or val_edges_false.shape[-1] != 2:
        raise ValueError(f"Expected val_edges_false shape [num_val, K, 2], got {val_edges_false.shape}")
    if test_edges_false.ndim != 3 or test_edges_false.shape[-1] != 2:
        raise ValueError(f"Expected test_edges_false shape [num_test, K, 2], got {test_edges_false.shape}")

    if len(val_edges) != val_edges_false.shape[0]:
        raise ValueError(
            f"Mismatch: len(val_edges)={len(val_edges)} but val_edges_false.shape[0]={val_edges_false.shape[0]}"
        )
    if len(test_edges) != test_edges_false.shape[0]:
        raise ValueError(
            f"Mismatch: len(test_edges)={len(test_edges)} but test_edges_false.shape[0]={test_edges_false.shape[0]}"
        )

    data = np.ones(train_edges.shape[0], dtype=np.float32)
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    adj_train.eliminate_zeros()

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
