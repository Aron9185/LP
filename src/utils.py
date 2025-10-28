import copy
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as sp
import scipy.stats as st
import torch
import torch.nn.functional as F
from dev import *
from matplotlib import colors
from model import dot_product_decode
from scipy.stats import gaussian_kde, vonmises
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity, kneighbors_graph
from sklearn.preprocessing import QuantileTransformer, normalize

# from sklearn.neighbors import kneighbors_graph
from torch.nn.functional import cosine_similarity
from torch_geometric.utils import (
    add_remaining_self_loops,
    degree,
    remove_self_loops,
    to_undirected,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add


def get_sim(embeds1, embeds2):
    # normalize embeddings across feature dimension
    embeds1 = F.normalize(embeds1)
    embeds2 = F.normalize(embeds2)
    sim = torch.mm(embeds1, embeds2.t())
    return sim

def neighbor_sampling(src_idx, dst_idx, node_dist, sim, 
                    max_degree, aug_degree):
    phi = sim[src_idx, dst_idx].unsqueeze(dim=1)
    phi = torch.clamp(phi, 0, 0.5)

    # print('phi', phi)
    mix_dist = node_dist[dst_idx]*phi + node_dist[src_idx]*(1-phi)

    new_tgt = torch.multinomial(mix_dist + 1e-12, int(max_degree)).to(phi.device)
    tgt_idx = torch.arange(max_degree).unsqueeze(dim=0).to(phi.device)

    new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = src_idx.repeat_interleave(aug_degree)
    return new_row, new_col

def degree_mask_edge(idx, sim, max_degree, node_degree, mask_prob):
    aug_degree = (node_degree * (1- mask_prob)).long().to(sim.device)
    sim_dist = sim[idx]

    # _, new_tgt = th.topk(sim_dist + 1e-12, int(max_degree))
    new_tgt = torch.multinomial(sim_dist + 1e-12, int(max_degree))
    tgt_idx = torch.arange(max_degree).unsqueeze(dim=0).to(sim.device)

    new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = idx.repeat_interleave(aug_degree)
    return new_row, new_col

def degree_aug(bias_Z, adj_labl, degree,num_nodes, edge_mask_rate_1, threshold, epoch):
    if(epoch < 200):
        return adj_labl, 1
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    
    max_degree = np.max(degree)
    src_idx = torch.LongTensor(np.argwhere(degree < threshold).flatten()).to(device)
    rest_idx = torch.LongTensor(np.argwhere(degree >= threshold).flatten()).to(device)
    rest_node_degree = degree[degree>=threshold]    # head node
    
    sim = get_sim(bias_Z, bias_Z)
    sim = torch.clamp(sim, 0, 1)
    sim = sim - torch.diag_embed(torch.diag(sim))
    src_sim = sim[src_idx]

    dst_idx = torch.multinomial(src_sim + 1e-12, 1).flatten().to(device)    # tail node

    rest_node_degree = torch.LongTensor(rest_node_degree)
    degree_dist = scatter_add(torch.ones(rest_node_degree.size()), rest_node_degree).to(device)
    prob = degree_dist.unsqueeze(dim=0).repeat(src_idx.size(0), 1)

    aug_degree = torch.multinomial(prob, 1).flatten().to(device)

    new_row_mix_1, new_col_mix_1 = neighbor_sampling(src_idx, dst_idx, adj_labl, sim, max_degree, aug_degree)   # tail node add
    new_row_rest_1, new_col_rest_1 = degree_mask_edge(rest_idx, sim, max_degree, rest_node_degree, edge_mask_rate_1)    # head node purify
    nsrc1 = torch.cat((new_row_mix_1, new_row_rest_1)).cpu()
    ndst1 = torch.cat((new_col_mix_1, new_col_rest_1)).cpu()
    
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[nsrc1, ndst1] = 1
    adj += torch.eye(num_nodes, dtype=torch.float32)
    adj = adj.to(device)
    return adj, 1

### 0619 modify 
def sparse_mx_to_edge_index(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    row = torch.from_numpy(sparse_mx.row.astype(np.int64))
    col = torch.from_numpy(sparse_mx.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
 
    return edge_index

def knn_graph(X, k=20, metric='minkowski'):
    X = X.cpu().detach().numpy()
    A = kneighbors_graph(X, n_neighbors=k, metric=metric)
    edge_index = sparse_mx_to_edge_index(A)
    edge_index, _ = remove_self_loops(edge_index)
    return edge_index


def merge_neighbors_efficient(adj_label, adj_knn, tail_idx, r, k):
    num_nodes = adj_label.size(0)
    tail_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_label.device)
    tail_mask[tail_idx] = True
    # 根据掩码提取尾节点的邻接信息
    tail_adj_label = adj_label[tail_mask]
    tail_adj_knn = adj_knn[tail_mask]
    old_avg_degree = torch.mean(torch.sum(tail_adj_label,dim=1).float())

    # 计算每个尾节点应该保留的邻居数（r%和1-r%）
    num_neighbors_label = (tail_adj_label.sum(dim=1) * (1-r)).ceil().int()
    num_neighbors_knn = math.ceil(r * k) #(tail_adj_knn.sum(dim=1) * (1 - r)).int()

    # 创建新的邻接矩阵
    new_adj = adj_label.clone()

    # 随机选择原始邻居进行保留
    total_label_neighbors = tail_adj_label.sum(1)
            # probs_label[total_label_neighbors == 0] = 0  # 避免除以零 有可能會有這個問題 但遇到再說zz
    probs_label = tail_adj_label / total_label_neighbors[:, None]  # 归一化概率
    sampled_label = torch.multinomial(probs_label, num_neighbors_label.max(), replacement=True)
    keep_label_mask = torch.zeros_like(tail_adj_label).scatter_(1, sampled_label, 1)

    # 随机选择KNN邻居进行添加
    total_knn_neighbors = tail_adj_knn.sum(1)
    probs_knn = tail_adj_knn / total_knn_neighbors[:, None]  # 归一化概率
    sampled_knn = torch.multinomial(probs_knn, num_neighbors_knn, replacement=False)
    keep_knn_mask = torch.zeros_like(tail_adj_knn).scatter_(1, sampled_knn, 1)

    new_adj[tail_mask] = keep_label_mask + keep_knn_mask
    # new_adj[tail_mask] = keep_label_mask #[tail_mask]  # 更新原始邻居
    # new_adj[tail_mask] = keep_knn_mask #[tail_mask]   # 更新KNN邻居
    all_mask = keep_label_mask + keep_knn_mask
    new_avg_degree = torch.mean(torch.sum(all_mask,dim=1).float())
    print(f"modified tail degree difference: {new_avg_degree - old_avg_degree}")
    return tail_mask.cpu(), keep_label_mask.cpu(), keep_knn_mask.cpu() #new_adj

def purify_head_nodes_efficient(bias_Z, adj_matrix, head_idx, r, new_adj=None):
   
    num_nodes = adj_matrix.size(0)
    head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_matrix.device)
    head_mask[head_idx] = True
    
    head_adj = adj_matrix[head_mask]
    num_to_keep = (head_adj.sum(dim=1)*(1-r)).ceil().int()
    old_avg_degree = torch.mean(torch.sum(head_adj,dim=1).float())
    
    sim_matrix = torch.matmul(bias_Z, bias_Z.t())
    sim_matrix += 2*abs(torch.min(sim_matrix))
    normalized_sim = sim_matrix / sim_matrix.sum(1)[:,None]
    normalized_sim_head = normalized_sim[head_mask]
    # print(normalized_sim_head)
    sampled_indices = torch.multinomial(normalized_sim_head, num_to_keep.max(), replacement=True)
    keep_label_mask = torch.zeros_like(head_adj).scatter_(1, sampled_indices, 1)
    
    # new_adj[head_mask] = 0
    # new_adj[head_mask] = keep_label_mask
    
    new_avg_degree = torch.mean(torch.sum(keep_label_mask,dim=1).float())
    print(f"modified head degree difference: {new_avg_degree - old_avg_degree}")

    return head_mask.cpu(), keep_label_mask.cpu()

def degree_aug_v2(bias_Z, adj_label, degree,num_nodes, edge_flip_rate, threshold, epoch):
    if(epoch < 200):
        return adj_label, 1
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    kg_edge_index = knn_graph(bias_Z, k=threshold, metric='euclidean')
    adj_knn = torch.sparse_coo_tensor(kg_edge_index, torch.ones(kg_edge_index.shape[1]), (num_nodes, num_nodes)).to_dense().to(device)

    head_idx = torch.LongTensor(np.argwhere(degree >= threshold).flatten()).to(device)
    
    # promote low degree node
    tail_idx = torch.LongTensor(np.argwhere(degree < threshold).flatten()).to(device)
    tail_node_degree = degree[degree<threshold]
    idx_degree_pair = np.column_stack((tail_idx.cpu().numpy(), tail_node_degree))
    # transformer = QuantileTransformer(output_distribution='normal', random_state=0)
    # normalized_tail_degrees = transformer.fit_transform(idx_degree_pair[:, 1].reshape(-1, 1)).flatten() + threshold
    # normalized_idx_degree_pair = np.column_stack((idx_degree_pair[:, 0], normalized_tail_degrees))    # pair[.][0] = idx, pair[.][1] = normalize degree 
    
    tail_mask, tail_keep_label_mask, keep_knn_mask = merge_neighbors_efficient(adj_label, adj_knn,tail_idx, edge_flip_rate, threshold)
    head_mask, head_keep_label_mask = purify_head_nodes_efficient(bias_Z, adj_label, head_idx, edge_flip_rate)
    
    new_adj=adj_label.clone()
    new_adj[tail_mask.to(device)] = tail_keep_label_mask.to(device) + keep_knn_mask.to(device)
    new_adj[head_mask.to(device)] = head_keep_label_mask.to(device)
    
    mask = torch.eye(new_adj.size(0), dtype=torch.bool).to(device)
    new_adj.masked_fill_(mask, 0)
    # print(new_adj)
    # demote high degree node
    # num_nodes = adj_label.size(0)
    # head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_label.device)
    # head_mask[head_idx] = True
    # head_adj_matrix = adj_[head_idx]
    
    return new_adj, 1

#### 0621 
#### try topk instead of multinomial
def merge_neighbors_v3(adj_label, adj_knn, tail_idx, r, k):
    num_nodes = adj_label.size(0)
    tail_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_label.device)
    tail_mask[tail_idx] = True

    # 提取尾节点的邻接信息
    tail_adj_label = adj_label[tail_mask]
    tail_adj_knn = adj_knn[tail_mask]

    # 创建新的邻接矩阵
    new_adj = adj_label.clone()
    old_degrees = []
    new_degrees = []
    for i, idx in enumerate(tail_idx):
        # 当前尾节点的原始邻居信息和KNN邻居信息
        current_label_neighbors = tail_adj_label[i]
        old_degrees.append(current_label_neighbors.sum().item())
        current_knn_neighbors = tail_adj_knn[i]

        # 计算应保留和新增的邻居数
        num_to_keep_label = int((current_label_neighbors.sum() * (1-r)).ceil().item())
        num_to_keep_knn = int((current_knn_neighbors.sum() * r).ceil().item())

        # 随机选择原始邻居进行保留
        if current_label_neighbors.sum() > 0:
            all_label_indices = torch.where(current_label_neighbors > 0)[0]
            rand_indices = torch.randperm(all_label_indices.size(0))[:num_to_keep_label]
            keep_label_indices = all_label_indices[rand_indices]
            label_mask = torch.zeros_like(current_label_neighbors)
            label_mask[keep_label_indices] = 1
        else:
            label_mask = torch.zeros_like(current_label_neighbors)

        # 随机选择KNN邻居进行添加
        if current_knn_neighbors.sum() > 0:
            all_knn_indices = torch.where(current_knn_neighbors > 0)[0]
            rand_indices = torch.randperm(all_knn_indices.size(0))[:num_to_keep_knn]
            keep_knn_indices = all_knn_indices[rand_indices]
            knn_mask = torch.zeros_like(current_knn_neighbors)
            knn_mask[keep_knn_indices] = 1
        else:
            knn_mask = torch.zeros_like(current_knn_neighbors)

        # 更新邻接矩阵
        all_mask = label_mask + knn_mask
        new_adj[idx] = all_mask  # 使用逻辑或合并保留和新增的邻居
        new_degrees.append(all_mask.sum().item())

    print(f"[Tail Node] origin avg degree: {np.mean(old_degrees)}, augmented avg degree: {np.mean(new_degrees)}")
    return new_adj
def purify_head_nodes_v3(bias_Z, adj_matrix, head_idx, r, new_adj):
    num_nodes = adj_matrix.size(0)
    head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_matrix.device)
    head_mask[head_idx] = True
    
    # sim_matrix = cosine_similarity(bias_Z, bias_Z)
    sim_matrix = torch.matmul(bias_Z, bias_Z.t())
    sim_matrix += 2*abs(torch.min(sim_matrix))
    normalized_sim = sim_matrix / sim_matrix.sum(1)[:, None]
    head_adj = adj_matrix[head_idx]
    num_to_keep = (head_adj.sum(dim=1) * (1 - r)).ceil().int()
    
    new_adj[head_idx] = 0  # Reset adjacency connections for head nodes
    
    for i, idx in enumerate(head_idx):
        current_sim = normalized_sim[idx]
        n_keep = num_to_keep[i].item()
        if n_keep > 0:
            # Use topk to find the indices of the highest similarity scores
            values, top_indices = torch.topk(current_sim, n_keep, largest=True, sorted=False)
            new_adj[idx, top_indices] = 1  # Set new connections in the adjacency matrix

    return new_adj

def degree_aug_v3(bias_Z, adj_label, degree,num_nodes, edge_flip_rate, threshold, epoch):
    if(epoch < 200):
        return adj_label, 1
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    kg_edge_index = knn_graph(bias_Z, k=threshold, metric='euclidean')
    adj_knn = torch.sparse_coo_tensor(kg_edge_index, torch.ones(kg_edge_index.shape[1]), (num_nodes, num_nodes)).to_dense().to(device)

    head_idx = torch.LongTensor(np.argwhere(degree >= threshold).flatten()).to(device)
    
    # promote low degree node
    tail_idx = torch.LongTensor(np.argwhere(degree < threshold).flatten()).to(device)
    tail_node_degree = degree[degree<threshold]
 
    new_adj = merge_neighbors_v3(adj_label, adj_knn,tail_idx, edge_flip_rate, threshold)
    new_adj = purify_head_nodes_v3(bias_Z, adj_label, head_idx, edge_flip_rate, new_adj)
        
    mask = torch.eye(new_adj.size(0), dtype=torch.bool).to(device)
    new_adj.masked_fill_(mask, 0)
    # print(new_adj)
    # demote high degree node
    # num_nodes = adj_label.size(0)
    # head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_label.device)
    # head_mask[head_idx] = True
    # head_adj_matrix = adj_[head_idx]
    
    return new_adj, 1

def degree_aug_v3p(bias_Z, adj_label, degree,num_nodes, edge_flip_rate, threshold, epoch):
    if(epoch < 200):
        return adj_label, 1
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    kg_edge_index = knn_graph(bias_Z, k=threshold, metric='euclidean')
    adj_knn = torch.sparse_coo_tensor(kg_edge_index, torch.ones(kg_edge_index.shape[1]), (num_nodes, num_nodes)).to_dense().to(device)

    head_idx = torch.LongTensor(np.argwhere(degree >= threshold).flatten()).to(device)
    
    # promote low degree node
    tail_idx = torch.LongTensor(np.argwhere(degree < threshold).flatten()).to(device)
    tail_node_degree = degree[degree<threshold]
 
    new_adj = merge_neighbors_v3p(adj_label, adj_knn,tail_idx, edge_flip_rate, threshold)
    new_adj = purify_head_nodes_v3p(bias_Z, adj_label, head_idx, edge_flip_rate, new_adj)
        
    mask = torch.eye(new_adj.size(0), dtype=torch.bool).to(device)
    new_adj.masked_fill_(mask, 0)
    # print(new_adj)
    # demote high degree node
    # num_nodes = adj_label.size(0)
    # head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_label.device)
    # head_mask[head_idx] = True
    # head_adj_matrix = adj_[head_idx]
    
    return new_adj, 1

#### 0628
#### Previous version (v2) is a little bit similar with uncover 
#### Try another version

def purify_merge(bias_Z, adj_matrix, adj_knn, head_idx, r, k, new_adj=None):
   
    num_nodes = adj_matrix.size(0)
    head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_matrix.device)
    head_mask[head_idx] = True
    
    head_adj = adj_matrix[head_mask]
    head_knn = adj_knn[head_mask]
    num_to_keep = (head_adj.sum(dim=1)*(1-r)).ceil().int()
    num_neighbors_knn = math.ceil(r * k) #(tail_adj_knn.sum(dim=1) * (1 - r)).int()
    old_avg_degree = torch.mean(torch.sum(head_adj,dim=1).float())
    
    sim_matrix = torch.matmul(bias_Z, bias_Z.t())
    sim_matrix += 2*abs(torch.min(sim_matrix))
    normalized_sim = sim_matrix / sim_matrix.sum(1)[:,None]
    normalized_sim_head = normalized_sim[head_mask]
    neighbor_sim_head = normalized_sim_head * head_adj
    # print(normalized_sim_head)
    sampled_indices = torch.multinomial(neighbor_sim_head, num_to_keep.max(), replacement=True)
    keep_label_mask = torch.zeros_like(head_adj).scatter_(1, sampled_indices, 1)
    
    
    total_knn_neighbors = head_knn.sum(1)
    probs_knn = head_knn / total_knn_neighbors[:, None]  # 归一化概率
    sampled_knn = torch.multinomial(probs_knn, num_neighbors_knn, replacement=False)
    keep_knn_mask = torch.zeros_like(head_knn).scatter_(1, sampled_knn, 1)


    all_mask = keep_label_mask + keep_knn_mask
    new_avg_degree = torch.mean(torch.sum(all_mask,dim=1).float())
    print(f"modified head degree difference: {new_avg_degree - old_avg_degree}")

    return head_mask.cpu(), keep_label_mask.cpu(), keep_knn_mask.cpu()

def degree_aug_v4(bias_Z, adj_label, degree,num_nodes, edge_flip_rate, degree_threshold, epoch):
    if(epoch < 200):
        return adj_label, 1
    print(f"degree threshold: {degree_threshold}, edge flip rate: {edge_flip_rate}")
    # print(degree_threshold.grad)
    # print(edge_flip_rate.grad)
    threshold = math.ceil(degree_threshold)
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    kg_edge_index = knn_graph(bias_Z, k=threshold, metric='euclidean')
    adj_knn = torch.sparse_coo_tensor(kg_edge_index, torch.ones(kg_edge_index.shape[1]), (num_nodes, num_nodes)).to_dense().to(device)

    head_idx = torch.LongTensor(np.argwhere(degree >= threshold).flatten()).to(device)
    
    # promote low degree node
    tail_idx = torch.LongTensor(np.argwhere(degree < threshold).flatten()).to(device)
    tail_node_degree = degree[degree<threshold]
    idx_degree_pair = np.column_stack((tail_idx.cpu().numpy(), tail_node_degree))
   
    tail_mask, tail_keep_label_mask, tail_keep_knn_mask = purify_merge(bias_Z, adj_label, adj_knn, tail_idx, edge_flip_rate, threshold)
    head_mask, head_keep_label_mask, head_keep_knn_mask = purify_merge(bias_Z, adj_label, adj_knn, head_idx, edge_flip_rate, threshold)
    
    new_adj=adj_label.clone()
    new_adj[tail_mask.to(device)] = tail_keep_label_mask.to(device) + tail_keep_knn_mask.to(device)
    new_adj[head_mask.to(device)] = head_keep_label_mask.to(device) + head_keep_knn_mask.to(device)
    
    mask = torch.eye(new_adj.size(0), dtype=torch.bool).to(device)
    new_adj.masked_fill_(mask, 0)
    
    return new_adj, 1

### 0629
### degree v4怪怪的，應該是說分兩個部分做puring/merge沒啥道理
### 感覺要就是直接所有的node都丟掉r個

def degree_aug_v5(bias_Z, adj_label, degree,num_nodes, edge_flip_rate, degree_threshold, epoch):
    if(epoch < 200):
        return adj_label, 1
    print(f"degree threshold: {degree_threshold}, edge flip rate: {edge_flip_rate}")
    # print(degree_threshold.grad)
    # print(edge_flip_rate.grad)
    threshold = math.ceil(degree_threshold)
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    kg_edge_index = knn_graph(bias_Z, k=threshold, metric='euclidean')
    adj_knn = torch.sparse_coo_tensor(kg_edge_index, torch.ones(kg_edge_index.shape[1]), (num_nodes, num_nodes)).to_dense().to(device)


    # head_idx = torch.LongTensor(np.argwhere(degree >= threshold).flatten()).to(device)
    
    # promote low degree node
    # tail_idx = torch.LongTensor(np.argwhere(degree < threshold).flatten()).to(device)
    # tail_node_degree = degree[degree<threshold]
    # idx_degree_pair = np.column_stack((tail_idx.cpu().numpy(), tail_node_degree))
    
    # tail_mask, tail_keep_label_mask, tail_keep_knn_mask = purify_merge(bias_Z, adj_label, adj_knn, tail_idx, edge_flip_rate, threshold)
    # head_mask, head_keep_label_mask, head_keep_knn_mask = purify_merge(bias_Z, adj_label, adj_knn, head_idx, edge_flip_rate, threshold)
    new_adj = merge_purify_v2(adj_label, adj_knn, bias_Z, edge_flip_rate, threshold)
    new_adj = new_adj.to(device)
    # new_adj[tail_mask.to(device)] = tail_keep_label_mask.to(device) + tail_keep_knn_mask.to(device)
    # new_adj[head_mask.to(device)] = head_keep_label_mask.to(device) + head_keep_knn_mask.to(device)
    
    mask = torch.eye(new_adj.size(0), dtype=torch.bool).to(device)
    new_adj.masked_fill_(mask, 1)
    
    return new_adj, 1

def degree_aug_v6(bias_Z, adj_label, degree,num_nodes, edge_flip_rate, degree_threshold, epoch):
    if(epoch < 200):
        return adj_label, 1
    print(f"degree threshold: {degree_threshold}, edge flip rate: {edge_flip_rate}")
    # print(degree_threshold.grad)
    # print(edge_flip_rate.grad)
    threshold = math.ceil(degree_threshold)
    device = adj_label.device #torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    kg_edge_index = knn_graph(bias_Z, k=threshold, metric='euclidean')
    adj_knn = torch.sparse_coo_tensor(kg_edge_index, torch.ones(kg_edge_index.shape[1]), (num_nodes, num_nodes)).to_dense().to(device)


    # head_idx = torch.LongTensor(np.argwhere(degree >= threshold).flatten()).to(device)
    
    # promote low degree node
    # tail_idx = torch.LongTensor(np.argwhere(degree < threshold).flatten()).to(device)
    # tail_node_degree = degree[degree<threshold]
    # idx_degree_pair = np.column_stack((tail_idx.cpu().numpy(), tail_node_degree))
    
    # tail_mask, tail_keep_label_mask, tail_keep_knn_mask = purify_merge(bias_Z, adj_label, adj_knn, tail_idx, edge_flip_rate, threshold)
    # head_mask, head_keep_label_mask, head_keep_knn_mask = purify_merge(bias_Z, adj_label, adj_knn, head_idx, edge_flip_rate, threshold)
    new_adj = merge_purify_v3(adj_label, adj_knn, bias_Z, edge_flip_rate, threshold)
    new_adj = new_adj.to(device)
    # new_adj[tail_mask.to(device)] = tail_keep_label_mask.to(device) + tail_keep_knn_mask.to(device)
    # new_adj[head_mask.to(device)] = head_keep_label_mask.to(device) + head_keep_knn_mask.to(device)
    
    # mask = torch.eye(new_adj.size(0), dtype=torch.bool).to(device)
    # new_adj.masked_fill_(mask, 1)
    
    return new_adj, 1

@torch.no_grad()
def _deg_excl_self(adj_dense: torch.Tensor) -> torch.Tensor:
    # degree excluding diagonal self-loops
    return (adj_dense.sum(dim=1) - torch.diag(adj_dense)).to(torch.long)

@torch.no_grad()
def degree_aug_fill_deficit_from_scores_budget(
    scores_frozen: torch.Tensor,      # [N,N] from SAME pretrained encoder (fixed)
    adj_label: torch.Tensor,          # [N,N] 0/1 dense WITH self-loops (current)
    num_nodes: int,
    degree_floor: int,                # target min degree (EXCLUDING self)
    order: str = "desc",              # "desc" or "asc"
    forbid_mask: torch.Tensor = None, # optional [N,N] bool pairs to forbid
    *,
    # ---- GLOBAL budget (total across whole run) ----
    E0: int,                          # ORIGINAL undirected |E| (fixed)
    aug_ratio: float,                 # GLOBAL cap: total adds <= aug_ratio*E0
    global_used_adds: int,            # cumulative adds already applied (count)
    # ---- PER-NODE cap (cumulative across whole run) ----
    deg0_excl_self: torch.Tensor,     # [N] ORIGINAL degrees (exclude self)
    aug_bound: float,                 # per-node cap fraction (relative to deg0)
    node_used_adds: torch.Tensor,     # [N] cumulative adds that touched each node
    # ---- OPTIONAL per-epoch cap (fraction of E0). Pass None to disable ----
    aug_ratio_epoch: float = None,
):
    """
    Add-only augmentation with:
      - GLOBAL cap: aug_ratio * E0 (cumulative across run)
      - PER-NODE cap: ceil(aug_bound * max(1, deg0[i]))  BUT at least current deficit
      - Optional per-epoch cap: aug_ratio_epoch * E0
    """
    device = adj_label.device
    N = num_nodes
    g = adj_label.clone()

    # Current degrees & low-degree set (exclude self)
    deg = _deg_excl_self(g)
    low = torch.nonzero(deg < degree_floor, as_tuple=False).view(-1)
    if low.numel() == 0:
        return g, 0, global_used_adds, node_used_adds

    # Build forbid mask: self + existing edges + external
    if forbid_mask is None:
        forbid_mask = torch.eye(N, dtype=torch.bool, device=device)
    else:
        forbid_mask = forbid_mask.to(device) | torch.eye(N, dtype=torch.bool, device=device)
    forbid_mask = forbid_mask | (g > 0)

    # ---- Per-node cumulative caps (based on ORIGINAL degrees) ----
    base = torch.maximum(deg0_excl_self.to(device), torch.ones_like(deg0_excl_self, device=device))
    node_caps = torch.ceil(aug_bound * base.float()).to(torch.long)

    # ensure enough headroom to reach the floor (cap >= current deficit)
    deficits = torch.clamp(degree_floor - deg, min=0)
    node_caps = torch.maximum(node_caps, deficits)

    # ------------------- CAP-AWARE, WIDER CANDIDATE POOL -------------------
    # Helpers that still have headroom this epoch (global cumulative)
    helper_headroom = (node_caps - node_used_adds.to(device))
    helper_ok = (helper_headroom > 0)   # [N] bool
    num_helpers_sat = int((helper_headroom <= 0).sum().item())
    print(f"[AUG] helpers_saturated={num_helpers_sat} / {N}")

    pool_mult = 5          # widen factor: try up to gap*pool_mult per node
    pool_abs_ceiling = 256 # hard ceiling to avoid huge lists (tune if needed)

    cand_pairs = []  # (score, a, b) with a<b
    for v in low.tolist():
        gap = int(degree_floor - deg[v].item())
        if gap <= 0:
            continue

        # forbid self + existing + external
        invalid = forbid_mask[v].clone()

        # additionally forbid helpers with no headroom
        # (do NOT forbid v itself; we count its own cap via eff_cap later)
        # invalid = invalid | (~helper_ok)
        # NEW: forbid saturated helpers only if they are not low
        invalid = invalid | ((~helper_ok) & (deg >= degree_floor))

        if invalid.all():
            continue

        s = scores_frozen[v].clone()
        avail_cnt = int((~invalid).sum().item())
        if avail_cnt <= 0:
            continue

        k_pool = min(max(gap * pool_mult, gap), avail_cnt, pool_abs_ceiling)

        if order == "desc":
            s.masked_fill_(invalid, float("-inf"))
            idx = torch.topk(s, k=k_pool, largest=True).indices
            sel_scores = s[idx]
        else:
            s.masked_fill_(invalid, float("+inf"))
            idx = torch.topk(-s, k=k_pool, largest=True).indices
            sel_scores = scores_frozen[v][idx]  # record original scores for ordering

        for u_i, sc in zip(idx.tolist(), sel_scores.tolist()):
            if u_i == v:
                continue
            a, b = (v, u_i) if v < u_i else (u_i, v)
            cand_pairs.append((float(sc), a, b))

    if not cand_pairs:
        return g, 0, global_used_adds, node_used_adds

    # De-duplicate: keep best score for each (a,b)
    best_for_pair = {}
    for sc, a, b in cand_pairs:
        key = (a, b)
        if key not in best_for_pair or (
            (order == "desc" and sc > best_for_pair[key]) or
            (order == "asc"  and sc < best_for_pair[key])
        ):
            best_for_pair[key] = sc

    # Sort globally by score
    cand_sorted = sorted(
        [(sc, a, b) for (a, b), sc in best_for_pair.items()],
        key=lambda t: t[0],
        reverse=(order == "desc")
    )

    # Budgets
    global_cap = max(0, int(round(aug_ratio * E0)))
    global_left = max(0, global_cap - global_used_adds)
    epoch_cap = None if aug_ratio_epoch is None else max(0, int(round(aug_ratio_epoch * E0)))

    # Diagnostics
    blocked_now = ((node_caps - node_used_adds.to(device)) <= 0) & (deg < degree_floor)
    print(f"[AUG] requested_add={len(cand_sorted)} | low_nodes={low.numel()} | "
          f"deg_floor={degree_floor} | order={order} | "
          f"GLOBAL used={global_used_adds}/{global_cap} left={global_left} | "
          f"node_cap(frac)={aug_bound} | epoch_cap={epoch_cap} | "
          f"blocked_by_cap={int(blocked_now.sum().item())}")

    applied = 0
    node_used = node_used_adds.clone()

    # Greedy selection under global & per-node caps (and optional epoch cap)
    for sc, a, b in cand_sorted:
        if global_left <= 0:
            break
        if epoch_cap is not None and applied >= epoch_cap:
            break
        if g[a, b] != 0:
            continue

        # CHANGED: dynamic effective caps (cap must also allow reaching the floor)
        # eff_cap_a = max(int(node_caps[a].item()), max(0, int(degree_floor - deg[a].item())))
        # eff_cap_b = max(int(node_caps[b].item()), max(0, int(degree_floor - deg[b].item())))
        # if node_used[a] >= eff_cap_a or node_used[b] >= eff_cap_b:
        #     continue
        cap_block_a = (deg[a] >= degree_floor) and (node_used[a] >= node_caps[a])
        cap_block_b = (deg[b] >= degree_floor) and (node_used[b] >= node_caps[b])
        if cap_block_a or cap_block_b:
            continue

        # Apply
        g[a, b] = 1
        g[b, a] = 1
        deg[a] += 1
        deg[b] += 1
        node_used[a] += 1
        node_used[b] += 1
        applied += 1
        global_left -= 1

    g.fill_diagonal_(1)
    g = torch.maximum(g, g.t())

    global_used_adds += applied
    node_used_adds.copy_(node_used)

    print(f"[AUG] applied_add={applied} | global_used={global_used_adds}/{global_cap} "
          f"| per-node_max_used={int(node_used_adds.max().item())}")
    return g, applied, global_used_adds, node_used_adds

@torch.no_grad()
def build_cluster_allow_mask(
    labels: np.ndarray,         # np.int32 array shape [N], -1 is noise
    mode: str,                  # "intra" or "inter"
    exclude_noise: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Returns a [N,N] bool matrix where True means the pair is ALLOWED by the cluster rule.
    """
    assert mode in ("intra", "inter")
    lbl = torch.from_numpy(labels.astype(np.int64))
    if device is not None:
        lbl = lbl.to(device)

    if mode == "intra":
        ok = (lbl[:, None] == lbl[None, :])
    else:  # inter
        ok = (lbl[:, None] != lbl[None, :])

    if exclude_noise:
        ok = ok & (lbl[:, None] != -1) & (lbl[None, :] != -1)

    # (Optional) we don't need to allow self-pairs; degree_aug will already forbid diag
    ok.fill_diagonal_(False)
    return ok

@torch.no_grad()
def density_cluster_embeddings(
    Z: torch.Tensor,
    method: str = "hdbscan",             # default now HDBSCAN
    eps: float | None = None,            # kept for backward-compat; IGNORED
    min_samples: int = 5,
    metric: str = "cosine",
    min_cluster_size: int | None = None, # NEW: HDBSCAN knob
) -> np.ndarray:
    """
    Run HDBSCAN clustering on encoder features Z (N,d).
    Returns np.ndarray labels (N,), where -1 = noise.

    Notes:
      - `eps` is ignored (HDBSCAN doesn't use eps).
      - If metric == "cosine", we L2-normalize Z and use Euclidean distances,
        which is equivalent to cosine distance ranking.
    """
    try:
        import hdbscan
    except ImportError as e:
        raise RuntimeError("Please `pip install hdbscan` to use HDBSCAN.") from e

    Zc = Z.detach().cpu().numpy()
    N = Zc.shape[0]

    # Normalize for stability and cosine geometry
    Zc_norm = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-12)

    # Heuristics for defaults
    if min_cluster_size is None:
        min_cluster_size = max(10, int(0.005 * N))  # ~0.5% of N, at least 10
    if min_samples is None:
        min_samples = max(5, int(0.002 * N))        # ~0.2% of N, at least 5

    if eps is not None:
        print("[CLUSTER] HDBSCAN ignores `eps`; provided value will be ignored.")

    # Use Euclidean on normalized vectors (≈ cosine). You can also set metric="cosine" directly.
    dist_metric = "euclidean" if metric == "cosine" else metric

    print(f"[CLUSTER] HDBSCAN metric={dist_metric} "
          f"min_cluster_size={min_cluster_size} min_samples={min_samples} N={N}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=dist_metric,
        core_dist_n_jobs=1,   # deterministic-ish and safe in many envs
    )
    labels = clusterer.fit_predict(Zc_norm if metric == "cosine" else Zc)

    # Debug: label histogram
    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"[CLUSTER] label hist: {dict(zip(uniq.tolist(), cnt.tolist()))}")

    return labels


@torch.no_grad()
def _scores_from_Z(Z: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Fallback confidence if you don’t want to use frozen_scores."""
    X = torch.nn.functional.normalize(Z, p=2, dim=1) if normalize else Z
    return X @ X.t()

@torch.no_grad()
def _svd_embed_from_scores(scores: torch.Tensor, k: int = 64) -> torch.Tensor:
    """
    If you don't have Z handy, derive an embedding from scores via SVD.
    Returns Z_k in R^{N×k}.
    """
    # scores is symmetric-ish; use top-k singular vectors
    U, S, Vh = torch.linalg.svd(scores, full_matrices=False)
    k = min(k, U.size(1))
    Zk = U[:, :k] * torch.sqrt(S[:k].clamp_min(1e-12))
    return Zk

@torch.no_grad()
def cluster_filtered_link_augmentation(
    Z_for_cluster: torch.Tensor,        # [N,d]
    adj_label: torch.Tensor,            # [N,N] 0/1 dense WITH self-loops
    deg0_excl_self: torch.Tensor,       # [N] ORIGINAL degrees (exclude self)
    E0: int,                            # ORIGINAL undirected |E|
    *,
    frozen_scores: torch.Tensor | None,
    forbid_mask: torch.Tensor | None,
    mode: str = "any",                  # "any" | "intra" | "inter"
    labels: np.ndarray | None = None,
    aug_ratio: float = 0.10,
    aug_bound: float = 0.10,
    topk_per_node: int = 64,
    exclude_noise: bool = True,
    charge_both_endpoints: bool = True,

    # ---- already-added relaxation knobs (keep yours) ----
    relax_when_stuck: bool = True,
    patience: int = 1,
    allow_noise_tier: bool = True,
    enable_nearby_cross: bool = True,
    nearby_topk: int = 10,
    late_score_q: float | None = 0.90,

    # ---- NEW: epoch-bounded spending & tier gating ----
    epoch: int | None = None,           # 0-based current epoch
    max_epochs: int | None = None,      # total epochs
    epoch_cap_mode: str = "linear",     # "linear" | "cosine"
    global_used_so_far: int = 0,        # cumulative adds before this epoch
    node_used_so_far: torch.Tensor | None = None,  # [N] cumulative per-node adds
    # tier release points as cumulative fraction of training progress
    # A, B, C, D become eligible when progress >= these frac thresholds
    tier_release_fracs: tuple[float, float, float, float] = (0.00, 0.25, 0.50, 0.75),
):
    """
    Same behavior as before, but:
      - Spending is throttled by epoch quota
      - Tiers A/B/C/D unlock by epoch progress
    """
    device = adj_label.device
    N = adj_label.size(0)
    g = adj_label.clone()
    deg = _deg_excl_self(g)

    # ----- scores (unchanged; keep your code here) -----
    # ... your existing frozen_scores handling ...
    scores = None
    if isinstance(frozen_scores, torch.Tensor) and tuple(frozen_scores.shape) == (N, N):
        scores = frozen_scores.to(device)
    if scores is None:
        scores = _scores_from_Z(Z_for_cluster)

    # ----- base forbid (diag, existing edges, user forbid) -----
    if forbid_mask is None:
        forbid_base = torch.eye(N, dtype=torch.bool, device=device)
    else:
        forbid_base = (forbid_mask.to(device) | torch.eye(N, dtype=torch.bool, device=device))
    forbid_base = forbid_base | (g > 0)

    # ----- budgets (persist across tiers) -----
    base = torch.maximum(deg0_excl_self.to(device), torch.ones_like(deg0_excl_self, device=device))
    node_caps = torch.ceil(aug_bound * base.float()).to(torch.long)
    global_cap = max(0, int(round(aug_ratio * E0)))

    # NEW: epoch quota
    def _epoch_target_frac(ep: int, T: int) -> float:
        # cumulative target by end of this epoch
        # linear: (ep+1)/T ; cosine: slow start, fast later
        if epoch_cap_mode == "cosine":
            import math
            return (1.0 - math.cos(math.pi * (ep + 1) / max(1, T))) * 0.5
        return float(ep + 1) / max(1, T)

    if epoch is not None and max_epochs is not None:
        target_cum = int(round(global_cap * _epoch_target_frac(epoch, max_epochs)))
        global_left = max(0, target_cum - int(global_used_so_far))
        print(f"[EPOCH] ep={epoch}/{max_epochs-1} | global_cap={global_cap} | target_cum={target_cum} "
              f"| used_so_far={global_used_so_far} | epoch_quota={global_left}")
    else:
        global_left = global_cap
        print(f"[EPOCH] ep-bounds disabled | global_cap={global_cap} | epoch_quota={global_left}")

    # carry over per-node usage if provided
    node_used = torch.zeros(N, dtype=torch.long, device=device) if node_used_so_far is None else node_used_so_far.to(device).clone()

    print(f"[AUG-{mode}] per-node cap frac={aug_bound} | topk={topk_per_node}")

    # ----- helpers (same as before; abridged here) -----
    def _topk_mask_per_row(S: torch.Tensor, k: int) -> torch.Tensor:
        k = max(0, min(k, S.size(1)))
        if k == 0: return torch.zeros_like(S, dtype=torch.bool)
        idx = torch.topk(S, k=k, dim=1, largest=True).indices
        mask = torch.zeros_like(S, dtype=torch.bool)
        rows = torch.arange(S.size(0), device=S.device).unsqueeze(1).expand_as(idx)
        mask[rows, idx] = True
        mask.fill_diagonal_(False)
        return mask

    def _build_cluster_allow(mode_local: str, excl_noise: bool) -> torch.Tensor:
        if mode_local == "none" or labels is None:
            return torch.ones((N, N), dtype=torch.bool, device=device).fill_diagonal_(False)
        return build_cluster_allow_mask(labels=labels, mode=mode_local, exclude_noise=excl_noise, device=device)

    def _add_with_forbid(current_forbid: torch.Tensor) -> int:
        nonlocal global_left
        cand = []
        for v in range(N):
            s = scores[v].clone()
            invalid = current_forbid[v]
            if invalid.all(): continue
            s.masked_fill_(invalid, float("-inf"))
            k = min(topk_per_node, int((~invalid).sum().item()))
            if k <= 0: continue
            idx = torch.topk(s, k=k, largest=True).indices
            sv = s[idx]
            for u, sc in zip(idx.tolist(), sv.tolist()):
                if u == v: continue
                a, b = (v, u) if v < u else (u, v)
                cand.append((float(sc), a, b))
        if not cand: return 0

        best = {}
        for sc, a, b in cand:
            if (a, b) not in best or sc > best[(a, b)]:
                best[(a, b)] = sc
        cand_sorted = sorted([(sc, a, b) for (a, b), sc in best.items()],
                             key=lambda t: t[0], reverse=True)

        applied = 0
        for sc, a, b in cand_sorted:
            if global_left <= 0: break
            if g[a, b] != 0: continue
            # per-node cap
            if charge_both_endpoints:
                if node_used[a] >= node_caps[a] or node_used[b] >= node_caps[b]:
                    continue
            else:
                aa, bb = (a, b) if deg[a] <= deg[b] else (b, a)
                if node_used[aa] >= node_caps[aa]:
                    continue
            # add
            g[a, b] = 1; g[b, a] = 1
            applied += 1; global_left -= 1
            deg[a] += 1; deg[b] += 1
            if charge_both_endpoints:
                node_used[a] += 1; node_used[b] += 1
            else:
                aa, bb = (a, b) if deg[a] <= deg[b] else (b, a)
                node_used[aa] += 1
        return applied

    nearby_mask = _topk_mask_per_row(scores, nearby_topk) if enable_nearby_cross else None
    score_thr = None
    if late_score_q is not None and 0.0 < late_score_q < 1.0:
        s_valid = scores.masked_fill(forbid_base, float("-inf"))
        vals = s_valid[s_valid > float("-inf")]
        if vals.numel() > 0:
            score_thr = torch.quantile(vals, late_score_q)

    # ----- epoch-gated tier schedule -----
    def _eligible_tiers():
        # If no labels or mode=="any": only D
        if labels is None or mode not in ("intra", "inter"):
            return [("D", dict(mode="none", exclude_noise=False, nearby_only=False))]
        A_rel, B_rel, C_rel, D_rel = tier_release_fracs
        if epoch is None or max_epochs is None:
            progress = 1.0  # allow all
        else:
            progress = float(epoch + 1) / max(1, max_epochs)
        tiers = []
        if progress >= A_rel:
            tiers.append(("A", dict(mode=mode, exclude_noise=exclude_noise, nearby_only=False)))
        if allow_noise_tier and progress >= B_rel:
            tiers.append(("B", dict(mode=mode, exclude_noise=False,    nearby_only=False)))
        if enable_nearby_cross and mode == "intra" and progress >= C_rel:
            tiers.append(("C", dict(mode="inter", exclude_noise=False, nearby_only=True)))
        if progress >= D_rel:
            tiers.append(("D", dict(mode="none",  exclude_noise=False, nearby_only=False)))
        return tiers

    tiers = _eligible_tiers()
    print(f"[TIER] epoch_progress={ (epoch+1)/max_epochs if (epoch is not None and max_epochs) else 1.0 :.3f} "
          f"| eligible={[t for t,_ in tiers]}")

    applied_total = 0
    zero_passes = 0
    tier_idx = 0
    while global_left > 0 and tier_idx < len(tiers):
        tier_name, cfg = tiers[tier_idx]

        forbid = forbid_base.clone()
        allow = _build_cluster_allow(cfg["mode"], cfg["exclude_noise"])
        if cfg.get("nearby_only", False) and nearby_mask is not None:
            allow = allow & nearby_mask
        forbid = forbid | (~allow)
        if tier_name == "D" and score_thr is not None:
            forbid = forbid | (scores < score_thr)

        added = _add_with_forbid(forbid)
        applied_total += added

        print(f"[AUG-{mode}|Tier {tier_name}] added={added} | epoch_used={ ( (int(round(aug_ratio*E0)) if (epoch is None or max_epochs is None) else int(round(global_cap * _epoch_target_frac(epoch, max_epochs))) ) - global_left) } | quota_left={global_left}")

        if added == 0:
            zero_passes += 1
            if relax_when_stuck and zero_passes >= patience:
                tier_idx += 1
                zero_passes = 0
            else:
                break
        else:
            zero_passes = 0
            # keep working on same tier until it stalls

    g.fill_diagonal_(1)
    g = torch.maximum(g, g.t())
    print(f"[AUG-{mode}] applied_add={applied_total} | epoch_quota_used={ ( (int(round(aug_ratio*E0)) if (epoch is None or max_epochs is None) else int(round(global_cap * _epoch_target_frac(epoch, max_epochs))) ) - global_left) } "
          f"| per-node_max_used={int(node_used.max().item())}")
    return g, applied_total, node_used  # return node_used so caller can persist it

# ---- pick one clustering method and compute once per run ----
# GMM on Z (embedding space)
import numpy as np
from sklearn.mixture import GaussianMixture


def gmm_labels(Z, K, tau=0.55, metric="cosine"):
    X = Z.detach().cpu().numpy()
    if metric == "cosine":
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    gmm = GaussianMixture(n_components=K, covariance_type="diag", random_state=0)
    gmm.fit(X)
    P = gmm.predict_proba(X)
    lbl = P.argmax(1).astype(np.int64)
    lbl[P.max(1) < tau] = -1  # low-confidence → noise
    return lbl

# Louvain on graph structure
import community as community_louvain  # pip install python-louvain
import networkx as nx


def louvain_labels(adj_dense_or_sparse):
    G = nx.from_scipy_sparse_array(adj_dense_or_sparse) if hasattr(adj_dense_or_sparse, "tocsr") \
        else nx.from_numpy_array(adj_dense_or_sparse)
    part = community_louvain.best_partition(G, random_state=0, resolution=1.0)
    N = G.number_of_nodes()
    return np.array([part[i] for i in range(N)], dtype=np.int64)

# ========================================================================

def robust_score_quantile_from_scores(
    scores: torch.Tensor,
    block_mask: torch.Tensor,     # True where we should ignore (e.g., forbid/edges/diag)
    q: float | None,
    max_elems: int = 2_000_000,   # sample size cap; tune if needed
) -> torch.Tensor | None:
    """
    Returns a scalar threshold at quantile q of the allowed scores.
    Works for huge matrices by sampling up to max_elems elements on CPU.
    """
    if q is None or not (0.0 < q < 1.0):
        return None

    # Mask out blocked entries
    s_valid = scores.masked_fill(block_mask, float("-inf"))
    vals = s_valid[s_valid > float("-inf")]   # 1-D view of allowed entries

    if vals.numel() == 0:
        return None

    v = vals.detach()
    # Always move heavy quantile to CPU float32
    if v.is_cuda:
        v = v.float().cpu()
    else:
        v = v.float()

    n = v.numel()
    if n > max_elems:
        # Random uniform sample without replacement (good enough for thresholding)
        idx = torch.randint(0, n, (max_elems,))
        v = v[idx]
        print(f"[AUG] quantile sampling: {v.numel()}/{n} elements for q={q}")

    thr = torch.quantile(v, q)
    # return on same device as scores for later comparisons
    return thr.to(scores.device)

@torch.no_grad()
def degree_deficit_snapshot(
    adj_label: torch.Tensor,          # [N,N] dense WITH self-loops
    degree_floor: int,                # target min degree (EXCLUDING self)
    *,
    out_dir: str = "logs/deg_deficit",
    tag: str = "before",              # "before" | "after"
    epoch: int | None = None,
    run_name: str = "default_run",
    topk: int = 50,
    writer=None,                      # optional: SummaryWriter
    save_snapshot: bool = False,      # <<< NEW: default False for fast per-epoch logging
):
    """
    Computes per-node degree deficit: max(0, degree_floor - deg_excl_self).
    Saves (always):
      - CSV: node, deg_excl_self, deficit   (one per epoch when called)
    Optionally (if save_snapshot=True):
      - Histogram PNG
      - Top-K deficits bar PNG
    Returns a metrics dict for the logger.
    """
    os.makedirs(out_dir, exist_ok=True)
    N = adj_label.size(0)
    deg = _deg_excl_self(adj_label)
    deficits = torch.clamp(degree_floor - deg, min=0).to(torch.long)

    total_def = int(deficits.sum().item())
    num_low = int((deficits > 0).sum().item())
    max_def = int(deficits.max().item())
    mean_def = float(deficits.float().mean().item())
    median_def = float(deficits.float().median().item())

    epoch_str = f"{epoch:03d}" if epoch is not None else "NA"
    base = os.path.join(out_dir, f"{run_name}_{tag}_e{epoch_str}")

    # CSV (per-epoch)
    nodes = torch.arange(N, device=adj_label.device)
    csv_path = base + ".csv"
    pd.DataFrame({
        "node": nodes.cpu().numpy(),
        "deg_excl_self": deg.cpu().numpy().astype(np.int64),
        "deficit": deficits.cpu().numpy().astype(np.int64),
    }).to_csv(csv_path, index=False)

    # Optional heavier plots
    hist_path, bar_path = None, None
    if save_snapshot:
        # Histogram
        hist_path = base + "_hist.png"
        plt.figure(figsize=(6.5, 4.0), dpi=130)
        bins = np.arange(-0.5, max_def + 1.5, 1.0) if max_def > 0 else np.arange(-0.5, 1.5, 1.0)
        plt.hist(deficits.cpu().numpy(), bins=bins)
        plt.xlabel("Degree deficit (floor - deg_excl_self)")
        plt.ylabel("Node count")
        plt.title(f"{run_name} | {tag} | epoch={epoch_str}\n"
                  f"total={total_def} | low_nodes={num_low} | max={max_def}")
        plt.tight_layout()
        plt.savefig(hist_path, bbox_inches="tight")
        plt.close()

        # Top-K bar
        if max_def > 0 and num_low > 0:
            deficits_np = deficits.cpu().numpy()
            top_idx = np.argsort(-deficits_np)[:topk]
            top_vals = deficits_np[top_idx]
            plt.figure(figsize=(max(6.5, min(14, 0.15 * len(top_idx) + 2)), 4.0), dpi=130)
            plt.bar(np.arange(len(top_idx)), top_vals)
            plt.xticks(np.arange(len(top_idx)), top_idx, rotation=90, fontsize=7)
            plt.xlabel("Node id (sorted by deficit)")
            plt.ylabel("Deficit")
            plt.title(f"{run_name} | {tag} | epoch={epoch_str} | top-{len(top_idx)}")
            plt.tight_layout()
            bar_path = base + f"_top{len(top_idx)}.png"
            plt.savefig(bar_path, bbox_inches="tight")
            plt.close()

    # Optional TB scalars
    if writer is not None:
        step = 0 if epoch is None else int(epoch)
        writer.add_scalar(f"{run_name}/deg_deficit_total/{tag}", total_def, step)
        writer.add_scalar(f"{run_name}/deg_deficit_num_low/{tag}", num_low, step)
        writer.add_scalar(f"{run_name}/deg_deficit_max/{tag}", max_def, step)
        writer.add_scalar(f"{run_name}/deg_deficit_mean/{tag}", mean_def, step)
        writer.add_scalar(f"{run_name}/deg_deficit_median/{tag}", median_def, step)

    print(f"[DEFICIT] tag={tag} epoch={epoch_str} N={N} "
          f"total={total_def} low_nodes={num_low} max={max_def} "
          f"mean={mean_def:.3f} median={median_def:.3f}")
    print(f"[DEFICIT] saved: {csv_path}")
    if save_snapshot:
        if hist_path: print(f"[DEFICIT] saved: {hist_path}")
        if bar_path:  print(f"[DEFICIT] saved: {bar_path}")

    return {
        "total_deficit": total_def,
        "num_low": num_low,
        "max_deficit": max_def,
        "mean_deficit": mean_def,
        "median_deficit": median_def,
        "csv_path": csv_path,
        "hist_path": hist_path,
        "bar_path": bar_path,
    }


class DegreeDeficitLogger:
    """Track deficit over epochs and emit series CSV + line charts at the end."""
    def __init__(self, out_dir: str = "logs/deg_deficit", run_name: str = "default_run", ema_alpha: float | None = None):
        self.out_dir = out_dir
        self.run_name = run_name
        self.ema_alpha = ema_alpha
        os.makedirs(self.out_dir, exist_ok=True)
        self.epochs, self.total, self.num_low, self.max_def, self.mean_def, self.median_def = [], [], [], [], [], []

    def log_point(self, epoch: int, metrics: dict):
        self.epochs.append(int(epoch))
        self.total.append(metrics["total_deficit"])
        self.num_low.append(metrics["num_low"])
        self.max_def.append(metrics["max_deficit"])
        self.mean_def.append(metrics["mean_deficit"])
        self.median_def.append(metrics["median_deficit"])

    def _ema(self, arr):
        if not self.ema_alpha or len(arr) < 2: return arr
        a = self.ema_alpha
        out = [arr[0]]
        for x in arr[1:]:
            out.append(a * x + (1 - a) * out[-1])
        return out

    def _normalize01(self, arr):
        lo, hi = min(arr), max(arr)
        if hi == lo: return [0.0 for _ in arr]
        return [(x - lo) / (hi - lo) for x in arr]

    def finalize_plots(self):
        if not self.epochs:
            return

        # sort by epoch in case user logged out of order
        order = np.argsort(self.epochs)
        ep   = list(np.array(self.epochs)[order])
        tot  = list(np.array(self.total)[order])
        nlow = list(np.array(self.num_low)[order])
        mx   = list(np.array(self.max_def)[order])
        mu   = list(np.array(self.mean_def)[order])
        med  = list(np.array(self.median_def)[order])

        # optional smoothing
        tot_s, nlow_s, mx_s, mu_s, med_s = map(self._ema, (tot, nlow, mx, mu, med))

        # series CSV
        series_csv = os.path.join(self.out_dir, f"{self.run_name}_series.csv")
        pd.DataFrame({
            "epoch": ep,
            "total_deficit": tot,
            "num_low": nlow,
            "max_deficit": mx,
            "mean_deficit": mu,
            "median_deficit": med,
        }).to_csv(series_csv, index=False)
        print(f"[DEFICIT] series CSV saved: {series_csv}")

        base = os.path.join(self.out_dir, f"{self.run_name}_series")

        # 1) Overlay (normalized) for quick glance
        plt.figure(figsize=(7.5, 4.5), dpi=130)
        plt.plot(ep, self._normalize01(tot_s),  marker="o", label="total")
        plt.plot(ep, self._normalize01(nlow_s), marker="o", label="#low")
        plt.plot(ep, self._normalize01(mx_s),   marker="o", label="max")
        plt.plot(ep, self._normalize01(mu_s),   marker="o", label="mean")
        plt.plot(ep, self._normalize01(med_s),  marker="o", label="median")
        plt.xlabel("Epoch"); plt.ylabel("Normalized value (0–1)")
        ttl = f"{self.run_name} | deficit metrics over time"
        if self.ema_alpha: ttl += f" (EMA α={self.ema_alpha})"
        plt.title(ttl); plt.legend()
        plt.tight_layout(); p0 = base + "_overlay.png"; plt.savefig(p0, bbox_inches="tight"); plt.close()

        # 2) Individual plots (5)
        def _one(y, ylabel, suffix):
            plt.figure(figsize=(6.8, 3.8), dpi=130)
            plt.plot(ep, y, marker="o")
            plt.xlabel("Epoch"); plt.ylabel(ylabel)
            plt.title(f"{self.run_name} | {ylabel} over time")
            plt.tight_layout()
            path = f"{base}_{suffix}.png"
            plt.savefig(path, bbox_inches="tight"); plt.close()
            print(f"[DEFICIT] time-series saved: {path}")

        _one(tot_s, "Total degree deficit", "total")
        _one(nlow_s, "# nodes with deficit > 0", "numlow")
        _one(mx_s,   "Max degree deficit", "max")
        _one(mu_s,   "Mean degree deficit", "mean")
        _one(med_s,  "Median degree deficit", "median")

        print(f"[DEFICIT] overlay saved: {p0}")


@torch.no_grad()
def drop_feature(x: torch.Tensor, drop_prob: float, inplace: bool = False, gen: torch.Generator | None = None):
    """
    Column-wise feature dropout:
      - With prob=drop_prob, zero out an entire feature column across all nodes.
      - Works for dense or sparse COO tensors.
    """
    if drop_prob <= 0:
        return x

    # Make dense if needed (sparse tensors have .to_dense(); dense don't)
    if getattr(x, "is_sparse", False):
        x = x.to_dense()

    # Ensure float dtype for zeroing
    if not torch.is_floating_point(x):
        x = x.float()

    if not inplace:
        x = x.clone()

    # Mask over feature dimension (columns)
    F = x.size(1)
    rng = gen if gen is not None else None
    col_mask = torch.rand(F, device=x.device, generator=rng) < drop_prob

    # Apply mask
    x[:, col_mask] = 0
    return x

def Graph_Modify_Constraint_feat(bias_Z, original_graph, k, bound, feat_sim):
    print("local structure modified function")
    aug_graph = dot_product_decode(bias_Z)
    constrainted_new_graph = original_graph.clone().to('cpu')

    # Modify/Flip the most accuate edges, error ranging from 0.0 ~ 0.2, bounded
    difference = torch.abs(aug_graph - original_graph) # difference[difference < 0] = 1.0 ###
    difference += torch.eye(difference.shape[0]).to(difference.device) * 2
    device = difference.device
    difference = difference.to('cpu')
    
    print(difference.device, constrainted_new_graph.device)
    link_mask = (difference < bound) & (feat_sim > 0.6) & (constrainted_new_graph == 0)
    unlink_mask = (difference < bound) & (feat_sim <= 0.6) & (constrainted_new_graph == 1) 
    link_mask = link_mask.type(torch.bool)
    unlink_mask = unlink_mask.type(torch.bool)

    link_indices = link_mask.nonzero(as_tuple=False)
    unlink_indices = unlink_mask.nonzero(as_tuple=False)
    total_possible_changes = link_indices.size(0) + unlink_indices.size(0)
    if total_possible_changes < k:
        print(f"total possible changes less than k: {total_possible_changes}")
        # print()
    if(link_indices.size(0)==0):
        print("indice shape = 0")
        return constrainted_new_graph.to(device), 0
    k_link = min(link_indices.size(0), k // 2)
    k_unlink = min(unlink_indices.size(0), k // 2)
    print(f"link indice size:{link_indices.size(0)}, unlink indice size:{unlink_indices.size(0)}")
    
    chosen_link_indices = link_indices[torch.randperm(link_indices.size(0))[:k_link]]
    print(f"chosen link indice #:{k_link}")
    print(chosen_link_indices[:5])
    chosen_unlink_indices = unlink_indices[torch.randperm(unlink_indices.size(0))[:k_unlink]]
    # chosen_indices = torch.cat([chosen_link_indices, chosen_unlink_indices])
    # constrainted_new_graph[chosen_indices] = 1 - constrainted_new_graph[chosen_indices]
    constrainted_new_graph[chosen_link_indices] = 1
    constrainted_new_graph[chosen_unlink_indices] = 0

    # constrainted_new_graph[link_mask] = 1
    # constrainted_new_graph[unlink_mask] = 0

    return constrainted_new_graph.to(device), chosen_link_indices.shape[0] / constrainted_new_graph.flatten().shape[0]

def Graph_Modify_Constraint_local(bias_Z, original_graph, k, bound, common_neighbors_count, cn_threshold):
    print("local structure modified function")
    aug_graph = dot_product_decode(bias_Z)
    constrainted_new_graph = original_graph.clone().to('cpu')

    # Modify/Flip the most accuate edges, error ranging from 0.0 ~ 0.2, bounded
    difference = torch.abs(aug_graph - original_graph) # difference[difference < 0] = 1.0 ###
    difference += torch.eye(difference.shape[0]).to(difference.device) * 2
    device = difference.device
    difference = difference.to('cpu')
    # for i in range(difference.shape[0]):
    #     for j in range(difference.shape[1]):
    #         if difference[i, j] < 0.5:
    #             if common_neighbors_count[i, j] > cn_threshold:
    #                 constrainted_new_graph[i, j] = 1
    #             else:
    #                 constrainted_new_graph[i, j] = 0
    print(difference.device, constrainted_new_graph.device)
    link_mask = (difference < bound) & (common_neighbors_count > cn_threshold) & (constrainted_new_graph == 0)
    unlink_mask = (difference < bound) & (common_neighbors_count <= cn_threshold) & (constrainted_new_graph == 1) 
    link_mask = link_mask.type(torch.bool)
    unlink_mask = unlink_mask.type(torch.bool)

    link_indices = link_mask.nonzero(as_tuple=False)
    unlink_indices = unlink_mask.nonzero(as_tuple=False)
    total_possible_changes = link_indices.size(0) + unlink_indices.size(0)
    if total_possible_changes < k:
        print(f"total possible changes less than k: {total_possible_changes}")
        # print()
    
    k_link = min(link_indices.size(0), k // 2)
    k_unlink = k - k_link
    chosen_link_indices = link_indices[torch.randperm(link_indices.size(0))[:k_link]]
    print(f"chosen link indice #:{k_link}")
    print(chosen_link_indices[:5])
    chosen_unlink_indices = unlink_indices[torch.randperm(unlink_indices.size(0))[:k_unlink]]
    # chosen_indices = torch.cat([chosen_link_indices, chosen_unlink_indices])
    # constrainted_new_graph[chosen_indices] = 1 - constrainted_new_graph[chosen_indices]
    constrainted_new_graph[chosen_link_indices] = 1
    constrainted_new_graph[chosen_unlink_indices] = 0

    # constrainted_new_graph[link_mask] = 1
    # constrainted_new_graph[unlink_mask] = 0

    return constrainted_new_graph.to(device), chosen_link_indices.shape[0] / constrainted_new_graph.flatten().shape[0]

def Graph_Modify_Constraint_exp(bias_Z, original_graph, k, bound):
    print("Thm experiment")
    aug_graph = dot_product_decode(bias_Z)
    constrainted_new_graph = original_graph.clone()
    # print(type(constrainted_new_graph))
    # Modify/Flip the most accuate edges, error ranging from 0.0 ~ 0.2, bounded
    difference = torch.abs(aug_graph - original_graph) # difference[difference < 0] = 1.0 ###
    difference += torch.eye(difference.shape[0]).to(difference.device) * 2
    
    edge_index = original_graph.to_sparse().indices()
    degree_weight = degree_drop_weights(edge_index)
    degree_weight_adj = edge_weight_to_adj(edge_index, degree_weight)
    degree_weight_adj_sc = torch.sigmoid(degree_weight_adj)
    # degree weight big => node centrality small => remove probability big 
    # difference big => not-well learned yet => choose smallest k
    difference = (0.4*difference + 0.6*(1-degree_weight_adj_sc))   
    # _, indices = torch.topk(degree_weight_adj_sc.flatten(), k, largest=True)
    # values = difference.flatten()[indices]
    values, indices = torch.topk(difference.flatten(), k, largest = False)
    indices_mask = ((values >= 0.0) & (values <= bound))
    mask = indices[indices_mask].type(torch.long)
    constrainted_new_graph.flatten()[mask] = 1 - constrainted_new_graph.flatten()[mask]

    print(f'Avg modified difference value: {torch.mean(difference.flatten()[mask])}')
    print(f'Modify percentage: {mask.shape[0] / constrainted_new_graph.flatten().shape[0]}')
    # print(type(constrainted_new_graph))
    return constrainted_new_graph, mask.shape[0] / constrainted_new_graph.flatten().shape[0]

def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights

def edge_weight_to_adj(edge_index, edge_weight, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    print(num_nodes)
    adj = torch.zeros((num_nodes, num_nodes), dtype=edge_weight.dtype, device=edge_weight.device)
    for i in range(len(edge_index[0])):
        # print(i)
        adj[edge_index[0][i]][edge_index[1][i]] = edge_weight[i]
        adj[edge_index[1][i]][edge_index[0][i]] = edge_weight[i]
    return adj

def aug_random_edge(input_adj, drop_percent=0.2):
    device = input_adj.device

    percent = drop_percent / 2
    indices = input_adj.nonzero()
    row_idx, col_idx = indices[0], indices[1]
    #row_idx, col_idx = input_adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i].item(), col_idx[i].item()))

    single_index_list = []
    to_remove = []  # List to store elements to remove
    for i in list(index_list):
        single_index_list.append(i)
        to_remove.append((i[1], i[0]))  # Store the pair to remove

    # Remove elements from index_list
    index_list = [pair for pair in index_list if pair not in to_remove]


    edge_num = len(row_idx) // 2
    add_drop_num = int(edge_num * percent / 2)
    aug_adj = input_adj.clone().to_dense()

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    for i in drop_idx:
        aug_adj[single_index_list[i][0], single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1], single_index_list[i][0]] = 0

    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        aug_adj[i[0], i[1]] = 1
        aug_adj[i[1], i[0]] = 1

    aug_adj = aug_adj.to(device)
    aug_adj = sp.csr_matrix(aug_adj.cpu().numpy())
    aug_adj = torch.tensor(aug_adj.todense(), device=device)

    return aug_adj, drop_percent

def vMF_KDE(dataset_str, Z):
    def nomalize_embedding(Z, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(Z, order, axis))
        l2[l2==0] = 1
        unit = Z / np.expand_dims(l2, axis)
        return unit

    transform = TSNE  # PCA
    trans = transform(n_components=2, learning_rate='auto', init='random')
    emb_transformed = pd.DataFrame(trans.fit_transform(Z.cpu().numpy()))
    emb_transformed = nomalize_embedding(emb_transformed)
    x = emb_transformed[0]
    y = emb_transformed[1]
    arc = np.arctan2(y, x)

    fig, ax = plt.subplots(figsize=(7, 1.4))
    ax.hist(arc, bins = 60)

    plt.show()
    plt.savefig(f'/home/retro/SECRET/pic/{dataset_str}_ARC_KDE.png')
    plt.clf()

def gaussion_KDE(dataset_str, Z):

    def nomalize_embedding(Z, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(Z, order, axis))
        l2[l2==0] = 1
        unit = Z / np.expand_dims(l2, axis)
        return unit

    transform = TSNE  # PCA
    trans = transform(n_components=2, learning_rate='auto', init='random')
    emb_transformed = pd.DataFrame(trans.fit_transform(Z.cpu().numpy()))
    norm_emb_transformed = nomalize_embedding(emb_transformed)

    x = norm_emb_transformed[0]
    y = norm_emb_transformed[1]

    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    z_norm = (z-min(z))/(max(z)-min(z))

    fig, ax = plt.subplots(figsize=(7,7))

    n = x.shape[0]
    c = 0.2
    colors = (1. - c) * plt.get_cmap("GnBu")(np.linspace(0., 1., n)) + c * np.ones((n, 4))
    ax.scatter(x, y, c=colors, s=30, alpha=z, marker='o')

    plt.show()
    plt.savefig(f'/home/retro/SECRET/pic/{dataset_str}_KDE.png')
    plt.clf()

def draw_adjacency_matrix_edge_weight_distribution(dataset_str, epoch, graph_type, adj):
    edge_weight_distribution = [
    torch.sum(adj == 0).item(),
    torch.sum((adj > 0.0) & (adj < 0.1)).item(),
    torch.sum((adj >= 0.1) & (adj < 0.2)).item(),
    torch.sum((adj >= 0.2) & (adj < 0.3)).item(),
    torch.sum((adj >= 0.3) & (adj < 0.4)).item(),
    torch.sum((adj >= 0.4) & (adj < 0.5)).item(),
    torch.sum((adj >= 0.5) & (adj < 0.6)).item(),
    torch.sum((adj >= 0.6) & (adj < 0.7)).item(),
    torch.sum((adj >= 0.7) & (adj < 0.8)).item(),
    torch.sum((adj >= 0.8) & (adj < 0.9)).item(),
    torch.sum((adj >= 0.9) & (adj < 1.0)).item(),
    torch.sum(adj == 1.0).item()]
    plt.plot(range(12), edge_weight_distribution, label = "edge_weight", marker="o")
    plt.xticks(ticks = range(12), labels = ['0.0', '<0.1', '<0.2', '<0.3', '<0.4', '<0.5', '<0.6', '<0.7', '<0.8', '<0.9', '<1.0', '1.0'])
    for i, j in zip(range(12), edge_weight_distribution):
        plt.annotate(str(j),xy=(i,j), fontsize = 7)
    plt.title(f'{dataset_str} {graph_type} epoch {epoch}')
    plt.savefig(f'/home/retro/SECRET/pic/edge_weight_distribution/{dataset_str}/{graph_type}_epoch{epoch}.png')
    plt.clf()

def Plot(dataset_str, roc_history, modification_ratio_history):
    # plt.ylim(0.98, 0.99)
    plt.plot(roc_history, label = "roc_history")
    plt.legend()
    plt.savefig(f'/home/retro/SECRET/pic/{dataset_str}_roc.png')
    plt.clf()
    plt.plot(modification_ratio_history, label = "modification_ratio_history")
    plt.legend()
    plt.savefig(f'/home/retro/SECRET/pic/{dataset_str}_modification.png')
    plt.clf()

def Visualize(dataset_str, Z, labels):
    transform = TSNE  # PCA

    trans = transform(n_components=2, learning_rate='auto', init='random')
    emb_transformed = pd.DataFrame(trans.fit_transform(Z.cpu().numpy()))
    labels = torch.argmax(torch.tensor(labels), dim=1)
    emb_transformed["label"] = labels

    alpha = 0.7

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        emb_transformed[0],
        emb_transformed[1],
        s = 10,
        c = emb_transformed["label"].astype("category"),
        cmap="jet",
        alpha = alpha,
    )
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    plt.title(f'{transform.__name__} visualization of embeddings for {dataset_str} dataset')
    plt.show()
    plt.savefig(f'/home/retro/SECRET/pic/{dataset_str}_visualize.png')
    plt.clf()

def Visualize_with_edge(dataset_str, Z, labels, edge_index):
    
    transform = TSNE  # PCA
    trans = transform(n_components=2, learning_rate='auto', init='random')
    emb_transformed = pd.DataFrame(trans.fit_transform(Z.cpu().numpy()))
    labels = torch.argmax(torch.tensor(labels), dim=1)
    emb_transformed["label"] = labels

    node_num = Z.shape[0]
    edge_num = edge_index.shape[1]

    edge_x = []
    edge_y = []
    for i in range(edge_num):
        u = edge_index[0][i].item()
        v = edge_index[1][i].item()

        x0 = emb_transformed[0][u]
        y0 = emb_transformed[1][u]

        x1 = emb_transformed[0][v]
        y1 = emb_transformed[1][v]

        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

    node_x = []
    node_y = []
    for i in range(node_num):
        x = emb_transformed[0][i]
        y = emb_transformed[1][i]

        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=[],
            size=10,
            line_width=2))

    node_trace.marker.color = emb_transformed["label"].astype("category")

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()