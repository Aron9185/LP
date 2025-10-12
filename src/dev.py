import torch
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from multiprocessing import Pool
import math
from model import dot_product_decode

def process_head_node(idx, normalized_sim, num_to_keep, new_adj):
    current_sim = normalized_sim[idx]
    n_keep = num_to_keep.item()
    if n_keep > 0:
        # Use topk to find the indices of the highest similarity scores
        _, top_indices = torch.topk(current_sim, n_keep, largest=True, sorted=False)
        new_adj[idx, top_indices] = 1  # Set new connections in the adjacency matrix

def purify_head_nodes_v3p(bias_Z, adj_matrix, head_idx, r, new_adj):
    num_nodes = adj_matrix.size(0)
    head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_matrix.device)
    head_mask[head_idx] = True
    
    # sim_matrix = cosine_similarity(bias_Z, bias_Z)
    sim_matrix = torch.matmul(bias_Z, bias_Z.t())
    sim_matrix += 2 * abs(torch.min(sim_matrix))
    normalized_sim = sim_matrix / sim_matrix.sum(1, keepdim=True)
    head_adj = adj_matrix[head_idx]
    num_to_keep = (head_adj.sum(dim=1) * (1 - r)).ceil().int()
    
    new_adj[head_idx] = 0  # Reset adjacency connections for head nodes

    # Multi-threading with 4 threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_head_node, idx, normalized_sim, num_to_keep[i], new_adj) for i, idx in enumerate(head_idx)]
        for future in futures:
            future.result()  # Wait for all threads to complete

    return new_adj


def process_tail_node(i, idx, tail_adj_label, tail_adj_knn, r, new_adj, old_degrees, new_degrees):
    # 当前尾节点的原始邻居信息和KNN邻居信息
    current_label_neighbors = tail_adj_label[i]
    old_degrees[i]=current_label_neighbors.sum().item()
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
    new_degrees[i]=all_mask.sum().item()

def merge_neighbors_v3p(adj_label, adj_knn, tail_idx, r, k):
    num_nodes = adj_label.size(0)
    tail_mask = torch.zeros(num_nodes, dtype=torch.bool, device=adj_label.device)
    tail_mask[tail_idx] = True

    # 提取尾节点的邻接信息
    tail_adj_label = adj_label[tail_mask]
    tail_adj_knn = adj_knn[tail_mask]

    # 创建新的邻接矩阵
    new_adj = adj_label.clone()
    old_degrees = [0]*tail_idx.size()[0]
    new_degrees = [0]*tail_idx.size()[0]

    # 多线程处理
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_tail_node, i, idx, tail_adj_label, tail_adj_knn, r, new_adj, old_degrees, new_degrees)
                   for i, idx in enumerate(tail_idx)]
        for future in futures:
            future.result()  # 等待所有线程完成

    print(f"[Tail Node] origin avg degree: {np.mean(old_degrees)}, augmented avg degree: {np.mean(new_degrees)}")
    return new_adj

# def merge_purify_v2(adj, adj_knn, bias_Z, r,k):
#     num_nodes = adj.size(0)
#     old_degrees = []
#     new_degrees = []
    
#     new_adj = adj.clone()
#     sim_matrix = torch.matmul(bias_Z, bias_Z.t())
#     sim_matrix += 2*abs(torch.min(sim_matrix))
#     normalized_sim = sim_matrix / sim_matrix.sum(1)[:,None]
#     neighbor_sim = normalized_sim * adj
    
#     for i in range(num_nodes):
#         # 当前尾节点的原始邻居信息和KNN邻居信息
#         current_label_neighbors = adj[i]
#         current_neighbors_sim = neighbor_sim[i]
#         old_degrees.append(current_label_neighbors.sum().item())
#         current_knn_neighbors = adj_knn[i]

#         # 计算应保留和新增的邻居数
#         num_to_keep_label = max(int((current_label_neighbors.sum() * (1-r)).ceil().item()),1)
#         num_to_keep_knn = int((current_knn_neighbors.sum() * r).ceil().item())

#         # 随机选择原始邻居进行保留
#         if current_label_neighbors.sum() > 0:
#             sampled_indices = torch.multinomial(current_neighbors_sim, num_to_keep_label, replacement=False)
#             label_mask = torch.zeros_like(current_label_neighbors).scatter_(0, sampled_indices, 1)
#         else:
#             label_mask = torch.zeros_like(current_label_neighbors)

#         # 随机选择KNN邻居进行添加
#         if current_knn_neighbors.sum() > 0:
#             all_knn_indices = torch.where(current_knn_neighbors > 0)[0]
#             rand_indices = torch.randperm(all_knn_indices.size(0))[:num_to_keep_knn]
#             keep_knn_indices = all_knn_indices[rand_indices]
#             knn_mask = torch.zeros_like(current_knn_neighbors)
#             knn_mask[keep_knn_indices] = 1
#         else:
#             knn_mask = torch.zeros_like(current_knn_neighbors)

#         # 更新邻接矩阵
#         all_mask = label_mask + knn_mask
#         new_adj[i] = torch.clamp(all_mask, max=1)
#         # new_adj[i] = all_mask  # 使用逻辑或合并保留和新增的邻居
#         new_degrees.append(all_mask.sum().item())

#     print(f"origin avg degree: {np.mean(old_degrees)}, augmented avg degree: {np.mean(new_degrees)}")
#     return new_adj

def merge_purify_v2(adj, adj_knn, bias_Z, r, k):
    num_nodes = adj.size(0)
    old_degrees = adj.sum(1).tolist()  # 可以直接在这里计算并转换为列表

    sim_matrix = torch.matmul(bias_Z, bias_Z.t())
    sim_matrix += 2 * abs(torch.min(sim_matrix))
    normalized_sim = sim_matrix / sim_matrix.sum(1, keepdim=True)
    neighbor_sim = normalized_sim * adj
    
    new_adj = torch.zeros_like(adj)  # 初始化新的邻接矩阵

    # 向量化处理所有节点的标签和KNN邻居选择
    label_degrees = adj.sum(1)
    knn_degrees = adj_knn.sum(1)
    num_to_keep_label = torch.clamp((label_degrees * (1 - r)).ceil(), min=1).int()
    num_to_keep_knn = (knn_degrees * r).ceil().int()

    # 使用循环（如果找到完全向量化的方法，则可以进一步优化）
    for i in range(num_nodes):
        if label_degrees[i] > 0:
            sampled_indices = torch.multinomial(neighbor_sim[i], num_to_keep_label[i].item(), replacement=False)
            new_adj[i, sampled_indices] = 1
        if knn_degrees[i] > 0:
            all_knn_indices = torch.where(adj_knn[i] > 0)[0]
            rand_indices = torch.randperm(all_knn_indices.size(0))[:num_to_keep_knn[i]]
            new_adj[i, all_knn_indices[rand_indices]] = 1

    new_degrees = new_adj.sum(1).tolist()  # 计算新的度
    print(f"origin avg degree: {np.mean(old_degrees)}, augmented avg degree: {np.mean(new_degrees)}")
    return new_adj


def merge_purify_v3(adj, adj_knn, bias_Z, r, k):
    # device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    device = adj.device
    print(device)
    num_nodes = adj.size(0)
    old_degrees = adj.sum(1).tolist()  # 可以直接在这里计算并转换为列表
    sim_matrix = dot_product_decode(bias_Z)

    neighbor_sim = sim_matrix * adj
    neighbor_sim[neighbor_sim < 0.5] = 0
    
    new_adj = neighbor_sim.clone().to(device)
    # new_adj = torch.zeros_like(adj)  # 初始化新的邻接矩阵

    # 向量化处理所有节点的标签和KNN邻居选择
    # label_degrees = adj.sum(1)
    knn_degrees = adj_knn.sum(1)
    # num_to_keep_label = torch.clamp((label_degrees * (1 - r)).ceil(), min=1).int()
    num_to_keep_knn = int(math.ceil(k * r))

    total_knn_neighbors = adj_knn.sum(1)
    probs_knn = adj_knn / total_knn_neighbors[:, None]  # 归一化概率
    sampled_knn = torch.multinomial(probs_knn, num_to_keep_knn, replacement=False)
    keep_knn_mask = torch.zeros_like(adj_knn).scatter_(1, sampled_knn, 1)
    new_adj += keep_knn_mask.float().to(device)  # 确保数据类型一致
    new_adj = torch.clamp(new_adj, 0, 1)
    mask = torch.eye(new_adj.size(0), dtype=torch.bool).to(device)
    new_adj.masked_fill_(mask, 1)
    new_degrees = new_adj.sum(1).tolist()  # 计算新的度
    #Aron
    #print(f"origin avg degree: {np.mean(old_degrees)}, augmented avg degree: {np.mean(new_degrees)}, minimum node degree: {np.min(new_degrees)}")
    print(f"origin avg degree: {np.mean(old_degrees)}, augmented avg degree: {np.mean(new_degrees)}")
    return new_adj


def process_node(args):
    i, neighbor_sim, num_to_keep_label, adj_knn, num_to_keep_knn, adj_shape = args
    # 确保在 CPU 上初始化张量
    node_adj = torch.zeros(adj_shape, device=adj_knn.device)  # 初始化单个节点的邻接矩阵
    
    if neighbor_sim.sum() > 0:
        sampled_indices = torch.multinomial(neighbor_sim, num_to_keep_label, replacement=False)
        node_adj[sampled_indices] = 1
    
    if adj_knn.sum() > 0:
        all_knn_indices = torch.where(adj_knn > 0)[0]
        rand_indices = torch.randperm(all_knn_indices.size(0))[:num_to_keep_knn]
        node_adj[all_knn_indices[rand_indices]] = 1
    
    return node_adj

def merge_purify_v2p(adj, adj_knn, bias_Z, r, k):
    num_nodes = adj.size(0)
    old_degrees = adj.sum(1).tolist()

    # 确保所有操作在 CPU 上进行
    sim_matrix = torch.matmul(bias_Z.cpu(), bias_Z.cpu().t())
    sim_matrix += 2 * abs(torch.min(sim_matrix))
    normalized_sim = sim_matrix / sim_matrix.sum(1, keepdim=True)
    neighbor_sim = normalized_sim * adj.cpu()

    label_degrees = adj.sum(1)
    # knn_degrees = adj_knn.sum(1)
    num_to_keep_label = torch.clamp((label_degrees * (1 - r)).ceil(), min=1).int()
    num_to_keep_knn = math.ceil(k * r)
    
    # 使用 multiprocessing 平行化处理
    with Pool() as pool:
        new_adj = torch.stack(pool.map(process_node, [
            (i, neighbor_sim[i], num_to_keep_label[i].item(), adj_knn[i].cpu(), num_to_keep_knn, adj.size(1))
            for i in range(num_nodes)
        ]))
    
    new_degrees = new_adj.sum(1).tolist()
    print(f"origin avg degree: {np.mean(old_degrees)}, augmented avg degree: {np.mean(new_degrees)}")
    return new_adj

def merge_purify_v2tp(adj, adj_knn, bias_Z, r, k):
    num_nodes = adj.size(0)
    old_degrees = adj.sum(1).tolist()

    sim_matrix = torch.matmul(bias_Z, bias_Z.t())
    sim_matrix += 2 * abs(torch.min(sim_matrix))
    normalized_sim = sim_matrix / sim_matrix.sum(1, keepdim=True)
    neighbor_sim = normalized_sim * adj

    label_degrees = adj.sum(1)
    num_to_keep_label = torch.clamp((label_degrees * (1 - r)).ceil(), min=1).int()
    num_to_keep_knn = math.ceil(k * r)

    # 使用 ThreadPoolExecutor 进行多线程处理
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_node, (i, neighbor_sim[i], num_to_keep_label[i].item(), adj_knn[i], num_to_keep_knn, adj.size(1))) for i in range(num_nodes)]
        new_adj = torch.stack([future.result() for future in futures])
    
    new_degrees = new_adj.sum(1).tolist()
    print(f"origin avg degree: {np.mean(old_degrees)}, augmented avg degree: {np.mean(new_degrees)}")
    return new_adj