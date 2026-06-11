import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from sklearn.cluster import KMeans

def loss_function(A_pred, adj_label, mean, logstd, norm, weight_tensor, alpha, beta, train_mask):
    n_nodes = A_pred.size(0)
    loss = alpha*norm*F.binary_cross_entropy(A_pred.view(-1)[train_mask], adj_label.to_dense().view(-1)[train_mask], weight = weight_tensor) # loss += alpha*F.mse_loss(A_pred, adj_label.to_dense())
    
    kl_divergence = 0.5/n_nodes  * (1 + 2*logstd - mean**2 - torch.exp(logstd)**2).sum(1).mean()
    loss -= beta * kl_divergence

    return loss

def intra_view_CL_loss(device, Z, adj_label, gamma, temperature):
    n_nodes = Z.shape[0]
    # Z_norm = torch.norm(Z, p = 2, dim = 1).unsqueeze(-1)# Z_norm = torch.where(Z_norm == 0., torch.ones_like(Z_norm), Z_norm) # unit_Z = Z / Z_norm# cos_sim = unit_Z @ unit_Z.t()# exp_cos_sim = torch.exp(torch.div(cos_sim, temperature))
    cos_sim = torch.sigmoid(torch.matmul(Z, Z.t()))
    exp_cos_sim = torch.exp(torch.div(cos_sim, temperature))
   
    # pos_indicator = torch.eye(n_nodes).to(device)
    # neg_indicator = torch.ones(n_nodes, n_nodes).to(device) - (torch.eye(n_nodes).to(device))
    # pos_indicator = adj_label.to_dense()
    # pos = torch.diag(torch.mm(exp_cos_sim, pos_indicator))
    
    pos = torch.diag(exp_cos_sim)
    neg = torch.sum(exp_cos_sim, dim = 1)
    neg = torch.where(neg == 0., torch.ones_like(neg), neg)
    
    CL_Loss = -1.0 * gamma * torch.mean(torch.log(torch.div(pos, neg)))
    
    return CL_Loss

def inter_view_CL_loss(device, Z, bias_Z, adj_label, gamma, temperature):
    n_nodes = Z.shape[0]
    # Z_norm = torch.norm(Z, p = 2, dim = 1).unsqueeze(-1) # Z_norm = torch.where(Z_norm == 0., torch.ones_like(Z_norm), Z_norm) # unit_Z = Z / Z_norm # bias_Z_norm = torch.norm(bias_Z, p = 2, dim = 1).unsqueeze(-1) # bias_Z_norm = torch.where(bias_Z_norm == 0., torch.ones_like(bias_Z_norm), bias_Z_norm) # unit_bias_Z = bias_Z / bias_Z_norm # cos_sim = unit_Z @ unit_bias_Z.t()
    cos_sim = torch.sigmoid(torch.matmul(Z, bias_Z.t()))
    exp_cos_sim = torch.exp(torch.div(cos_sim, temperature))

    # neg_indicator = torch.ones(n_nodes, n_nodes).to(device) - (adj_label.to_dense())
    # neg = torch.diag(torch.mm(exp_cos_sim, neg_indicator))

    pos_indicator = adj_label.to_dense()
    pos = torch.diag(torch.mm(exp_cos_sim, pos_indicator))
    neg = torch.sum(exp_cos_sim, dim = 1)
    neg = torch.where(neg == 0., torch.ones_like(neg), neg)
    
    CL_Loss = -1.0 * gamma * torch.mean(torch.log(torch.div(pos, neg)))

    return CL_Loss


def symmetric_node_infonce_loss(Z1, Z2, gamma, temperature):
    """Symmetric node-level InfoNCE with same node across views as the positive."""
    if Z1.shape != Z2.shape:
        raise ValueError(f"InfoNCE views must have the same shape, got {tuple(Z1.shape)} and {tuple(Z2.shape)}")
    if Z1.dim() != 2:
        raise ValueError(f"InfoNCE views must be rank-2 tensors, got dim={Z1.dim()}")
    n_nodes = Z1.size(0)
    if n_nodes == 0:
        return Z1.sum() * 0.0
    temp = max(float(temperature), 1e-6)
    z1 = F.normalize(Z1, p=2, dim=1)
    z2 = F.normalize(Z2, p=2, dim=1)
    logits = torch.matmul(z1, z2.t()) / temp
    labels = torch.arange(n_nodes, device=Z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return float(gamma) * 0.5 * (loss_12 + loss_21)


def Cluster(device, Z, nb_classes):
    kmeans = KMeans(n_clusters = nb_classes)
    kmeans.fit(Z)
    
    y = kmeans.predict(Z)
    pseudo_pos_indicator = torch.zeros((y.shape[0], y.shape[0]), dtype = torch.float32, requires_grad = False).to(device)
    for i in range(len(pseudo_pos_indicator)):
        mask = y == y[i]
        pseudo_pos_indicator[i][mask] = 1.0

    return pseudo_pos_indicator
