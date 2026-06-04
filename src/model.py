import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch_geometric.nn import APPNP, SuperGATConv


#TODO: other message passing layer: APPNP, Normalize after linear, clamp std, dropout
class VGNAE_ENCODER(nn.Module):
	def __init__(self, in_channels, hidden_channels_1, out_channels, dropout, device):
		super(VGNAE_ENCODER, self).__init__()
		self.in_channels = in_channels
		self.hidden_channels_1 = hidden_channels_1
		self.out_channels = out_channels
		self.dropout = dropout
		# self.scaling_factor = scaling_factor

		self.preprocess = nn.Linear(in_channels, hidden_channels_1)
		self.activation = F.relu

		self.std_linear_1 = nn.Linear(hidden_channels_1, out_channels)
		self.propagate_logstd_1 = APPNP(K = 2, alpha = 0.5, dropout = dropout)

		self.mean_linear_1 = nn.Linear(hidden_channels_1, out_channels)
		self.propagate_1 = APPNP(K = 2, alpha = 0.5, dropout = dropout)

		self.projection_layer = nn.Linear(out_channels, out_channels)

		self.batch_norm = nn.BatchNorm1d(out_channels)
		# self.layer_norm = nn.LayerNorm(hidden_channels_1)
		
		self.device = device
		for m in self.modules():
			self.weights_init(m)
	
	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def encode(self, x, edge_index, edge_weight):
		if self.training:
			x = self.preprocess(x) # x = F.normalize(x, dim=1)
			x = self.activation(x)
			x = F.dropout(x, p = self.dropout, training = self.training)
			
			hid_std = self.std_linear_1(x)
			self.logstd = self.propagate_logstd_1(hid_std, edge_index, edge_weight)
			self.logstd = self.logstd.clamp(min = -10.0, max = 10.0)

			hidden_repr = self.mean_linear_1(x) + torch.randn(x.size(0), self.out_channels).to(self.device)*torch.exp(self.logstd)
			# hidden_repr = self.batch_norm(hidden_repr)
			self.Z =  self.propagate_1(hidden_repr, edge_index, edge_weight) + torch.randn(x.size(0), self.out_channels).to(self.device)*torch.exp(self.logstd)
			self.Z = self.batch_norm(self.Z)
			self.mean = self.projection_layer(self.Z)

			gaussian_noise = torch.randn(x.size(0), self.out_channels).to(self.device)
			sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean

			return sampled_z
		else:
			x = self.preprocess(x) # x = F.normalize(x, dim=1)
			x = self.activation(x)
			x = F.dropout(x, p = self.dropout, training = self.training)

			hidden_repr = self.mean_linear_1(x)
			# hidden_repr = self.batch_norm(hidden_repr)
			self.Z =  self.propagate_1(hidden_repr, edge_index, edge_weight)
			self.Z = self.batch_norm(self.Z)
			self.mean = self.projection_layer(self.Z)

			return self.mean

	def forward(self, x, edge_index, edge_weight = None):
		Z = self.encode(x, edge_index, edge_weight)
		return Z

class MaskGAE_ENCODER(nn.Module):
	def __init__(self, in_channels, hidden_channels_1, out_channels, dropout, device, mask_rate=0.3):
		super(MaskGAE_ENCODER, self).__init__()
		self.in_channels = in_channels
		self.hidden_channels_1 = hidden_channels_1
		self.out_channels = out_channels
		self.dropout = dropout
		self.device = device
		self.mask_rate = float(mask_rate)

		self.mask_token = nn.Parameter(torch.zeros(in_channels))
		self.preprocess = nn.Linear(in_channels, hidden_channels_1)
		self.activation = F.relu
		self.mean_linear_1 = nn.Linear(hidden_channels_1, out_channels)
		self.propagate_1 = APPNP(K=2, alpha=0.5, dropout=dropout)
		self.projection_layer = nn.Linear(out_channels, out_channels)
		self.batch_norm = nn.BatchNorm1d(out_channels)
		self.feature_decoder = nn.Sequential(
			nn.Linear(out_channels, hidden_channels_1),
			nn.ReLU(),
			nn.Linear(hidden_channels_1, in_channels),
		)

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def _dense_features(self, x):
		return x.to_dense() if getattr(x, "is_sparse", False) else x

	def encode(self, x, edge_index, edge_weight=None):
		x_orig = self._dense_features(x).to(self.device)
		x_in = x_orig
		mask = torch.zeros(x_orig.size(0), dtype=torch.bool, device=x_orig.device)

		if self.training and self.mask_rate > 0.0 and x_orig.size(0) > 0:
			mask = torch.rand(x_orig.size(0), device=x_orig.device) < self.mask_rate
			if not bool(mask.any()):
				mask[torch.randint(0, x_orig.size(0), (1,), device=x_orig.device)] = True
			x_in = x_orig.clone()
			x_in[mask] = self.mask_token.to(dtype=x_orig.dtype, device=x_orig.device)

		x_hid = self.preprocess(x_in)
		x_hid = self.activation(x_hid)
		x_hid = F.dropout(x_hid, p=self.dropout, training=self.training)

		hidden_repr = self.mean_linear_1(x_hid)
		self.Z = self.propagate_1(hidden_repr, edge_index, edge_weight)
		self.Z = self.batch_norm(self.Z)
		self.mean = self.projection_layer(self.Z)
		self.logstd = torch.zeros_like(self.mean)

		self.masked_feature_pred = self.feature_decoder(self.mean)
		self.masked_feature_target = x_orig.detach()
		self.masked_feature_mask = mask
		return self.mean

	def masked_feature_loss(self):
		pred = getattr(self, "masked_feature_pred", None)
		target = getattr(self, "masked_feature_target", None)
		mask = getattr(self, "masked_feature_mask", None)
		if pred is None or target is None or mask is None or not bool(mask.any()):
			device = pred.device if pred is not None else self.mask_token.device
			return torch.zeros((), dtype=self.mask_token.dtype, device=device)

		pred_m = pred[mask]
		target_m = target[mask].to(dtype=pred_m.dtype, device=pred_m.device)
		if bool((target_m >= 0).all()) and bool((target_m <= 1).all()):
			return F.binary_cross_entropy_with_logits(pred_m, target_m)
		return F.mse_loss(pred_m, target_m)

	def forward(self, x, edge_index, edge_weight=None):
		Z = self.encode(x, edge_index, edge_weight)
		return Z

class CIMAGELite_ENCODER(MaskGAE_ENCODER):
	def __init__(
		self,
		in_channels,
		hidden_channels_1,
		out_channels,
		dropout,
		device,
		mask_rate=0.3,
		num_factors=8,
		num_clusters=16,
		cluster_alpha=1.0,
	):
		super(CIMAGELite_ENCODER, self).__init__(
			in_channels,
			hidden_channels_1,
			out_channels,
			dropout,
			device,
			mask_rate=mask_rate,
		)
		self.num_factors = int(num_factors)
		self.num_clusters = int(num_clusters)
		self.cluster_alpha = float(cluster_alpha)
		if self.num_factors <= 0:
			raise ValueError("num_factors must be positive")
		if out_channels % self.num_factors != 0:
			raise ValueError(
				f"out_channels={out_channels} must be divisible by num_factors={self.num_factors}"
			)
		if self.num_clusters <= 0:
			raise ValueError("num_clusters must be positive")

		self.factor_dim = out_channels // self.num_factors
		self.factor_decoder = nn.Sequential(
			nn.Linear(out_channels, hidden_channels_1),
			nn.ReLU(),
			nn.Linear(hidden_channels_1, out_channels),
		)
		self.cluster_centers = nn.Parameter(torch.empty(self.num_clusters, out_channels))
		nn.init.xavier_uniform_(self.cluster_centers)

		for m in self.factor_decoder.modules():
			self.weights_init(m)

	def _factor_reconstruction_loss(self, z):
		if self.num_factors <= 1:
			return z.sum() * 0.0
		N = z.size(0)
		z_factors = z.view(N, self.num_factors, self.factor_dim)
		if self.training:
			factor_mask = torch.rand(self.num_factors, device=z.device) < 0.5
			if not bool(factor_mask.any()):
				factor_mask[torch.randint(0, self.num_factors, (1,), device=z.device)] = True
			if bool(factor_mask.all()):
				factor_mask[torch.randint(0, self.num_factors, (1,), device=z.device)] = False
		else:
			factor_mask = torch.zeros(self.num_factors, dtype=torch.bool, device=z.device)
			factor_mask[self.num_factors // 2:] = True

		visible = z_factors.clone()
		visible[:, factor_mask, :] = 0.0
		recon = self.factor_decoder(visible.reshape(N, -1)).view(N, self.num_factors, self.factor_dim)
		self.cimage_factor_mask = factor_mask
		return F.mse_loss(recon[:, factor_mask, :], z_factors[:, factor_mask, :].detach())

	def _cluster_assignment_loss(self, z):
		diff = z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)
		dist2 = (diff ** 2).sum(dim=2)
		q = (1.0 + dist2 / self.cluster_alpha) ** (-(self.cluster_alpha + 1.0) / 2.0)
		q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
		p = (q ** 2) / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
		p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
		self.cimage_cluster_q = q
		return F.kl_div(torch.log(q.clamp_min(1e-12)), p.detach(), reduction="batchmean")

	def encode(self, x, edge_index, edge_weight=None):
		z = super().encode(x, edge_index, edge_weight)
		self.cimage_factor_loss = self._factor_reconstruction_loss(z)
		self.cimage_cluster_loss = self._cluster_assignment_loss(z)
		return z

	def cimage_losses(self):
		factor = getattr(self, "cimage_factor_loss", None)
		cluster = getattr(self, "cimage_cluster_loss", None)
		if factor is None:
			factor = torch.zeros((), dtype=self.mask_token.dtype, device=self.mask_token.device)
		if cluster is None:
			cluster = torch.zeros((), dtype=self.mask_token.dtype, device=self.mask_token.device)
		return factor, cluster

class CIMAGEFull_ENCODER(nn.Module):
	def __init__(
		self,
		in_channels,
		hidden_channels_1,
		out_channels,
		dropout,
		device,
		mask_rate=0.3,
		num_factors=8,
		num_clusters=16,
		pseudo_label_threshold=0.90,
		factor_select_ratio=0.50,
		mrmr_redundancy_weight=0.20,
		cluster_balance_weight=0.05,
		sce_power=2.0,
	):
		super(CIMAGEFull_ENCODER, self).__init__()
		self.in_channels = in_channels
		self.hidden_channels_1 = hidden_channels_1
		self.out_channels = out_channels
		self.dropout = dropout
		self.device = device
		self.edge_mask_rate = float(mask_rate)
		self.num_factors = int(num_factors)
		self.num_clusters = int(num_clusters)
		self.pseudo_label_threshold = float(pseudo_label_threshold)
		self.factor_select_ratio = float(factor_select_ratio)
		self.mrmr_redundancy_weight = float(mrmr_redundancy_weight)
		self.cluster_balance_weight = float(cluster_balance_weight)
		self.sce_power = float(sce_power)

		if self.num_factors <= 0:
			raise ValueError("num_factors must be positive")
		if out_channels % self.num_factors != 0:
			raise ValueError(
				f"out_channels={out_channels} must be divisible by num_factors={self.num_factors}"
			)
		if self.num_clusters <= 0:
			raise ValueError("num_clusters must be positive")

		self.factor_dim = out_channels // self.num_factors
		self.preprocess = nn.Linear(in_channels, hidden_channels_1)
		self.activation = F.relu
		self.factor_linears = nn.ModuleList([
			nn.Linear(hidden_channels_1, self.factor_dim) for _ in range(self.num_factors)
		])
		self.factor_propagates = nn.ModuleList([
			APPNP(K=2, alpha=0.5, dropout=dropout) for _ in range(self.num_factors)
		])
		self.factor_projections = nn.ModuleList([
			nn.Linear(self.factor_dim, self.factor_dim) for _ in range(self.num_factors)
		])
		self.batch_norm = nn.BatchNorm1d(out_channels)
		self.factor_decoder = nn.Sequential(
			nn.Linear(out_channels, hidden_channels_1),
			nn.ReLU(),
			nn.Linear(hidden_channels_1, out_channels),
		)
		self.cluster_head = nn.Linear(out_channels, self.num_clusters)

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def _dense_features(self, x):
		return x.to_dense() if getattr(x, "is_sparse", False) else x

	def masked_feature_loss(self):
		return torch.zeros((), dtype=self.preprocess.weight.dtype, device=self.preprocess.weight.device)

	def _edge_mask(self, edge_index, edge_weight):
		if (not self.training) or self.edge_mask_rate <= 0.0 or edge_index is None or edge_index.numel() == 0:
			return edge_index, edge_weight
		num_edges = edge_index.size(1)
		keep = torch.rand(num_edges, device=edge_index.device) >= self.edge_mask_rate
		keep = keep | (edge_index[0] == edge_index[1])
		if not bool(keep.any()):
			return edge_index, edge_weight
		edge_weight_masked = None if edge_weight is None else edge_weight[keep]
		return edge_index[:, keep], edge_weight_masked

	def _soft_modularity_loss(self, q, edge_index, edge_weight):
		if edge_index is None or edge_index.numel() == 0:
			return q.sum() * 0.0
		src = edge_index[0]
		dst = edge_index[1]
		valid = src != dst
		src = src[valid]
		dst = dst[valid]
		if src.numel() == 0:
			return q.sum() * 0.0
		if edge_weight is None:
			w = q.new_ones(src.numel())
		else:
			w = edge_weight.to(device=q.device, dtype=q.dtype)[valid]
		deg = q.new_zeros(q.size(0))
		deg.index_add_(0, src, w)
		total_w = deg.sum().clamp_min(1e-12)
		q_src = q.index_select(0, src)
		q_dst = q.index_select(0, dst)
		soft_intra_edges = (w * (q_src * q_dst).sum(dim=1)).sum()
		degree_cluster_mass = (deg.unsqueeze(1) * q).sum(dim=0)
		expected = degree_cluster_mass.pow(2).sum() / total_w
		modularity = (soft_intra_edges - expected) / total_w
		cluster_mass = q.mean(dim=0)
		balance = (cluster_mass - (1.0 / self.num_clusters)).pow(2).sum()
		return -modularity + self.cluster_balance_weight * balance

	@torch.no_grad()
	def _factor_context_mask(self, z_factors, q):
		num_factors = z_factors.size(1)
		conf, labels = q.detach().max(dim=1)
		use_nodes = conf >= self.pseudo_label_threshold
		if int(use_nodes.sum().item()) < max(4, num_factors) or labels[use_nodes].unique().numel() < 2:
			use_nodes = torch.ones_like(conf, dtype=torch.bool)

		x = z_factors.detach()[use_nodes]
		y = labels[use_nodes]
		relevance = z_factors.new_zeros(num_factors)
		if x.size(0) > 1 and y.unique().numel() > 1:
			for f_idx in range(num_factors):
				xf = x[:, f_idx, :]
				overall = xf.mean(dim=0, keepdim=True)
				total = (xf - overall).pow(2).sum().clamp_min(1e-12)
				between = xf.new_tensor(0.0)
				for lbl in y.unique():
					group = xf[y == lbl]
					if group.numel() == 0:
						continue
					diff = group.mean(dim=0, keepdim=True) - overall
					between = between + group.size(0) * diff.pow(2).sum()
				relevance[f_idx] = between / total
		else:
			relevance.fill_(1.0)

		redundancy = z_factors.new_zeros(num_factors)
		centered = x - x.mean(dim=0, keepdim=True)
		flat = centered.reshape(centered.size(0), num_factors, -1)
		norms = flat.pow(2).sum(dim=(0, 2)).sqrt().clamp_min(1e-12)
		for f_idx in range(num_factors):
			sims = []
			for g_idx in range(num_factors):
				if f_idx == g_idx:
					continue
				sim = (flat[:, f_idx, :] * flat[:, g_idx, :]).sum().abs() / (norms[f_idx] * norms[g_idx])
				sims.append(sim)
			if sims:
				redundancy[f_idx] = torch.stack(sims).mean()

		scores = relevance - self.mrmr_redundancy_weight * redundancy
		if not torch.isfinite(scores).all():
			scores = torch.arange(num_factors, device=z_factors.device, dtype=z_factors.dtype)
		scores = scores - scores.min()
		if float(scores.max().item()) > 0.0:
			scores = scores / scores.max().clamp_min(1e-12)

		source_count = int(round(self.factor_select_ratio * num_factors))
		source_count = min(num_factors - 1, max(1, source_count))
		source_idx = torch.topk(scores, k=source_count, largest=True).indices
		source_mask = torch.zeros(num_factors, dtype=torch.bool, device=z_factors.device)
		source_mask[source_idx] = True
		target_mask = ~source_mask
		if not bool(target_mask.any()):
			target_mask[source_idx[-1]] = True
		return target_mask, scores

	def _scaled_cosine_error(self, pred, target):
		pred_f = pred.reshape(-1, pred.size(-1))
		target_f = target.reshape(-1, target.size(-1))
		cos = F.cosine_similarity(pred_f, target_f, dim=-1, eps=1e-8)
		return (1.0 - cos).clamp_min(0.0).pow(self.sce_power).mean()

	def _factor_reconstruction_loss(self, z, q):
		if self.num_factors <= 1:
			return z.sum() * 0.0
		z_factors = z.view(z.size(0), self.num_factors, self.factor_dim)
		target_mask, scores = self._factor_context_mask(z_factors, q)
		visible = z_factors.clone()
		visible[:, target_mask, :] = 0.0
		recon = self.factor_decoder(visible.reshape(z.size(0), -1)).view(
			z.size(0), self.num_factors, self.factor_dim
		)
		self.cimage_factor_mask = target_mask
		self.cimage_factor_scores = scores
		return self._scaled_cosine_error(recon[:, target_mask, :], z_factors[:, target_mask, :].detach())

	def encode(self, x, edge_index, edge_weight=None):
		x_dense = self._dense_features(x).to(self.device)
		edge_index_masked, edge_weight_masked = self._edge_mask(edge_index, edge_weight)

		x_hid = self.preprocess(x_dense)
		x_hid = self.activation(x_hid)
		x_hid = F.dropout(x_hid, p=self.dropout, training=self.training)

		factors = []
		for linear, propagate, project in zip(self.factor_linears, self.factor_propagates, self.factor_projections):
			h = linear(x_hid)
			h = propagate(h, edge_index_masked, edge_weight_masked)
			h = project(h)
			factors.append(h)
		z = torch.cat(factors, dim=1)
		self.Z = self.batch_norm(z)
		self.mean = self.Z
		self.logstd = torch.zeros_like(self.mean)

		q = F.softmax(self.cluster_head(self.mean), dim=1)
		self.cimage_cluster_q = q
		self.cimage_cluster_loss = self._soft_modularity_loss(q, edge_index, edge_weight)
		self.cimage_factor_loss = self._factor_reconstruction_loss(self.mean, q)
		return self.mean

	def cimage_losses(self):
		factor = getattr(self, "cimage_factor_loss", None)
		cluster = getattr(self, "cimage_cluster_loss", None)
		if factor is None:
			factor = torch.zeros((), dtype=self.preprocess.weight.dtype, device=self.preprocess.weight.device)
		if cluster is None:
			cluster = torch.zeros((), dtype=self.preprocess.weight.dtype, device=self.preprocess.weight.device)
		return factor, cluster

	def forward(self, x, edge_index, edge_weight=None):
		return self.encode(x, edge_index, edge_weight)

class VGAE_ENCODER(nn.Module):
	def __init__(self, input_dim, hidden1, hidden2, dropout, device):
		super(VGAE_ENCODER, self).__init__()
		self.device = device
		self.hidden1 = hidden1
		self.hidden2 = hidden2
		self.base_gcn = GraphConvSparse(input_dim, hidden1, dropout, activation = F.relu)
		self.gcn_mean = GraphConvSparse(hidden1, hidden2, dropout, activation = lambda x:x)
		self.gcn_logstddev = GraphConvSparse(hidden1, hidden2, dropout, activation = lambda x:x)
		self.projection_layer = nn.Linear(hidden2, hidden2)

		for m in self.modules():
			self.weights_init(m)
	
	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def encode(self, X, adj):
		if self.training:
			
			hidden = self.base_gcn(X, adj)
			self.Z = self.gcn_mean(hidden, adj)
			self.mean = self.projection_layer(self.Z)

			self.logstd = self.gcn_logstddev(hidden, adj)
			self.logstd = self.logstd.clamp(max = 10)

			gaussian_noise = torch.randn(X.size(0), self.hidden2).to(self.device)
			sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean

			return sampled_z
		else:
			hidden = self.base_gcn(X, adj)
			self.Z = self.gcn_mean(hidden, adj)
			self.mean = self.projection_layer(self.Z)

			return self.mean
		
	def forward(self, X, adj):
		Z = self.encode(X, adj)
		return Z
		
class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, dropout, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
		self.reset_parameters()
		self.activation = activation
		self.dropout = dropout

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.weight)

	def forward(self, inputs, adj):
		# inputs = F.dropout(inputs, self.dropout, self.training)
		x = torch.mm(inputs, self.weight)
		x = torch.mm(adj, x)
		outputs = self.activation(x)
		return outputs

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	# dotproduct = torch.matmul(Z,Z.t())
	# mean = torch.mean(dotproduct)
	# mean = torch.mean(dotproduct, dim = 1).reshape(dotproduct.shape[0], 1)
	# A_pred = torch.sigmoid(dotproduct - mean)
	return A_pred

# class Decoder(nn.Module):
# 	def __init__(self, input_dim, output_dim):
# 		super(Decoder, self).__init__()

# 		self.projection = nn.Linear(input_dim, output_dim)

# 		for m in self.modules():
# 			self.weights_init(m)
	
# 	def weights_init(self, m):
# 		if isinstance(m, nn.Linear):
# 			torch.nn.init.xavier_uniform_(m.weight.data)
# 			if m.bias is not None:
# 				m.bias.data.fill_(0.0)

# 	def forward(self, z):
# 		z = self.projection(z)
# 		return dot_product_decode(z)

class MLP(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(MLP, self).__init__()
		self.projection = nn.Linear(input_dim, output_dim)
		self.channel_1 = nn.Linear(input_dim, output_dim)
		self.channel_2 = nn.Linear(input_dim, output_dim)
		self.channel_3 = nn.Linear(input_dim, output_dim)
		self.activation = F.relu

		for m in self.modules():
			self.weights_init(m)
	
	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)
	
	def freeze(self):
		for param in self.parameters():
			param.grad = None
			param.requires_grad_(False)
			param.requires_grad = False

	def unfreeze(self):
		for param in self.parameters():
			param.requires_grad_(True)
			param.requires_grad = True

	def forward(self, x):
		hidden = self.activation(self.projection(x))
		# hidden = F.dropout(hidden, p = 0.5, training = self.training)

		# out_1 = self.channel_1(hidden)
		# out_2 = self.channel_2(hidden)
		# out_3 = self.channel_3(hidden)

		out_1 = self.channel_1(hidden) + x
		out_2 = self.channel_2(hidden) + x
		out_3 = self.channel_3(hidden) + x
		
		# out_1 = (torch.randn(x.size(0), x.size(1)).cuda()) * torch.exp(self.channel_1(hidden).clamp(min = -10.0, max = 10.0)) + x
		# out_2 = (torch.randn(x.size(0), x.size(1)).cuda()) * torch.exp(self.channel_2(hidden).clamp(min = -10.0, max = 10.0)) + x
		# out_3 = (torch.randn(x.size(0), x.size(1)).cuda()) * torch.exp(self.channel_3(hidden).clamp(min = -10.0, max = 10.0)) + x

		# return [out_1, out_2, out_3]
		return out_1

class LogReg(nn.Module):
	def __init__(self, ft_in, nb_classes):
		super(LogReg, self).__init__()
		self.fc1 = nn.Linear(ft_in, int(ft_in/2))
		self.fc2 = nn.Linear(int(ft_in/2), int(ft_in/4))
		self.fc3 = nn.Linear(int(ft_in/4), nb_classes)

		self.activation = F.relu

		for m in self.modules():
			self.weights_init(m)
	
	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)
	
	def forward(self, seq):
		hid = self.activation(self.fc1(seq))
		hid = F.dropout(hid, p = 0.5, training = self.training)
		hid = self.activation(self.fc2(hid))
		hid = F.dropout(hid, p = 0.5, training = self.training)
		ret = self.fc3(hid)
		return ret
