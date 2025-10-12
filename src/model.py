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