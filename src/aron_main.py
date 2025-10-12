import argparse
import atexit
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from aron_train import logist_regressor_classification, train_classifier, train_encoder
from input_data import load_data
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from utils import Plot, Visualize, Visualize_with_edge, gaussion_KDE, vMF_KDE

parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--dataset-str', type=str, default = 'cora', help='type of dataset.')
parser.add_argument('--epochs', type=int, default = 700, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default = 256, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default = 64, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default = 0.001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default = 0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default = 5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--aug_graph_weight', type=float, default = 1.0, help='augmented graph weight')
parser.add_argument('--aug_ratio', type=float, default = 0.1, help='augmented ratio')
parser.add_argument('--aug_bound', type=float, default = 0.1, help='augmented edge bound')
parser.add_argument('--alpha', type=float, default = 1.0, help='Reconstruction Loss Weight')
parser.add_argument('--beta', type=float, default = 1.0, help='KL Divergence Weight')
parser.add_argument('--gamma', type=float, default = 1.0, help='Contrastive Loss Weight')
parser.add_argument('--delta', type=float, default = 1.0, help='Inter Contrastive Loss Weight')
parser.add_argument('--temperature', type=float, default = 1.0, help='Contrastive Temperature')
parser.add_argument('--logging', dest='logging', action='store_true',
                    help='Enable file logging')
parser.add_argument('--date', type=str, default = "0000", help='date')
parser.add_argument('--ver', type=str, default = "origin", help='modified version') # [origin, thm_exp, uncover]
parser.add_argument('--idx', type=str, default = "1", help='index') # [1,2,3,4,5]
parser.add_argument('--degree_threshold', type=float, default = 0.5, help='degree threshold') # [1,2,3,4,5]
parser.add_argument('--loss_ver', type=str, default = "origin", help='loss version') # [origin, nei]
parser.add_argument('--feat_mask_ratio', type=float, default = 0.1, help='feature augmented ratio')
parser.add_argument('--scaling', type=float, default = 1.0, help='scaling factor')

# Aron
parser.add_argument('--pretrain_epochs', type=int, default=100, help='Pretraining epochs on original graph (only for aron_desc/aron_asc).')
parser.add_argument('--frozen_scores', type=str, default='', help='Path to save/load frozen score matrix (.pt).')
parser.add_argument('--pretrained_ckpt', type=str, default='', help='Path to save/load pretrained encoder state_dict (.pt).')
# also use: --ver aron_desc or --ver aron_asc

args = parser.parse_args()

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)

def main():
    print(f'Dataset: {args.dataset_str}')

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    # if args.dataset_str == 'pubmed':
    #     device = torch.device('cpu')

    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset_str)
    
    # print(f"adj:{adj}")
    # print(f"idx_train:{idx_train}")
    # print(f"idx_val:{idx_val}")
    # print(f"idx_test:{idx_test}")
    
    # Perform random sampling on the edges for 'ogbl-collab'
    """if args.dataset_str == 'ogbl-collab':
        # Extract edge indices from the adjacency matrix
        row, col = adj.nonzero()
        num_edges = row.shape[0]  # Total number of edges
        
        # Define the fraction of edges to sample (e.g., 50%)
        sampling_ratio = 0.5
        num_sampled_edges = int(sampling_ratio * num_edges)

        # Randomly sample edge indices
        sampled_indices = np.random.choice(num_edges, num_sampled_edges, replace=False)

        # Create a new sparse adjacency matrix with the sampled edges
        sampled_row = row[sampled_indices]
        sampled_col = col[sampled_indices]
        adj = sp.coo_matrix((np.ones(num_sampled_edges), (sampled_row, sampled_col)), shape=adj.shape)"""
    
    Z, roc_history, modification_ratio_history, edge_index = train_encoder(
        args.dataset_str, device, args.epochs, adj, features, args.hidden1, args.hidden2, args.dropout, args.lr, args.weight_decay,
        args.aug_graph_weight, args.aug_ratio, args.aug_bound, args.alpha, args.beta, args.gamma, args.delta, args.temperature,
        labels, idx_train, idx_val, idx_test, args.ver, args.degree_threshold, args.loss_ver, args.feat_mask_ratio,
        args.pretrain_epochs, args.frozen_scores, args.pretrained_ckpt
    )
    
    #Plot(args.dataset_str, roc_history, modification_ratio_history)
    gaussion_KDE(args.dataset_str, Z)
    vMF_KDE(args.dataset_str, Z)
    
    if labels is not None:
        train_classifier(device, Z, labels, idx_train, idx_val, idx_test)
        logist_regressor_classification(device, Z, labels, idx_train, idx_val, idx_test)
        Visualize(args.dataset_str, Z, labels)
        # Visualize_with_edge(args.dataset_str, Z, labels, from_scipy_sparse_matrix(adj)[0])
       


if __name__ == '__main__':
    if args.logging == True:
        old_stdout = sys.stdout
        
        if args.logging:
            log_dir = Path('/home/retro/SECRET/log') / str(args.date)
            log_dir.mkdir(parents=True, exist_ok=True)

            log_path = log_dir / f"{args.dataset_str}_{args.ver}_r{args.aug_ratio}_fmr{args.feat_mask_ratio}_d{args.degree_threshold}_{args.idx}_loss_{args.loss_ver}.log"
            old_stdout = sys.stdout
            log_file = open(log_path, "w")
            sys.stdout = log_file
        
        set_random_seed(args.seed)
        main()

        sys.stdout = old_stdout
        log_file.close()
    else:
        set_random_seed(args.seed)

        main()
        