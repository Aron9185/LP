import argparse
import atexit
import io
import logging
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
parser.add_argument("--seed", type=int, default=1, help="Random seed.")
parser.add_argument("--dataset", type=str, default="cora", help="type of dataset.")
parser.add_argument(
    "--epochs", type=int, default=700, help="Number of epochs to train."
)
parser.add_argument(
    "--hidden1", type=int, default=256, help="Number of units in hidden layer 1."
)
parser.add_argument(
    "--hidden2", type=int, default=64, help="Number of units in hidden layer 2."
)
parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
parser.add_argument(
    "--dropout", type=float, default=0.3, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument(
    "--aug_graph_weight", type=float, default=1.0, help="augmented graph weight"
)
parser.add_argument("--aug_ratio", type=float, default=0.1, help="augmented ratio")
parser.add_argument("--aug_bound", type=float, default=0.1, help="augmented edge bound")
parser.add_argument(
    "--alpha", type=float, default=1.0, help="Reconstruction Loss Weight"
)
parser.add_argument("--beta", type=float, default=1.0, help="KL Divergence Weight")
parser.add_argument("--gamma", type=float, default=1.0, help="Contrastive Loss Weight")
parser.add_argument(
    "--delta", type=float, default=1.0, help="Inter Contrastive Loss Weight"
)
parser.add_argument(
    "--temperature", type=float, default=1.0, help="Contrastive Temperature"
)
parser.add_argument(
    "--logging", dest="logging", action="store_true", help="Enable file logging"
)
parser.add_argument("--date", type=str, default="0000", help="date")
parser.add_argument(
    "--ver", type=str, default="origin", help="modified version"
)  # [origin, thm_exp, uncover]
parser.add_argument("--idx", type=str, default="1", help="index")  # [1,2,3,4,5]
parser.add_argument(
    "--degree_threshold", type=float, default=0.5, help="degree threshold"
)  # [1,2,3,4,5]
parser.add_argument(
    "--loss_ver", type=str, default="origin", help="loss version"
)  # [origin, nei]
parser.add_argument(
    "--feat_mask_ratio", type=float, default=0.1, help="feature augmented ratio"
)
parser.add_argument("--scaling", type=float, default=1.0, help="scaling factor")

# Aron
parser.add_argument(
    "--pretrain_epochs",
    type=int,
    default=100,
    help="Pretraining epochs on original graph (only for aron_desc/aron_asc).",
)
parser.add_argument(
    "--frozen_scores",
    type=str,
    default="",
    help="Path to save/load frozen score matrix (.pt).",
)
parser.add_argument(
    "--pretrained_ckpt",
    type=str,
    default="",
    help="Path to save/load pretrained encoder state_dict (.pt).",
)
parser.add_argument(
    "--dbscan_eps", type=float, default=None, help="DBSCAN eps; None=auto"
)
parser.add_argument("--dbscan_min_samples", type=int, default=5)
parser.add_argument(
    "--dbscan_metric", type=str, default="cosine", choices=["cosine", "euclidean"]
)
parser.add_argument("--topk_per_node", type=int, default=64)
parser.add_argument(
    "--aug_ratio_epoch",
    type=float,
    default=None,
    help="Optional per-epoch budget as a fraction of |E0|; None disables throttle.",
)
# also use: --ver aron_desc or --ver aron_asc

args = parser.parse_args()


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)


def main():
    print(f"Dataset: {args.dataset}")

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda:0")
    # if args.dataset_str == 'pubmed':
    #     device = torch.device('cpu')

    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)

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

    Z, roc_hist, mod_hist, edge_index = train_encoder(
        dataset_str=args.dataset,
        device=device,
        num_epoch=args.epochs,  # or your existing name
        adj=adj,  # dense WITH self-loops
        features=features,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        aug_graph_weight=args.aug_graph_weight,
        aug_ratio=args.aug_ratio,
        aug_bound=args.aug_bound,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        temperature=args.temperature,
        labels=labels,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        ver=args.ver,
        degree_ratio=args.degree_threshold,  # you subtract 1 inside to get excl-self floor
        loss_ver=args.loss_ver,
        feat_maske_ratio=args.feat_mask_ratio,
        pretrain_epochs=args.pretrain_epochs,
        frozen_scores_path=args.frozen_scores,
        pretrained_ckpt_path=args.pretrained_ckpt,
        # NEW
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        dbscan_metric=args.dbscan_metric,
        topk_per_node=args.topk_per_node,
        aug_ratio_epoch=args.aug_ratio_epoch,
        run_tag=args.idx,
        seed=args.seed,
    )

    # Plot(args.dataset_str, roc_history, modification_ratio_history)
    gaussion_KDE(args.dataset, Z)
    vMF_KDE(args.dataset, Z)

    if labels is not None:
        train_classifier(device, Z, labels, idx_train, idx_val, idx_test)
        logist_regressor_classification(device, Z, labels, idx_train, idx_val, idx_test)
        Visualize(args.dataset, Z, labels)
        # Visualize_with_edge(args.dataset_str, Z, labels, from_scipy_sparse_matrix(adj)[0])


class TqdmOnlyStderr(io.TextIOBase):
    """Forward everything to log_file; forward only tqdm-style carriage-return updates to terminal."""

    def __init__(self, term_stderr, log_file, show_tracebacks_on_terminal=False):
        self.term = term_stderr
        self.log = log_file
        self.show_tracebacks = show_tracebacks_on_terminal
        self._saw_traceback = False

    def write(self, data: str):
        # Always write to the log file (strip \r so the log is clean)
        self.log.write(data.replace("\r", ""))
        self.log.flush()

        # Heuristic: forward ONLY tqdm refreshes to terminal
        is_tqdm_update = ("\r" in data and not data.endswith("\n")) or (
            "it/s" in data and "\n" not in data
        )

        # Optionally show tracebacks on terminal too
        if self.show_tracebacks:
            if "Traceback (most recent call last):" in data:
                self._saw_traceback = True
            if self._saw_traceback:
                self.term.write(data)
                self.term.flush()
                if data.endswith("\n"):
                    self._saw_traceback = False
                return len(data)

        if is_tqdm_update:
            self.term.write(data)
            self.term.flush()
        return len(data)

    def flush(self):
        self.term.flush()
        self.log.flush()

    # Make tqdm think this is a TTY so it renders properly
    def isatty(self):
        return True


# --- inside your main guard ---
if __name__ == "__main__":
    if args.logging:
        # Prepare log path
        log_dir = Path("/home/retro/ARON/log") / str(args.date)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = (
            log_dir
            / f"{args.dataset}_{args.ver}_r{args.aug_ratio}_fmr{args.feat_mask_ratio}_d{args.degree_threshold}_{args.idx}_loss_{args.loss_ver}.log"
        )

        # Open line-buffered file
        log_file = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")

        # Keep terminal streams
        term_out, term_err = sys.stdout, sys.stderr

        # Route all prints to the log only
        sys.stdout = log_file

        # Route stderr to (log + terminal only for tqdm)
        sys.stderr = TqdmOnlyStderr(
            term_err, log_file, show_tracebacks_on_terminal=True
        )  # set False to hide tracebacks

        # Ensure Python logging does NOT spam terminal: send it to stdout (which we redirected to file)
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],  # goes to log_file only
        )

        try:
            set_random_seed(args.seed)
            main()
        finally:
            # Restore streams
            sys.stdout = term_out
            sys.stderr = term_err
            log_file.close()
    else:
        set_random_seed(args.seed)
        main()
