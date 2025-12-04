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
import os

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
parser.add_argument("--ver", type=str, default="origin",
    help="version: origin | aron_* | remove_only_intra[_c0p|_noncore|_cp][_fracX] | remove_only_inter[_fracX] | remove_only_both[_fracX]"
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
parser.add_argument("--cluster_method", choices=["none", "gmm", "louvain"], default="gmm")
parser.add_argument("--cluster_mode",   choices=["any", "intra", "inter"], default="any")
parser.add_argument("--gmm_k", type=int, default=16)        # pick your K
parser.add_argument("--gmm_tau", type=float, default=0.55)  # confidence→noise
parser.add_argument(
    "--c0p_prune_frac",
    type=float,
    default=0.10,   # 0.10 = 移除 outlier 的同群內邊 10%；設 0 可關閉
    help="For outlier nodes (non-core with radius > thr), drop this fraction of INTRA-cluster edges to compact C0p (0 disables)."
)
parser.add_argument(
    "--sweep_scope",
    type=str,
    default="cp_all",
    choices=["cp_all", "c0p_only", "cp_minus_c0p"],
    help="Edge-candidate scope for offline sweep after training."
)

parser.add_argument("--restricted", action="store_true")
parser.add_argument("--restrict_alpha", type=float, default=0.8)
parser.add_argument("--restrict_gamma", type=float, default=1.0)

parser.add_argument("--run_tag", type=str, default="", help="Optional run identifier tag (used for log naming)")
parser.add_argument("--sweep_mode", action="store_true",
    help="Do NOT hijack stdout/stderr; print metrics to stdout so external runners can capture. Also disables tqdm.")

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
        # NEW cluster controls
        cluster_method=args.cluster_method,
        cluster_mode=args.cluster_mode,
        # NEW: restricted augmentation knobs
        restricted=args.restricted,
        restrict_alpha=args.restrict_alpha,
        restrict_gamma=args.restrict_gamma,
        c0p_prune_frac=args.c0p_prune_frac,
        sweep_scope=args.sweep_scope,
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
        self.log.write(data.replace("\r", ""))
        self.log.flush()

        is_tqdm_update = ("\r" in data and not data.endswith("\n")) or ("it/s" in data and "\n" not in data)

        if self.show_tracebacks:
            if "Traceback (most recent call last):" in data:
                self._saw_traceback = True
            if self._saw_traceback:
                self.term.write(data); self.term.flush()
                if data.endswith("\n"):
                    self._saw_traceback = False
                return len(data)

        if is_tqdm_update:
            self.term.write(data); self.term.flush()
        return len(data)

    def flush(self):
        self.term.flush()
        self.log.flush()

    def isatty(self):
        return True


if __name__ == "__main__":
    # optional: align folder name with runner
    LOG_ROOT = "log"

    if args.sweep_mode:
        # Disable tqdm noise when sweeping
        os.environ["TQDM_DISABLE"] = "1"

        # IMPORTANT: do NOT redirect stdout/stderr here.
        # Let the external sweep script capture everything into its own log file.
        set_random_seed(args.seed)
        main()

    elif args.logging:
        # mirror script: logs/<DATESTR>/...
        log_dir = Path(LOG_ROOT) / str(args.date)
        log_dir.mkdir(parents=True, exist_ok=True)

        a_str   = f"{float(args.restrict_alpha):.2f}"
        g_str   = f"{float(args.restrict_gamma):.2f}"
        c0p_str = f"{float(args.c0p_prune_frac):.2f}"

        # keep your existing r/fmr/d and suffix tags
        r_str   = f"{float(args.aug_ratio):.1f}"
        fmr_str = f"{float(args.aug_bound):.1f}"
        d_str   = f"{float(args.degree_threshold):.1f}"

        cm = getattr(args, "cluster_method", "none")
        cm_suffix = f"_{cm}" if cm and cm != "none" else ""
        loss_tag  = f"_loss_{args.loss_ver}" if getattr(args, "loss_ver", "") else ""
        tag_suffix = f"_tag{args.run_tag}" if args.run_tag else ""

        # NEW filename (core pattern first, extras after)
        fname = (
            f"{args.dataset}_{args.ver}"
            f"_a{a_str}_g{g_str}_c0p{c0p_str}_seed{args.seed}"            # <-- collector-critical part
            f"_r{r_str}_fmr{fmr_str}_d{d_str}"                            # your original fields
            f"{cm_suffix}_idx{args.idx}{tag_suffix}{loss_tag}.log"
        )
        log_path = log_dir / fname

        log_file = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")

        term_out, term_err = sys.stdout, sys.stderr
        sys.stdout = log_file
        sys.stderr = TqdmOnlyStderr(term_err, log_file, show_tracebacks_on_terminal=True)

        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

        print("===== RUN META =====")
        print(f"date={args.date} time={time.strftime('%F %T')}")
        print(f"dataset={args.dataset} ver={args.ver} mode={args.cluster_mode} method={args.cluster_method}")
        print(f"seed={args.seed} idx={args.idx} run_tag={args.run_tag or '<none>'}")
        print(f"aug_ratio={args.aug_ratio} aug_bound={args.aug_bound} degree_thr={args.degree_threshold}")
        print(f"topk_per_node={args.topk_per_node} aug_ratio_epoch={args.aug_ratio_epoch}")
        if args.restricted:
            print(f"restricted=1 alpha={args.restrict_alpha} gamma={args.restrict_gamma}")
        else:
            print("restricted=0")
        print("====================")

        try:
            set_random_seed(args.seed)
            main()
        finally:
            sys.stdout = term_out
            sys.stderr = term_err
            log_file.close()

    else:
        set_random_seed(args.seed)
        main()
