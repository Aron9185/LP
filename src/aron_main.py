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
from aron_train_edit_decoder import logist_regressor_classification, train_classifier, train_encoder
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
parser.add_argument(
    "--split_mode",
    type=str,
    default="random",
    choices=["random", "heart", "cimage_paper"],
    help=(
        "random = old mask_test_edges path, heart = official HeaRT split path, "
        "cimage_paper = PyG RandomLinkSplit(num_val=0.1, num_test=0.05)"
    ),
)
parser.add_argument(
    "--heart_data_dir",
    type=str,
    default="dataset",
    help="Root directory containing {dataset}/train_pos.txt, valid_pos.txt, test_pos.txt, heart_valid_*.npy, heart_test_*.npy",
)
parser.add_argument(
    "--heart_filename",
    type=str,
    default="samples.npy",
    help="HeaRT filename suffix. samples.npy -> heart_valid_samples.npy / heart_test_samples.npy",
)
parser.add_argument(
    "--heart_eval_every",
    type=int,
    default=None,
    help="Override HeaRT validation interval. Default keeps the training-code policy.",
)
parser.add_argument(
    "--heart_val_frac",
    type=float,
    default=None,
    help="Override HeaRT validation fraction used during training. Use 1.0 for full validation.",
)
parser.add_argument(
    "--heart_checkpoint_metric",
    type=str,
    default="roc",
    choices=["roc", "ap", "hit1", "hit3", "hit10", "hit20", "hit50", "hit100"],
    help="Validation metric used to select HeaRT checkpoints.",
)
parser.add_argument(
    "--random_checkpoint_metric",
    type=str,
    default="roc",
    choices=["roc", "ap", "hit1", "hit3", "hit10", "hit20", "hit50", "hit100"],
    help="Validation metric used to select random/cimage-paper split checkpoints.",
)
parser.add_argument(
    "--lp_train_graph",
    type=str,
    default="train",
    choices=["train", "full"],
    help=(
        "Graph used for link-prediction training tensors. "
        "train = no-leak split graph; full = CIMAGE-style leakage protocol where "
        "the encoder/reconstruction see all observed edges while eval still uses the held-out split."
    ),
)

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

# === NEW: static pre-prune knobs ===
parser.add_argument(
    "--pre_prune_frac",
    type=float,
    default=0.0,
    help="Static pre-prune: fraction of candidate edges to drop BEFORE training (0 disables).",
)
parser.add_argument(
    "--pre_prune_scope",
    type=str,
    default="cp_all",
    choices=["cp_all", "c0p_only", "cp_minus_c0p"],
    help="Scope for static pre-prune (cp_all / c0p_only / cp_minus_c0p).",
)

parser.add_argument("--run_tag", type=str, default="", help="Optional run identifier tag (used for log naming)")
parser.add_argument("--sweep_mode", action="store_true",
    help="Do NOT hijack stdout/stderr; print metrics to stdout so external runners can capture. Also disables tqdm.")
parser.add_argument(
    "--ae_backbone",
    type=str,
    default="vgnae",
    choices=["vgnae", "vgae", "maskgae", "cimage", "cimage_lite", "cimage_full"],
    help="Autoencoder backbone. vgae aliases VGNAE; cimage aliases cimage_full.",
)
parser.add_argument("--maskgae_mask_rate", type=float, default=0.3, help="Node-feature mask rate used when --ae_backbone maskgae.")
parser.add_argument("--maskgae_feature_weight", type=float, default=1.0, help="Weight for MaskGAE masked-feature reconstruction.")
parser.add_argument("--cimage_factor_weight", type=float, default=0.1, help="Weight for CIMAGE latent factor reconstruction.")
parser.add_argument("--cimage_cluster_weight", type=float, default=0.1, help="Weight for CIMAGE clustering loss.")
parser.add_argument("--cimage_num_factors", type=int, default=8, help="Number of CIMAGE latent factors.")
parser.add_argument("--cimage_num_clusters", type=int, default=16, help="Number of CIMAGE pseudo-label clusters.")
parser.add_argument("--cimage_cluster_alpha", type=float, default=1.0, help="Student-t cluster assignment alpha for CIMAGE-lite.")
parser.add_argument("--cimage_pseudo_label_threshold", type=float, default=0.90, help="Confidence threshold for CIMAGE full pseudo-label factor scoring.")
parser.add_argument("--cimage_factor_select_ratio", type=float, default=0.50, help="Fraction of CIMAGE full factors used as the visible context.")
parser.add_argument("--cimage_mrmr_redundancy_weight", type=float, default=0.20, help="Redundancy penalty used by CIMAGE full factor selection.")
parser.add_argument("--cimage_cluster_balance_weight", type=float, default=0.05, help="Balance penalty used by CIMAGE full modularity clustering.")
parser.add_argument("--cimage_sce_power", type=float, default=2.0, help="Power for CIMAGE full scaled cosine error factor reconstruction.")

# Edited decoder / decoded-graph augmentation
parser.add_argument("--use_edited_decoder", action="store_true", help="Enable the edited decoder branch.")
parser.add_argument("--decoder_type", type=str, default="bilinear", choices=["bilinear", "mlp_pair", "pair_mlp_struct"])
parser.add_argument("--decoder_normalize_input", dest="decoder_normalize_input", action="store_true", help="L2-normalize decoder input embeddings before pair scoring.")
parser.add_argument("--no_decoder_normalize_input", dest="decoder_normalize_input", action="store_false", help="Use raw decoder input embeddings without L2 normalization.")
parser.add_argument("--score_source", type=str, default="dot", choices=["dot", "decoder", "pred_decoder"], help="Score validation/test edges with dot-product embeddings, the edit decoder, or the prediction decoder.")
parser.add_argument("--mlp_pair_max_rows", type=int, default=16, help="Row chunk size for pair decoders to control GPU memory.")
parser.add_argument("--decoder_objective", type=str, default="hybrid", choices=["recon", "hybrid"], help="Decoder edit objective.")
parser.add_argument("--decoder_recon_weight", type=float, default=1.0)
parser.add_argument("--decoder_keep_weight", type=float, default=1.0, help="Weight for structure-preserving BCE outside rewrite scope.")
parser.add_argument("--decoder_add_rank_weight", type=float, default=1.0, help="Weight for ranking valid add candidates above bad additions.")
parser.add_argument("--decoder_remove_rank_weight", type=float, default=1.0, help="Weight for ranking valid kept edges above removable edges.")
parser.add_argument("--decoder_rank_margin", type=float, default=0.2, help="Margin used for decoder ranking losses.")
parser.add_argument(
    "--decoder_rank_strategy",
    type=str,
    default="easy",
    choices=["easy", "heart_like"],
    help="Mining strategy for decoder ranking negatives. heart_like uses hard boundary negatives with shared-endpoint fallback.",
)
parser.add_argument(
    "--decoder_rank_neg_k",
    type=int,
    default=8,
    help="Number of hard negatives per positive anchor for decoder heart-like ranking.",
)
parser.add_argument(
    "--decoder_rank_pool_factor",
    type=int,
    default=4,
    help="Boundary-pool multiplier for heart-like decoder ranking mining.",
)
parser.add_argument("--heart_rank_weight", type=float, default=0.0, help="Weight for HeaRT-style train-positive vs hard-nonedge decoder ranking.")
parser.add_argument("--heart_rank_margin", type=float, default=0.2, help="Margin for HeaRT-style decoder ranking.")
parser.add_argument("--heart_rank_neg_k", type=int, default=8, help="Hard negatives per train positive for HeaRT-style decoder ranking.")
parser.add_argument("--heart_rank_pool_factor", type=int, default=4, help="Hard-negative pool multiplier for HeaRT-style decoder ranking.")
parser.add_argument(
    "--prediction_decoder_type",
    type=str,
    default="none",
    choices=[
        "none",
        "pair_residual_struct",
        "pair_residual_struct_ncnc",
        "pair_residual_struct_ncnc_multi",
        "pair_residual_struct_ocn",
    ],
    help="Optional prediction decoder trained separately from the edit decoder.",
)
parser.add_argument("--prediction_rank_weight", type=float, default=1.0, help="Weight for prediction-decoder HeaRT ranking loss.")
parser.add_argument("--prediction_bce_weight", type=float, default=0.1, help="Weight for sampled train-edge BCE on prediction-decoder logits.")
parser.add_argument("--prediction_rank_margin", type=float, default=0.2, help="Margin for prediction-decoder ranking loss.")
parser.add_argument("--prediction_rank_neg_k", type=int, default=16, help="Hard negatives per train positive for prediction-decoder ranking.")
parser.add_argument("--prediction_rank_pool_factor", type=int, default=8, help="Hard-negative pool multiplier for prediction-decoder ranking.")
parser.add_argument("--prediction_joint_start_epoch", type=int, default=-1, help="Epoch when prediction-decoder loss may backpropagate into the encoder. -1 uses decoded rewrite start.")
parser.add_argument("--prediction_encoder_weight", type=float, default=0.0, help="Weight for the late joint prediction-decoder loss on encoder embeddings.")
parser.add_argument("--compactness_weight", type=float, default=0.2, help="Loss weight for cluster compactness (pull).")
parser.add_argument("--compactness_objective", type=str, default="hybrid", choices=["radius", "prototype", "hybrid"], help="Compactness objective for edited latent training.")
parser.add_argument("--compactness_radius_metric", type=str, default="cosine", choices=["cosine", "mahalanobis"], help="Radius metric used by compactness diagnostics/objective.")
parser.add_argument("--preserve_weight", type=float, default=0.0)
parser.add_argument("--separate_edit_training", action="store_true", help="Use two-stage training: task learning before edit_start_epoch, then edit-only optimization afterward.")
parser.add_argument("--edit_phase_retain_recon_weight", type=float, default=0.0, help="Optional reconstruction-retention weight during phase-2 edit training.")
parser.add_argument("--edit_phase_retain_cl_weight", type=float, default=0.0, help="Optional contrastive-retention weight during phase-2 edit training.")
parser.add_argument(
    "--phase2_task_main_loss",
    action="store_true",
    help="In phase 2, keep the full task loss as the main objective and treat edit loss as a regularizer.",
)
parser.add_argument(
    "--edit_phase_edit_weight",
    type=float,
    default=0.10,
    help="Weight of edit_total_loss when --phase2_task_main_loss is enabled.",
)
parser.add_argument("--editor_pull_strength", type=float, default=0.10, help="Direct latent pulling strength (the augmentation trigger).")
parser.add_argument(
    "--editor_push_scope",
    type=str,
    default="none",
    choices=["none", "noncompact_cp", "noise", "noncompact_cp_and_noise"],
    help="Optional direct latent push-away scope applied after pulling.",
)
parser.add_argument("--editor_noncompact_push_strength", type=float, default=0.0, help="Push strength for non-C0p non-noise CP nodes.")
parser.add_argument("--editor_noise_push_strength", type=float, default=0.0, help="Push strength for GMM noise nodes.")
parser.add_argument("--editor_push_preserve_norm", dest="editor_push_preserve_norm", action="store_true", help="Rescale pushed embeddings back to their original norm.")
parser.add_argument("--no_editor_push_preserve_norm", dest="editor_push_preserve_norm", action="store_false", help="Do not preserve embedding norms after push-away edits.")
parser.add_argument("--editor_edit_scale", type=float, default=0.0)
parser.add_argument("--edit_start_epoch", type=int, default=10)
parser.add_argument("--edit_train_start_epoch", type=int, default=-1, help="Epoch to start edit-decoder training. -1 follows edit_start_epoch for backward compatibility.")
parser.add_argument("--decoded_rewrite_start_epoch", type=int, default=-1, help="Epoch to start applying decoded graph rewrites. -1 follows edit_start_epoch for backward compatibility.")
parser.add_argument("--decoded_rewrite_every", type=int, default=1, help="Apply decoded graph rewrites every N epochs after rewrite start. Default keeps every-epoch rewrites.")
parser.add_argument("--eval_log_every", type=int, default=5)
parser.add_argument("--train_eval_every", type=int, default=1, help="Run non-HeaRT validation/test evaluation every N epochs. Default keeps per-epoch evaluation.")
parser.add_argument("--skip_train_acc", action="store_true", help="Skip per-epoch full-matrix train accuracy during training evaluation.")
parser.add_argument("--decoder_diag_every", type=int, default=-1, help="Run decoder score diagnostics every N epochs; -1 means every evaluated epoch, 0 disables training-time diagnostics.")
parser.add_argument("--edit_metric_every", type=int, default=1, help="Run edit compactness/radius diagnostics every N epochs; 0 disables training-time edit metrics.")
parser.add_argument("--edge_eval", dest="edge_eval", action="store_true", help="Evaluate validation/test edges with edge-only scoring instead of materializing a full score matrix.")
parser.add_argument("--full_matrix_eval", dest="edge_eval", action="store_false", help="Materialize a full score matrix for validation/test evaluation.")
parser.add_argument("--freeze_c0p_at_edit_start", dest="freeze_c0p_at_edit_start", action="store_true", help="Freeze GMM/C0p targets once editing starts.")
parser.add_argument("--dynamic_c0p_targets", dest="freeze_c0p_at_edit_start", action="store_false", help="Recompute GMM/C0p targets every time instead of freezing them.")
parser.add_argument("--use_decoded_graph_augment", action="store_true", help="Decode the pulled latent into a rewritten graph, then re-encode on that graph.")
parser.add_argument("--decoded_add_ratio", type=float, default=0.02, help="Per-epoch add budget for decoded graph rewrite, measured as a fraction of E0.")
parser.add_argument("--decoded_remove_ratio", type=float, default=0.00, help="Per-epoch remove budget for decoded graph rewrite, measured as a fraction of E0.")
parser.add_argument("--decoded_add_threshold", type=float, default=None, help="Add decoded edges with score >= this threshold. Overrides add_ratio when set.")
parser.add_argument("--decoded_remove_threshold", type=float, default=None, help="Remove decoded edges with score <= this threshold. Overrides remove_ratio when set.")
parser.add_argument("--decoded_add_quantile", type=float, default=None, help="Add decoded edges from the top-q valid non-edge scores. Example 0.002 keeps the top 0.2%%.")
parser.add_argument("--decoded_remove_quantile", type=float, default=None, help="Remove decoded edges from the bottom-q valid existing-edge scores. Example 0.001 keeps the lowest 0.1%%.")
parser.add_argument("--decoded_max_add_per_round", type=int, default=None, help="Hard cap on decoded edge additions per rewrite round.")
parser.add_argument("--decoded_max_remove_per_round", type=int, default=None, help="Hard cap on decoded edge removals per rewrite round.")
parser.add_argument("--decoded_graph_aug_bound", type=float, default=-1.0, help="Per-node cap fraction for decoded graph additions. Set <= 0 to disable the cap entirely.")
parser.add_argument("--decoded_add_degree_target", type=int, default=-1, help="If >0, prioritize decoded additions that lift rewrite-mask nodes toward this minimum degree.")
parser.add_argument("--decoded_add_degree_target_scope", type=str, default="total", choices=["total", "intra_cluster"], help="Degree used by --decoded_add_degree_target: total graph degree or same-cluster induced degree.")
parser.add_argument("--decoded_add_degree_target_nodes", type=str, default="rewrite", choices=["rewrite", "cp", "cluster_deficit", "rewrite_or_cluster_deficit"], help="Nodes repaired by --decoded_add_degree_target: rewrite mask, all non-noise CP nodes, CP nodes below target, or rewrite mask plus CP deficits.")
parser.add_argument("--decoded_guarantee_degree_target", action="store_true", help="Let the degree-target repair pass exceed add_ratio and per-node caps so target nodes reach the requested degree whenever valid candidates exist.")
parser.add_argument("--decoded_degree_floor", type=int, default=None, help="Minimum degree floor (excluding self-loops) when removing decoded edges. Defaults to the run's degree threshold floor.")
parser.add_argument("--decoded_allow_cross_cluster", dest="decoded_same_cluster_only", action="store_false", help="Allow decoded rewrites across clusters.")
parser.add_argument("--decoded_same_cluster_only", dest="decoded_same_cluster_only", action="store_true", help="Restrict decoded rewrites to same-cluster pairs only.")
parser.add_argument("--decoded_require_c0p_endpoint", dest="decoded_require_c0p_endpoint", action="store_true", help="Require at least one endpoint of a rewritten edge to be in C0p.")
parser.add_argument("--decoded_no_c0p_endpoint", dest="decoded_require_c0p_endpoint", action="store_false", help="Do not require C0p membership for rewritten edges.")
parser.add_argument("--decoded_require_c0p_noncompact_endpoint", action="store_true", help="Require decoded rewritten edges to connect one C0p endpoint to one non-C0p CP endpoint.")
parser.add_argument("--decoded_accumulate_into_base", dest="decoded_accumulate_into_base", action="store_true", help="Persist decoded graph rewrites into the base training graph across epochs.")
parser.add_argument("--decoded_temporary_view_only", dest="decoded_accumulate_into_base", action="store_false", help="Use the decoded rewritten graph only for the current augmented view; do not persist it into the base graph.")
parser.add_argument("--decoded_require_both_c0p", action="store_true", help="Require both endpoints of a rewritten edge to be in C0p.")
parser.add_argument("--pull_mask_scope", type=str, default="cp", choices=["cp", "c0p"], help="Scope for latent pulling.")
parser.add_argument("--compactness_mask_scope", type=str, default="cp", choices=["cp", "c0p"], help="Scope for compactness mask computation.")
parser.add_argument("--rewrite_endpoint_scope", type=str, default="c0p", choices=["cp", "c0p"], help="Scope for the rewriting endpoint restriction.")
parser.add_argument("--phase2_freeze_encoder", dest="phase2_freeze_encoder", action="store_true", help="Freeze the encoder and train only the edited decoder in phase 2.")
parser.add_argument("--phase2_tune_encoder", dest="phase2_freeze_encoder", action="store_false", help="Keep updating the encoder in phase 2 instead of freezing it.")
parser.add_argument("--edit_phase_encoder_lr_scale", type=float, default=0.0, help="Relative encoder LR used in phase 2 when the encoder is not frozen. 0 disables encoder updates.")
parser.add_argument("--decoded_edit_end_epoch", type=int, default=-1, help="Stop decoded rewrites after this epoch. -1 keeps rewrites active through the end.")
parser.add_argument("--decoder_warmup_in_phase1", dest="decoder_warmup_in_phase1", action="store_true", help="Warm up the decoder during phase 1 before using decoded rewrites.")
parser.add_argument("--no_decoder_warmup_in_phase1", dest="decoder_warmup_in_phase1", action="store_false", help="Disable decoder warm-up during phase 1.")
parser.add_argument("--decoder_warmup_recon_weight", type=float, default=1.0, help="Reconstruction weight used for decoder warm-up during phase 1.")
parser.add_argument("--decoder_warmup_use_pulled_latent", action="store_true", help="Use pulled latent instead of base latent during decoder warm-up.")
parser.add_argument("--phase2_decoder_inference_only", dest="phase2_decoder_inference_only", action="store_true", help="In phase 2, freeze decoder training and use it only to infer decoded rewrites.")
parser.add_argument("--no_phase2_decoder_inference_only", dest="phase2_decoder_inference_only", action="store_false", help="Allow decoder training losses to remain active in phase 2.")
parser.add_argument("--skip_oom_epoch", action="store_true", help="If a CUDA OOM occurs during backward/step, clear cache and skip that epoch instead of aborting the run.")
parser.set_defaults(
    freeze_c0p_at_edit_start=True,
    decoded_same_cluster_only=False,
    decoded_require_c0p_endpoint=False,
    decoded_accumulate_into_base=True,
    phase2_freeze_encoder=True,
    decoder_warmup_in_phase1=True,
    phase2_decoder_inference_only=True,
    decoder_normalize_input=True,
    editor_push_preserve_norm=True,
    edge_eval=True,
)

# also use: --ver aron_desc or --ver aron_asc

args = parser.parse_args()


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    print(f"Dataset: {args.dataset}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        ae_backbone=args.ae_backbone,
        maskgae_mask_rate=args.maskgae_mask_rate,
        maskgae_feature_weight=args.maskgae_feature_weight,
        cimage_factor_weight=args.cimage_factor_weight,
        cimage_cluster_weight=args.cimage_cluster_weight,
        cimage_num_factors=args.cimage_num_factors,
        cimage_num_clusters=args.cimage_num_clusters,
        cimage_cluster_alpha=args.cimage_cluster_alpha,
        cimage_pseudo_label_threshold=args.cimage_pseudo_label_threshold,
        cimage_factor_select_ratio=args.cimage_factor_select_ratio,
        cimage_mrmr_redundancy_weight=args.cimage_mrmr_redundancy_weight,
        cimage_cluster_balance_weight=args.cimage_cluster_balance_weight,
        cimage_sce_power=args.cimage_sce_power,
        split_mode=args.split_mode,
        heart_data_dir=args.heart_data_dir,
        heart_filename=args.heart_filename,
        heart_eval_every=args.heart_eval_every,
        heart_val_frac=args.heart_val_frac,
        heart_checkpoint_metric=args.heart_checkpoint_metric,
        random_checkpoint_metric=args.random_checkpoint_metric,
        lp_train_graph=args.lp_train_graph,
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
        gmm_k=args.gmm_k,
        gmm_tau=args.gmm_tau,
        # NEW: restricted augmentation knobs
        restricted=args.restricted,
        restrict_alpha=args.restrict_alpha,
        restrict_gamma=args.restrict_gamma,
        c0p_prune_frac=args.c0p_prune_frac,
        sweep_scope=args.sweep_scope,
        pre_prune_frac=args.pre_prune_frac,       # NEW
        pre_prune_scope=args.pre_prune_scope,     # NEW
        seed=args.seed,
        use_edited_decoder=args.use_edited_decoder,
        decoder_type=args.decoder_type,
        decoder_normalize_input=args.decoder_normalize_input,
        score_source=args.score_source,
        mlp_pair_max_rows=args.mlp_pair_max_rows,
        decoder_objective=args.decoder_objective,
        decoder_recon_weight=args.decoder_recon_weight,
        decoder_keep_weight=args.decoder_keep_weight,
        decoder_add_rank_weight=args.decoder_add_rank_weight,
        decoder_remove_rank_weight=args.decoder_remove_rank_weight,
        decoder_rank_margin=args.decoder_rank_margin,
        decoder_rank_strategy=args.decoder_rank_strategy,
        decoder_rank_neg_k=args.decoder_rank_neg_k,
        decoder_rank_pool_factor=args.decoder_rank_pool_factor,
        heart_rank_weight=args.heart_rank_weight,
        heart_rank_margin=args.heart_rank_margin,
        heart_rank_neg_k=args.heart_rank_neg_k,
        heart_rank_pool_factor=args.heart_rank_pool_factor,
        prediction_decoder_type=args.prediction_decoder_type,
        prediction_rank_weight=args.prediction_rank_weight,
        prediction_bce_weight=args.prediction_bce_weight,
        prediction_rank_margin=args.prediction_rank_margin,
        prediction_rank_neg_k=args.prediction_rank_neg_k,
        prediction_rank_pool_factor=args.prediction_rank_pool_factor,
        prediction_joint_start_epoch=args.prediction_joint_start_epoch,
        prediction_encoder_weight=args.prediction_encoder_weight,
        compactness_weight=args.compactness_weight,
        compactness_objective=args.compactness_objective,
        compactness_radius_metric=args.compactness_radius_metric,
        preserve_weight=args.preserve_weight,
        separate_edit_training=args.separate_edit_training,
        edit_phase_retain_recon_weight=args.edit_phase_retain_recon_weight,
        edit_phase_retain_cl_weight=args.edit_phase_retain_cl_weight,
        phase2_task_main_loss=args.phase2_task_main_loss,
        edit_phase_edit_weight=args.edit_phase_edit_weight,
        editor_pull_strength=args.editor_pull_strength,
        editor_push_scope=args.editor_push_scope,
        editor_noncompact_push_strength=args.editor_noncompact_push_strength,
        editor_noise_push_strength=args.editor_noise_push_strength,
        editor_push_preserve_norm=args.editor_push_preserve_norm,
        editor_edit_scale=args.editor_edit_scale,
        edit_start_epoch=args.edit_start_epoch,
        edit_train_start_epoch=args.edit_train_start_epoch,
        decoded_rewrite_start_epoch=args.decoded_rewrite_start_epoch,
        decoded_rewrite_every=args.decoded_rewrite_every,
        eval_log_every=args.eval_log_every,
        train_eval_every=args.train_eval_every,
        skip_train_acc=args.skip_train_acc,
        decoder_diag_every=args.decoder_diag_every,
        edit_metric_every=args.edit_metric_every,
        edge_eval=args.edge_eval,
        freeze_c0p_at_edit_start=args.freeze_c0p_at_edit_start,
        use_decoded_graph_augment=args.use_decoded_graph_augment,
        decoded_add_ratio=args.decoded_add_ratio,
        decoded_remove_ratio=args.decoded_remove_ratio,
        decoded_add_threshold=args.decoded_add_threshold,
        decoded_remove_threshold=args.decoded_remove_threshold,
        decoded_add_quantile=args.decoded_add_quantile,
        decoded_remove_quantile=args.decoded_remove_quantile,
        decoded_max_add_per_round=args.decoded_max_add_per_round,
        decoded_max_remove_per_round=args.decoded_max_remove_per_round,
        decoded_same_cluster_only=args.decoded_same_cluster_only,
        decoded_require_c0p_endpoint=args.decoded_require_c0p_endpoint,
        decoded_require_both_c0p=args.decoded_require_both_c0p,
        decoded_require_c0p_noncompact_endpoint=args.decoded_require_c0p_noncompact_endpoint,
        decoded_graph_aug_bound=args.decoded_graph_aug_bound,
        decoded_add_degree_target=args.decoded_add_degree_target,
        decoded_add_degree_target_scope=args.decoded_add_degree_target_scope,
        decoded_add_degree_target_nodes=args.decoded_add_degree_target_nodes,
        decoded_guarantee_degree_target=args.decoded_guarantee_degree_target,
        decoded_degree_floor=args.decoded_degree_floor,
        decoded_accumulate_into_base=args.decoded_accumulate_into_base,
        decoded_edit_end_epoch=args.decoded_edit_end_epoch,
        phase2_freeze_encoder=args.phase2_freeze_encoder,
        edit_phase_encoder_lr_scale=args.edit_phase_encoder_lr_scale,
        decoder_warmup_in_phase1=args.decoder_warmup_in_phase1,
        decoder_warmup_recon_weight=args.decoder_warmup_recon_weight,
        decoder_warmup_use_pulled_latent=args.decoder_warmup_use_pulled_latent,
        phase2_decoder_inference_only=args.phase2_decoder_inference_only,
        skip_oom_epoch=args.skip_oom_epoch,
        pull_mask_scope=args.pull_mask_scope,
        compactness_mask_scope=args.compactness_mask_scope,
        rewrite_endpoint_scope=args.rewrite_endpoint_scope,
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

        def _fmt_num(x):
            if x is None:
                return "na"
            if isinstance(x, bool):
                return "1" if x else "0"
            if isinstance(x, int):
                return str(x)
            try:
                xf = float(x)
                if xf.is_integer():
                    return str(int(xf))
                return f"{xf:g}".replace("-", "m")
            except Exception:
                return str(x)

        def _clean_token(s: str) -> str:
            return (
                str(s)
                .replace("/", "-")
                .replace(" ", "")
                .replace(".", "p")
                .replace("__", "_")
            )

        phase_tag = "2stage" if args.separate_edit_training else "joint"
        edit_tag = f"edit-{args.decoder_type}" if args.use_edited_decoder else "base"
        target_tag = "freezeC0p" if args.freeze_c0p_at_edit_start else "dynC0p"
        cluster_tag = f"{args.cluster_method}-{args.cluster_mode}" if getattr(args, "cluster_method", None) else "cluster-na"
        loss_tag = f"task-{args.loss_ver}" if getattr(args, "loss_ver", "") else "task-na"
        weight_tag = f"dr{_fmt_num(args.decoder_recon_weight)}_cp{_fmt_num(args.compactness_weight)}_pv{_fmt_num(args.preserve_weight)}"
        stage_tag = f"es{_fmt_num(args.edit_start_epoch)}"
        objective_tag = f"dobj-{args.decoder_objective}_cobj-{args.compactness_objective}_rmet-{args.compactness_radius_metric}"

        extra_tags = []
        if args.score_source != "dot":
            extra_tags.append(f"score-{args.score_source}")
        if args.prediction_decoder_type != "none":
            extra_tags.append(f"pred-{args.prediction_decoder_type}")
        if args.separate_edit_training:
            extra_tags.append(f"rr{_fmt_num(args.edit_phase_retain_recon_weight)}")
            extra_tags.append(f"rc{_fmt_num(args.edit_phase_retain_cl_weight)}")

        if args.use_decoded_graph_augment:
            rewrite_mode = "accum" if args.decoded_accumulate_into_base else "temp"
            cluster_scope = "samecl" if args.decoded_same_cluster_only else "crosscl"
            endpoint_scope = (
                "c0p-noncompact" if args.decoded_require_c0p_noncompact_endpoint else
                "bothc0p" if args.decoded_require_both_c0p else
                ("onec0p" if args.decoded_require_c0p_endpoint else "noc0p")
            )
            add_tag = (
                f"aq{_fmt_num(args.decoded_add_quantile)}" if args.decoded_add_quantile is not None else
                f"at{_fmt_num(args.decoded_add_threshold)}" if args.decoded_add_threshold is not None else
                f"ar{_fmt_num(args.decoded_add_ratio)}"
            )
            remove_tag = (
                f"rq{_fmt_num(args.decoded_remove_quantile)}" if args.decoded_remove_quantile is not None else
                f"rt{_fmt_num(args.decoded_remove_threshold)}" if args.decoded_remove_threshold is not None else
                f"rrm{_fmt_num(args.decoded_remove_ratio)}"
            )
            extra_tags.extend([
                "rewrite",
                rewrite_mode,
                cluster_scope,
                endpoint_scope,
                add_tag,
                remove_tag,
            ])
            if int(args.decoded_add_degree_target) > 0:
                extra_tags.append(f"dtarget{args.decoded_add_degree_target}")
                if args.decoded_add_degree_target_scope != "total":
                    extra_tags.append(f"dtargetscope-{args.decoded_add_degree_target_scope}")
                if args.decoded_add_degree_target_nodes != "rewrite":
                    extra_tags.append(f"dtargetnodes-{args.decoded_add_degree_target_nodes}")
                if args.decoded_guarantee_degree_target:
                    extra_tags.append("dtarget-guarantee")
        else:
            extra_tags.append("norewrite")

        extra_tags.extend([
            f"ver-{args.ver}",
            f"seed{args.seed}",
            f"idx{args.idx}",
        ])
        if args.run_tag:
            extra_tags.append(f"tag-{args.run_tag}")
        if args.ae_backbone != "vgnae":
            extra_tags.append(f"ae-{args.ae_backbone}")
        if args.editor_push_scope != "none":
            extra_tags.append(f"push-{args.editor_push_scope}")
            extra_tags.append(f"ncpush{_fmt_num(args.editor_noncompact_push_strength)}")
            extra_tags.append(f"noisepush{_fmt_num(args.editor_noise_push_strength)}")

        fname_parts = [
            args.dataset,
            phase_tag,
            edit_tag,
            stage_tag,
            target_tag,
            cluster_tag,
            loss_tag,
            objective_tag,
            weight_tag,
            *extra_tags,
        ]
        fname = "_".join(_clean_token(p) for p in fname_parts if p) + ".log"
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
        print(f"ae_backbone={args.ae_backbone} maskgae_mask_rate={args.maskgae_mask_rate} maskgae_feature_weight={args.maskgae_feature_weight}")
        print(f"cimage_factor_weight={args.cimage_factor_weight} cimage_cluster_weight={args.cimage_cluster_weight} cimage_num_factors={args.cimage_num_factors} cimage_num_clusters={args.cimage_num_clusters} cimage_cluster_alpha={args.cimage_cluster_alpha}")
        print(f"cimage_pseudo_label_threshold={args.cimage_pseudo_label_threshold} cimage_factor_select_ratio={args.cimage_factor_select_ratio} cimage_mrmr_redundancy_weight={args.cimage_mrmr_redundancy_weight} cimage_cluster_balance_weight={args.cimage_cluster_balance_weight} cimage_sce_power={args.cimage_sce_power}")
        print(f"aug_ratio={args.aug_ratio} aug_bound={args.aug_bound} degree_thr={args.degree_threshold}")
        print(f"topk_per_node={args.topk_per_node} aug_ratio_epoch={args.aug_ratio_epoch}")
        if args.restricted:
            print(f"restricted=1 alpha={args.restrict_alpha} gamma={args.restrict_gamma}")
        else:
            print("restricted=0")
        print(f"edited_decoder={int(args.use_edited_decoder)} decoded_graph_augment={int(args.use_decoded_graph_augment)} freeze_c0p={int(args.freeze_c0p_at_edit_start)} accumulate_base={int(args.decoded_accumulate_into_base)} separate_edit_training={int(args.separate_edit_training)}")
        print(f"edit_phase_retain_recon_weight={args.edit_phase_retain_recon_weight} edit_phase_retain_cl_weight={args.edit_phase_retain_cl_weight}")
        print(f"phase2_task_main_loss={int(args.phase2_task_main_loss)} edit_phase_edit_weight={args.edit_phase_edit_weight}")
        print(f"phase2_freeze_encoder={int(args.phase2_freeze_encoder)} edit_phase_encoder_lr_scale={args.edit_phase_encoder_lr_scale}")
        print(f"compactness_objective={args.compactness_objective} compactness_radius_metric={args.compactness_radius_metric} compactness_weight={args.compactness_weight}")
        print(f"editor_push_scope={args.editor_push_scope} noncompact_push={args.editor_noncompact_push_strength} noise_push={args.editor_noise_push_strength} push_preserve_norm={int(args.editor_push_preserve_norm)}")
        print(f"decoded_edit_end_epoch={args.decoded_edit_end_epoch} decoder_warmup_in_phase1={int(args.decoder_warmup_in_phase1)} decoder_warmup_recon_weight={args.decoder_warmup_recon_weight} decoder_warmup_use_pulled_latent={int(args.decoder_warmup_use_pulled_latent)} phase2_decoder_inference_only={int(args.phase2_decoder_inference_only)}")
        print(f"decoded_add_ratio={args.decoded_add_ratio} decoded_remove_ratio={args.decoded_remove_ratio} add_thr={args.decoded_add_threshold} remove_thr={args.decoded_remove_threshold} add_q={args.decoded_add_quantile} remove_q={args.decoded_remove_quantile} max_add={args.decoded_max_add_per_round} max_remove={args.decoded_max_remove_per_round}")
        print(f"decoded_add_ratio={args.decoded_add_ratio} decoded_remove_ratio={args.decoded_remove_ratio} same_cluster_only={int(args.decoded_same_cluster_only)} c0p_endpoint={int(args.decoded_require_c0p_endpoint)} both_c0p={int(args.decoded_require_both_c0p)} c0p_noncompact_endpoint={int(args.decoded_require_c0p_noncompact_endpoint)} per_node_cap={args.decoded_graph_aug_bound} add_degree_target={args.decoded_add_degree_target} add_degree_target_scope={args.decoded_add_degree_target_scope} add_degree_target_nodes={args.decoded_add_degree_target_nodes} guarantee_degree_target={int(args.decoded_guarantee_degree_target)}")
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
