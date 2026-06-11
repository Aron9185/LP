import os
import sys

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from loss import symmetric_node_infonce_loss
from aron_train_edit_decoder import (
    _build_decoded_pair_context,
    _deg_excl_self,
    graph_edge_jaccard,
    sample_constraint_preserving_two_view_graph,
)


def test_symmetric_node_infonce_prefers_same_node_pairs():
    z = torch.eye(4)
    matched = symmetric_node_infonce_loss(z, z.clone(), gamma=1.0, temperature=0.2)
    shuffled = symmetric_node_infonce_loss(z, z[[1, 0, 3, 2]], gamma=1.0, temperature=0.2)

    assert torch.isfinite(matched)
    assert matched < shuffled


def test_constraint_preserving_view_respects_degree_floor():
    g = torch.eye(3)
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        g[i, j] = 1.0
        g[j, i] = 1.0
    ctx = _build_decoded_pair_context(
        g,
        labels=None,
        node_mask=None,
        degree_floor=1,
        same_cluster_only=False,
        require_c0p_endpoint=False,
    )

    view, stats = sample_constraint_preserving_two_view_graph(
        g,
        ctx,
        E0=3,
        add_ratio=0.0,
        remove_ratio=1.0,
        degree_floor=1,
    )

    assert torch.equal(view, view.t())
    assert torch.all(torch.diag(view) == 1)
    assert int(_deg_excl_self(view).min().item()) >= 1
    assert stats["degree_violations"] == 0
    assert graph_edge_jaccard(g, view) < 1.0


def test_constraint_preserving_view_adds_only_valid_pairs():
    g = torch.eye(4)
    labels = torch.tensor([0, 0, 1, 1]).numpy()
    c0p_mask = torch.tensor([True, False, True, False])
    ctx = _build_decoded_pair_context(
        g,
        labels=labels,
        node_mask=c0p_mask,
        degree_floor=0,
        same_cluster_only=True,
        require_c0p_endpoint=True,
    )

    view, stats = sample_constraint_preserving_two_view_graph(
        g,
        ctx,
        E0=2,
        add_ratio=2.0,
        remove_ratio=0.0,
        degree_floor=0,
    )

    added = ((view > 0) & ~(g > 0)).triu(1).nonzero(as_tuple=False).tolist()
    assert sorted([tuple(pair) for pair in added]) == [(0, 1), (2, 3)]
    assert stats["constraint_add_violations"] == 0
