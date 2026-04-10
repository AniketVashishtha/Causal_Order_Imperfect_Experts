"""
Entropy-based cycle removal for triplet voting graphs.

After the triplet merge step produces a (potentially cyclic) directed graph
along with per-edge probability distributions, this module:

  1. Computes the entropy of each edge's voting distribution.
  2. Assigns edge weights as 1/(entropy + epsilon), so high-confidence
     edges get higher weight.
  3. Uses binary search on a weight threshold to find the largest
     acyclic subgraph.
"""

import math
import networkx as nx


EPSILON = 0.0001


def calculate_entropy(probabilities):
    """Compute Shannon entropy (base 2) of a probability distribution."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def compute_edge_weights(predicted_edges, edgewise_scores):
    """
    Assign an inverse-entropy weight to each predicted edge.

    Args:
        predicted_edges: list of (source, target) edge tuples from the
            triplet merge step.
        edgewise_scores: dict mapping (nodeA, nodeB) -> [p(A->B), p(B->A), p(none)],
            as returned by merge_triplet_votes.

    Returns:
        dict: mapping each edge tuple -> weight (float).
    """
    valid_tuples = set(predicted_edges + [tuple(reversed(e)) for e in predicted_edges])
    filtered_scores = {k: v for k, v in edgewise_scores.items() if k in valid_tuples}

    probabilities = list(filtered_scores.values())
    entropies = [calculate_entropy(prob) for prob in probabilities]

    weights = [(1.0 / (ent + EPSILON)) for ent in entropies]

    edge_weights = dict(zip(predicted_edges, weights))
    return edge_weights


def _find_max_min_weights(edge_weights):
    """Return (max_weight, min_weight) from an edge weight dict."""
    values = list(edge_weights.values())
    return max(values), min(values)


def _remove_below_threshold(edge_weights, threshold):
    """Keep only edges with weight >= threshold."""
    return {k: v for k, v in edge_weights.items() if v >= threshold}


def remove_cycles_by_threshold(edge_weights):
    """
    Binary-search on the weight threshold to find the largest acyclic subgraph.

    Iteratively narrows a [min_weight, max_weight] window: if the subgraph
    above the midpoint threshold is acyclic, lower the threshold (include more
    edges); otherwise raise it (remove more edges).

    Args:
        edge_weights: dict mapping edge tuple -> weight.

    Returns:
        tuple: (acyclic_edges, threshold_used)
          - acyclic_edges: list of (source, target) edge tuples forming a DAG.
          - threshold_used: the final weight threshold.
    """
    max_w, min_w = _find_max_min_weights(edge_weights)
    threshold = 0

    while round(max_w, 5) > round(min_w, 5):
        mid = (min_w + max_w) / 2
        threshold = mid
        filtered = _remove_below_threshold(edge_weights, threshold)
        dgraph = list(filtered.keys())
        directed_graph = nx.DiGraph(dgraph)
        is_acyclic = nx.is_directed_acyclic_graph(directed_graph)

        if is_acyclic:
            max_w = mid
        else:
            min_w = mid

    final_filtered = _remove_below_threshold(edge_weights, threshold)
    acyclic_edges = list(final_filtered.keys())
    return acyclic_edges, threshold


def remove_cycles(predicted_edges, edgewise_scores):
    """
    High-level function: given triplet merge output, remove cycles and
    return a DAG.

    Args:
        predicted_edges: list of directed edge tuples from merge step.
        edgewise_scores: dict from merge_triplet_votes.

    Returns:
        tuple: (acyclic_edges, threshold, edge_weights)
    """
    edge_weights = compute_edge_weights(predicted_edges, edgewise_scores)
    acyclic_edges, threshold = remove_cycles_by_threshold(edge_weights)

    is_dag = nx.is_directed_acyclic_graph(nx.DiGraph(acyclic_edges))
    print(f"Cycle removal: threshold={threshold:.4f}, "
          f"edges kept={len(acyclic_edges)}/{len(predicted_edges)}, "
          f"is_acyclic={is_dag}")

    return acyclic_edges, threshold, edge_weights
