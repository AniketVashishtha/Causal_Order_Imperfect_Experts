"""
Run the triplet-based causal discovery experiment on a benchmark graph.

Usage:
    python -m causal_discovery.run_triplet --graph child --subgraph-model gpt-4o-mini --expert-model gpt-4o

Use --help for all options.
"""

import argparse
import pickle
import networkx as nx

from .graphs import get_graph
from .strategies.triplet import run_triplet_experiment
from .utils.metrics import (
    structural_hamming_distance,
    topological_divergence_2,
    count_cycles,
    count_isolated_nodes,
)
from .utils.cycle_remover import remove_cycles


def main():
    parser = argparse.ArgumentParser(
        description="Triplet-based LLM causal discovery")
    parser.add_argument("--graph", type=str, required=True,
                        help="Graph name (e.g. cancer, asia, child, covid, alzheimers, ...)")
    parser.add_argument("--subgraph-model", type=str, default="gpt-4o-mini",
                        help="Weaker model for triplet subgraph orientation (default: gpt-4o-mini)")
    parser.add_argument("--expert-model", type=str, default="gpt-4o",
                        help="Stronger model for tie-breaking clashes (default: gpt-4o)")
    parser.add_argument("--delay", type=int, default=12,
                        help="Seconds between API calls (default: 12)")
    parser.add_argument("--save-edgewise", type=str, default=None,
                        help="Path to save edgewise distribution as pickle")
    parser.add_argument("--no-cycle-removal", action="store_true",
                        help="Skip entropy-based cycle removal")
    args = parser.parse_args()

    graph_config = get_graph(args.graph)
    nodes = graph_config["nodes"]
    gt_edges = graph_config["ground_truth_edges"]

    print(f"\n{'='*60}")
    print(f"Graph: {args.graph} ({len(nodes)} nodes, {len(gt_edges)} GT edges)")
    print(f"Subgraph model (weak learner): {args.subgraph_model}")
    print(f"Expert model (tie-breaker):    {args.expert_model}")
    print(f"{'='*60}\n")

    result = run_triplet_experiment(
        graph_config=graph_config,
        subgraph_model=args.subgraph_model,
        expert_model=args.expert_model,
        delay=args.delay,
    )

    predicted = result["predicted_edges"]
    edgewise_dist = result["edgewise_dist"]

    print(f"\n--- Raw Triplet Merge Results ---")
    print(f"Predicted edges: {len(predicted)}")

    shd = structural_hamming_distance(gt_edges, predicted, nodes)
    print(f"SHD: {shd}")

    connected, n_isolated = count_isolated_nodes(predicted, nodes)
    print(f"Connected nodes: {len(connected)}, Isolated: {n_isolated}")

    G = nx.DiGraph(predicted)
    n_cycles = len(sorted(nx.simple_cycles(G)))
    print(f"Cycles: {n_cycles}")

    if edgewise_dist:
        print(f"\nEdgewise distribution entries: {len(edgewise_dist)}")

    if args.save_edgewise and edgewise_dist:
        with open(args.save_edgewise, 'wb') as f:
            pickle.dump(edgewise_dist, f)
        print(f"Saved edgewise distribution to {args.save_edgewise}")

    if not args.no_cycle_removal and n_cycles > 0 and edgewise_dist:
        print(f"\n--- Cycle Removal ---")
        acyclic_edges, threshold, _ = remove_cycles(predicted, edgewise_dist)

        shd_acyclic = structural_hamming_distance(gt_edges, acyclic_edges, nodes)
        print(f"SHD after cycle removal: {shd_acyclic}")

        try:
            td = topological_divergence_2(gt_edges, acyclic_edges)
            print(f"Topological Divergence: {td}")
        except Exception:
            print("Topological Divergence: could not compute (graph may still have issues)")

        connected_a, n_isolated_a = count_isolated_nodes(acyclic_edges, nodes)
        print(f"Connected nodes: {len(connected_a)}, Isolated: {n_isolated_a}")
    elif n_cycles == 0:
        try:
            td = topological_divergence_2(gt_edges, predicted)
            print(f"Topological Divergence: {td}")
        except Exception:
            pass

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
