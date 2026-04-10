"""
Run the pairwise LLM causal discovery experiment on a benchmark graph.

Usage:
    python -m causal_discovery.run_pairwise --graph cancer --prompt-type cot --model gpt-4o

Use --help for all options.
"""

import argparse
import networkx as nx

from .graphs import get_graph
from .strategies.pairwise import run_pairwise_experiment, PROMPT_TYPES
from .utils.metrics import (
    structural_hamming_distance,
    topological_divergence,
    count_cycles,
    count_isolated_nodes,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise LLM causal discovery")
    parser.add_argument("--graph", type=str, required=True,
                        help="Graph name (e.g. cancer, asia, child, ...)")
    parser.add_argument("--prompt-type", type=str, default="cot",
                        choices=list(PROMPT_TYPES.keys()),
                        help="Prompting strategy (default: cot)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model name (default: gpt-4o)")
    parser.add_argument("--max-tokens", type=int, default=400,
                        help="Max response tokens (default: 400)")
    parser.add_argument("--delay", type=int, default=12,
                        help="Seconds between API calls (default: 12)")
    args = parser.parse_args()

    graph_config = get_graph(args.graph)
    nodes = graph_config["nodes"]
    gt_edges = graph_config["ground_truth_edges"]

    print(f"\n{'='*60}")
    print(f"Graph: {args.graph} ({len(nodes)} nodes, {len(gt_edges)} GT edges)")
    print(f"Model: {args.model}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"{'='*60}\n")

    result = run_pairwise_experiment(
        graph_config=graph_config,
        prompt_type=args.prompt_type,
        model=args.model,
        max_tokens=args.max_tokens,
        delay=args.delay,
    )

    predicted = result["predicted_edges"]

    print(f"\n--- Pairwise Results ---")
    print(f"Predicted edges: {len(predicted)}")

    shd = structural_hamming_distance(gt_edges, predicted, nodes)
    print(f"SHD: {shd}")

    connected, n_isolated = count_isolated_nodes(predicted, nodes)
    print(f"Connected nodes: {len(connected)}, Isolated: {n_isolated}")

    G = nx.DiGraph(predicted)
    n_cycles = len(sorted(nx.simple_cycles(G)))
    print(f"Cycles: {n_cycles}")

    if n_cycles == 0:
        try:
            td = topological_divergence(gt_edges, predicted)
            print(f"Topological Divergence: {td}")
        except Exception:
            pass

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
