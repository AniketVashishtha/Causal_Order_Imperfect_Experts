"""
Pairwise causal discovery strategy.

Pipeline:
  1. Generate all C(n,2) node pairs.
  2. For each pair, query the LLM with the chosen prompt type.
  3. Parse the A/B/C answer to orient the edge (or skip if C).
  4. Collect the resulting directed graph.
"""

import time
from tqdm import tqdm

from ..prompts.pairwise import (
    simple_pairwise_prompt,
    pairwise_with_context_prompt,
    all_directed_edges_prompt,
    markov_blanket_prompt,
    cot_pairwise_prompt,
)
from ..utils.helpers import parse_answer_tag
from ..utils.llm_client import query_llm


PROMPT_TYPES = {
    "simple": "simple_pairwise",
    "cot": "cot_pairwise",
    "context": "pairwise_with_context",
    "all_directed": "all_directed_edges",
    "markov_blanket": "markov_blanket",
}


def _build_messages(prompt_type, X, Y, context=None, nodes=None,
                    current_graph=None, directed_edges=None,
                    X_edges=None, Y_edges=None):
    """Build the messages list for the chosen prompt type."""
    if prompt_type == "simple":
        return simple_pairwise_prompt(X, Y)
    elif prompt_type == "cot":
        return cot_pairwise_prompt(X, Y, context=context, nodes=nodes)
    elif prompt_type == "context":
        return pairwise_with_context_prompt(X, Y, current_graph=current_graph)
    elif prompt_type == "all_directed":
        return all_directed_edges_prompt(X, Y, directed_edges=directed_edges)
    elif prompt_type == "markov_blanket":
        return markov_blanket_prompt(X, Y, X_edges=X_edges, Y_edges=Y_edges)
    else:
        available = ", ".join(PROMPT_TYPES.keys())
        raise ValueError(f"Unknown prompt_type '{prompt_type}'. Available: {available}")


def run_pairwise_experiment(graph_config, prompt_type="cot", model="gpt-4o",
                            max_tokens=400, delay=12):
    """
    Run the pairwise causal discovery pipeline on a graph.

    Args:
        graph_config: dict with keys 'nodes', 'ground_truth_edges',
                      'descriptions' (or None), 'context'.
        prompt_type: one of 'simple', 'cot', 'context', 'all_directed',
                     'markov_blanket'.
        model: LLM model name.
        max_tokens: max response tokens.
        delay: seconds between API calls for rate limiting.

    Returns:
        dict with keys:
          - 'predicted_edges': list of directed edge tuples
    """
    nodes = graph_config["nodes"]
    context = graph_config["context"]

    all_pairs = [(a, b) for idx, a in enumerate(nodes) for b in nodes[idx + 1:]]
    print(f"Total pairs: {len(all_pairs)}")

    result_graph = []
    directed_so_far = []

    backoff_ceil = 10
    backoff_base = 2
    backoff_count = 0

    for pair in tqdm(all_pairs, desc="Querying pairs"):
        X, Y = pair
        rerun = True
        while rerun:
            try:
                time.sleep(delay)
                messages = _build_messages(
                    prompt_type=prompt_type, X=X, Y=Y,
                    context=context, nodes=nodes,
                    current_graph=directed_so_far,
                    directed_edges=directed_so_far,
                )

                answer = query_llm(messages, model=model, max_completion_tokens=max_tokens)
                ans = parse_answer_tag(answer)

                if ans == 'A':
                    result_graph.append(pair)
                    directed_so_far.append(pair)
                elif ans == 'B':
                    result_graph.append(pair[::-1])
                    directed_so_far.append(pair[::-1])
                rerun = False

            except Exception as e:
                print(f"Error: {e}")
                backoff_count = min(backoff_count + 1, backoff_ceil)
                sleep_time = backoff_base ** backoff_count
                print(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)

    return {
        "predicted_edges": result_graph,
    }
