"""
Subgroup-based causal discovery strategy (triplet / quadruplet / …).

Pipeline:
  1. Generate all C(n, k) subgroups from the node set (k = 3 for triplets,
     k = 4 for quadruplets, etc.).
  2. For each subgroup, query the LLM to produce a small k-node DAG.
  3. Merge subgroup-level DAGs via majority voting over edge directions.
  4. Break ties with a CoT pairwise tie-breaker query.

The merge step also produces an edgewise probability distribution that
can be passed to the cycle remover.
"""

import itertools
import re
import time
from tqdm import tqdm

from ..prompts.triplet import (
    generate_subgraph_prompt,
    generate_subgraph_with_descr_prompt,
    cot_tiebreaker_prompt,
)
from ..utils.helpers import parse_answer_tag, str_2_lst
from ..utils.llm_client import query_llm


def generate_all_subgroups(nodes, subgroup_size=3):
    """Generate all combinations of `subgroup_size` nodes."""
    return list(itertools.combinations(nodes, subgroup_size))


def generate_all_triplets(nodes):
    """Generate all combinations of 3 nodes (convenience alias)."""
    return generate_all_subgroups(nodes, subgroup_size=3)


def query_triplet_subgraph(triplet, context, descriptions,
                           model="gpt-4o-mini", max_tokens=300, delay=12):
    """
    Query the LLM to orient edges within a single triplet of nodes.

    Uses a weaker/cheaper model by default since the merge step aggregates
    many such responses via majority voting.

    Args:
        triplet: tuple of 3 node names.
        context: modelling context string.
        descriptions: dict mapping node name -> description, or None.
        model: LLM model name for subgraph orientation (weaker model).
        max_tokens: max response tokens.
        delay: seconds to wait before the API call (rate limiting).

    Returns:
        str: the raw DAG string from the LLM (content of <Answer> tag).
    """
    if descriptions:
        descr_sub = {k: descriptions[k] for k in triplet if k in descriptions}
        messages = generate_subgraph_with_descr_prompt(
            nodes=list(triplet), context=context, descr_nodes=descr_sub)
    else:
        messages = generate_subgraph_prompt(
            nodes=list(triplet), context=context)

    time.sleep(delay)
    answer = query_llm(messages, model=model, max_completion_tokens=max_tokens)
    dag_str = parse_answer_tag(answer)
    return dag_str


def merge_triplet_votes(subgroup_list, subgraph_list, nodes, context,
                        expert_model="gpt-4o", delay=12):
    """
    Merge per-triplet DAGs into a single graph via majority voting.

    For each pair of nodes (X, Y), counts how many triplet subgraphs
    contain X->Y, Y->X, or neither. The majority direction wins.
    Ties are broken by a CoT pairwise LLM query using a stronger
    "expert" model.

    Also computes the edgewise probability distribution used for
    entropy-based cycle removal.

    Args:
        subgroup_list: list of triplet tuples (same order as subgraph_list).
        subgraph_list: list of parsed edge lists for each triplet.
        nodes: full list of node names.
        context: modelling context string.
        expert_model: LLM model name for tie-breaking (stronger model).
        delay: seconds to wait before tie-breaker API calls.

    Returns:
        tuple: (final_graph, edgewise_dist)
          - final_graph: list of (source, target) directed edge tuples.
          - edgewise_dist: dict mapping (nodeA, nodeB) -> [p(A->B), p(B->A), p(none)].
    """
    final_graph = []
    edgewise_dist = {}

    for i in tqdm(range(len(nodes)), desc="Merging (outer)"):
        x = nodes[i]
        for j in tqdm(range(i + 1, len(nodes)), desc="Merging (inner)", leave=False):
            y = nodes[j]
            x_y_list = []
            y_x_list = []
            x_y_ind = []

            subgraph_index = []
            xy_group = [t for t in subgroup_list if x in t and y in t]
            for ele in xy_group:
                subgraph_index.append(subgroup_list.index(ele))

            for index in subgraph_index:
                x_y_list.append([item for item in subgraph_list[index] if item == (x, y)])
                x_y_list = [item for item in x_y_list if item != []]
                y_x_list.append([item for item in subgraph_list[index] if item == (y, x)])
                y_x_list = [item for item in y_x_list if item != []]

                sublist = [item for item in subgraph_list[index]
                           if (x, y) not in subgraph_list[index]
                           and (y, x) not in subgraph_list[index]]
                if sublist:
                    x_y_ind.append(sublist)

            lists = {
                "A_2_B": x_y_list,
                "B_2_A": y_x_list,
                "No_Conn": x_y_ind,
            }

            total_den = len(x_y_list) + len(y_x_list) + len(x_y_ind)
            if total_den > 0:
                edgewise_dist[(nodes[i], nodes[j])] = [
                    len(x_y_list) / total_den,
                    len(y_x_list) / total_den,
                    len(x_y_ind) / total_den,
                ]

            max_len = max(len(v) for v in lists.values())
            keys = [k for k, v in lists.items() if len(v) == max_len]

            if len(keys) == 1:
                max_key = keys[0]
            else:
                max_key = _tiebreaker_query(x, y, nodes, context,
                                            model=expert_model, delay=delay)

            if max_key == "A_2_B":
                final_graph.append((x, y))
            elif max_key == "B_2_A":
                final_graph.append((y, x))

    return final_graph, edgewise_dist


def _tiebreaker_query(X, Y, nodes, context, model="gpt-4o", delay=12):
    """
    Use a CoT pairwise prompt to break a tie between edge directions.

    Returns:
        str: one of "A_2_B", "B_2_A", or "No_Conn".
    """
    messages = cot_tiebreaker_prompt(X, Y, nodes, context)

    time.sleep(delay)
    answer = query_llm(messages, model=model, max_completion_tokens=400)
    ans = parse_answer_tag(answer)

    if ans == 'A':
        return "A_2_B"
    elif ans == 'B':
        return "B_2_A"
    else:
        return "No_Conn"


def run_triplet_experiment(graph_config, subgraph_model="gpt-4o-mini",
                           expert_model="gpt-4o", delay=12,
                           subgroup_size=3):
    """
    Run the full subgroup-based causal discovery pipeline on a graph.

    The pipeline uses two models:
      - subgraph_model (weaker/cheaper): orients edges within each subgroup.
        Since many subgroup responses are aggregated via majority voting, a
        weaker model suffices here.
      - expert_model (stronger): resolves ties when the majority vote is
        inconclusive for a particular edge pair.

    Args:
        graph_config: dict with keys 'nodes', 'ground_truth_edges',
                      'descriptions' (or None), 'context'.
        subgraph_model: LLM model name for subgroup orientation.
        expert_model: LLM model name for tie-breaking.
        delay: seconds between API calls for rate limiting.
        subgroup_size: number of nodes per subgroup (3 = triplet, 4 = quadruplet).

    Returns:
        dict with keys:
          - 'predicted_edges': list of directed edge tuples
          - 'edgewise_dist': probability distribution per edge pair
          - 'subgroup_list': the subgroups used
          - 'subgraph_results': raw DAG strings per subgroup
    """
    nodes = graph_config["nodes"]
    context = graph_config["context"]
    descriptions = graph_config.get("descriptions")

    subgroup_list = generate_all_subgroups(nodes, subgroup_size=subgroup_size)
    label = {3: "triplets", 4: "quadruplets"}.get(subgroup_size,
                                                    f"{subgroup_size}-subgroups")
    print(f"Total {label}: {len(subgroup_list)}")

    subgraph_results = []
    subgr_dict = {}

    for group in tqdm(subgroup_list, desc=f"Querying {label}"):
        dag_str = query_triplet_subgraph(
            triplet=group,
            context=context,
            descriptions=descriptions,
            model=subgraph_model,
            delay=delay,
        )
        if dag_str is not None:
            subgraph_results.append(dag_str)
            subgr_dict[group] = dag_str
        else:
            print(f"Warning: no answer for triplet {group}")
            subgraph_results.append("[]")
            subgr_dict[group] = "[]"

    subgroup_list_final = list(subgr_dict.keys())
    parsed_subgraphs = [str_2_lst(x) for x in subgraph_results]

    final_graph, edgewise_dist = merge_triplet_votes(
        subgroup_list=subgroup_list_final,
        subgraph_list=parsed_subgraphs,
        nodes=nodes,
        context=context,
        expert_model=expert_model,
        delay=delay,
    )

    return {
        "predicted_edges": final_graph,
        "edgewise_dist": edgewise_dist,
        "subgroup_list": subgroup_list_final,
        "subgraph_results": subgraph_results,
    }
