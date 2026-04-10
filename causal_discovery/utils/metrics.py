"""
Evaluation metrics for comparing predicted causal graphs against ground truth.
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple


def structural_hamming_distance(ground_truth, predicted, nodes):
    """
    Calculates the Structural Hamming Distance (SHD) between two DAGs
    represented as lists of directed edges.

    SHD counts three types of errors:
      - Reversed edges (edge exists in both but in opposite directions)
      - Missing edges (in ground truth but not in predicted)
      - Extra edges (in predicted but not in ground truth)

    Args:
        ground_truth: list of (source, target) tuples for the true DAG.
        predicted: list of (source, target) tuples for the predicted DAG.
        nodes: list of node names in the graph.

    Returns:
        int: the Structural Hamming Distance.
    """
    edges1 = list(ground_truth)
    edges2 = list(predicted)

    SHD = 0

    reverse_edge1_list = []
    reverse_edge2_list = []
    for edge in edges1:
        node1 = edge[0]
        node2 = edge[1]
        if (node2, node1) in edges2:
            SHD += 1
            reverse_edge1_list.append(edge)
            reverse_edge2_list.append((node2, node1))

    for item in reverse_edge1_list:
        if item in edges1:
            edges1.remove(item)

    for item in reverse_edge2_list:
        if item in edges2:
            edges2.remove(item)

    for edge in edges1:
        if edge not in edges2:
            SHD += 1

    for edge in edges2:
        if edge not in edges1:
            SHD += 1

    return SHD


def topological_ordering(graph_edges):
    """
    Compute a topological ordering of nodes in a DAG.

    Args:
        graph_edges: list of (source, target) directed edge tuples.

    Returns:
        list: topologically sorted node list.
    """
    G = nx.DiGraph()
    G.add_edges_from(graph_edges)
    return list(nx.topological_sort(G))


def topological_divergence(ground_truth_edges, predicted_edges):
    """
    Compute topological divergence: counts how many ground-truth edges
    go against the topological order of the predicted graph.

    Args:
        ground_truth_edges: edge list of the ground truth DAG.
        predicted_edges: edge list of the predicted DAG.

    Returns:
        int: number of order-violating edges.
    """
    order = topological_ordering(predicted_edges)
    divergence = 0
    for i in range(1, len(order)):
        for j in range(0, i):
            if (order[i], order[j]) in ground_truth_edges:
                divergence += 1
    return divergence


def count_cycles(graph_edges):
    """
    Count the number of simple cycles in a directed graph.

    Args:
        graph_edges: list of directed edge tuples.

    Returns:
        int: number of distinct simple cycles.
    """
    G = nx.DiGraph(graph_edges)
    cycles = sorted(nx.simple_cycles(G))
    return len(cycles)


def count_isolated_nodes(graph_edges, nodes):
    """
    Count nodes that do not appear in any edge.

    Args:
        graph_edges: list of directed edge tuples.
        nodes: full list of node names.

    Returns:
        tuple: (list of connected node names, number of isolated nodes)
    """
    connected = list(set(item for edge in graph_edges for item in edge))
    return connected, len(nodes) - len(connected)


def plot_graph(graph_edges, title=None):
    """
    Visualise a directed graph using networkx + matplotlib.

    Args:
        graph_edges: list of directed edge tuples.
        title: optional plot title.
    """
    G = nx.DiGraph()
    G.add_edges_from(graph_edges)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1500, font_size=8,
            arrows=True, arrowsize=15)
    if title:
        plt.title(title)
    plt.show()
