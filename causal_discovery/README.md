# LLM-Based Causal Discovery

Code for the paper: *[Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference](https://arxiv.org/pdf/2310.15117)*

This repository implements LLM-based strategies for causal graph discovery from node names alone:

1. **Pairwise** — Query an LLM for each pair of nodes to determine edge direction. Multiple prompting variants are supported (simple, chain-of-thought, context-augmented, etc.).
2. **Triplet** — Query an LLM on all triplets (3-node subsets), then merge the resulting subgraphs via majority voting. An entropy-based cycle removal step produces a final DAG.


## Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=" "
```

## Usage

### Triplet strategy

The triplet approach uses two models: a **weaker model** orients each 3-node
subgraph (since many responses are aggregated via majority voting), and a
**stronger expert model** resolves ties when votes are split.

```bash
python -m causal_discovery.run_triplet \
    --graph child \
    --subgraph-model gpt-4o-mini \
    --expert-model gpt-4o \
    --delay 12 \
    --save-edgewise child_edgewise.pkl
```

### Quadruplet strategy

Same approach as Triplet but with 4-node subgroups. More context per query
but significantly more subgroups (C(n,4) vs C(n,3)).

```bash
python -m causal_discovery.run_quadruplet \
    --graph child \
    --subgraph-model gpt-4o-mini \
    --expert-model gpt-4o \
    --delay 12 \
    --save-edgewise child_quad_edgewise.pkl
```

### Pairwise strategy

```bash
python -m causal_discovery.run_pairwise \
    --graph cancer \
    --prompt-type cot \
    --model gpt-4o \
    --delay 12
```

### Available prompt types (pairwise)

| Key              | Description                                        |
|------------------|----------------------------------------------------|
| `simple`         | Basic pairwise A/B/C prompt                        |
| `cot`            | Chain-of-thought with few-shot Cancer & CHD examples|
| `context`        | Augmented with already-oriented edges              |
| `all_directed`   | Augmented with full set of directed edges so far   |
| `markov_blanket` | Augmented with Markov blanket of each node         |

### Available benchmark graphs

| Name           | Nodes | Edges |
|----------------|-------|-------|
| `cancer`       | 5     | 4     |
| `asia`         | 8     | 8     |
| `earthquake`   | 5     | 4     |
| `survey`       | 6     | 6     |
| `maths`        | 5     | 5     |
| `child`        | 20    | 25    |
| `covid`        | 11    | 20    |
| `alzheimers`   | 11    | 21    |
| `insurance`    | 27    | 52    |
| `sangiovese`   | 15    | 55    |
| `neuropathic`  | 22    | 25    |

## Project Structure

```
causal_discovery/
├── graphs/
│   └── definitions.py        # Benchmark graph nodes, edges, descriptions
├── prompts/
│   ├── pairwise.py           # Pairwise prompt builders
│   └── triplet.py            # Triplet / subgraph prompt builders
├── strategies/
│   ├── pairwise.py           # Pairwise strategy orchestration
│   └── triplet.py            # Triplet strategy orchestration
├── utils/
│   ├── metrics.py            # SHD, topological divergence, plotting
│   ├── cycle_remover.py      # Entropy-based cycle removal
│   ├── llm_client.py         # OpenAI API wrapper
│   └── helpers.py            # Response parsing utilities
├── run_pairwise.py           # CLI entry point for pairwise experiments
├── run_triplet.py            # CLI entry point for triplet experiments
├── run_quadruplet.py         # CLI entry point for quadruplet experiments
├── requirements.txt
└── README.md
```

## Evaluation Metrics

- **SHD (Structural Hamming Distance)**: Counts reversed, missing, and extra edges between predicted and ground-truth DAGs.
- **Topological Divergence**: Number of ground-truth edges that violate the topological ordering of the predicted graph.

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{
    anonymous2024can,
    title={Can Large Language Models Build Causal Graphs?},
    author={Anonymous},
    booktitle={Conference},
    year={2024}
}
```
