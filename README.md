# Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference

**Paper:** [OpenReview](https://openreview.net/pdf?id=9juyeCqL0u) | [arXiv](https://arxiv.org/pdf/2310.15117)

Large Language Models (LLMs) have been used as experts to infer causal graphs, often by repeatedly applying a pairwise prompt that asks about the causal relationship of each variable pair. However, such experts, including human domain experts, cannot distinguish between direct and indirect effects given a pairwise prompt. Therefore, instead of the graph, we propose that causal order be used as a more stable output interface for utilizing expert knowledge. Even when querying a perfect expert with a pairwise prompt, we show that the inferred graph can have significant errors whereas the causal order is always correct. In practice, however, LLMs are imperfect experts and we find that pairwise prompts lead to multiple cycles and do not yield a valid order. Hence, we propose a prompting strategy that introduces an auxiliary variable for every variable pair and instructs the LLM to avoid cycles within this triplet. Across multiple real-world graphs, such a triplet-based method yields a more accurate order than the pairwise prompt, using both LLMs and human annotators as experts. Since the triplet method queries an expert repeatedly with different auxiliary variables for each variable pair, it also increases robustness---the triplet method with significantly smaller models such as Phi-3 and Llama-3 8B outperforms a pairwise prompt with GPT-4. For practical usage, we show how the expert-provided causal order from the triplet method can be used to reduce error in downstream graph discovery and effect inference tasks.

![image](https://github.com/user-attachments/assets/5fd9a2b6-c743-4145-ab12-b6c745b78452)

---

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

## Strategies

This repository implements three LLM-based strategies for causal graph discovery from node names alone:

1. **Pairwise** — Query an LLM for each pair of nodes to determine edge direction. Multiple prompting variants are supported (simple, chain-of-thought, context-augmented, etc.).
2. **Triplet** — Query an LLM on all triplets (3-node subsets), then merge the resulting subgraphs via majority voting. An entropy-based cycle removal step produces a final DAG.
3. **Quadruplet** — Same voting strategy as Triplet but uses 4-node subsets (C(n,4) combinations), providing more context per subgroup query at the cost of more API calls.

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
└── run_quadruplet.py         # CLI entry point for quadruplet experiments
```

## Evaluation Metrics

- **SHD (Structural Hamming Distance)**: Counts reversed, missing, and extra edges between predicted and ground-truth DAGs.
- **Topological Divergence**: Number of ground-truth edges that violate the topological ordering of the predicted graph.

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{
    vashishtha2025causal,
    title={Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference},
    author={Aniket Vashishtha},
    booktitle={ICLR},
    year={2025}
}
```
