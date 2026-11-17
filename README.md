# Dynamic Routing over Specialized Small Language Models

Research project (TER, Master MLDM): can we **route each user query to a small specialized language model** to cut compute and energy cost, while keeping answer quality close to bigger general-purpose models?

## Idea
A lightweight classifier reads the query, finds its type, and sends it to the best small model for that type. Only one small model runs per query.

| Query type | Specialized model |
|---|---|
| Code | DeepSeek-Coder-6.7B-Instruct |
| Mathematics | Qwen2.5-Math-1.5B-Instruct |
| Dialogue | Phi-3.5-mini-Instruct |

The router is compared with two general-purpose baselines that answer everything: **Llama-3.1-8B-Instruct** and **Mistral-7B-Instruct-v0.3**.

## Method
1. **Router**: a Random Forest classifier on text features (compared with a logistic regression), saved in `models/text_router.joblib`.
2. **Evaluation set**: 450 queries, 150 per category (code, math, dialogue). The same queries are given to the routing system and to both baselines.
3. **Measures** for each system: answer quality score, inference time, GPU memory (VRAM), and energy / CO2 measured with [CodeCarbon](https://codecarbon.io/) on 2 x Tesla T4 GPUs.

## Results

### The router is accurate
| Router | Macro-F1 (seed 42) | Macro-F1 (5 seeds, mean +/- std) |
|---|---|---|
| **Random Forest** | **0.922** | **0.967 +/- 0.026** |
| Logistic regression | 0.876 | 0.885 +/- 0.025 |

On the 450 evaluation queries, the router sent **97.1%** of them to the right category.

### Answer quality (score from the experiment, higher is better)
| Category | Routing | Llama-3.1-8B | Mistral-7B |
|---|---|---|---|
| Code | 0.822 | 0.822 | **0.848** |
| Dialogue | 0.842 | 0.834 | **0.855** |
| Math | **0.820** | 0.533 | 0.193 |

### Cost
| System | Energy (kWh) | CO2 (kg) | Avg time per query (s) | Avg VRAM (MB) |
|---|---|---|---|---|
| **Routing** | **0.374** | **0.169** | 24-34 | 1,784-3,516 |
| Llama-3.1-8B | 0.565 | 0.256 | 36-37 | about 3,690 |
| Mistral-7B | 0.502 | 0.234 | 20-26 | about 2,365 |

### Reading the results
- **Math is where routing helps the most**: the specialized math model reaches 0.82, while the two general models are at 0.53 and 0.19.
- On **code and dialogue, routing matches Llama-3.1-8B** (equal on code, +0.7 points on dialogue) but **Mistral-7B is slightly better** (+2.6 and +1.3 points).
- **Energy**: the routing system used about **34% less energy than Llama-3.1-8B** (0.374 vs 0.565 kWh) and about **25% less than Mistral-7B** (0.374 vs 0.502 kWh) for the full run.
- Routing is **not the fastest or the lightest in memory**: Mistral-7B is quicker and uses less VRAM per query on code and dialogue. The gain is mostly energy and math quality.

## Limitations
- One run per system, 150 queries per category: differences of a few points in quality are probably not significant.
- Energy values include the whole run (model loading included), measured on cloud GPUs, so they are not a pure per-query measure.
- Only 3 query types. Queries mixing several types (for example "write code to solve this equation") are not handled.
- Routing errors (about 3% here) send a query to a model that is not specialized for it.

## Repository structure
```text
├── models/
│   └── text_router.joblib          # trained router
├── results/
│   ├── routing_results.csv         # answers and measures of the routing system
│   ├── baseline_results_*.csv      # answers and measures of the two baselines
│   ├── router_comparison.csv       # Random Forest vs logistic regression
│   ├── router_seed_runs.csv        # router stability over 5 seeds
│   ├── router_stability_summary.csv
│   ├── full_comparison_summary.csv # quality, VRAM and time by category and system
│   ├── routing_summary.csv
│   ├── emissions.csv               # CodeCarbon energy and CO2 logs
│   └── ...
├── requirements.txt
└── README.md
```
The experiment notebook is not stored in the repo.

## How to run
```bash
pip install -r requirements.txt
```
The generation step loads the models from Hugging Face (`transformers`, `bitsandbytes`), so a GPU is needed to reproduce it.

Note: `text_router.joblib` was saved from a notebook and depends on a custom class (`CombinedFeatureTransformer`) that is defined in that notebook, not in this repo. To load the router with `joblib.load`, this class must be defined first. All the results above can be read directly from the CSV files in `results/`.
