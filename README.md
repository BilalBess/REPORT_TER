# Dynamic Routing over Specialized Small Language Models

This project investigates whether routing user queries to specialized Small Language Models (SLMs) can reduce computational and energy costs while maintaining answer quality compared with larger general-purpose language models.

## Approach

The system classifies each query into one of three categories:

- Code
- Mathematics
- Dialogue

A Random Forest router then selects a specialized model:

- Code → DeepSeek-Coder-6.7B-Instruct
- Mathematics → Qwen2.5-Math-1.5B-Instruct
- Dialogue → Phi-3.5-mini-Instruct

The routing system is compared with two general-purpose baselines:

- Llama-3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3

## Repository structure

```text
├── models/
│   └── text_router.joblib
├── results/
│   ├── routing_results.csv
│   ├── router_comparison.csv
│   ├── emissions.csv
│   ├── full_comparison_summary.csv
│   └── ...
├── notebooks/
│   └── experiment.ipynb
├── requirements.txt
└── README.md