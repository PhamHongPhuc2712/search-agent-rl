# search-agent-rl

An AI search answer engine: a small open LLM, fine-tuned (SFT) and then trained with reinforcement learning (GRPO), that plans, searches Wikipedia, takes notes and answers with citations. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full design.

**Status:** M0 (repo, frozen splits, eval harness) done.

## Setup

```bash
uv sync            # Python 3.12 environment
make data          # download HotpotQA, MuSiQue, TopiOCQA into data/raw (~750 MB)
make verify        # check the downloads match the frozen splits
make test          # tests with coverage
```

## Data

| Dataset | Train | Validation (frozen, from train) | Test (official dev, sealed) |
|---|---|---|---|
| HotpotQA | 89,447 | 1,000 | 7,405 |
| MuSiQue (answerable) | 18,938 | 1,000 | 2,417 |
| TopiOCQA | 44,200 turns | 1,250 turns (100 conversations) | 2,514 turns |

- Validation is chosen by hashing ids, so it is identical everywhere. The ids and SHA-256 hashes of the raw files are committed in `splits/`, and loading fails if either changes.
- The test split only loads with `--final`.
- Train/test question overlap is in `docs/results/overlap.json`: none for HotpotQA and MuSiQue. TopiOCQA's matches are almost all generic follow-ups ("where was he born?") whose meaning depends on the conversation; 19 are true duplicate first-turn questions.

## Evaluation

```bash
uv run python -m search_agent_rl.eval.harness --system gold --dataset musique
```

EM/F1 follow the official scorers: HotpotQA's `hotpot_evaluate_v1.py` for HotpotQA, SQuAD v2 style for MuSiQue and TopiOCQA. Every result has 95% bootstrap intervals and a breakdown by hop count, and is written to `docs/results/<split>/<dataset>/<system>.json`.
