.PHONY: test lint data splits verify overlap sanity

HF := https://huggingface.co/datasets
RAW := data/raw

test:
	uv run pytest --cov=search_agent_rl --cov-report=term-missing

lint:
	uv run ruff check . && uv run ruff format --check .

# Download HotpotQA and MuSiQue (FlashRAG release) and TopiOCQA (McGill-NLP release).
data:
	@for f in hotpotqa/train.jsonl hotpotqa/dev.jsonl musique/train.jsonl musique/dev.jsonl; do \
		mkdir -p $(RAW)/$$(dirname $$f); \
		curl -sSfL -o $(RAW)/$$f $(HF)/RUC-NLPIR/FlashRAG_datasets/resolve/main/$$f; \
	done
	@mkdir -p $(RAW)/topiocqa
	@for s in train valid; do \
		curl -sSfL -o $(RAW)/topiocqa/$$s.jsonl $(HF)/McGill-NLP/TopiOCQA/resolve/main/data/topiocqa_$$s.jsonl; \
	done

splits:
	uv run python -m search_agent_rl.data.cli make-splits

verify:
	uv run python -m search_agent_rl.data.cli verify

overlap:
	uv run python -m search_agent_rl.data.cli overlap

# Gold must score EM=1 and empty EM~0 on every validation split (see EmptySystem).
sanity:
	@for d in hotpotqa musique topiocqa; do \
		for s in gold empty; do \
			uv run python -m search_agent_rl.eval.harness --system $$s --dataset $$d --out $${TMPDIR:-/tmp}/sarl-sanity; \
		done; \
	done
