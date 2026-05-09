# Is Your Prompt Poisoning Code?

This repository accompanies the paper:

> **Is Your Prompt Poisoning Code? Defect Induction Rates and Security Mitigation Strategies**
> Bin Wang, YiLu Zhong, MiDi Wan, WenJie Yu, YuanBing Ouyang, Yenan Huang, Hui Li†
> *Empirical Software Engineering* (accepted)
> arXiv: [2510.22944](https://arxiv.org/abs/2510.22944)
>
> † Corresponding author: Hui Li

It contains the **CWE-BENCH-PYTHON** dataset and the experiment runners used to measure how prompt normativity affects the security of LLM-generated code, together with the runners for the Chain-of-Thought, ReAct, and synonym-perturbation variants reported in the paper.

## What is in this repository

| Path | Contents |
|---|---|
| `DataSet/base/` | CWE-BENCH-PYTHON: 33 CWE × 5 tasks × 4 prompt-normativity levels (L0–L3), one JSON per CWE |
| `DataSet/syn/` | Synonym-perturbed prompt variants used in the SynV1 / SynV2 robustness experiments |
| `DataSet/added_Java_and_cpp/` | Supplementary Java / C++ test cases used in the cross-language extension |
| `experimental_code/` | Experiment runners (Baseline / CoT / ReAct / Synonym), model registry, DB layer, utilities |

Raw experimental outputs (judged SQL views, vote aggregations, hyper-parameter sweep JSONL) and the rendered figures from the paper are **not redistributed via this repository**. They can be regenerated end-to-end by running the code below; please contact the authors if you need access to a specific intermediate artifact for reproduction or re-analysis.

## Quickstart

### 1. Install

```bash
git clone https://github.com/<your-org>/<this-repo>.git
cd <this-repo>
pip install -r experimental_code/requirements.txt
```

Tested on Python 3.10+.

### 2. Configure secrets

The runners require a MySQL backend and one or more LLM endpoints. **All hosts, paths, and credentials are read from environment variables** — nothing is committed to the repository.

```bash
cd experimental_code
cp .env.example .env
# edit .env to fill in your DB credentials and model API keys
set -a && source .env && set +a
```

The full set of variables is documented in `experimental_code/.env.example`. Default values point to `localhost` placeholders only; you must supply your own credentials.

### 3. Prepare the database

The pipeline persists every (CWE × task × prompt level × code model) trial to a single MySQL table whose name is configurable via `DB_TABLE_NAME` (default `result_table_710`). The schema is created on first run by the peewee model in `experimental_code/my_db.py`.

### 4. Run experiments

Pipelines are independent except for **ReAct**, which consumes the rows produced by the Baseline run. Recommended order:

```bash
cd experimental_code
python run_base.py    # Baseline:        prompt → code → security judgment
python run_cot.py     # Chain-of-Thought variant of the Baseline prompt
python run_react.py   # Self-correcting rewrite of Baseline outputs
python run_syn.py     # Synonym-perturbation experiment (SynV1 / SynV2)
```

Each runner uses a 200-thread pool by default — adjust `MAX_WORKERS` inside the runner's `__main__` if your model providers have lower rate limits.

## Environment variables

| Variable | Purpose |
|---|---|
| `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` | MySQL connection |
| `DB_TABLE_NAME` | Result table (defaults to `result_table_710`) |
| `VLLM_HOST` | Host where local vLLM / Ollama servers listen |
| `VLLM_MODEL_BASE` | Filesystem prefix for open-source model weights, as passed to vLLM `--model` |
| `VLLM_RELAY_API_KEY` | Bearer token for the local vLLM relay (leave blank if not required) |
| `KIMI_API_URL`, `KIMI_API_KEY` | Moonshot Kimi endpoint |
| `LLM_PROXY_URL`, `LLM_PROXY_API_KEY` | OpenAI-compatible relay used for Gemini, GPT-4o, Grok, Claude |

## Models

The model registry in `experimental_code/custom_model.py` covers, locally hosted: Qwen3:32b, DeepSeek-Coder-V2-Instruct, CodeQwen1.5-7B, mixtral:8x7b, codegemma-1.1-7b-it, Llama-4-Scout-17B-16E-Instruct, starcoder2-15b, Qwen3-14B; and hosted: Kimi K2, Gemini 2.5 Flash, GPT-4o, Grok-3, Claude Sonnet 4. Models that are not relevant to a given run can be removed via the `exclude_code_model_ids` list in each runner's `__main__`.

The LLM judge used in the paper is `gemini-2.5-flash`. To use a different judge, change `judgment_model_id` at the bottom of each runner.

## Dataset card — CWE-BENCH-PYTHON

- **Scope.** 33 CWEs, each containing 5 task scenarios. Every task is rendered at four prompt normativity levels (L0 fully specified → L3 highly disordered), giving 33 × 5 × 4 = 660 base prompts plus their synonym-perturbed counterparts in `DataSet/syn/`.
- **Per-file structure.** Each `CWE-XXX.json` contains `structured_output` (task titles), `prompt_templates` (judge prompt that elicits a JSON verdict with `vulnerability_exists ∈ {0, 1}`), and `taskN` / `taskN_process` blocks describing each scenario and its design rationale.
- **Languages.** Primary benchmark is Python; `DataSet/added_Java_and_cpp/` extends a subset of CWEs to Java and C++ for the cross-language analysis.

## Responsible use

CWE-BENCH-PYTHON intentionally elicits insecure code patterns from language models in order to measure their susceptibility to prompt-quality degradation. The dataset is released for **research use only**. Please:

- Do not use it as training data for production code-generation systems.
- Follow your provider's acceptable-use policy when interacting with hosted LLM APIs.
- Treat any credential-shaped strings that appear in model outputs as sensitive and do not republish them verbatim — they may be memorized fragments of public training data.

## License

- **Code** (`experimental_code/`): MIT License — see [`LICENSE`](LICENSE).
- **Dataset** (`DataSet/`): Creative Commons Attribution 4.0 International — see [`LICENSE-DATA`](LICENSE-DATA).

## Citation

If you use this dataset or code, please cite:

```bibtex
@article{wang2025prompt,
  title         = {Is Your Prompt Poisoning Code? Defect Induction Rates and Security Mitigation Strategies},
  author        = {Wang, Bin and Zhong, YiLu and Wan, MiDi and Yu, WenJie and Ouyang, YuanBing and Huang, Yenan and Li, Hui},
  journal       = {Empirical Software Engineering},
  year          = {2025},
  note          = {To appear},
  eprint        = {2510.22944},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://arxiv.org/abs/2510.22944}
}
```

A [`CITATION.cff`](CITATION.cff) file is also included so that GitHub renders a one-click citation widget on the repository page.
