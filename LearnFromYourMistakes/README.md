# Learn from Your Mistakes: Tree-like Self-Play for Secure Code LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/Data-CC--BY--4.0-green.svg)](LICENSE-DATA)

Implementation and evaluation for the paper *"Learn from Your Mistakes:
Tree-like Self-Play for Secure Code LLMs"*.

## Abstract

While Large Language Models (LLMs) have demonstrated remarkable performance in code generation, their effectiveness is undermined by a propensity to replicate subtle yet critical security vulnerabilities endemic to their training data. Current security alignment techniques often treat code as an indivisible unit, making it difficult to precisely correct localized, single-token errors that cause vulnerabilities. In this work, we introduce Tree-like Self-Play (TSP), a novel training framework that reframes secure code generation as a targeted self-play process. TSP models secure code generation as a tree-structured sequential decision problem, where each node represents a state in the generation process. At these "CWE Risk Nodes," the model engages in self-play by generating multiple code variants as branches. The main player's core task is to learn from its own mistakes by distinguishing the secure "golden path" from the potentially vulnerable branches.

## Repository Contents

| Directory | Description |
|-----------|-------------|
| `experimental_code/step0_annotation/` | CWE risk node annotation via GPT-4o |
| `experimental_code/step1_inference/` | vLLM code generation at CWE risk nodes |
| `experimental_code/step2_data_processing/` | DPO preference pair creation and format conversion |
| `experimental_code/step3_training/` | DPO training via LLaMA-Factory |
| `experimental_code/evaluation/rq1_securityeval/` | RQ1 evaluation: SecurityEval benchmark + CodeQL analysis |
| `experimental_code/evaluation/rq1_cweeval/` | RQ1 evaluation: CWE_Eval benchmark + LLM evaluation |
| `experimental_code/evaluation/rq2_generalization/` | RQ2 evaluation: Generalization to unseen CWE types |
| `experimental_code/evaluation/rq3_ablation/` | RQ3 evaluation: Ablation study and cross-language transfer |

## Key Results

- **Security Pass Rate (SPR@1)**: CodeLlama-7B improved to **75.8%** (vs. 57.0% for SFT)
- **Unseen CWE Generalization**: **32% reduction** in vulnerabilities on novel CWE types
- **Cross-Lingual Transfer**: Security knowledge transfers from C/C++ to Python, JavaScript, Go, and Ruby
- **Minimal Coding Impact**: Negligible effect on general-purpose coding ability (HumanEval)

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU with at least 24GB VRAM (for 7B models)
- [vLLM](https://github.com/vllm-project/vllm) for inference
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for DPO training
- [CodeQL](https://codeql.github.com/) (for RQ1 SecurityEval evaluation)

### Installation

```bash
cd experimental_code

# Install Python dependencies
pip install -r requirements.txt

# Set up LLaMA-Factory (required for training)
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

# Configure environment
cp .env.example .env
# Edit .env with your API keys and model paths
```

### Running the TSP Pipeline

The TSP pipeline consists of three steps:

**Step 0 -- CWE Risk Node Annotation** (requires GPT-4o API access):

```bash
cd step0_annotation
bash run_annotation.sh
```

**Step 1 -- Code Generation at Risk Nodes**:

```bash
cd step1_inference
bash run_inference.sh
```

**Step 2-3 -- Preference Pair Creation + DPO Training** (single entry point):

```bash
cd step3_training
bash tsp_pipeline.sh \
    -i /path/to/annotated_data_with_nodes.json \
    -o ./output \
    -m CodeLlama-7b-Instruct-hf
```

### Reproducing Evaluation Experiments

Each RQ evaluation directory contains an `inference/run_inference.sh` script:

```bash
# RQ1: SecurityEval + CodeQL
cd evaluation/rq1_securityeval/inference
MODEL_NAME=codellama7b_tsp bash run_inference.sh

# RQ1: CWE_Eval + LLM evaluation
cd evaluation/rq1_cweeval/inference
MODEL_NAME=codellama7b_tsp bash run_inference.sh

# RQ2: Unseen CWE generalization
cd evaluation/rq2_generalization/inference
MODEL_NAME=codellama7b_tsp bash run_inference.sh

# RQ3: Ablation / cross-language
cd evaluation/rq3_ablation/inference
MODEL_NAME=codellama7b_tsp bash run_inference.sh
```

### Supported Models

| Model | Config File |
|-------|-------------|
| CodeLlama-7B-Instruct | `step3_training/config/codellama_7b.yaml` |
| Qwen2.5-Coder-7B-Instruct | `step3_training/config/qwencoder_7b.yaml` |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for GPT-4o annotation (Step 0) |
| `LLM_EVAL_API_KEY` | API key for LLM-based vulnerability evaluation |
| `LLM_EVAL_API_URL` | API URL for LLM-based vulnerability evaluation |
| `LLAMA_FACTORY_DIR` | Path to LLaMA-Factory installation |
| `TSP_MODEL_PATH` | Override default model path |

## Citation

If you use this work, please cite:

```bibtex
@article{tsp2025,
  title={Learn from Your Mistakes: Tree-like Self-Play for Secure Code LLMs},
  author={},
  journal={},
  year={2025}
}
```

## License

- **Code**: MIT License — see [LICENSE](LICENSE)
- **Datasets**: CC-BY-4.0 License — see [LICENSE-DATA](LICENSE-DATA)
