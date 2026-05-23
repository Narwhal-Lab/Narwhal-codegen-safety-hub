# Learn from Your Mistakes: Tree-like Self-Play for Secure Code LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/Data-CC--BY--4.0-green.svg)](LICENSE-DATA)

Implementation for the paper *"Learn from Your Mistakes:
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

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU with at least 24GB VRAM (for 7B models)
- [vLLM](https://github.com/vllm-project/vllm) for inference
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for DPO training

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

The TSP pipeline consists of four steps:

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

### Supported Models

| Model | Config File |
|-------|-------------|
| CodeLlama-7B-Instruct | `step3_training/config/codellama_7b.yaml` |
| Qwen2.5-Coder-7B-Instruct | `step3_training/config/qwencoder_7b.yaml` |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for GPT-4o annotation (Step 0) |
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
