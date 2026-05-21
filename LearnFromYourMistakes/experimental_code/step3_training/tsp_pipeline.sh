#!/bin/bash

# TSP: Tree-like Self-Play Pipeline (Step 3 - Main Entry Point)
# Pipeline for training security-enhanced code generation models
# Version: 3.0

# Display help information
function show_help {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -i, --input FILE     Specify input dataset file path"
    echo "  -o, --output DIR     Specify output directory path"
    echo "  -m, --model MODEL    Specify base model for training"
    echo "  -h, --help          Display this help information"
    echo
    echo "Supported Models:"
    echo "  - CodeLlama-7b-Instruct-hf (default)"
    echo "  - Qwen/Qwen2.5-Coder-7B-Instruct"
    echo "  - Qwen/Qwen2.5-Coder-7B"
    echo
    echo "Examples:"
    echo "  # Run with CodeLlama (default)"
    echo "  $0 -i dataset.json -o /path/to/output"
    echo
    echo "  # Run with Qwen2.5-Coder-7B-Instruct"
    echo "  $0 -i dataset.json -o /path/to/output -m Qwen/Qwen2.5-Coder-7B-Instruct"
    echo
}

# Function to select config file based on model
function get_config_file {
    local model_path="$1"
    local config_dir="$SCRIPT_DIR/config"

    case "$model_path" in
        *"Qwen"*"Coder"*)
            echo "$config_dir/qwencoder_7b.yaml"
            ;;
        *"CodeLlama"*|*"codellama"*)
            echo "$config_dir/codellama_7b.yaml"
            ;;
        *)
            echo "$config_dir/codellama_7b.yaml"
            ;;
    esac
}

# Set default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-$WORKSPACE_DIR/LLaMA-Factory}"

MODEL_PATH="CodeLlama-7b-Instruct-hf"
INPUT_DATA="${TSP_INPUT_DATA:-$WORKSPACE_DIR/data/sec-new-desc_annotated_with_nodes.json}"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--input)
            INPUT_DATA="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ -z "$INPUT_DATA" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: Missing required parameters"
    show_help
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/inference"
mkdir -p "$OUTPUT_DIR/tsp_data"

# Set up file paths
NODES_DATA="$OUTPUT_DIR/inference/nodes_data.json"
GENERATED_DATA="$OUTPUT_DIR/inference/generation_output.json"
PREFERENCE_PAIRS="$OUTPUT_DIR/inference/preference_pairs.json"
TSP_DATA="$OUTPUT_DIR/tsp_data/tsp_training.json"
TRAIN_CONFIG=$(get_config_file "$MODEL_PATH")

echo "====================================================="
echo "        TSP: Tree-like Self-Play Training"
echo "====================================================="
echo "Input file: $INPUT_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Base model: $MODEL_PATH"
echo "Config file: $TRAIN_CONFIG"
echo "====================================================="

# Check if input file exists
if [ ! -f "$INPUT_DATA" ]; then
    echo "Error: Input file '$INPUT_DATA' does not exist"
    exit 1
fi

# Check if config file exists
if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "Error: Config file '$TRAIN_CONFIG' does not exist"
    exit 1
fi

# Step 1: Run model inference to generate code variations
echo "Step [1/3]: Running code generation at vulnerability nodes..."
if [ ! -f "$GENERATED_DATA" ]; then
    python "$WORKSPACE_DIR/step1_inference/inference_with_template.py" \
        --data_file "$INPUT_DATA" \
        --output_file "$GENERATED_DATA" \
        --model "$MODEL_PATH"
fi

if [ ! -f "$GENERATED_DATA" ]; then
    echo "Error: Code generation failed"
    exit 1
fi
echo "[OK] Code generation completed"

# Step 2: Create preference pairs from generated code
echo ""
echo "Step [2/3]: Creating preference pairs..."
python -c "
import sys
sys.path.append('$WORKSPACE_DIR/step2_data_processing')
from utils import make_preference_pair, convert_to_dpo_format

make_preference_pair('$GENERATED_DATA', '$PREFERENCE_PAIRS')
convert_to_dpo_format('$PREFERENCE_PAIRS', '$TSP_DATA')
"

if [ ! -f "$TSP_DATA" ]; then
    echo "Error: Preference pair creation failed"
    exit 1
fi
echo "[OK] Preference pairs created and converted to TSP format"

# Step 3: Train model using TSP
echo ""
echo "Step [3/3]: Running TSP training..."

# Check if LLaMA-Factory is available
if [ ! -d "$LLAMA_FACTORY_DIR" ]; then
    echo "Error: LLaMA-Factory not found at $LLAMA_FACTORY_DIR"
    echo "Please clone it: git clone https://github.com/hiyouga/LLaMA-Factory.git $LLAMA_FACTORY_DIR"
    echo "Or set LLAMA_FACTORY_DIR environment variable to your installation path."
    exit 1
fi

# Create temp config with current model path
mkdir -p "$LLAMA_FACTORY_DIR/data"
cp "$TSP_DATA" "$LLAMA_FACTORY_DIR/data/"
TEMP_CONFIG="$OUTPUT_DIR/tsp_data/train_config.yaml"
cp "$TRAIN_CONFIG" "$TEMP_CONFIG"
sed -i "s|model_name_or_path:.*|model_name_or_path: $MODEL_PATH|g" "$TEMP_CONFIG"
sed -i "s|output_dir:.*|output_dir: $OUTPUT_DIR/model|g" "$TEMP_CONFIG"

# Run training
cd "$LLAMA_FACTORY_DIR"
llamafactory-cli train "$TEMP_CONFIG"

if [ $? -eq 0 ]; then
    echo "[OK] Training completed successfully!"
else
    echo "[FAIL] Training failed"
    exit 1
fi

# Clean up
rm -f "$TEMP_CONFIG"

echo ""
echo "====================================================="
echo "        TSP Training Pipeline Complete!"
echo "====================================================="
echo "Generated files:"
echo "  - Generated code: $GENERATED_DATA"
echo "  - Preference pairs: $PREFERENCE_PAIRS"
echo "  - Training data: $TSP_DATA"
echo "Trained model saved in: $OUTPUT_DIR/model"
echo "====================================================="
