#!/bin/bash
# TSP Inference Pipeline (Step 1)
# Paths can be overridden via environment variables

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${TSP_DATA_DIR:-$WORKSPACE_DIR/data}"
OUTPUT_DIR="${TSP_OUTPUT_DIR:-$WORKSPACE_DIR/output}"
INPUT_FILE="${INPUT_FILE:-$DATA_DIR/sec-new-desc_annotated_with_nodes.json}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/sec-new-desc_annotated_with_nodes_generation.json}"
MODEL_PATH="${MODEL_PATH:-CodeLlama-7b-Instruct-hf}"

python "$SCRIPT_DIR/inference_with_template.py" \
    --data_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model "$MODEL_PATH"
