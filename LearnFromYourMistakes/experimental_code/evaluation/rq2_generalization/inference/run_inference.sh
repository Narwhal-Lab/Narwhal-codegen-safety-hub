#!/bin/bash
# RQ2 (Unseen CWE Generalization): Inference + Database Conversion

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${TSP_DATA_DIR:-$EVAL_DIR/../../data/evaluation/unseen_cwe}"
OUTPUT_DIR="${TSP_OUTPUT_DIR:-$SCRIPT_DIR/output}"
MODEL_PATH="${MODEL_PATH:-CodeLlama-7b-Instruct-hf}"
MODEL_NAME="${MODEL_NAME:-codellama7b_tsp}"

mkdir -p "$OUTPUT_DIR"

echo "===== RQ2 Generalization: Inference ====="
echo "Model: $MODEL_PATH"
echo "Input: $DATA_DIR/dataset.json"
echo "Output: $OUTPUT_DIR/${MODEL_NAME}_inference_output.json"

python "$SCRIPT_DIR/inference_with_template.py" \
    --model "$MODEL_PATH" \
    --input "$DATA_DIR/dataset.json" \
    --output "$OUTPUT_DIR/${MODEL_NAME}_inference_output.json"

echo ""
echo "===== RQ2 Generalization: Converting to database ====="
python "$SCRIPT_DIR/convert_to_database.py" "$MODEL_NAME"

echo "Done."
