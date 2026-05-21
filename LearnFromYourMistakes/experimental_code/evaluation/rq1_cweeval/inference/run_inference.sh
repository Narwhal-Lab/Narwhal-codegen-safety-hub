#!/bin/bash
# RQ1 (CWE_Eval): Inference + Database Conversion + LLM Evaluation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${TSP_DATA_DIR:-$EVAL_DIR/../../data/evaluation/CWEEval}"
OUTPUT_DIR="${TSP_OUTPUT_DIR:-$SCRIPT_DIR/output}"
MODEL_PATH="${MODEL_PATH:-CodeLlama-7b-Instruct-hf}"
MODEL_NAME="${MODEL_NAME:-codellama7b_tsp}"

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Start execution time: $(date)" | tee "$LOG_DIR/execution.log"
echo "Model name: $MODEL_NAME" | tee -a "$LOG_DIR/execution.log"
echo "Model path: $MODEL_PATH" | tee -a "$LOG_DIR/execution.log"

# Inference phase
echo "===== Starting inference phase =====" | tee -a "$LOG_DIR/inference.log"
python "$SCRIPT_DIR/inference_with_template.py" \
    --model "$MODEL_PATH" \
    --input "$DATA_DIR/cwe_evaluate.json" \
    --output "$OUTPUT_DIR/${MODEL_NAME}_inference_output.json" \
    2>&1 | tee -a "$LOG_DIR/inference.log"

# Convert to database
echo "===== Starting database conversion phase =====" | tee -a "$LOG_DIR/conversion.log"
python "$SCRIPT_DIR/convert_to_database.py" "$MODEL_NAME" \
    2>&1 | tee -a "$LOG_DIR/conversion.log"

echo "Done. Output: $OUTPUT_DIR/${MODEL_NAME}_inference_output.json"
