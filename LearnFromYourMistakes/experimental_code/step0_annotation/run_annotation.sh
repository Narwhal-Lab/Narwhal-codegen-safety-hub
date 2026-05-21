#!/bin/bash
# API Annotation Pipeline (Step 0)
# Input/output paths can be overridden via environment variables

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${TSP_DATA_DIR:-$WORKSPACE_DIR/data}"
OUTPUT_DIR="${TSP_OUTPUT_DIR:-$WORKSPACE_DIR/data}"
INPUT_FILE="${INPUT_FILE:-$DATA_DIR/sec-new-desc.json}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/sec-new-desc_annotated.json}"

python "$SCRIPT_DIR/api_annotation.py" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE"

echo "$OUTPUT_FILE generated"
