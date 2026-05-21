#!/bin/bash

# Define array of model names to analyze
MODELS=(
    "codellama-7b-self-play-dpo-4-2"
    # Add more models here
)

# CodeQL path
CODEQL_PATH=""

# Iterate through all models
for MODEL_NAME in "${MODELS[@]}"; do
    echo "==============================================="
    echo "Processing model: ${MODEL_NAME}"
    echo "==============================================="
    
    # Check if test case directory exists
    TEST_DIR="./SecurityEval/Testcases_${MODEL_NAME}"
    if [ ! -d "$TEST_DIR" ]; then
        echo "Warning: Directory does not exist - $TEST_DIR"
        echo "Skipping this model..."
        echo
        continue
    fi
    
    # Switch to test case directory
    cd "$TEST_DIR" || continue
    echo "Changed to directory: $(pwd)"
    
    # Create CodeQL database
    echo "Creating CodeQL database..."
    "$CODEQL_PATH" database create --overwrite --language=python "./SecurityEval/Databases/Testcases_${MODEL_NAME}"
    
    # Check if database creation was successful
    if [ $? -ne 0 ]; then
        echo "Error: Database creation failed"
        continue
    fi
    
    # Switch to database directory
    cd "./SecurityEval/Databases" || continue
    echo "Changed to directory: $(pwd)"
    
    # Run analysis job
    JOB_SCRIPT="./SecurityEval/Databases/job_${MODEL_NAME}.sh"
    if [ -f "$JOB_SCRIPT" ]; then
        echo "Executing analysis job script: $JOB_SCRIPT"
        bash "$JOB_SCRIPT"
    else
        echo "Warning: Job script does not exist - $JOB_SCRIPT"
    fi
    
    echo "Model ${MODEL_NAME} processing complete"
    echo
done

echo "All model analyses complete!"


