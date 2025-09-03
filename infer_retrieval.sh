#!/bin/bash

# =================================================================
# Author:      Xinyu Lu
# Email:       xinyulu@stu.xmu.edu.cn
# Date:        2025-09-03
# Description: This script evaluates the retrieval performance of
#              the vib2mol model for different spectral types on
#              various datasets. It performs both initial retrieval
#              and re-ranking.
# =================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Common Parameters ---
MODEL_NAME="vib2mol"
RERANK_TOPK=5            # Number of top-k results to consider for re-ranking

# --- Parse Command-Line Arguments ---
FORMULA_ENABLED=""
RERANK_ENABLED=""
for arg in "$@"; do
  if [ "$arg" == "--use_formula" ]; then
    FORMULA_ENABLED="--use_formula"
  elif [ "$arg" == "--rerank" ]; then
    RERANK_ENABLED="-rerank"
  fi
done

# --- Function to run a single evaluation ---
run_evaluation() {
    local DATASET=$1
    local SPECTRAL_TYPE=$2
    local TEST_MODEL_PATH=$3

    # Construct the command dynamically
    local command_base="python infer_retrieval.py \
      --model \"${MODEL_NAME}\" \
      --ds \"${DATASET}\" \
      --spectral_types \"${SPECTRAL_TYPE}\" \
      --test_model_path \"${TEST_MODEL_PATH}\""
      
    local full_command="${command_base}"
    if [ -n "${RERANK_ENABLED}" ]; then
      full_command+=" ${RERANK_ENABLED} --topk \"${RERANK_TOPK}\" --rank_model_path \"${TEST_MODEL_PATH}\""
    fi
    
    if [ -n "${FORMULA_ENABLED}" ]; then
      full_command+=" ${FORMULA_ENABLED}"
    fi

    eval "${full_command}"
}

# --- Main Script Execution ---
echo "Starting evaluation script for vib2mol retrieval performance..."
echo "---------------------------------------------------------"

# Define test configurations in an array
declare -a configs=(
    # Dataset QM9s
    "qm9s ir checkpoints/qm9s/ir-kekule_smiles/vib2mol/2025-07-08-02-55-db43f7/epoch999.pth"
    "qm9s raman checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-06-12-15-bbe117/epoch999.pth"
    "qm9s raman-ir checkpoints/qm9s/ir-raman-kekule_smiles/vib2mol/2025-07-11-06-32-9005f4/epoch999.pth"
    
    # Dataset VB-mols
    "mols ir checkpoints/mols/ir-kekule_smiles/vib2mol/2025-07-08-02-55-6ac6f2/epoch999.pth"
    "mols raman checkpoints/mols/raman-kekule_smiles/vib2mol/2025-07-08-02-56-69bd7e/epoch999.pth"
    "mols raman-ir checkpoints/mols/ir-raman-kekule_smiles/vib2mol/2025-07-12-01-38-833d2d/epoch999.pth"
    
    # Dataset SDBS
    "sdbs ir checkpoints/sdbs/ir-kekule_smiles/vib2mol/2025-07-16-15-32-ea6ef7/epoch999.pth"
    "sdbs raman checkpoints/sdbs/raman-kekule_smiles/vib2mol/2025-07-16-15-33-d68938/epoch999.pth"
    "sdbs raman-ir checkpoints/sdbs/ir-raman-kekule_smiles/vib2mol/2025-07-16-15-24-9a8d24/epoch999.pth"

    # Dataset NIST
    "nist ir checkpoints/nist/ir-kekule_smiles/vib2mol/2025-07-16-16-03-187b97/epoch999.pth"
)

# Initialize a counter for the tests
counter=1
total_tests=${#configs[@]}

# Iterate through the configurations and run the evaluation function
for config in "${configs[@]}"; do
    echo ""
    echo "--- Running Test #$counter of $total_tests ---"
    read -r DATASET SPECTRAL_TYPE TEST_MODEL_PATH <<< "${config}"
    
    echo "Evaluating retrieval performance for ${DATASET}-${SPECTRAL_TYPE}..."
    run_evaluation "${DATASET}" "${SPECTRAL_TYPE}" "${TEST_MODEL_PATH}"
    
    echo "${DATASET}-${SPECTRAL_TYPE} evaluation completed."
    echo "---------------------------------------------------------"

    ((counter++))
done

echo "All evaluations completed."
echo "---------------------------------------------------------"