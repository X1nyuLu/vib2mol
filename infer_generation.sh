#!/bin/bash

# =================================================================
# Author:      Xinyu Lu
# Email:       xinyulu@stu.xmu.edu.cn
# Date:        2025-09-03
# Description: This script evaluates the de novo generation
#              performance of the vib2mol model on various datasets.
# =================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Default Parameters ---
MODEL_NAME="vib2mol"
RERANK_ENABLED=""
RERANK_TOPK=5
BEAM_SIZE=10
USE_FORMULA_ENABLED=""

# --- Parse Command-Line Arguments ---
# This loop handles flags like --rerank, --use_formula, and others
for arg in "$@"; do
  case $arg in
    --rerank)
      RERANK_ENABLED="--rerank"
      ;;
    --use_formula)
      USE_FORMULA_ENABLED="--use_formula"
      ;;
    --topk=*)
      RERANK_TOPK="${arg#*=}"
      ;;
    --beam_size=*)
      BEAM_SIZE="${arg#*=}"
      ;;
    *)
      # Ignore unknown arguments for future flexibility
      ;;
  esac
done

# --- Function to run a single evaluation ---
run_generation_evaluation() {
    local DATASET=$1
    local SPECTRAL_TYPE=$2
    local TEST_MODEL_PATH=$3
    local RANK_MODEL_PATH=$4

    echo "Evaluating generation performance for ${DATASET}-${SPECTRAL_TYPE}..."

    # Dynamically build the command string
    local command_base="python infer_generation.py \
      --model \"${MODEL_NAME}\" \
      --ds \"${DATASET}\" \
      --spectral_types \"${SPECTRAL_TYPE}\" \
      --beam_size \"${BEAM_SIZE}\" \
      --test_model_path \"${TEST_MODEL_PATH}\""
      
    local full_command="${command_base}"
    
    # Add reranking parameters if enabled
    if [ -n "${RERANK_ENABLED}" ]; then
      full_command+=" ${RERANK_ENABLED} --topk \"${RERANK_TOPK}\" --rank_model_path \"${RANK_MODEL_PATH}\""
    fi
    
    # Add chemical formula flag if enabled
    if [ -n "${USE_FORMULA_ENABLED}" ]; then
      full_command+=" ${USE_FORMULA_ENABLED}"
    fi

    eval "${full_command}"
}

# --- Main Script Execution ---
echo "Starting vib2mol generation performance evaluation script..."
echo "---------------------------------------------------------"

# Define test configurations in an array
# Each line represents a single test case with its specific parameters
declare -a configs=(
    # --- QM9s Dataset ---
    "qm9s ir checkpoints/qm9s/ir-kekule_smiles/vib2mol/2025-07-10-01-45-46b8b6/epoch999.pth checkpoints/qm9s/ir-kekule_smiles/vib2mol/2025-07-08-02-55-db43f7/epoch999.pth"
    "qm9s ir checkpoints/qm9s/ir-kekule_smiles-formula/vib2mol/2025-07-12-02-02-4afc55/epoch999.pth checkpoints/qm9s/ir-kekule_smiles/vib2mol/2025-07-08-02-55-db43f7/epoch999.pth"
    "qm9s raman checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-07-07-51-af1552/epoch999.pth checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-06-12-15-bbe117/epoch999.pth"
    "qm9s raman checkpoints/qm9s/raman-kekule_smiles-formula/vib2mol/2025-07-12-02-01-7839c0/epoch999.pth checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-06-12-15-bbe117/epoch999.pth"
    "qm9s raman-ir checkpoints/qm9s/ir-raman-kekule_smiles-formula/vib2mol/2025-07-19-03-47-7c9f30/epoch999.pth checkpoints/qm9s/ir-raman-kekule_smiles/vib2mol/2025-07-11-06-32-9005f4/epoch999.pth"

    # --- VB-mols Dataset ---
    "mols ir checkpoints/mols/ir-kekule_smiles/vib2mol/2025-07-10-07-42-dde12a/epoch999.pth checkpoints/mols/ir-kekule_smiles/vib2mol/2025-07-08-02-55-6ac6f2/epoch999.pth"
    "mols ir checkpoints/mols/ir-kekule_smiles-formula/vib2mol/2025-07-14-07-06-e44b51/epoch999.pth checkpoints/mols/ir-kekule_smiles/vib2mol/2025-07-08-02-55-6ac6f2/epoch999.pth"
    "mols raman checkpoints/mols/raman-kekule_smiles/vib2mol/2025-07-10-07-46-68d484/epoch999.pth checkpoints/mols/raman-kekule_smiles/vib2mol/2025-07-08-02-56-69bd7e/epoch999.pth"
    "mols raman checkpoints/mols/raman-kekule_smiles-formula/vib2mol/2025-07-14-07-06-8a587e/epoch999.pth checkpoints/mols/raman-kekule_smiles/vib2mol/2025-07-08-02-56-69bd7e/epoch999.pth"
    "mols ir-raman checkpoints/mols/ir-raman-kekule_smiles-formula/vib2mol/2025-07-14-07-07-e1c1f4/epoch999.pth checkpoints/mols/ir-raman-kekule_smiles/vib2mol/2025-07-12-01-38-833d2d/epoch999.pth"
)

# Initialize counter for the tests
counter=1
total_tests=${#configs[@]}

# Iterate through the configurations and run the evaluation function
for config in "${configs[@]}"; do
    echo ""
    echo "--- Running Test #$counter of $total_tests ---"
    read -r DATASET SPECTRAL_TYPE TEST_MODEL_PATH RANK_MODEL_PATH <<< "${config}"
    
    run_generation_evaluation "${DATASET}" "${SPECTRAL_TYPE}" "${TEST_MODEL_PATH}" "${RANK_MODEL_PATH}"
    
    echo "${DATASET}-${SPECTRAL_TYPE} evaluation completed."
    echo "---------------------------------------------------------"

    ((counter++))
done

echo "All evaluations completed."
echo "---------------------------------------------------------"