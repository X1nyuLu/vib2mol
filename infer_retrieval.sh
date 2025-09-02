#!/bin/bash

# This script evaluates the retrieval performance of the vib2mol model
# for different spectral types on the QM9s dataset.
# It performs both initial retrieval and re-ranking.

# Set common parameters to variables for easier modification and readability
MODEL_NAME="vib2mol"
DATASET="qm9s"
RERANK_ENABLED="-rerank" # Flag to enable re-ranking
RERANK_TOPK=5            # Number of top-k results to consider for re-ranking

echo "Starting evaluation script for vib2mol retrieval performance..."
echo "---------------------------------------------------------"

# --- Evaluation for QM9s-IR (Infrared) ---
echo "Evaluating retrieval performance for QM9s-IR..."
SPECTRAL_TYPE_IR="ir" # Note: Your original script used 'raman' for QM9s-IR.
                      # If it's truly IR, please use 'ir'.
                      # Assuming based on path "ir-kekule_smiles" that it should be 'ir'.
TEST_MODEL_PATH_IR="checkpoints/qm9s/ir-kekule_smiles/vib2mol/2025-07-08-02-55-db43f7/epoch999.pth"
RANK_MODEL_PATH_IR="${TEST_MODEL_PATH_IR}" # Often the same model is used for ranking

python infer_retrieval.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE_IR}" \
  --test_model_path "${TEST_MODEL_PATH_IR}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH_IR}"

echo "QM9s-IR evaluation completed."
echo "---------------------------------------------------------"


# --- Evaluation for QM9s-Raman ---
echo "Evaluating retrieval performance for QM9s-Raman..."
SPECTRAL_TYPE_RAMAN="raman"
TEST_MODEL_PATH_RAMAN="checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-06-12-15-bbe117/epoch999.pth"
RANK_MODEL_PATH_RAMAN="${TEST_MODEL_PATH_RAMAN}" # Often the same model is used for ranking

python infer_retrieval.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE_RAMAN}" \
  --test_model_path "${TEST_MODEL_PATH_RAMAN}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH_RAMAN}"

echo "QM9s-Raman evaluation completed."
echo "---------------------------------------------------------"


# --- Evaluation for QM9s-Raman-IR (Combined Spectra) ---
echo "Evaluating retrieval performance for QM9s-Raman-IR (Combined Spectra)..."
SPECTRAL_TYPE_RAMAN_IR="raman-ir"
TEST_MODEL_PATH_RAMAN_IR="checkpoints/qm9s/ir-raman-kekule_smiles/vib2mol/2025-07-11-06-32-9005f4/epoch999.pth"
RANK_MODEL_PATH_RAMAN_IR="${TEST_MODEL_PATH_RAMAN_IR}" # Often the same model is used for ranking

python infer_retrieval.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE_RAMAN_IR}" \
  --test_model_path "${TEST_MODEL_PATH_RAMAN_IR}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH_RAMAN_IR}"

echo "QM9s-Raman-IR evaluation completed."
echo "---------------------------------------------------------"


DATASET="mols"
# --- Evaluation for VB-mols-IR ---
echo "Evaluating retrieval performance for VB-mols-IR ..."
SPECTRAL_TYPE_RAMAN_IR="ir"
TEST_MODEL_PATH_RAMAN_IR="checkpoints/mols/ir-kekule_smiles/vib2mol/2025-07-08-02-55-6ac6f2/epoch999.pth"
RANK_MODEL_PATH_RAMAN_IR="${TEST_MODEL_PATH_RAMAN_IR}" # Often the same model is used for ranking

python infer_retrieval.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE_RAMAN_IR}" \
  --test_model_path "${TEST_MODEL_PATH_RAMAN_IR}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH_RAMAN_IR}"

echo "VB-mols-IR evaluation completed."
echo "---------------------------------------------------------"


# --- Evaluation for VB-mols-Raman ---
echo "Evaluating retrieval performance for VB-mols-Raman ..."
SPECTRAL_TYPE_RAMAN_IR="raman"
TEST_MODEL_PATH_RAMAN_IR="checkpoints/mols/raman-kekule_smiles/vib2mol/2025-07-08-02-56-69bd7e/epoch999.pth"
RANK_MODEL_PATH_RAMAN_IR="${TEST_MODEL_PATH_RAMAN_IR}" # Often the same model is used for ranking

python infer_retrieval.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE_RAMAN_IR}" \
  --test_model_path "${TEST_MODEL_PATH_RAMAN_IR}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH_RAMAN_IR}" \
  --use_formula

echo "VB-mols-Raman evaluation completed."
echo "---------------------------------------------------------"


# --- Evaluation for VB-mols-IR-Raman ---
echo "Evaluating retrieval performance for VB-mols-IR-Raman ..."
SPECTRAL_TYPE_RAMAN_IR="raman-ir"
TEST_MODEL_PATH_RAMAN_IR="checkpoints/mols/ir-raman-kekule_smiles/vib2mol/2025-07-12-01-38-833d2d/epoch999.pth"
RANK_MODEL_PATH_RAMAN_IR="${TEST_MODEL_PATH_RAMAN_IR}" # Often the same model is used for ranking

python infer_retrieval.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE_RAMAN_IR}" \
  --test_model_path "${TEST_MODEL_PATH_RAMAN_IR}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH_RAMAN_IR}"

echo "VB-mols-IR-Raman evaluation completed."
echo "---------------------------------------------------------"

echo "All evaluations finished."




DATASET="sdbs"
# --- Evaluation for SDBS-IR ---
echo "Evaluating retrieval performance for SDBS-IR ..."
SPECTRAL_TYPE_RAMAN_IR="ir"
TEST_MODEL_PATH_RAMAN_IR="checkpoints/sdbs/ir-kekule_smiles/vib2mol/2025-07-16-15-32-ea6ef7/epoch999.pth"
RANK_MODEL_PATH_RAMAN_IR="${TEST_MODEL_PATH_RAMAN_IR}" # Often the same model is used for ranking

python infer_retrieval.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE_RAMAN_IR}" \
  --test_model_path "${TEST_MODEL_PATH_RAMAN_IR}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH_RAMAN_IR}"

echo "SDBS-IR evaluation completed."
echo "---------------------------------------------------------"


# --- Evaluation for SDBS-Raman ---
echo "Evaluating retrieval performance for SDBS-Raman ..."
SPECTRAL_TYPE_RAMAN_IR="raman"
TEST_MODEL_PATH_RAMAN_IR="checkpoints/sdbs/raman-kekule_smiles/vib2mol/2025-07-16-15-33-d68938/epoch999.pth"
RANK_MODEL_PATH_RAMAN_IR="${TEST_MODEL_PATH_RAMAN_IR}" # Often the same model is used for ranking

python infer_retrieval.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE_RAMAN_IR}" \
  --test_model_path "${TEST_MODEL_PATH_RAMAN_IR}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH_RAMAN_IR}"

echo "SDBS-Raman evaluation completed."
echo "---------------------------------------------------------"


# --- Evaluation for SDBS-IR-Raman ---
echo "Evaluating retrieval performance for SDBS-IR-Raman ..."
SPECTRAL_TYPE_RAMAN_IR="raman-ir"
TEST_MODEL_PATH_RAMAN_IR="checkpoints/sdbs/ir-raman-kekule_smiles/vib2mol/2025-07-16-15-24-9a8d24/epoch999.pth"
RANK_MODEL_PATH_RAMAN_IR="${TEST_MODEL_PATH_RAMAN_IR}" # Often the same model is used for ranking

python infer_retrieval.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE_RAMAN_IR}" \
  --test_model_path "${TEST_MODEL_PATH_RAMAN_IR}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH_RAMAN_IR}"

echo "SDBS-IR-Raman evaluation completed."
echo "---------------------------------------------------------"

echo "All evaluations finished."


DATASET="nist"
# --- Evaluation for SDBS-IR ---
echo "Evaluating retrieval performance for NIST-IR ..."
SPECTRAL_TYPE_RAMAN_IR="ir"
TEST_MODEL_PATH_RAMAN_IR="checkpoints/nist/ir-kekule_smiles/vib2mol/2025-07-16-16-03-187b97/epoch999.pth"
RANK_MODEL_PATH_RAMAN_IR="${TEST_MODEL_PATH_RAMAN_IR}" # Often the same model is used for ranking

python infer_retrieval.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE_RAMAN_IR}" \
  --test_model_path "${TEST_MODEL_PATH_RAMAN_IR}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH_RAMAN_IR}"

echo "NIST-IR evaluation completed."
echo "---------------------------------------------------------"
