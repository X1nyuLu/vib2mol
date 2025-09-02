# #!/bin/bash

# Set common parameters for easier modification and readability
MODEL_NAME="vib2mol"
DATASET="qm9s"
RERANK_ENABLED="--rerank" # Flag to enable re-ranking
RERANK_TOPK=5            # Number of top-k results to consider for re-ranking
BEAM_SIZE=10             # Beam size for molecular generation

echo "Starting vib2mol generation performance evaluation script..."
echo "---------------------------------------------------------"

# --- Evaluation for QM9s-IR (Infrared) ---
echo "Evaluating generation performance for QM9s-IR..."

SPECTRAL_TYPE="ir"
# Re-ranking model path for IR
RANK_MODEL_PATH="checkpoints/qm9s/ir-kekule_smiles/vib2mol/2025-07-08-02-55-db43f7/epoch999.pth" 
# Test model path for IR (without chemical formula)
TEST_MODEL_PATH="checkpoints/qm9s/ir-kekule_smiles/vib2mol/2025-07-10-01-45-46b8b6/epoch999.pth"

# Run inference for IR (without chemical formula)
python infer_lm.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE}" \
  --beam_size "${BEAM_SIZE}" \
  --test_model_path "${TEST_MODEL_PATH}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH}"

echo "Evaluating generation performance for QM9s-IR (with chemical formula)..."

# Test model path for IR (with chemical formula)
TEST_MODEL_PATH="checkpoints/qm9s/ir-kekule_smiles-formula/vib2mol/2025-07-12-02-02-4afc55/epoch999.pth"

# Run inference for IR (with chemical formula)
# Note: Using the same RANK_MODEL_PATH here. 
python infer_lm.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE}" \
  --beam_size "${BEAM_SIZE}" \
  --test_model_path "${TEST_MODEL_PATH}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH}" \
  --use_formula

echo "QM9s-IR evaluation completed."
echo "---------------------------------------------------------"


# --- Evaluation for QM9s-Raman ---
echo "Evaluating generation performance for QM9s-Raman..."
SPECTRAL_TYPE="raman"
# Test model path for Raman (without chemical formula)
TEST_MODEL_PATH="checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-07-07-51-af1552/epoch999.pth"
# Re-ranking model path for Raman
RANK_MODEL_PATH="checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-06-12-15-bbe117/epoch999.pth" 

# Run inference for Raman (without chemical formula)
python infer_lm.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE}" \
  --beam_size "${BEAM_SIZE}" \
  --test_model_path "${TEST_MODEL_PATH}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH}"

echo "Evaluating generation performance for QM9s-Raman (with chemical formula)..."

# Test model path for Raman (with chemical formula)
TEST_MODEL_PATH="checkpoints/qm9s/raman-kekule_smiles-formula/vib2mol/2025-07-12-02-01-7839c0/epoch999.pth"

# Run inference for Raman (with chemical formula)
# Note: Using the same RANK_MODEL_PATH here. 
python infer_lm.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE}" \
  --beam_size "${BEAM_SIZE}" \
  --test_model_path "${TEST_MODEL_PATH}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH}" \
  --use_formula

echo "QM9s-Raman evaluation completed."
echo "---------------------------------------------------------"

# --- Evaluation for QM9s-Raman-IR (with chemical formula) ---
echo "Evaluating generation performance for QM9s-Raman-IR (with chemical formula)..."
SPECTRAL_TYPE="ir-raman"
TEST_MODEL_PATH="checkpoints/qm9s/ir-raman-kekule_smiles-formula/vib2mol/2025-07-19-03-47-7c9f30/epoch999.pth"
RANK_MODEL_PATH="checkpoints/qm9s/ir-raman-kekule_smiles/vib2mol/2025-07-11-06-32-9005f4/epoch999.pth" 

python infer_lm.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE}" \
  --beam_size "${BEAM_SIZE}" \
  --test_model_path "${TEST_MODEL_PATH}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH}" \
  --use_formula

echo "QM9s-Raman-IR evaluation completed."
echo "---------------------------------------------------------"



# --- Evaluation for VB-mols dataset ---
# Reset common parameters for the new dataset 'mols'
MODEL_NAME="vib2mol" # Redefining for clarity, though it's the same value
DATASET="mols"
RERANK_ENABLED="--rerank" 
RERANK_TOPK=5
BEAM_SIZE=10

# echo "---------------------------------------------------------"
# echo "Starting evaluation for VB-mols dataset..."

# --- Evaluation for VB-mols-IR (Infrared) ---
echo "Evaluating generation performance for VB-mols-IR..."

SPECTRAL_TYPE="ir"
# Re-ranking model path for VB-mols IR
RANK_MODEL_PATH="checkpoints/mols/ir-kekule_smiles/vib2mol/2025-07-08-02-55-6ac6f2/epoch999.pth" 
# Test model path for VB-mols IR (without chemical formula)
TEST_MODEL_PATH="checkpoints/mols/ir-kekule_smiles/vib2mol/2025-07-10-07-42-dde12a/epoch999.pth"

# Run inference for VB-mols IR (without chemical formula)
python infer_lm.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE}" \
  --beam_size "${BEAM_SIZE}" \
  --test_model_path "${TEST_MODEL_PATH}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH}"

# echo "Evaluating generation performance for VB-mols-IR (with chemical formula)..."

# Test model path for VB-mols IR (with chemical formula)
TEST_MODEL_PATH="checkpoints/mols/ir-kekule_smiles-formula/vib2mol/2025-07-14-07-06-e44b51/epoch999.pth"

# Run inference for VB-mols IR (with chemical formula)
# Note: Using the same RANK_MODEL_PATH here. If a dedicated re-ranking model for formula-integrated data exists, please replace.
python infer_lm.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE}" \
  --beam_size "${BEAM_SIZE}" \
  --test_model_path "${TEST_MODEL_PATH}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH}" \
  --use_formula

# echo "VB-mols-IR evaluation completed."
# echo "---------------------------------------------------------"


# --- Evaluation for VB-mols-Raman ---
echo "Evaluating generation performance for VB-mols-Raman..."
SPECTRAL_TYPE="raman"
# Re-ranking model path for VB-mols Raman
RANK_MODEL_PATH="checkpoints/mols/raman-kekule_smiles/vib2mol/2025-07-08-02-56-69bd7e/epoch999.pth" 
# Test model path for VB-mols Raman (without chemical formula)
TEST_MODEL_PATH="checkpoints/mols/raman-kekule_smiles/vib2mol/2025-07-10-07-46-68d484/epoch999.pth"

# Run inference for VB-mols Raman (without chemical formula)
python infer_lm.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE}" \
  --beam_size "${BEAM_SIZE}" \
  --test_model_path "${TEST_MODEL_PATH}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH}"

echo "Evaluating generation performance for VB-mols-Raman (with chemical formula)..."

# Test model path for VB-mols Raman (with chemical formula)
TEST_MODEL_PATH="checkpoints/mols/raman-kekule_smiles-formula/vib2mol/2025-07-14-07-06-8a587e/epoch999.pth"

# Run inference for VB-mols Raman (with chemical formula)
# Note: Using the same RANK_MODEL_PATH here. If a dedicated re-ranking model for formula-integrated data exists, please replace.
python infer_lm.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE}" \
  --beam_size "${BEAM_SIZE}" \
  --test_model_path "${TEST_MODEL_PATH}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH}" \
  --use_formula

echo "VB-mols-Raman evaluation completed."
echo "---------------------------------------------------------"


# --- Evaluation for VB-mols-Raman-IR (with chemical formula) ---
echo "Evaluating generation performance for VB-mols-Raman-IR (with chemical formula)..."
SPECTRAL_TYPE="ir-raman"
TEST_MODEL_PATH="checkpoints/mols/ir-raman-kekule_smiles-formula/vib2mol/2025-07-14-07-07-e1c1f4/epoch999.pth"
RANK_MODEL_PATH="checkpoints/mols/ir-raman-kekule_smiles/vib2mol/2025-07-12-01-38-833d2d/epoch999.pth" 

python infer_lm.py \
  --model "${MODEL_NAME}" \
  --ds "${DATASET}" \
  --spectral_types "${SPECTRAL_TYPE}" \
  --beam_size "${BEAM_SIZE}" \
  --test_model_path "${TEST_MODEL_PATH}" \
  "${RERANK_ENABLED}" \
  --topk "${RERANK_TOPK}" \
  --rank_model_path "${RANK_MODEL_PATH}" \
  --use_formula

echo "VB-mols-Raman-IR evaluation completed."
echo "---------------------------------------------------------"