# =================================================================
# Author:      Xinyu Lu (Translated to PowerShell)
# Date:        2025-09-03
# Description: This script evaluates the retrieval performance of
#              the vib2mol model for different spectral types on
#              various datasets. It performs both initial retrieval
#              and re-ranking.
# =================================================================

# Exit immediately if a command fails
$ErrorActionPreference = "Stop"

# --- Common Parameters ---
$MODEL_NAME = "vib2mol"
$RERANK_TOPK = 5            # Number of top-k results to consider for re-ranking

# --- Parse Command-Line Arguments ---
$FORMULA_ENABLED = $false
$RERANK_ENABLED = $false

foreach ($arg in $args) {
    if ($arg -eq "--use_formula") {
        $FORMULA_ENABLED = $true
    }
    elseif ($arg -eq "--rerank") {
        $RERANK_ENABLED = $true
    }
}

# --- Function to run a single evaluation ---
function Run-Evaluation {
    param (
        [string]$Dataset,
        [string]$SpectralType,
        [string]$TestModelPath
    )

    # Construct the argument list for Python
    # Using an array for arguments is cleaner and safer than string concatenation in PowerShell
    $pythonArgs = @(
        "infer_retrieval.py",
        "--model", $MODEL_NAME,
        "--ds", $Dataset,
        "--spectral_types", $SpectralType,
        "--test_model_path", $TestModelPath
    )

    if ($RERANK_ENABLED) {
        $pythonArgs += "-rerank"
        $pythonArgs += "--topk"
        $pythonArgs += $RERANK_TOPK
        $pythonArgs += "--rank_model_path"
        $pythonArgs += $TestModelPath
    }

    if ($FORMULA_ENABLED) {
        $pythonArgs += "--use_formula"
    }

    # Execute the python script
    # We use & to call the operator and pass the array of arguments
    & python $pythonArgs
}

# --- Main Script Execution ---
Write-Host "Starting evaluation script for vib2mol retrieval performance..."
Write-Host "---------------------------------------------------------"

# Define test configurations using an array of objects for better readability
$configs = @(
    # Dataset QM9s
    @{ ds="qm9s"; type="ir"; path="checkpoints/qm9s/ir-kekule_smiles/vib2mol/2025-07-08-02-55-db43f7/epoch999.pth" },
    @{ ds="qm9s"; type="raman"; path="checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-06-12-15-bbe117/epoch999.pth" },
    @{ ds="qm9s"; type="raman-ir"; path="checkpoints/qm9s/ir-raman-kekule_smiles/vib2mol/2025-07-11-06-32-9005f4/epoch999.pth" },
    
    # Dataset VB-mols
    @{ ds="mols"; type="ir"; path="checkpoints/mols/ir-kekule_smiles/vib2mol/2025-07-08-02-55-6ac6f2/epoch999.pth" },
    @{ ds="mols"; type="raman"; path="checkpoints/mols/raman-kekule_smiles/vib2mol/2025-07-08-02-56-69bd7e/epoch999.pth" },
    @{ ds="mols"; type="raman-ir"; path="checkpoints/mols/ir-raman-kekule_smiles/vib2mol/2025-07-12-01-38-833d2d/epoch999.pth" },
    
    # Dataset SDBS
    @{ ds="sdbs"; type="ir"; path="checkpoints/sdbs/ir-kekule_smiles/vib2mol/2025-07-16-15-32-ea6ef7/epoch999.pth" },
    @{ ds="sdbs"; type="raman"; path="checkpoints/sdbs/raman-kekule_smiles/vib2mol/2025-07-16-15-33-d68938/epoch999.pth" },
    @{ ds="sdbs"; type="raman-ir"; path="checkpoints/sdbs/ir-raman-kekule_smiles/vib2mol/2025-07-16-15-24-9a8d24/epoch999.pth" },

    # Dataset NIST
    @{ ds="nist"; type="ir"; path="checkpoints/nist/ir-kekule_smiles/vib2mol/2025-07-16-16-03-187b97/epoch999.pth" }
)

# Initialize a counter for the tests
$counter = 1
$total_tests = $configs.Count

# Iterate through the configurations and run the evaluation function
foreach ($config in $configs) {
    Write-Host ""
    Write-Host "--- Running Test #$counter of $total_tests ---"
    
    $DATASET = $config.ds
    $SPECTRAL_TYPE = $config.type
    $TEST_MODEL_PATH = $config.path

    Write-Host "Evaluating retrieval performance for ${DATASET}-${SPECTRAL_TYPE}..."
    Run-Evaluation -Dataset $DATASET -SpectralType $SPECTRAL_TYPE -TestModelPath $TEST_MODEL_PATH
    
    Write-Host "${DATASET}-${SPECTRAL_TYPE} evaluation completed."
    Write-Host "---------------------------------------------------------"

    $counter++
}

Write-Host "All evaluations completed."
Write-Host "---------------------------------------------------------"