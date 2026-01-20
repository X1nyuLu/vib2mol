# =================================================================
# Author:      Xinyu Lu (Translated to PowerShell)
# Date:        2025-09-03
# Description: This script evaluates the de novo generation
#              performance of the vib2mol model on various datasets.
# =================================================================

# Stop script if any command fails
$ErrorActionPreference = "Stop"

# --- Default Parameters ---
$MODEL_NAME = "vib2mol"
$RERANK_ENABLED = $false
$RERANK_TOPK = 5
$BEAM_SIZE = 10
$USE_FORMULA_ENABLED = $false

# --- Parse Command-Line Arguments ---
foreach ($arg in $args) {
    if ($arg -eq "--rerank") {
        $RERANK_ENABLED = $true
    }
    elseif ($arg -eq "--use_formula") {
        $USE_FORMULA_ENABLED = $true
    }
    elseif ($arg -like "--topk=*") {
        $RERANK_TOPK = $arg.Split('=')[1]
    }
    elseif ($arg -like "--beam_size=*") {
        $BEAM_SIZE = $arg.Split('=')[1]
    }
}

# --- Function to run a single evaluation ---
function Run-GenerationEvaluation {
    param (
        [string]$Dataset,
        [string]$SpectralType,
        [string]$TestModelPath,
        [string]$RankModelPath
    )

    Write-Host "`nEvaluating generation performance for $Dataset-$SpectralType..." -ForegroundColor Cyan

    # Build Argument List for Python
    $pythonArgs = @(
        "infer_generation.py",
        "--model", $MODEL_NAME,
        "--ds", $Dataset,
        "--spectral_types", $SpectralType,
        "--beam_size", $BEAM_SIZE,
        "--test_model_path", $TestModelPath
    )

    # Add reranking parameters if enabled
    if ($RERANK_ENABLED) {
        $pythonArgs += "--rerank"
        $pythonArgs += "--topk"
        $pythonArgs += $RERANK_TOPK
        $pythonArgs += "--rank_model_path"
        $pythonArgs += $RankModelPath
    }

    # Add chemical formula flag if enabled
    if ($USE_FORMULA_ENABLED) {
        $pythonArgs += "--use_formula"
    }

    # Execute Python
    Start-Process -FilePath "python" -ArgumentList $pythonArgs -Wait -NoNewWindow
}

# --- Main Script Execution ---
Write-Host "Starting vib2mol generation performance evaluation script..."
Write-Host "---------------------------------------------------------"

# Define test configurations using an array of objects for better readability in PowerShell
$configs = @(
    # --- QM9s Dataset ---
    @{ ds="qm9s"; type="ir"; test="checkpoints/qm9s/ir-kekule_smiles/vib2mol/2025-07-10-01-45-46b8b6/epoch999.pth"; rank="checkpoints/qm9s/ir-kekule_smiles/vib2mol/2025-07-08-02-55-db43f7/epoch999.pth" },
    @{ ds="qm9s"; type="ir"; test="checkpoints/qm9s/ir-kekule_smiles-formula/vib2mol/2025-07-12-02-02-4afc55/epoch999.pth"; rank="checkpoints/qm9s/ir-kekule_smiles/vib2mol/2025-07-08-02-55-db43f7/epoch999.pth" },
    @{ ds="qm9s"; type="raman"; test="checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-07-07-51-af1552/epoch999.pth"; rank="checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-06-12-15-bbe117/epoch999.pth" },
    @{ ds="qm9s"; type="raman"; test="checkpoints/qm9s/raman-kekule_smiles-formula/vib2mol/2025-07-12-02-01-7839c0/epoch999.pth"; rank="checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-06-12-15-bbe117/epoch999.pth" },
    @{ ds="qm9s"; type="raman-ir"; test="checkpoints/qm9s/ir-raman-kekule_smiles-formula/vib2mol/2025-07-19-03-47-7c9f30/epoch999.pth"; rank="checkpoints/qm9s/ir-raman-kekule_smiles/vib2mol/2025-07-11-06-32-9005f4/epoch999.pth" },

    # --- VB-mols Dataset ---
    @{ ds="mols"; type="ir"; test="checkpoints/mols/ir-kekule_smiles/vib2mol/2025-07-10-07-42-dde12a/epoch999.pth"; rank="checkpoints/mols/ir-kekule_smiles/vib2mol/2025-07-08-02-55-6ac6f2/epoch999.pth" },
    @{ ds="mols"; type="ir"; test="checkpoints/mols/ir-kekule_smiles-formula/vib2mol/2025-07-14-07-06-e44b51/epoch999.pth"; rank="checkpoints/mols/ir-kekule_smiles/vib2mol/2025-07-08-02-55-6ac6f2/epoch999.pth" },
    @{ ds="mols"; type="raman"; test="checkpoints/mols/raman-kekule_smiles/vib2mol/2025-07-10-07-46-68d484/epoch999.pth"; rank="checkpoints/mols/raman-kekule_smiles/vib2mol/2025-07-08-02-56-69bd7e/epoch999.pth" },
    @{ ds="mols"; type="raman"; test="checkpoints/mols/raman-kekule_smiles-formula/vib2mol/2025-07-14-07-06-8a587e/epoch999.pth"; rank="checkpoints/mols/raman-kekule_smiles/vib2mol/2025-07-08-02-56-69bd7e/epoch999.pth" },
    @{ ds="mols"; type="ir-raman"; test="checkpoints/mols/ir-raman-kekule_smiles-formula/vib2mol/2025-07-14-07-07-e1c1f4/epoch999.pth"; rank="checkpoints/mols/ir-raman-kekule_smiles/vib2mol/2025-07-12-01-38-833d2d/epoch999.pth" }
)

$counter = 1
$totalTests = $configs.Count

foreach ($config in $configs) {
    Write-Host "`n--- Running Test #$counter of $totalTests ---" -ForegroundColor Yellow
    
    Run-GenerationEvaluation `
        -Dataset $config.ds `
        -SpectralType $config.type `
        -TestModelPath $config.test `
        -RankModelPath $config.rank
    
    Write-Host "$($config.ds)-$($config.type) evaluation completed."
    Write-Host "---------------------------------------------------------"
    $counter++
}

Write-Host "`nAll evaluations completed." -ForegroundColor Green
Write-Host "---------------------------------------------------------"