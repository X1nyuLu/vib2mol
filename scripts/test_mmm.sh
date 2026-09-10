#!/usr/bin/env bash
# Test the four bundled MMM checkpoints against existing full test datasets.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

if [[ $# -gt 1 ]]; then
  printf 'Usage: bash scripts/test_mmm.sh [all|qm9s|mols]\n' >&2
  exit 2
fi
case "${1:-all}" in
  all) datasets=(qm9s mols) ;;
  qm9s) datasets=(qm9s) ;;
  mols) datasets=(mols) ;;
  -h|--help)
    printf 'Usage: bash scripts/test_mmm.sh [all|qm9s|mols]\n'
    printf 'Overrides: VIB2MOL_PYTHON, VIB2MOL_BATCH_SIZE, VIB2MOL_DEVICE\n'
    exit 0
    ;;
  *)
    printf 'Choose all, qm9s, or mols.\n' >&2
    exit 2
    ;;
esac

vib2mol_python="${VIB2MOL_PYTHON:-python}"
vib2mol_batch_size="${VIB2MOL_BATCH_SIZE:-32}"
vib2mol_device="${VIB2MOL_DEVICE:-cuda:0}"

# Check all selected paths before starting any lengthy inference.
for dataset in "${datasets[@]}"; do
  for suite in retrieval_mmm generation_mmm; do
    "$vib2mol_python" scripts/run_inference.py "$suite" \
      --dataset "$dataset" --device "$vib2mol_device" \
      --batch-size "$vib2mol_batch_size" --dry-run
  done
done

mkdir -p test_outputs
output_dir="$(mktemp -d test_outputs/mmm.XXXXXX)"
printf 'Test output directory: %s\n' "$output_dir"

"$vib2mol_python" -m unittest discover -s scripts/tests -v \
  2>&1 | tee "$output_dir/unit_tests.log"

for dataset in "${datasets[@]}"; do
  for suite in retrieval_mmm generation_mmm; do
    "$vib2mol_python" -u scripts/run_inference.py "$suite" \
      --dataset "$dataset" --device "$vib2mol_device" \
      --batch-size "$vib2mol_batch_size" \
      2>&1 | tee "$output_dir/${suite}_${dataset}.log"
  done
done

printf 'All selected MMM evaluations completed. Logs: %s\n' "$output_dir"
