#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash batch/run_spectra_single.sh <SID> [extra args forwarded]
#
# Example sanity check:
#   bash batch/run_spectra_single.sh 563732 --alpha-keep 0 --max-rays 4 --filter-mode noflip --no-plots --verbose

SID="${1:?Need SID as first argument}"
shift || true

REPO="$HOME/m61-tng"

CUTOUT_ROOT="/scratch/tsingh65/TNG50-1_snap99"
ORIENT_OUT_BASE="/scratch/tsingh65/m61-tng/outputs"
SPECTRA_OUT_BASE=""   # empty -> write under ORIENT_OUT_BASE/sid<SID>/

module purge
module load hwloc-2.9.3-gcc-11.2.0

# conda
conda deactivate 2>/dev/null || true
conda activate trident

export MPLBACKEND=Agg
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python -u "$REPO/notebooks/run_spectra_one_sid.py" \
  --sid "$SID" \
  --snap 99 \
  --cutout-root "$CUTOUT_ROOT" \
  --orient-out-base "$ORIENT_OUT_BASE" \
  ${SPECTRA_OUT_BASE:+--spectra-out-base "$SPECTRA_OUT_BASE"} \
  "$@"