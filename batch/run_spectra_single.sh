#!/usr/bin/env bash
set -euo pipefail

SID="${1:?Need SID as first argument}"
shift || true

REPO="$HOME/m61-tng"

CUTOUT_ROOT="/scratch/tsingh65/TNG50-1_snap99"
ORIENT_OUT_BASE="/scratch/tsingh65/m61-tng/outputs"
SPECTRA_OUT_BASE=""   # empty -> write under ORIENT_OUT_BASE/sid<SID>/

module purge
module load hwloc-2.9.3-gcc-11.2.0

# --- conda init for non-interactive shells ---
if [[ -f "$HOME/.bashrc" ]]; then
  # safe to source; may contain module messages
  source "$HOME/.bashrc" >/dev/null 2>&1 || true
fi

# Prefer conda hook if available
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" >/dev/null 2>&1 || true
fi

# Fallback: source conda.sh directly (common on clusters)
if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "/home/tsingh65/.conda/etc/profile.d/conda.sh" ]]; then
    source "/home/tsingh65/.conda/etc/profile.d/conda.sh"
  else
    echo "FATAL: could not find conda.sh; conda not initializable in this shell."
    exit 2
  fi
fi

# DO NOT conda deactivate here (it is triggering libxml2_deactivate.sh 'unbound variable' under set -u)
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