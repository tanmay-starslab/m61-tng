#!/bin/bash
#SBATCH --job-name=spec_one
#SBATCH --output=spec_one_%j.out
#SBATCH --error=spec_one_%j.err
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=04:00:00

set -euo pipefail

SID="${1:?Usage: sbatch run_spectra_single.sh <SID>}"

REPO="/home/tsingh65/m61-tng"
OUT_BASE="/scratch/tsingh65/m61-tng/outputs"
CUTOUT_ROOT="/scratch/tsingh65/TNG50-1_snap99"

module purge
module load hwloc-2.9.3-gcc-11.2.0 2>/dev/null || true
module load mamba
eval "$(conda shell.bash hook)"
conda activate trident

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CONDA_PREFIX/bin:$PATH"

export HDF5_USE_FILE_LOCKING=FALSE
export HDF5_DISABLE_VERSION_CHECK=2
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export PYTHONPATH="${REPO}/notebooks:${PYTHONPATH:-}"

export TRIDENT_RAY_TMP="${SLURM_TMPDIR:-${OUT_BASE}/sid${SID}/_tmp_trident}"
mkdir -p "$TRIDENT_RAY_TMP"

cd "$REPO"

python -u "${REPO}/notebooks/run_spectra_one_sid.py" \
  --sid "$SID" \
  --snap 99 \
  --out-base "$OUT_BASE" \
  --cutout-root "$CUTOUT_ROOT" \
  --verbose
