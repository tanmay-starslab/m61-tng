#!/bin/bash
#SBATCH --chdir=/scratch/tsingh65/m61-tng/outputs
#SBATCH --output=/scratch/tsingh65/m61-tng/outputs/logs/spec_arr_%A_%a.out
#SBATCH --error=/scratch/tsingh65/m61-tng/outputs/logs/spec_arr_%A_%a.err
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=2-18:00:00
#SBATCH --array=1-20  # set at submission time to topn

set -euo pipefail

REPO="/home/tsingh65/m61-tng"
OUT_BASE="/scratch/tsingh65/m61-tng/outputs"
CUTOUT_ROOT="/scratch/tsingh65/TNG50-1_snap99"
MATCHES_CSV="/home/tsingh65/m61-tng/data/m61_closest_matches_3d.csv"
export MATCHES_CSV

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

mkdir -p "${OUT_BASE}/logs"

SID="$(python - <<'PY'
import pandas as pd, os
matches = os.environ["MATCHES_CSV"]
i = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
df = pd.read_csv(matches)
for c in ["SubhaloID","subhalo_id","sid","SubfindID"]:
    if c in df.columns:
        sid_col = c
        break
else:
    sid_col = df.columns[0]
print(int(df.loc[i, sid_col]))
PY
)"

export TRIDENT_RAY_TMP="${SLURM_TMPDIR:-${OUT_BASE}/sid${SID}/_tmp_trident}"
mkdir -p "$TRIDENT_RAY_TMP"

cd "$REPO"

python -u "${REPO}/notebooks/run_spectra_one_sid.py" \
  --sid "$SID" \
  --snap 99 \
  --out-base "$OUT_BASE" \
  --cutout-root "$CUTOUT_ROOT" \
  --verbose
