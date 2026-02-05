#!/usr/bin/env bash
#SBATCH -J m61_spectra
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /scratch/tsingh65/m61-tng/logs/%x_%A_%a.out
#SBATCH -e /scratch/tsingh65/m61-tng/logs/%x_%A_%a.err

set -euo pipefail

REPO="$HOME/m61-tng"

CUTOUT_ROOT="/scratch/tsingh65/TNG50-1_snap99"
ORIENT_OUT_BASE="/scratch/tsingh65/m61-tng/outputs"

# Provide a plain text list of SIDs (one per line). If missing, auto-generate from matches CSV.
SID_LIST="$REPO/data/sids_topN.txt"
MATCHES_CSV="$REPO/data/m61_closest_matches_3d.csv"

mkdir -p /scratch/tsingh65/m61-tng/logs

if [[ ! -f "$SID_LIST" ]]; then
  python - <<'PY'
import pandas as pd, os
matches = os.environ["MATCHES_CSV"]
sid_list = os.environ["SID_LIST"]
df = pd.read_csv(matches)
# pick a reasonable SID column
for c in ["SubhaloID","subhalo_id","sid","SubfindID"]:
    if c in df.columns:
        col = c
        break
else:
    raise RuntimeError(f"No SubhaloID-like column in {matches}; columns={list(df.columns)}")
sids = df[col].dropna().astype(int).tolist()
with open(sid_list,"w") as f:
    for s in sids:
        f.write(f"{s}\n")
print("WROTE", sid_list, "N=", len(sids))
PY
fi

SID="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SID_LIST")"
if [[ -z "${SID}" ]]; then
  echo "No SID for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${SID_LIST}"
  exit 2
fi

module purge
module load hwloc-2.9.3-gcc-11.2.0

conda deactivate 2>/dev/null || true
conda activate trident

export MPLBACKEND=Agg
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Defaults here are "production-ish". Override via sbatch command line by exporting EXTRA_ARGS.
EXTRA_ARGS="${EXTRA_ARGS:---verbose}"

python -u "$REPO/notebooks/run_spectra_one_sid.py" \
  --sid "$SID" \
  --snap 99 \
  --cutout-root "$CUTOUT_ROOT" \
  --orient-out-base "$ORIENT_OUT_BASE" \
  $EXTRA_ARGS