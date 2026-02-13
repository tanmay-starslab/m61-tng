#!/bin/bash
#SBATCH --job-name=spec_sid488530
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/tsingh65/m61-tng/outputs/sid488530/logs_sbatch/spec_%j.out
#SBATCH --error=/scratch/tsingh65/m61-tng/outputs/sid488530/logs_sbatch/spec_%j.err

# NOTE:
# The libxml2_deactivate.sh "unbound variable" happens because:
#   set -u  + conda's activate/deactivate scripts + unset internal vars
# Solution:
#   keep -e and pipefail, but disable nounset only for conda activation.
#   also disable conda "activate.d/deactivate.d" scripts for this job (they are not needed for trident),
#   which completely removes the libxml2_deactivate failure mode.

set -eo pipefail   # intentionally NOT using -u globally

# -----------------------------
# CONFIG
# -----------------------------
SID="488530"
SNAP="99"
RUN_LABELS="L4Rvir"

REPO="/home/tsingh65/m61-tng"
CUTOUT_ROOT="/scratch/tsingh65/TNG50-1_snap99"
ORIENT_OUT_BASE="/scratch/tsingh65/m61-tng/outputs"

FILTER_MODES=("noflip" "flip")

# must match Trident DB (your dump confirms these exact labels exist)
LINES_CSV="Si II 1190,Si II 1193,Si III 1206,N V 1239,Si II 1260,O I 1302,C II 1335,Si IV,H I 1216"

# -----------------------------
# PATHS / PRECHECKS
# -----------------------------
OUT_SID_DIR="${ORIENT_OUT_BASE}/sid${SID}"
LOGDIR="${OUT_SID_DIR}/logs_sbatch"
mkdir -p "${LOGDIR}"

CUTOUT_H5="${CUTOUT_ROOT}/out_sub_${SID}/cutout_ALLFIELDS_sphere_2p1Rvir_sub${SID}.hdf5"

echo "=== JOB START ==="
date
echo "HOST: $(hostname)"
echo "PWD : $(pwd)"
echo "JOBID: ${SLURM_JOB_ID:-NA}"
echo "SID=${SID} SNAP=${SNAP} RUN_LABELS=${RUN_LABELS}"
echo "CUTOUT_H5=${CUTOUT_H5}"
echo "REPO=${REPO}"
echo "ORIENT_OUT_BASE=${ORIENT_OUT_BASE}"
echo "LINES=${LINES_CSV}"
echo "================="

if [[ ! -d "${REPO}" ]]; then
  echo "FATAL: REPO not found: ${REPO}"
  exit 2
fi
if [[ ! -f "${REPO}/notebooks/run_spectra_one_sid.py" ]]; then
  echo "FATAL: missing ${REPO}/notebooks/run_spectra_one_sid.py"
  exit 2
fi
if [[ ! -f "${CUTOUT_H5}" ]]; then
  echo "FATAL: missing cutout: ${CUTOUT_H5}"
  exit 2
fi
if [[ ! -d "${OUT_SID_DIR}" ]]; then
  echo "FATAL: rays base dir missing (orient outputs not found): ${OUT_SID_DIR}"
  echo "Expected rays CSV under: ${OUT_SID_DIR}/rays_and_recipes_sid${SID}_snap${SNAP}_${RUN_LABELS}/rays_sid${SID}.csv"
  exit 2
fi

# -----------------------------
# THREAD / HDF5 STABILITY
# -----------------------------
export HDF5_USE_FILE_LOCKING=FALSE
export HDF5_DISABLE_VERSION_CHECK=2
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# -----------------------------
# ENV SETUP (SOL)
# Keep it like your older working job:
#   module load mamba
#   eval "$(conda shell.bash hook)"
#   conda activate trident
#
# The only extra hardening:
#   disable activate.d/deactivate.d scripts for this job to avoid libxml2 hooks entirely.
# -----------------------------
module purge
module load mamba

eval "$(conda shell.bash hook)"

# prevent conda from running env activation/deactivation scripts (where your error comes from)
export CONDA_NO_PLUGINS=true

conda activate trident

echo "[ENV] CONDA_PREFIX=${CONDA_PREFIX}"
echo "[ENV] python=$(command -v python)"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# sanity imports (if these fail, you will see it in .err)
python - <<'PY'
import trident, yt
print("trident:", getattr(trident, "__version__", "unknown"))
print("yt:", yt.__version__)
PY

# -----------------------------
# RUN BOTH MODES
# -----------------------------
for MODE in "${FILTER_MODES[@]}"; do
  echo "=== RUN MODE=${MODE} ==="
  date

  python -u "${REPO}/notebooks/run_spectra_one_sid.py" \
    --sid "${SID}" \
    --snap "${SNAP}" \
    --cutout-root "${CUTOUT_ROOT}" \
    --orient-out-base "${ORIENT_OUT_BASE}" \
    --run-labels "${RUN_LABELS}" \
    --filter-mode "${MODE}" \
    --lines "${LINES_CSV}" \
    --no-plots \
    --verbose

  echo "=== DONE MODE=${MODE} ==="
  date
done

echo "=== JOB END ==="
date