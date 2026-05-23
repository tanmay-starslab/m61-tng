#!/bin/bash
#SBATCH --job-name=m61_spec_arr
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=1-22:00:00
#SBATCH --output=/scratch/tsingh65/m61-tng/logs/spec_arr_%A_%a.out
#SBATCH --error=/scratch/tsingh65/m61-tng/logs/spec_arr_%A_%a.err
#SBATCH --array=1-1   # <-- OVERRIDE AT SUBMIT TIME (see bottom)

# Notes:
# - Array maps SLURM_ARRAY_TASK_ID -> SID_LIST line number.
# - Auto-builds SID_LIST from /data/sborthak/m61/cutouts/out_sub_* directories.
# - By default, includes all discovered SIDs.
# - Optional skip-if-done logic can be enabled, but defaults to rerun everything.

set -eo pipefail   # intentionally NOT using -u globally (conda hooks + nounset == pain)

# -----------------------------
# CONFIG
# -----------------------------
SNAP="99"
RUN_LABELS="L2Rvir"         # supported: L2Rvir,L3Rvir,L4Rvir
FILTER_MODES=("noflip" "flip")

# Must match your local Trident line-name conventions (integer-label / aliases).
LINES_CSV="Si II 1190,Si II 1193,Si III 1206,N V 1239,Si II 1260,O I 1302,C II 1335,Si IV 1403,H I 1216"

REPO="/home/tsingh65/m61-tng"
PATCHED_TRIDENT_REPO="/home/tsingh65/src/trident"
CUTOUT_ROOT="/data/sborthak/m61/cutouts"
ORIENT_OUT_BASE="/scratch/tsingh65/m61-tng/outputs"

# Leave empty to include all discovered SIDs. Set a numeric SID to exclude one explicitly.
EXCLUDE_SID=""

# Where to store logs + generated SID list
GLOBAL_LOGDIR="/scratch/tsingh65/m61-tng/logs"
SID_LIST="${REPO}/data/sids_from_cutouts_snap${SNAP}.txt"
REBUILD_SID_LIST=1

# Skip work if already processed (default 0 = rerun all SIDs)
SKIP_IF_DONE=0

mkdir -p "${GLOBAL_LOGDIR}"

# -----------------------------
# PRECHECKS
# -----------------------------
if [[ ! -d "${REPO}" ]]; then
  echo "FATAL: REPO not found: ${REPO}"
  exit 2
fi
if [[ ! -f "${REPO}/notebooks/run_spectra_one_sid.py" ]]; then
  echo "FATAL: missing ${REPO}/notebooks/run_spectra_one_sid.py"
  exit 2
fi
if [[ ! -d "${CUTOUT_ROOT}" ]]; then
  echo "FATAL: CUTOUT_ROOT missing: ${CUTOUT_ROOT}"
  exit 2
fi

# -----------------------------
# BUILD SID LIST (FROM CUTOUT DIRECTORIES)
# -----------------------------
if [[ "${REBUILD_SID_LIST}" -eq 1 || ! -f "${SID_LIST}" ]]; then
  mkdir -p "$(dirname "${SID_LIST}")"
  SID_LIST_LOCK="${SID_LIST}.lock"

  if mkdir "${SID_LIST_LOCK}" 2>/dev/null; then
    trap 'rmdir "${SID_LIST_LOCK}" 2>/dev/null || true' EXIT
    echo "[INFO] Generating SID_LIST: ${SID_LIST}"
    SID_LIST_TMP="${SID_LIST}.tmp.${SLURM_JOB_ID:-manual}.${SLURM_ARRAY_TASK_ID:-0}"

    # Find directories like: /data/sborthak/m61/cutouts/out_sub_488530
    # Extract numeric SID and sort unique.
    find "${CUTOUT_ROOT}" -maxdepth 1 -type d -name "out_sub_*" -printf "%f\n" \
      | sed -E 's/^out_sub_([0-9]+)$/\1/' \
      | awk '/^[0-9]+$/' \
      | sort -n -u > "${SID_LIST_TMP}"

    mv "${SID_LIST_TMP}" "${SID_LIST}"
    echo "[INFO] WROTE ${SID_LIST}  N=$(wc -l < "${SID_LIST}")"
    rmdir "${SID_LIST_LOCK}" 2>/dev/null || true
    trap - EXIT
  else
    echo "[INFO] Waiting for SID_LIST rebuild lock: ${SID_LIST_LOCK}"
    for _wait_i in $(seq 1 150); do
      if [[ ! -d "${SID_LIST_LOCK}" && -s "${SID_LIST}" ]]; then
        break
      fi
      sleep 2
    done
    if [[ -d "${SID_LIST_LOCK}" || ! -s "${SID_LIST}" ]]; then
      echo "FATAL: timed out waiting for SID_LIST rebuild: ${SID_LIST}"
      exit 2
    fi
    echo "[INFO] Reusing rebuilt SID_LIST: ${SID_LIST}  N=$(wc -l < "${SID_LIST}")"
  fi
fi

# -----------------------------
# MAP ARRAY INDEX -> SID
# -----------------------------
SID="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SID_LIST}" | tr -d '[:space:]')"
if [[ -z "${SID}" || ! "${SID}" =~ ^[0-9]+$ ]]; then
  echo "[WARN] Empty/invalid SID for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; skipping."
  exit 0
fi

if [[ -n "${EXCLUDE_SID}" && "${SID}" == "${EXCLUDE_SID}" ]]; then
  echo "[INFO] SID=${SID} matched EXCLUDE_SID=${EXCLUDE_SID}. Skipping by explicit request."
  exit 0
fi

# -----------------------------
# PER-SID PATHS
# -----------------------------
OUT_SID_DIR="${ORIENT_OUT_BASE}/sid${SID}"
LOGDIR_SID="${OUT_SID_DIR}/logs_sbatch"
mkdir -p "${LOGDIR_SID}"

CUTOUT_H5="${CUTOUT_ROOT}/out_sub_${SID}/cutout_ALLFIELDS_sphere_2p1Rvir_sub${SID}.hdf5"
if [[ ! -f "${CUTOUT_H5}" ]]; then
  echo "[ERROR] Missing cutout for SID=${SID}: ${CUTOUT_H5}"
  exit 1
fi

# Optional: ensure orient outputs exist (rays CSV). If missing, skip.
# (Your python will error anyway; skipping here avoids wasting queue time.)
IFS=',' read -r -a RUN_LABEL_ARR <<< "${RUN_LABELS}"
for RL in "${RUN_LABEL_ARR[@]}"; do
  RAYS_CSV="${OUT_SID_DIR}/rays_and_recipes_sid${SID}_snap${SNAP}_${RL}/rays_sid${SID}.csv"
  if [[ ! -f "${RAYS_CSV}" ]]; then
    echo "[WARN] Missing rays CSV for SID=${SID} RUN_LABEL=${RL}: ${RAYS_CSV}"
    echo "[WARN] Continuing anyway so this SID is still attempted."
  fi
done

# Optional: skip if already processed (summary csv exists for each run-label + both modes)
if [[ "${SKIP_IF_DONE}" -eq 1 ]]; then
  all_done=1
  for RL in "${RUN_LABEL_ARR[@]}"; do
    for MODE in "${FILTER_MODES[@]}"; do
      # Your module writes: <output_base>/rays_and_spectra_sid<SID>_snap<SNAP>_<RUN_LABEL>/summary_all_rays.csv
      # and it's mode-filtered per run in the same dir. We treat existence as "done enough".
      SUMMARY="${OUT_SID_DIR}/rays_and_spectra_sid${SID}_snap${SNAP}_${RL}/summary_all_rays.csv"
      if [[ ! -s "${SUMMARY}" ]]; then
        all_done=0
      fi
    done
  done
  if [[ "${all_done}" -eq 1 ]]; then
    echo "[INFO] SID=${SID} already has summary_all_rays.csv for all run-labels; skipping."
    exit 0
  fi
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

# Trident ray scratch (your python prefers SLURM_TMPDIR; this keeps it explicit)
export TRIDENT_RAY_TMP="${SLURM_TMPDIR:-${OUT_SID_DIR}/_tmp_trident}"
mkdir -p "${TRIDENT_RAY_TMP}"

# -----------------------------
# ENV SETUP (SOL)
# -----------------------------
module purge
module load mamba
eval "$(conda shell.bash hook)"

# disable conda plugins / activate.d/deactivate.d hooks for this job
export CONDA_NO_PLUGINS=true

conda activate trident

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${PATCHED_TRIDENT_REPO}:${PYTHONPATH:-}"

# -----------------------------
# LOG HEADER
# -----------------------------
echo "=== JOB START ==="
date
echo "HOST: $(hostname)"
echo "PWD : $(pwd)"
echo "JOBID: ${SLURM_JOB_ID:-NA}"
echo "ARRAY_TASK: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "SID=${SID} SNAP=${SNAP} RUN_LABELS=${RUN_LABELS}"
echo "CUTOUT_H5=${CUTOUT_H5}"
echo "REPO=${REPO}"
echo "PATCHED_TRIDENT_REPO=${PATCHED_TRIDENT_REPO}"
echo "ORIENT_OUT_BASE=${ORIENT_OUT_BASE}"
echo "LINES=${LINES_CSV}"
echo "DOPPLER_ONLY_REDSHIFT=True"
echo "TRIDENT_RAY_TMP=${TRIDENT_RAY_TMP}"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "python=$(command -v python)"
echo "================="

python - <<'PY'
import inspect
import trident, yt
print("trident:", getattr(trident, "__version__", "unknown"))
print("trident_file:", getattr(trident, "__file__", "unknown"))
sig = inspect.signature(trident.SpectrumGenerator.make_spectrum)
print("supports_use_doppler_redshift_only:", "use_doppler_redshift_only" in sig.parameters)
print("use_doppler_redshift_only_requested:", True)
print("yt:", yt.__version__)
PY

# Also tee a per-SID logfile (in addition to SLURM %A_%a logs)
RUNLOG="${LOGDIR_SID}/spec_sid${SID}_job${SLURM_JOB_ID:-NA}_task${SLURM_ARRAY_TASK_ID:-NA}.log"
exec > >(tee -a "${RUNLOG}") 2>&1

# -----------------------------
# RUN BOTH MODES
# -----------------------------
for MODE in "${FILTER_MODES[@]}"; do
  echo "=== RUN MODE=${MODE} SID=${SID} ==="
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

  echo "=== DONE MODE=${MODE} SID=${SID} ==="
  date
done

echo "=== JOB END SID=${SID} ==="
date

# -----------------------------
# SUBMIT NOTE (manual):
#   N=$(wc -l < /home/tsingh65/m61-tng/data/sids_from_cutouts_snap99.txt)
#   sbatch --array=1-$N this_file.sh
# -----------------------------
