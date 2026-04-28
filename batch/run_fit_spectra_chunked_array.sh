#!/bin/bash
#SBATCH --job-name=m61_fit_chunk
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/home/tsingh65/m61-tng/logs/fit_spectra/chunk_%A_%a.out
#SBATCH --error=/home/tsingh65/m61-tng/logs/fit_spectra/chunk_%A_%a.err
#SBATCH --array=1-1%50   # OVERRIDE AT SUBMIT TIME, e.g. --array=1-${N_JOBS}%50

set -o pipefail

SNAP="99"
RUN_LABEL="L4Rvir"
BASE_DIR="/scratch/tsingh65/m61-tng/outputs"
OUTPUT_SUBDIR="fitted_spectra_snr10_bin3"

REPO="/home/tsingh65/m61-tng"
TASK_LIST="${REPO}/batch/fit_spectra_tasks_snap${SNAP}_${RUN_LABEL}.tsv"
LOGDIR="${REPO}/logs/fit_spectra"

# One SLURM array task processes this many one-spectrum task-list rows.
# Override at submit time with: sbatch --export=TASKS_PER_JOB=20 ...
TASKS_PER_JOB="${TASKS_PER_JOB:-25}"

mkdir -p "${LOGDIR}"

if [[ ! -d "${REPO}" ]]; then
  echo "FATAL: REPO not found: ${REPO}"
  exit 2
fi

if [[ ! -f "${TASK_LIST}" ]]; then
  echo "FATAL: TASK_LIST not found: ${TASK_LIST}"
  echo "Generate it with: python ${REPO}/scripts/make_fit_task_list.py"
  exit 2
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "FATAL: SLURM_ARRAY_TASK_ID is not set."
  exit 2
fi

if [[ ! "${TASKS_PER_JOB}" =~ ^[0-9]+$ || "${TASKS_PER_JOB}" -lt 1 ]]; then
  echo "FATAL: TASKS_PER_JOB must be a positive integer. Got: ${TASKS_PER_JOB}"
  exit 2
fi

N_TASKS="$(($(wc -l < "${TASK_LIST}") - 1))"
if [[ "${N_TASKS}" -lt 1 ]]; then
  echo "FATAL: No task rows found in ${TASK_LIST}"
  exit 2
fi

START_TASK="$(( (SLURM_ARRAY_TASK_ID - 1) * TASKS_PER_JOB + 1 ))"
END_TASK="$(( SLURM_ARRAY_TASK_ID * TASKS_PER_JOB ))"
if [[ "${START_TASK}" -gt "${N_TASKS}" ]]; then
  echo "[INFO] START_TASK=${START_TASK} exceeds N_TASKS=${N_TASKS}; nothing to do."
  exit 0
fi
if [[ "${END_TASK}" -gt "${N_TASKS}" ]]; then
  END_TASK="${N_TASKS}"
fi

export HDF5_USE_FILE_LOCKING=FALSE
export HDF5_DISABLE_VERSION_CHECK=2
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

module purge || exit 2
module load mamba || exit 2
eval "$(conda shell.bash hook)" || exit 2

export CONDA_NO_PLUGINS=true
conda activate trident || exit 2

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

cd "${REPO}" || exit 2

echo "=== FIT CHUNK JOB START ==="
date
echo "HOST: $(hostname)"
echo "PWD : $(pwd)"
echo "JOBID: ${SLURM_JOB_ID:-NA}"
echo "ARRAY_TASK: ${SLURM_ARRAY_TASK_ID}"
echo "TASK_LIST=${TASK_LIST}"
echo "TASKS_PER_JOB=${TASKS_PER_JOB}"
echo "N_TASKS=${N_TASKS}"
echo "TASK_RANGE=${START_TASK}-${END_TASK}"
echo "SNAP=${SNAP} RUN_LABEL=${RUN_LABEL}"
echo "BASE_DIR=${BASE_DIR}"
echo "OUTPUT_SUBDIR=${OUTPUT_SUBDIR}"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "python=$(command -v python)"
echo "==========================="

N_OK=0
N_FAIL=0

for TASK_INDEX in $(seq "${START_TASK}" "${END_TASK}"); do
  echo "=== RUN TASK_INDEX=${TASK_INDEX} ==="
  date

  python -u "${REPO}/scripts/fit_saved_spectra_batch.py" \
    --task-list "${TASK_LIST}" \
    --task-index "${TASK_INDEX}" \
    --snap "${SNAP}" \
    --run-label "${RUN_LABEL}" \
    --base-dir "${BASE_DIR}" \
    --output-subdir "${OUTPUT_SUBDIR}"

  RC="$?"
  if [[ "${RC}" -eq 0 ]]; then
    N_OK="$((N_OK + 1))"
    echo "=== DONE TASK_INDEX=${TASK_INDEX} RC=0 ==="
  else
    N_FAIL="$((N_FAIL + 1))"
    echo "=== FAILED TASK_INDEX=${TASK_INDEX} RC=${RC}; continuing chunk ==="
  fi
  date
done

echo "=== FIT CHUNK JOB END ==="
date
echo "TASK_RANGE=${START_TASK}-${END_TASK}"
echo "N_OK=${N_OK}"
echo "N_FAIL=${N_FAIL}"

if [[ "${N_FAIL}" -gt 0 ]]; then
  exit 1
fi

exit 0

# Submit examples:
#   N_TASKS=$(($(wc -l < /home/tsingh65/m61-tng/batch/fit_spectra_tasks_snap99_L4Rvir.tsv) - 1))
#   TASKS_PER_JOB=25
#   N_JOBS=$(((N_TASKS + TASKS_PER_JOB - 1) / TASKS_PER_JOB))
#   sbatch --array=1-4%4 /home/tsingh65/m61-tng/batch/run_fit_spectra_chunked_array.sh
#   sbatch --array=1-${N_JOBS}%50 /home/tsingh65/m61-tng/batch/run_fit_spectra_chunked_array.sh
#   sbatch --array=1-${N_JOBS}%100 /home/tsingh65/m61-tng/batch/run_fit_spectra_chunked_array.sh
#
# Production intentionally omits --make-plots. Use the one-spectrum script with
# --make-plots manually for diagnostic spot checks.
