#!/bin/bash
#SBATCH --job-name=m61_fit_spec
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/home/tsingh65/m61-tng/logs/fit_spectra/fit_%A_%a.out
#SBATCH --error=/home/tsingh65/m61-tng/logs/fit_spectra/fit_%A_%a.err
#SBATCH --array=1-1%100   # OVERRIDE AT SUBMIT TIME, e.g. --array=1-1000%100

set -eo pipefail

SNAP="99"
RUN_LABEL="L4Rvir"
BASE_DIR="/scratch/tsingh65/m61-tng/outputs"
OUTPUT_SUBDIR="fitted_spectra_snr10_bin3"

REPO="/home/tsingh65/m61-tng"
TASK_LIST="${REPO}/batch/fit_spectra_tasks_snap${SNAP}_${RUN_LABEL}.tsv"
LOGDIR="${REPO}/logs/fit_spectra"

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

export HDF5_USE_FILE_LOCKING=FALSE
export HDF5_DISABLE_VERSION_CHECK=2
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

module purge
module load mamba
eval "$(conda shell.bash hook)"

export CONDA_NO_PLUGINS=true
conda activate trident

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

cd "${REPO}"

echo "=== FIT JOB START ==="
date
echo "HOST: $(hostname)"
echo "PWD : $(pwd)"
echo "JOBID: ${SLURM_JOB_ID:-NA}"
echo "ARRAY_TASK: ${SLURM_ARRAY_TASK_ID}"
echo "TASK_LIST=${TASK_LIST}"
echo "SNAP=${SNAP} RUN_LABEL=${RUN_LABEL}"
echo "BASE_DIR=${BASE_DIR}"
echo "OUTPUT_SUBDIR=${OUTPUT_SUBDIR}"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "python=$(command -v python)"
echo "====================="

python -u "${REPO}/scripts/fit_saved_spectra_batch.py" \
  --task-list "${TASK_LIST}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --snap "${SNAP}" \
  --run-label "${RUN_LABEL}" \
  --base-dir "${BASE_DIR}" \
  --output-subdir "${OUTPUT_SUBDIR}"

echo "=== FIT JOB END ==="
date

# Submit examples:
#   N=$(($(wc -l < /home/tsingh65/m61-tng/batch/fit_spectra_tasks_snap99_L4Rvir.tsv) - 1))
#   sbatch --array=1-5%5 /home/tsingh65/m61-tng/batch/run_fit_spectra_array.sh
#   sbatch --array=1-${N}%100 /home/tsingh65/m61-tng/batch/run_fit_spectra_array.sh
