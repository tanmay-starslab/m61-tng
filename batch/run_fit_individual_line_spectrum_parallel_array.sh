#!/bin/bash
#SBATCH --job-name=m61_fit_ray
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=10G
#SBATCH --time=01:30:00
#SBATCH --output=/home/tsingh65/m61-tng/logs/fit_individual_line_parallel/fit_ray_%A_%a.out
#SBATCH --error=/home/tsingh65/m61-tng/logs/fit_individual_line_parallel/fit_ray_%A_%a.err
#SBATCH --array=1-1%200   # OVERRIDE AT SUBMIT TIME

set -eo pipefail

SNAP="99"
RUN_LABEL="L2Rvir"
BASE_DIR="/scratch/tsingh65/m61-tng/outputs"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-fitted_individual_line_spectra_parallel_snr10_bin3}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-9}}"
USE_SID_INDEX="${USE_SID_INDEX:-0}"
SID="${SID:-}"
FIT_MODE="${FIT_MODE:-all}"
FIT_ALPHA="${FIT_ALPHA:-all}"

REPO="/home/tsingh65/m61-tng"
TASK_LIST="${TASK_LIST:-${REPO}/batch/fit_individual_line_spectra_tasks_snap${SNAP}_${RUN_LABEL}.tsv}"
LOGDIR="${REPO}/logs/fit_individual_line_parallel"

mkdir -p "${LOGDIR}"

if [[ "${USE_SID_INDEX}" != "1" && ! -f "${TASK_LIST}" ]]; then
  echo "FATAL: TASK_LIST not found: ${TASK_LIST}"
  echo "Generate it with: python ${REPO}/scripts/make_fit_individual_line_task_list.py --run-label ${RUN_LABEL}"
  exit 2
fi
if [[ "${USE_SID_INDEX}" == "1" && -z "${SID}" ]]; then
  echo "FATAL: USE_SID_INDEX=1 requires SID=<subhalo_id>."
  exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "FATAL: SLURM_ARRAY_TASK_ID is not set."
  exit 2
fi

export HDF5_USE_FILE_LOCKING=FALSE
export HDF5_DISABLE_VERSION_CHECK=2
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/m61_fit_parallel_mpl_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# Each worker is a process. Keep BLAS/OpenMP single-threaded to avoid oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module purge
module load mamba
eval "$(conda shell.bash hook)"
export CONDA_NO_PLUGINS=true
conda activate trident
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

cd "${REPO}"

echo "=== PARALLEL INDIVIDUAL-LINE FIT START ==="
date
echo "HOST=$(hostname)"
echo "JOBID=${SLURM_JOB_ID:-NA}"
echo "ARRAY_TASK=${SLURM_ARRAY_TASK_ID}"
echo "TASK_LIST=${TASK_LIST}"
echo "RUN_LABEL=${RUN_LABEL}"
echo "OUTPUT_SUBDIR=${OUTPUT_SUBDIR}"
echo "N_WORKERS=${N_WORKERS}"
echo "USE_SID_INDEX=${USE_SID_INDEX}"
echo "SID=${SID}"
echo "FIT_MODE=${FIT_MODE}"
echo "FIT_ALPHA=${FIT_ALPHA}"
echo "python=$(command -v python)"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "=========================================="

if [[ "${USE_SID_INDEX}" == "1" ]]; then
  python -u "${REPO}/scripts/fit_individual_line_spectrum_parallel.py" fit-sid-index \
    --sid "${SID}" \
    --spectrum-index "${SLURM_ARRAY_TASK_ID}" \
    --snap "${SNAP}" \
    --run-label "${RUN_LABEL}" \
    --base-dir "${BASE_DIR}" \
    --output-subdir "${OUTPUT_SUBDIR}" \
    --mode "${FIT_MODE}" \
    --alpha "${FIT_ALPHA}" \
    --n-workers "${N_WORKERS}"
else
  python -u "${REPO}/scripts/fit_individual_line_spectrum_parallel.py" fit-task \
    --task-list "${TASK_LIST}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --snap "${SNAP}" \
    --run-label "${RUN_LABEL}" \
    --base-dir "${BASE_DIR}" \
    --output-subdir "${OUTPUT_SUBDIR}" \
    --n-workers "${N_WORKERS}"
fi

echo "=== PARALLEL INDIVIDUAL-LINE FIT END ==="
date

# Suggested production submission after generating the task list:
#   python /home/tsingh65/m61-tng/scripts/make_fit_individual_line_task_list.py --run-label L2Rvir
#   N=$(($(wc -l < /home/tsingh65/m61-tng/batch/fit_individual_line_spectra_tasks_snap99_L2Rvir.tsv) - 1))
#   sbatch --array=1-${N}%200 /home/tsingh65/m61-tng/batch/run_fit_individual_line_spectrum_parallel_array.sh
