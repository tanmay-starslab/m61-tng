#!/bin/bash
#SBATCH --job-name=ism_slit
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/tsingh65/m61-tng/outputs/logs_sbatch/ism_slit_%A_%a.out
#SBATCH --error=/scratch/tsingh65/m61-tng/outputs/logs_sbatch/ism_slit_%A_%a.err

set -eo pipefail

REPO="${REPO:-/home/tsingh65/m61-tng}"
SCRIPT="${REPO}/scripts/stellar_ism_velocity_slit_profiles.py"
SID_LIST="${SID_LIST:-${REPO}/data/sids_from_cutouts_snap99.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/tsingh65/m61-tng/outputs}"
CUTOUT_ROOT="${CUTOUT_ROOT:-/data/sborthak/m61/cutouts}"
SNAP="${SNAP:-99}"
RUN_LABEL="${RUN_LABEL:-L2Rvir}"
SIGHTLINE_ID="${SIGHTLINE_ID:-J122138+043026}"
ALPHAS="${ALPHAS:-all}"
MODES="${MODES:-noflip,flip}"
S_MIN_KPC="${S_MIN_KPC:--60}"
S_MAX_KPC="${S_MAX_KPC:-60}"
N_BINS="${N_BINS:-1000}"
SLIT_WIDTH_KPC="${SLIT_WIDTH_KPC:-2}"
MAKE_PLOTS="${MAKE_PLOTS:-0}"
SAVE_CSV="${SAVE_CSV:-0}"
SAVE_JSON="${SAVE_JSON:-0}"
SAVE_SUMMARY_CSV="${SAVE_SUMMARY_CSV:-0}"
MAP_WIDTH_KPC="${MAP_WIDTH_KPC:-120}"
MAP_NPIX="${MAP_NPIX:-500}"

mkdir -p "${OUTPUT_ROOT}/logs_sbatch"

export HDF5_USE_FILE_LOCKING=FALSE
export HDF5_DISABLE_VERSION_CHECK=2
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/m61_matplotlib_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
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

SID="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SID_LIST}" | awk '{print $1}')"
if [[ -z "${SID}" ]]; then
  echo "No SID found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${SID_LIST}" >&2
  exit 2
fi

cd "${REPO}"

echo "=== STELLAR ISM VELOCITY SLIT START ==="
date
echo "HOST=$(hostname)"
echo "JOBID=${SLURM_JOB_ID:-NA}"
echo "TASK=${SLURM_ARRAY_TASK_ID:-NA}"
echo "SID=${SID}"
echo "RUN_LABEL=${RUN_LABEL}"
echo "ALPHAS=${ALPHAS}"
echo "MODES=${MODES}"
echo "SIGNED_SLIT=${S_MIN_KPC}..${S_MAX_KPC} kpc; N_BINS=${N_BINS}; WIDTH=${SLIT_WIDTH_KPC} kpc"
echo "python=$(command -v python)"
echo "======================================="

PLOT_ARG=()
if [[ "${MAKE_PLOTS}" == "1" ]]; then
  PLOT_ARG=(--make-plots)
fi
CSV_ARG=()
if [[ "${SAVE_CSV}" == "1" ]]; then
  CSV_ARG=(--save-csv)
fi
JSON_ARG=()
if [[ "${SAVE_JSON}" == "1" ]]; then
  JSON_ARG=(--save-json)
fi
SUMMARY_CSV_ARG=()
if [[ "${SAVE_SUMMARY_CSV}" == "1" ]]; then
  SUMMARY_CSV_ARG=(--save-summary-csv)
fi

python -u "${SCRIPT}" \
  --sid "${SID}" \
  --snap "${SNAP}" \
  --run-label "${RUN_LABEL}" \
  --sightline-id "${SIGHTLINE_ID}" \
  --alphas "${ALPHAS}" \
  --modes "${MODES}" \
  --output-root "${OUTPUT_ROOT}" \
  --cutout-root "${CUTOUT_ROOT}" \
  --s-min-kpc "${S_MIN_KPC}" \
  --s-max-kpc "${S_MAX_KPC}" \
  --n-bins "${N_BINS}" \
  --slit-width-kpc "${SLIT_WIDTH_KPC}" \
  --map-width-kpc "${MAP_WIDTH_KPC}" \
  --map-npix "${MAP_NPIX}" \
  "${PLOT_ARG[@]}" \
  "${CSV_ARG[@]}" \
  "${JSON_ARG[@]}" \
  "${SUMMARY_CSV_ARG[@]}"

echo "=== STELLAR ISM VELOCITY SLIT END ==="
date
