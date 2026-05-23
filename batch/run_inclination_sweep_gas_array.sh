#!/bin/bash
#SBATCH --job-name=m61_inc_gas
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/tsingh65/m61-tng/outputs/sid488530/inclination_sweep_alpha000_180_gas_stars_movies/logs_sbatch/gas_project_%A_%a.out
#SBATCH --error=/scratch/tsingh65/m61-tng/outputs/sid488530/inclination_sweep_alpha000_180_gas_stars_movies/logs_sbatch/gas_project_%A_%a.err
#SBATCH --array=0-1447%45

set -eo pipefail

REPO="/home/tsingh65/m61-tng"
SCRIPT="${REPO}/scripts/m61_inclination_sweep_alpha_movies.py"
OUT="/scratch/tsingh65/m61-tng/outputs/sid488530/inclination_sweep_alpha000_180_gas_stars_movies"

mkdir -p "${OUT}/logs_sbatch" "${OUT}/logs"

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

echo "=== M61 INCLINATION SWEEP GAS PROJECTION START ==="
date
echo "HOST=$(hostname)"
echo "JOBID=${SLURM_JOB_ID:-NA}"
echo "ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-NA}"
echo "OUT=${OUT}"
echo "python=$(command -v python)"
echo "=================================================="

python -u "${SCRIPT}" project-gas \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --npix 1024 \
  --output-dir "${OUT}"

echo "=== M61 INCLINATION SWEEP GAS PROJECTION END ==="
date
