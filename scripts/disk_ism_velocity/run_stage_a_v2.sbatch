#!/bin/bash
#SBATCH --job-name=dismA_v2
#SBATCH --partition=public
#SBATCH --account=grp_sborthak
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/logs/stageA_v2_%A_%a.out
#SBATCH --error=/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/logs/stageA_v2_%A_%a.err
#SBATCH --array=1-20%20
set -eo pipefail
module purge; module load mamba
eval "$(conda shell.bash hook)"; export CONDA_NO_PLUGINS=true
conda activate trident || true
export PY=/home/tsingh65/.conda/envs/trident/bin/python
export LD_LIBRARY_PATH="/home/tsingh65/.conda/envs/trident/lib:${LD_LIBRARY_PATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/scratch/tsingh65/m61-tng/scripts:/scratch/tsingh65/m61-tng/scripts/disk_velocity_v2:/home/tsingh65/m61-tng/scripts/disk_ism_velocity
LIST=/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/all_sids.txt
SID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
echo "=== Stage A v2  task=${SLURM_ARRAY_TASK_ID}  SID=${SID} ==="
"$PY" /home/tsingh65/m61-tng/scripts/disk_ism_velocity/build_sid_rc_v2.py "${SID}"
