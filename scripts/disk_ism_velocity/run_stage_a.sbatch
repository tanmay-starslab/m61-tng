#!/bin/bash
#SBATCH --job-name=dism_stageA
#SBATCH --partition=public
#SBATCH --account=grp_sborthak
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/logs/stageA_%A_%a.out
#SBATCH --error=/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/logs/stageA_%A_%a.err
#SBATCH --array=1-15%6
# Stage A: per-SID multi-tracer rotation curves (build_sid_rc.py). SID list = stage_a_sids.txt.
set -euo pipefail
module purge; module load mamba
eval "$(conda shell.bash hook)"; export CONDA_NO_PLUGINS=true
conda activate trident
export HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/scratch/tsingh65/m61-tng/scripts:/scratch/tsingh65/m61-tng/scripts/disk_velocity_v2
LIST=/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/stage_a_sids.txt
SID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
echo "=== Stage A  task=${SLURM_ARRAY_TASK_ID}  SID=${SID} ==="
python /home/tsingh65/m61-tng/scripts/disk_ism_velocity/build_sid_rc.py "${SID}"
