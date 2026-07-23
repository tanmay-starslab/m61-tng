#!/bin/bash
#SBATCH --job-name=dism_stageB
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/logs/stageB_%A_%a.out
#SBATCH --error=/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/logs/stageB_%A_%a.err
#SBATCH --array=1-20%20
# Stage B: per-sightline v_ISM for all 720 orientations of one SID. SID list = all_sids.txt.
set -eo pipefail   # NB: no -u; conda's (de)activate scripts reference unbound vars
module purge; module load mamba
eval "$(conda shell.bash hook)"; export CONDA_NO_PLUGINS=true
conda activate trident || true
export PY=/home/tsingh65/.conda/envs/trident/bin/python   # full path: robust vs flaky activation
export LD_LIBRARY_PATH="/home/tsingh65/.conda/envs/trident/lib:${LD_LIBRARY_PATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
export PYTHONPATH=/scratch/tsingh65/m61-tng/scripts:/scratch/tsingh65/m61-tng/scripts/disk_velocity_v2:/home/tsingh65/m61-tng/scripts/disk_ism_velocity
LIST=/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/all_sids.txt
SID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
echo "=== Stage B  task=${SLURM_ARRAY_TASK_ID}  SID=${SID} ==="
"$PY" /home/tsingh65/m61-tng/scripts/disk_ism_velocity/stage_b_vism.py "${SID}"
