#!/bin/bash
set -eo pipefail

REPO="/home/tsingh65/m61-tng"
OUT="/scratch/tsingh65/m61-tng/outputs/sid488530/inclination_sweep_alpha000_180_gas_stars_movies"
SCRIPT="${REPO}/scripts/m61_inclination_sweep_alpha_movies.py"

mkdir -p "${OUT}/logs_sbatch" "${OUT}/logs"

module purge
module load mamba
eval "$(conda shell.bash hook)"
export CONDA_NO_PLUGINS=true
conda activate trident
python -u "${SCRIPT}" write-index --npix 1024 --output-dir "${OUT}"

gas_job=$(sbatch --parsable "${REPO}/batch/run_inclination_sweep_gas_array.sh")
star_job=$(sbatch --parsable "${REPO}/batch/run_inclination_sweep_stars_array.sh")
gas_render_job=$(sbatch --parsable --dependency=afterok:${gas_job} "${REPO}/batch/run_inclination_sweep_gas_render_array.sh")
star_render_job=$(sbatch --parsable --dependency=afterok:${star_job} "${REPO}/batch/run_inclination_sweep_stars_render_array.sh")

cat <<EOF
Submitted inclination-sweep movie workflow
output_dir       = ${OUT}
gas projection   = ${gas_job}
star projection  = ${star_job}
gas render       = ${gas_render_job} (afterok:${gas_job})
star render      = ${star_render_job} (afterok:${star_job})

Monitor with:
  squeue -u ${USER} -j ${gas_job},${star_job},${gas_render_job},${star_render_job}
EOF
