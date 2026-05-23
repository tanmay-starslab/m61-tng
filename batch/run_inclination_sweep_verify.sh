#!/bin/bash
#SBATCH --job-name=m61_inc_verify
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/tsingh65/m61-tng/outputs/sid488530/inclination_sweep_alpha000_180_gas_stars_movies/logs_sbatch/verify_%A.out
#SBATCH --error=/scratch/tsingh65/m61-tng/outputs/sid488530/inclination_sweep_alpha000_180_gas_stars_movies/logs_sbatch/verify_%A.err

set -eo pipefail

OUT="/scratch/tsingh65/m61-tng/outputs/sid488530/inclination_sweep_alpha000_180_gas_stars_movies"
MANIFEST="${OUT}/final_video_manifest.txt"

mkdir -p "${OUT}/logs_sbatch"

{
  echo "M61 inclination sweep alpha movie final manifest"
  date
  echo
  echo "NPZ counts:"
  for comp in gas stars; do
    count=$(find "${OUT}" -path "*/${comp}/data/*.npz" | wc -l)
    echo "  ${comp}: ${count}"
  done
  echo
  echo "Frame counts by inclination/component:"
  for inc in inc000 inc023 inc045 inc075 inc090 inc135 inc170 inc180; do
    gas_frames=$(find "${OUT}/${inc}/gas/frames/combined" -maxdepth 1 -name 'frame_*.png' 2>/dev/null | wc -l)
    star_frames=$(find "${OUT}/${inc}/stars/frames/combined" -maxdepth 1 -name 'frame_*.png' 2>/dev/null | wc -l)
    echo "  ${inc}: gas_frames=${gas_frames} star_frames=${star_frames}"
  done
  echo
  echo "Videos:"
  find "${OUT}" -path '*videos/*.mp4' -printf '  %p\n' | sort
  echo
  echo "Video file sizes:"
  find "${OUT}" -path '*videos/*.mp4' -exec ls -lh {} \; | sort -k9
} > "${MANIFEST}"

cat "${MANIFEST}"
