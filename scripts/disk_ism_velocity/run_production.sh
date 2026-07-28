#!/bin/bash
# Production driver: wait Stage A -> submit + wait Stage B -> combine -> summary.
# Runs as a background process; sequences the SLURM stages (no fragile afterok chain).
# Usage: run_production.sh <stageA_array_jobid>
set -eo pipefail
D=/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity
SCR=/home/tsingh65/m61-tng/scripts/disk_ism_velocity
PY=/home/tsingh65/.conda/envs/trident/bin/python
export PYTHONPATH=/scratch/tsingh65/m61-tng/scripts:/scratch/tsingh65/m61-tng/scripts/disk_velocity_v2:$SCR
export HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
STAGEA_JOB="$1"
log() { echo "[$(date +%F_%H:%M:%S)] $*"; }

log "driver start; Stage A job=$STAGEA_JOB"
# 1. wait for the Stage A array to leave the queue
while squeue -j "$STAGEA_JOB" -h 2>/dev/null | grep -q .; do sleep 120; done
NRC=$(ls "$D"/rotation_curves/rc_sid*.npz 2>/dev/null | wc -l)
log "Stage A cleared queue; rotation curves present = $NRC/20"
# report any SID still missing an RC
while read -r s; do [ -f "$D/rotation_curves/rc_sid${s}.npz" ] || log "  WARN no RC for SID $s"; done < "$D/all_sids.txt"

# 2. Stage B for all 20 SIDs (tasks whose RC is missing exit 3 harmlessly)
JOBB=$(sbatch --parsable "$SCR/run_stage_b.sbatch")
log "Stage B submitted job=$JOBB; waiting"
sleep 30
while squeue -j "$JOBB" -h 2>/dev/null | grep -q .; do sleep 60; done
NVI=$(ls "$D"/vism_tables/vism_sid*.csv 2>/dev/null | wc -l)
log "Stage B cleared queue; per-SID v_ISM tables = $NVI/20"
while read -r s; do [ -f "$D/vism_tables/vism_sid${s}.csv" ] || log "  WARN no vism table for SID $s"; done < "$D/all_sids.txt"

# 3. combine -> master
$PY "$SCR/combine_vism.py"
log "combine done"
$PY - <<PYEOF
import pandas as pd
d = pd.read_csv("$D/vism_tables/vism_master_all_sightlines.csv")
print("MASTER rows", len(d), "| SIDs", d["sid"].nunique())
print("v_mode:", d["v_mode"].value_counts().to_dict())
print("primary v_ISM non-null:", int(d["v_ism_primary"].notna().sum()))
dc = d[d["v_mode"] == "direct_cool"]
if len(dc):
    print("direct_cool |v_ISM - SiII_dip| median:", round((dc["v_ism_primary"]-dc["SiII_dip"]).abs().median(), 1), "km/s over", len(dc))
PYEOF
log "DRIVER COMPLETE"
