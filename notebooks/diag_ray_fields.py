#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import yt
import trident

SID = 143885
SNAP = 99
RUN_LABEL = "L3Rvir"
CUTOUT_H5 = f"/scratch/tsingh65/TNG50-1_snap99/out_sub_{SID}/cutout_ALLFIELDS_sphere_2p1Rvir_sub{SID}.hdf5"
RAYS_CSV  = f"/scratch/tsingh65/m61-tng/outputs/sid{SID}/rays_and_recipes_sid{SID}_snap{SNAP}_{RUN_LABEL}/rays_sid{SID}.csv"

print("[LOAD]", CUTOUT_H5)
ds = yt.load(CUTOUT_H5)

# add only what you need
trident.add_ion_fields(ds, ions=["H I"])

df = pd.read_csv(RAYS_CSV)
row = df.iloc[0]

p0 = np.array([row["p0_X_ckpch_abs"], row["p0_Y_ckpch_abs"], row["p0_Z_ckpch_abs"]], float)
p1 = np.array([row["p1_X_ckpch_abs"], row["p1_Y_ckpch_abs"], row["p1_Z_ckpch_abs"]], float)

p0a = ds.arr(p0, "code_length")
p1a = ds.arr(p1, "code_length")

tmpdir = os.environ.get("SLURM_TMPDIR") or f"/scratch/tsingh65/m61-tng/outputs/sid{SID}/_tmp_trident"
os.makedirs(tmpdir, exist_ok=True)
rayfile  = os.path.join(tmpdir, f"diag_ray_sid{SID}.h5")
trajfile = os.path.join(tmpdir, f"diag_traj_sid{SID}.txt")

for p in (rayfile, trajfile):
    try: os.remove(p)
    except FileNotFoundError: pass

print("[RAY] making simple ray ...")
ray = trident.make_simple_ray(ds, start_position=p0a, end_position=p1a,
                              data_filename=rayfile, solution_filename=trajfile)

print("[RAY TYPE]", type(ray))
print("[FIELD_LIST LEN]", len(getattr(ray, "field_list", [])))

# print a few fields so you see ftypes actually present on the light-ray dataset
print("\n[SAMPLE field_list]")
for f in list(ray.field_list)[:40]:
    print(" ", f)

ad = ray.all_data()

def try_field(f):
    try:
        a = ad[f]
        v = a.to_value() if hasattr(a, "to_value") else np.asarray(a)
        print(f"[OK] {f:>18}  shape={v.shape}  dtype={v.dtype}  units={getattr(a,'units','')}")
        if v.size:
            print(f"     min={np.nanmin(v):.3e}  max={np.nanmax(v):.3e}")
        return True
    except Exception as e:
        print(f"[FAIL] {f:>18}  -> {type(e).__name__}: {e}")
        return False

print("\n[TEST coordinate-like fields via all_data()]")
for f in [
    ("gas","x"), ("index","x"), ("all","x"),
    ("gas","y"), ("index","y"), ("all","y"),
    ("gas","z"), ("index","z"), ("all","z"),
]:
    try_field(f)

print("\n[TEST dl-like fields via all_data()]")
for f in [
    ("gas","dl"), ("index","dl"), ("all","dl"),
    ("gas","segment_length"), ("index","segment_length"),
]:
    try_field(f)

print("\n[TEST ion field]")
try_field(("gas","H_p0_number_density"))