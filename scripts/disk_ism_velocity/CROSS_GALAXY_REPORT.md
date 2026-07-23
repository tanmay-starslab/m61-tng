# Disk-ISM velocity across all 20 galaxies — final report

Branch `feat/disk-ism-velocity`. Per-orientation line-of-sight disk-ISM velocity for the
**14,400 mock-QSO sightlines** (20 TNG50 subhalos × 360 α × {flip, noflip}, single QSO
J122138+043026 at native ρ ≈ 25.6 kpc). Purpose: a velocity reference to subtract from the
Si II absorption so the ISM component can be separated from HVC/IVC clouds.

## Method (final)

For each sightline, from the combined `all_rays_L2Rvir.h5` Trident ray (ion grid + per-line
spectra):

1. **Corrected velocity:** `v_rest = -velocity_los/1e5 - v_sys`, positive = recession,
   `v_sys = SubhaloVel·los`. (Ray v_rest lands on the Si II/C II dip — sign verified.)
2. **Disk frame:** normal = **cone axis of the α-los series** (see "Disk-normal fix"), giving
   inc = 23° for every galaxy. Disk-plane cells = `|z_disk| < 2 kpc AND R_disk < R_edge`.
3. **R_edge = R95** of the cold-gas (T<1e4) disk mass profile, per galaxy (Stage A).
4. **PRIMARY v_ISM = cool-gas (T<1e4) density column-weighted `v_rest`** of the disk-plane
   cells. Gate: needs a real cool ISM column (`n_cool_disk_cells ≥ 1 AND N_SiII_disk > 0`).
5. **Fallback (`v_ism_R95edge`):** when the sightline crosses beyond the cool disk
   (`R_cross ≥ R_edge`) or carries no cool ISM, use the fiducial rotation curve at R95
   projected onto the LOS, `v_fid(R_edge)·proj`.
6. Cross-checks stored per sightline: all-gas density, Si II, and H I weightings, plus the
   rotation-curve MODEL `v_phi(R)·proj` and the Si II spectrum dip.

`compute_vism_fields()` in `ray_ism_diagnostic.py` is the single source of truth used by both
the diagnostic plots and the production `stage_b_vism.py`.

## Two bugs found and fixed during verification

1. **Disk-normal fix (correctness-critical).** The orientation JSON's `normal_used_hat` is
   inconsistent with the sightlines — off by up to **49°** for several galaxies (413372:43,
   143886:45, 348901:49, 352426:41; others ~0–4). Using it gave unphysical v_ISM up to
   ±227 km/s (apparent inclination 16–65° instead of 23°). The true disk normal is the
   **cone axis of the per-mode α-los series** (smallest-variance direction of the unit-los
   points), which yields **inc = 23° (std 0) for all 20 SIDs × both modes**. flip/noflip share
   the axis; one mode must be used (pooling both puts los on opposite hemispheres and breaks
   the recovery). Fixed in `build_sid_rc.disk_normal_from_los`.
2. **Cool-gas weighting.** The primary direct v_ISM now weights disk-plane gas by **cool-gas
   (T<1e4) density**, not all-gas density, so warm/hot gas no longer contaminates the ISM
   velocity when no cool disk gas is present. (Plus an sbatch fix: use the absolute env python;
   conda activation in batch left a numpy-less `python` on PATH.)

## Results (14,400 sightlines, verified)

- **6,366 direct-cool (44%)** — sightlines that cross cool disk gas; **8,034 R95-edge (56%)**.
- Direct-cool **v_ISM matches the Si II dip to 10.6 km/s median** (9.3 at ≥11 cool cells; 17.6
  at 1 cell — agreement improves cleanly with sampling). This is the ISM velocity to subtract.
- Geometry is physical: `|proj_phi|` = 0.36 (constant per SID, as expected — the anchor azimuth
  and los rotate together with α); R95-edge `|v_ISM| ≤ 134`.
- Direct-cool values reaching ~300 km/s are **real, well-sampled** (median 15 cool cells) — cool
  gas with large non-circular motions at the outer disk (~28 kpc); they still track the Si II
  absorption. The direct method reports the gas that is actually there (its whole advantage over
  the biased rotation-curve model, which runs ~15–30 km/s slow at these radii).
- **6 galaxies (143884, 348901, 375073, 388544, 432106, 438148) have compact cool disks**
  (R_edge 19–24 kpc); every ρ≈25.6 kpc sightline crosses beyond the cool disk → all R95-edge,
  correctly flagged. No cool ISM to subtract there.

## Answer to "shouldn't the main absorber be the ISM?"

At ρ ≈ 25.6 kpc it depends on the sightline, and the tool separates the two cases:
- **In-disk sightlines (44%)** cross cool disk gas; its velocity (direct v_ISM) sits on the
  Si II absorption → the ISM is identified, and any offset dips are HVC/IVC.
- **Beyond-disk sightlines (56%)** graze/miss the outer cool disk; there the strong Si II is a
  CGM/HVC cloud, not disk ISM, and `in_disk=False` + the R95-edge reference flag this. This is
  physics at a large impact parameter, not a sign/geometry error.

## Master table

`/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/vism_tables/vism_master_all_sightlines.csv`
(14,400 rows). Key columns: `sid, mode, alpha_deg, rho_kpc, v_sys, R_edge, R_cross, in_disk,
proj_phi, v_ism_model, v_ism_direct_cool, v_ism_direct_density, v_ism_SiII, v_ism_HI,
v_ism_R95edge, v_ism_primary, v_mode ('direct_cool'|'R95-edge'), f_disk_SiII, n_cool_disk_cells,
N_SiII_disk, N_HI_disk, has_cool, SiII_dip`. **Subtract `v_ism_primary`** from the absorption
velocity to get cloud-vs-disk velocity; use `v_mode`/`in_disk` to know if a real disk ISM was
measured (direct_cool) or only a disk-edge rotation reference (R95-edge).

## Reproduce
```
# Stage A (rotation curves + R_edge, per SID):   sbatch --array=1-20%6 run_stage_a.sbatch
# Stage B (per-sightline v_ISM, per SID):        sbatch --array=1-20%20 run_stage_b.sbatch
# Combine -> master:                             python combine_vism.py
# Or the whole chain:                            bash run_production.sh <stageA_jobid>
# Diagnostic plots / validation gallery:         python gen_gallery.py
```

## Diagnostic plots
- Ray diagnostics (multi-ion spectra + ray velocity profile + geometry):
  `outputs/disk_ism_velocity/ray_diagnostics/raydiag_sid<SID>_<mode>_alpha<AAA>.png`
- Si II overlays: `outputs/disk_ism_velocity/validation/overlay_*.png`
- Gallery summary: `outputs/disk_ism_velocity/ray_diagnostics/gallery_summary.csv`
