# Disk-ISM velocity — ray diagnosis & cross-galaxy report

Branch `feat/disk-ism-velocity`. Investigating whether the disk ISM is the main
absorber at v_ISM, verifying the velocity sign, using multiple ions and the ray.

## Method (rectified)
- **Direct v_ISM** from the Trident ray (`.../combined/all_rays_L2Rvir.h5`):
  **gas-density (`density·dl`) column-weighted `v_rest`** of disk-plane cells:
  **|z_disk| < 2 kpc AND R_disk < R_edge**.
- **R_edge = R95 of the cold-gas (T<1e4) disk mass profile**, computed per galaxy.
- Sign (verified): `v_rest = -velocity_los/1e5 - v_sys`, positive = recession,
  `v_sys = SubhaloVel·los`. Ray column-weighted v matches the spectrum dip to <10 km/s.
- Cross-checks reported alongside: Si II- and H I-column weighting, and the
  rotation-curve MODEL v_ISM (`v_phi(R)·proj`).

## Findings
1. **Sign convention is correct.** Direct v_ISM lands on the Si II / C II dip.
2. **The direct method works across galaxies.** e.g. 342448 flip32 = -55 (dip -61),
   noflip100 = +93 (dip +91, sign flips); 482889 flip200 = -113 (dip -113).
   Density, Si II and H I weightings agree to a few km/s **when cool disk gas is present**.
3. **The rotation-curve MODEL v_ISM is biased ~15-30 km/s too negative** (outer-disk
   gas at rho~25 sub-rotates vs the median curve) — so we use the DIRECT value and keep
   the model only as a secondary "expected rotation" reference.
4. **At the QSO impact parameter (rho=25.6 kpc) many sightlines cross the disk at/beyond
   its edge.** R_cross (disk radius at the midplane crossing) varies **24-40 kpc** with
   orientation; per-galaxy R_edge ranges **23-29 kpc**. The **in_disk flag (R_cross<R_edge)
   correctly identifies which sightlines have a measurable disk ISM.** Example: 438148 has a
   compact disk (R_edge=23) so ALL tested sightlines are beyond the disk -> v_ISM = nan (correct).
5. **The Si II disk fraction (f_disk) is often low (0-0.85).** At this large rho the disk ISM
   is frequently a minor absorber and the strong Si II is a CGM/cloud component — this is
   physics, not a bug, and it is exactly the ISM-vs-cloud separation the tool is meant to show.
   When cool disk gas is absent, all-gas density weighting picks up warm/hot gas (a meaningless
   "ISM" velocity) -> **REFINEMENT NEEDED: weight by cool-gas (T<1e4) density, or require a
   minimum disk Si II/H I column, and flag sightlines with no cool ISM.**

## Verdict
The pipeline measures the disk-ISM velocity correctly **where a cool disk is crossed**, and
**flags** sightlines beyond the disk (no ISM to subtract). The "main absorber = ISM" picture
holds for in-disk sightlines with cool gas; at rho=25.6 kpc many orientations graze/miss the
outer disk, so the strongest absorber there is a CGM cloud — correctly separated from v_ISM.

## Diagnostic plots (spectra) & tables
- **Ray diagnostics (multi-ion spectra + ray velocity profile + geometry):**
  `/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/ray_diagnostics/raydiag_sid<SID>_<mode>_alpha<AAA>.png`
- **Si II overlays (v_ISM on the raw spectrum):**
  `/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/validation/overlay_*.png`
- **Disk radial extent (surface density + R_edge):**
  `/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/disk_extent/disk_extent_sid<SID>.png`
- **Numeric summary:**
  `/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/ray_diagnostics/ray_diag_summary.csv`
- Rotation curves (Stage A): `/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/rotation_curves/`

Galaxies tested so far (Stage A done): 342448, 482889, 452978, 438148, 413372.

## Next
- Weight by cool-gas density (T<1e4) or require a minimum disk cool-gas column; flag no-ISM sightlines.
- Stage A for the remaining 15 SIDs (sbatch), then Stage B (direct v_ISM) over all 14,400 sightlines.
