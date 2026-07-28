# v_ISM v3 — two per-ORIENTATION methods (corrected)

Motivation: v2 used the azimuthally-AVERAGED rotation curve, so it gave ~one velocity per
galaxy. v3 measures a **custom velocity for each orientation and each galaxy** from the actual
cutout gas. Two variants are built; both give a value for EVERY (orientation, galaxy). All
outputs are in v3 directories; nothing else is overwritten.

Both average the three supervisor tracers — cold gas (T<10⁴), SF gas (SFR>0), young stars
(age<300 Myr) — weighted by density (gas) / mass (stars). The LOS velocity is computed
**directly in the orientation, no projection**: `v_rest = +(v_pec_rest · los)` (sign
calibrated so it matches the ray/direct convention and the Si II dip).

Sky frame per orientation: `rhat = unit(anchor−center)` (impact-parameter direction, ⟂ los),
`los`, `uhat = rhat×los`. For any particle: s = pos·rhat (impact coordinate), w = pos·uhat
(transverse), depth = pos·los, z = pos·n_disk. Verified: the impact azimuth sweeps the full
disk as α varies, so both methods are genuinely per-orientation.

## The two methods

**v3b — binned impact-parameter slit (the corrected method).** A rectangular slit through the
centre along `rhat`, s∈[−40,+40] kpc, half-width R_SLIT=5 kpc (in uhat), disk layer |z|<3 kpc,
depth |depth|<40 kpc. The 3-tracer density/mass-weighted LOS velocity is computed in **2-kpc
bins of s → a velocity profile v(s)**. **v3b = the profile read at the exact impact parameter
s=ρ** (interpolated), NOT averaged over the slit — this is the fix. (Averaging the ±40 kpc slit
cancels rotation, because the impact side and its diametric opposite project with opposite
sign; reading the single impact-parameter bin does not.) If ρ is beyond the outermost populated
bin (the disk ends first), v3b = the **disk-edge bin value** (flagged `v3b_edge_fallback`).

**v3a — along the actual sightline.** 3-tracer LOS velocity in a tube of radius R_TUBE=5 kpc
around the real sightline (impact param ρ), disk layer |z|<3 kpc. If the tube is empty
(sightline beyond the disk), v3a falls back to the v3b disk-edge value (flagged
`v3a_edge_fallback`). This guarantees a value for every sightline.

## Validation vs the Si II 1260 dip (in-disk sightlines, 20 galaxies)

| method | median − dip | **σ − dip** | n | comment |
|---|---|---|---|---|
| **v3a — along the sightline** | −1.4 | **21.0** | 9158 | per-orientation, lands on the absorption |
| **v3b — binned slit @ impact** | −1.5 | **19.4** | 9158 | per-orientation, lands on the absorption |
| v1 direct cool-gas (reference) | −1.1 | 15.5 | 6366 | cool-gas-only along the ray |
| v2 — galaxy rotation curve | −3.6 | 46.2 | 9158 | ~one value per galaxy |

**Key result:** v3a and v3b are two independent constructions that **agree** (σ 21 vs 19) and
both land on the absorption ~2.5× tighter than v2, with **values for all 9158 in-disk
sightlines** (vs 6366 for the cool-gas-only direct method). The consistency v3a≈v3b (see
`diagnostics_v3/v3_compare.png`, panels b & c — both tight 1:1 with the dip) is the referee-proof
check. Three partially-compact galaxies (143885, 143886, 452978) show higher σ for BOTH methods
(they agree) because ~half of their sightlines fall at/beyond the disk edge — the fallback
regime, where the ISM velocity is genuinely uncertain; those are flagged.

Fully-compact galaxies (348901, 375073, 388544, 432106, 438148: disk ends well before ρ≈26 kpc)
have no in-disk sightlines; every orientation still gets a value via the disk-edge fallback,
as requested.

## Provenance & reproducibility
Columns and velocities come from the simulation **cutout** gas/stars sampled per orientation,
not from pyGad Voigt fits. Weighting: gas by density, young stars by mass; v_rest sign
validated against the Si II dip. NOTE: `build_vism_v3.py` must be run at LOW concurrency (≤2 per
node) — the ±40 kpc slit needs the full cutout in memory, and running 4+ big cutout loads
concurrently on one node truncates them and corrupts the slit profile (v3a's compact tube
survives, v3b does not). Re-running the affected galaxies singly fixes it.

## Files (all new / v3)
- `build_vism_v3.py` → `vism_tables_v3/vism_v3_sid<SID>.csv` + `slitprof_sid<SID>.npz` (the full
  slit profiles); `combine_v3.py` → `vism_master_v3.csv` (v3a, v3b, components, flags, +
  v2/direct/dip).
- `v3_compare.py` → `diagnostics_v3/v3_compare.png`.
- `make_paper_figures_v3.py v3a|v3b` → `paper_figures_v3a/`, `paper_figures_v3b/` (9 HVC figures
  + fig10 detection-rate; in-band ions, wavelength labels, no titles).
- `ray_diagnostic_v3.py` → `diagnostics_v3/ray_diagnostics_v3/` (per-sightline audit: spectra +
  ray profile + **the v3b slit profile v(s) with the read-at-ρ point** + numbers);
  `ray_diag_v3_overview.py` → overview + contact sheets.

## Recommendation
v3a and v3b are equivalent and both valid; **v3b (the binned slit read at the impact parameter)
is the method you specified** and comes with the disk-edge fallback, so it has a value for
every orientation and galaxy. v3a is the direct cross-check. Use either for the science with
confidence; report the edge-fallback flag so edge/compact sightlines are transparent.
