# v_ISM v3 — two per-ORIENTATION methods

Motivation: v2 used the azimuthally-AVERAGED rotation curve, so it gave ~one velocity per
galaxy. v3 measures a custom velocity for each orientation from the actual cutout gas. Two
variants were built (both requested); all outputs are in v3 directories, nothing overwritten.

Both average the three supervisor tracers — cold gas (T<10⁴), SF gas (SFR>0), young stars
(age<300 Myr) — weighted by density (gas) / mass (stars), with the galaxy-rest-frame LOS
velocity `v_rest = +(v_pec_rest · los)` (sign calibrated so it matches the ray/direct
convention and the Si II dip). They differ only in **which gas is sampled**:

- **v3a — along the sightline** (`build_vism_v3.py`, tube R_TUBE=3 kpc around the actual LOS,
  near the disk plane |z_disk|<2 kpc, |path|<40 kpc). This is the gas the QSO actually shines
  through.
- **v3b — centre→impact probe-line** (cylinder R_AP=5 kpc around `center + s·d_hat`,
  s∈[−40,+40], `d_hat=unit(anchor−center)`). The literal "line to the impact point, extended
  ±40 kpc".

The impact azimuth sweeps the full disk as α varies (verified: φ_disk = φ₀+α at fixed
R_anchor=25.3 kpc), so both are genuinely custom per orientation.

## Which one to trust — validation vs the Si II 1260 dip (in-disk sightlines, 20 galaxies)

| method | median − dip | **σ − dip** | comment |
|---|---|---|---|
| **v3a — 3-tracer along the sightline** | −1.0 | **18.3** | per-orientation AND lands on the absorption |
| v1 direct cool-gas (reference) | −1.1 | 15.5 | best; cool-gas-only along the ray |
| v2 — galaxy rotation curve | −3.6 | 46.2 | ~one value per galaxy |
| v3b — centre→impact line | −5.9 | 131.4 | custom, but does NOT track the absorption |

**Key result:** v3a is the per-orientation method that works — σ=18 km/s vs the dip (2.5×
tighter than v2, ≈ the direct method), with more valid sightlines (7777 vs 6366) because it
uses all three tracers. **v3b lands poorly (σ=131)**: a line through the centre samples the
impact point *and its diametric opposite*, where disk rotation projects with opposite sign, so
averaging the ±40 kpc line cancels the rotation signal (see `diagnostics_v3/v3_compare.png`,
panel c — a blob near v3b≈0 regardless of the dip). This is physical, not a bug: the ISM
absorption forms where the **sightline crosses the disk**, which v3a samples directly.

Compact galaxies whose disk ends before ρ≈26 kpc (143884, 348901, 375073, 388544, 432106,
438148) have few/no in-disk sightlines, so v3a is (correctly) undefined there — there is no
disk ISM at that impact parameter to measure.

## Provenance
As in v1/v2, columns and velocities come from the simulation gas (here the **cutout**
particles, sampled per orientation), not from pyGad Voigt fits. Weighting: gas by density,
young stars by mass; v_rest sign validated against the Si II dip.

## Files (all new)
- `build_vism_v3.py` → `vism_tables_v3/vism_v3_sid<SID>.csv`; `combine_v3.py` →
  `vism_master_v3.csv` (v_ism_v3a, v_ism_v3b + components + v2/direct/dip for comparison).
- `v3_compare.py` → `diagnostics_v3/v3_compare.png` (the validation figure).
- `make_paper_figures_v3.py v3a|v3b` → `paper_figures_v3a/`, `paper_figures_v3b/` (9 HVC
  figures + fig10 detection-rate, in-band ions, wavelength labels, no titles).
- `ray_diagnostic_v3.py` → `diagnostics_v3/ray_diagnostics_v3/` (per-sightline audit: all
  v_ISM estimates on the spectra + ray profile + numbers).

## Recommendation
Use **v3a** for the science (per-orientation, tracks the absorption). v3b is provided as
requested but should be treated as a diagnostic only. The v1 direct method remains the tightest
reference; v3a's advantage is the 3-tracer definition and broader sightline coverage.
