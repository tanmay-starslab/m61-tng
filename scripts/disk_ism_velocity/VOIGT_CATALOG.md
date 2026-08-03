# The Voigt component catalog — detections, definitions and caveats

This is the data product behind every transition-resolved (per-line) figure in the paper.
It flattens the pipeline's **own** spectral-fitting output; it does not introduce a new
detection criterion.

## Provenance

Every mock COS-G130M spectrum was fit line-by-line with pyGad Voigt profiles at S/N = 10,
bin = 3, by the spectral-fitting stage:

```
outputs/sid<SID>/rays_and_spectra_sid<SID>_snap99_L2Rvir/
  fitted_individual_line_spectra_parallel_snr10_bin3/per_spectrum_h5/<mode>/alpha<NNN>/
    L2Rvir_sid<SID>_J122138+043026_<mode>_alpha<A>_spectrum_individual_line_fits.h5
```

`build_voigt_catalog.py` reads all of them and writes

```
outputs/disk_ism_velocity/voigt_catalog/
  voigt_components.parquet     one row per fitted Voigt COMPONENT
  voigt_line_status.parquet    one row per (sightline, line) fit  <- the DENOMINATOR
```

Coverage is complete: **20 galaxies x 720 orientations x 9 transitions = 129,600 line-fits,
310,217 components** (299,065 detected, 11,152 upper limits), 14,400 sightlines, no missing
files. The fits are on the **L2Rvir** ray product — the same one `v_ISM` was built on.

Transitions: H I 1216, O I 1302, C II 1335, Si II 1190, Si II 1193, Si II 1260,
Si III 1206, Si IV 1403, N V 1239.

## Definitions (do not substitute your own)

| term | definition |
|---|---|
| **detection** | a fitted component with `uplim == False`. pyGad flags upper limits with `UpLim`; a line with no real component has `non_detection == True` in `voigt_line_status`. |
| **covering fraction** | (# sightlines with >= 1 detected component meeting the condition) / (# sightlines on which that line was fit). **Pure sightline counting — never weighted by column, EW or anything else.** |
| **ISM / IVC / HVC** | \|dv\| < 40 / 40–100 / >= 100 km/s |
| **denominator** | every line was fit on all 720 sightlines per galaxy, so `non_detection` is a genuine zero, not a missing measurement. Both `detection` and `non_detection` count (`m61_voigt.FIT_RAN`). |

Because each transition is fit **independently**, the three Si II lines are three separate
measurements of the same gas: the weak transitions simply fail to recover components the
strong one finds. That is the honest, pipeline-native origin of the per-line differences —
no curve-of-growth modelling or oscillator-strength thresholding is involved.

## Velocity convention (VERIFIED)

pyGad's `v_kms` is the **raw** observed velocity. The rest-frame velocity on the same
convention as `v_ISM` and the ray velocities is

```
v_rest = -v_kms - v_sys
dv_v3a = v_rest - v_ism_v3a        # THE kinematic offset from the disk ISM
dv_v3b = v_rest - v_ism_v3b
dv_v1  = v_rest - v_ism_direct_cool
```

Verified 2026-08-03 against the Si II 1260 flux minimum (`SiII_dip`) over four galaxies:
this convention gives sigma = 7–33 km/s; every sign/offset alternative gives 200–640 km/s.
It is the same convention as `fig10_accumulate.py`.

## Traps

* **`dv_kms` is the UNCERTAINTY on `v_kms`, not a velocity offset.** It sits next to
  `logN`/`dlogN` in the pyGad output and is easy to misread. Stored here as **`v_err_kms`**.
  The kinematic offset is `dv_v3a` / `dv_v3b`.
* **`sat` is a PER-FIT flag, not per-component.** It is constant across every component of a
  (sightline, line) fit (verified in 100% of groups) and means "this fit region contains
  saturation". `Chisq` is likewise per fit region (constant within 78% of groups; the rest
  have several regions). **Never quote a "fraction of saturated components".**

## H I 1216 is not comparable to the metal lines

H I was fit over a **±1200 km/s** window with **b_max = 150 km/s**; the metals used **±800**
with b_max 50–100. Consequently **59.5%** of H I components sit *at* the b ceiling and
**60.6%** lie beyond \|dv\| > 500 km/s. These are multi-component fits to the damped,
saturated Lyman-alpha profile — not physical high-velocity clouds.

The effect on the HVC covering fraction is large for H I and negligible for the metals:

| line | f_c raw | f_c clean | b at ceiling | beyond 500 km/s |
|---|---|---|---|---|
| H I 1216 | 0.930 | **0.486** | 59.5% | 60.6% |
| O I 1302 | 0.295 | 0.280 | 5.1% | 1.2% |
| C II 1335 | 0.402 | 0.389 | 1.6% | 1.5% |
| Si II 1190 | 0.324 | 0.313 | 3.6% | 0.9% |
| Si II 1193 | 0.358 | 0.344 | 6.2% | 0.9% |
| Si II 1260 | 0.432 | 0.392 | 14.0% | 1.3% |
| Si III 1206 | 0.455 | 0.443 | 2.4% | 1.4% |
| Si IV 1403 | 0.391 | 0.382 | 1.7% | 1.3% |
| N V 1239 | 0.270 | 0.260 | 1.8% | 1.6% |

`clean` = `~b_at_ceiling & ~beyond_common_window`, the default science sample. **Quote H I
only with this caveat.** Si II 1260's 14% ceiling rate is itself physical: the strong,
saturated line needs broad components to absorb its core — the same effect that drives the
covering-fraction ordering.

## Headline results

* Si II covering fraction orders by line strength, from the fits alone:
  **0.392 / 0.344 / 0.313** (clean) for λ1260 / λ1193 / λ1190 — f·λ = 1487 / 686 / 330.
  A single-transition "HVC covering fraction" is therefore uncertain at the ~25% level
  purely through the choice of line.
* **v3a and v3b agree to better than 0.002 per line** (Si II 1260: 0.433 vs 0.432 raw).
  The observables are insensitive to which per-orientation v_ISM variant is adopted.

## What the fits cannot tell you

The Voigt catalog has **no galactocentric radial velocity**. Blueshifted vs redshifted
relative to the disk ISM is an *observable proxy* for inflow/outflow and is labelled as such
in the figures. The physical inflow/outflow decomposition (gas `v_r`, metallicity,
temperature, disk geometry) comes from the per-cell gas catalog via `m61_lines.py`, which is
ION-level — for a column-weighted gas statistic the three Si II transitions are degenerate.

## Files

* `m61_voigt.py` — line registry, loaders, covering-fraction helpers (the definitions live here)
* `build_voigt_catalog.py` — `python build_voigt_catalog.py <sid>` / `--combine`
* `m61_lines.py` — gas-catalog loader for the ION-level physical figures (**not** detection)
