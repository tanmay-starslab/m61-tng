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
v_rest = -(v_obs_kms + v_zeropoint_kms) - v_sys     # zero-point: see below
dv_v3a = v_rest - v_ism_v3a        # THE kinematic offset from the disk ISM
dv_v3b = v_rest - v_ism_v3b
dv_v1  = v_rest - v_ism_direct_cool
```

Verified 2026-08-03 against the Si II 1260 flux minimum (`SiII_dip`) over four galaxies:
the sign/offset convention gives sigma = 7–33 km/s; every alternative gives 200–640 km/s.
It is the same convention as `fig10_accumulate.py`.

### Si II 1260 rest-wavelength zero-point (CORRECTED IN THE CATALOG)

pyGad's line list does not always match the wavelengths Trident used to synthesise the
spectra. Recovering the rest wavelength pyGad actually assumed, from its own output,
`lam0_pygad = lambda_A / (1 + v_obs/c)`:

| line | pyGad lambda_0 | Trident lambda_0 | diff |
|---|---|---|---|
| H I, O I, C II, Si II 1190, Si II 1193, Si III, Si IV, N V | (match) | (match) | **0.0000 A** |
| **Si II 1260** | **1260.5220** | 1260.4220 | **+0.1000 A** |

0.1000 A = **+23.76 km/s**, reddening *every* Si II 1260 component. Uncorrected its median
`dv` is **+24.67** km/s against **+0.39 … +1.20** for all eight other transitions;
corrected it is **+0.91**. `build_voigt_catalog.py` now recovers this per row and corrects
`v_rest`, storing `lam0_pygad` and `v_zeropoint_kms` so the fix stays auditable and applies
itself if any other line list ever drifts.

This was the cause of two things previously misattributed:
* The Si II 1260 red/blue excess. It was briefly blamed on an S II lambda1259.519 blend at
  dv = +214.8 km/s. **That blend does not exist**: `batch/run_spectra_array.sh:29` synthesises
  exactly nine transitions (Si II 1190/1193/1260, Si III 1206, Si IV 1403, C II 1335, O I 1302,
  N V 1239, H I 1216) — no S II, no Fe II 1260.533, no C I 1260.736 — so no species can blend
  into the window. Searching the corrected catalog for a residual finds none: the +215+-20 km/s
  excess over lambda1193 is +11.4% (Poisson sigma 3.5%), statistically identical to the mirror
  window at -215 (+13.6%). The whole effect was the zero-point. Uncorrected `f_blue - f_red`
  = -0.098 for Si II 1260 against -0.003…-0.022 for every other line; corrected, **-0.017**,
  in line with lambda1193 (-0.009) and lambda1190 (-0.010).
* The "+23.5 km/s median offset" seen when the velocity convention was first validated
  against the Si II dip. The convention was right; this systematic was the whole offset.

## Traps

* **H I's fitted `logN` is the fitter's upper bound, not a measurement.** 57.5% of clean H I
  components sit within 0.05 dex of the `fit_logN_max` = 19.52 ceiling (median 19.485), so any
  H I logN distribution, median or `logN_1/2` is reporting the bound. Metals are <= 2.8%.
  Flagged as `logN_at_ceiling` — not dropped, because a damped saturated line genuinely does
  have a high column; we simply cannot measure how high.
* **`dv_kms` is the UNCERTAINTY on `v_kms`, not a velocity offset.** It sits next to
  `logN`/`dlogN` in the pyGad output and is easy to misread. Stored here as **`v_err_kms`**.
  The kinematic offset is `dv_v3a` / `dv_v3b`.
* **`sat` is a PER-FIT flag, not per-component.** It is constant across every component of a
  (sightline, line) fit (verified in 100% of groups) and means "this fit region contains
  saturation". `Chisq` is likewise per fit region (constant within 78% of groups; the rest
  have several regions). **Never quote a "fraction of saturated components".**

## H I 1216 is not comparable to the metal lines

H I was fit over a **±1200 km/s** window with **b_max = 150 km/s**. The metals used ±800 with
b_max 50–100, **except Si II 1190 and Si II 1193, which used ±500** — those two set the
`beyond_common_window` bound, it is not an arbitrary choice.
Consequently **59.5%** of H I components sit *at* the b ceiling and
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
| Si II 1260 | 0.421 | 0.382 | 14.0% | 1.3% |
| Si III 1206 | 0.455 | 0.443 | 2.4% | 1.4% |
| Si IV 1403 | 0.391 | 0.382 | 1.7% | 1.3% |
| N V 1239 | 0.270 | 0.260 | 1.8% | 1.6% |

`clean` = `~b_at_ceiling & ~beyond_common_window`, the default science sample. **Quote H I
only with this caveat.** Si II 1260's 14% ceiling rate is itself physical: the strong,
saturated line needs broad components to absorb its core — the same effect that drives the
covering-fraction ordering.

## Headline results

* Si II covering fraction orders by line strength, from the fits alone:
  **0.382 / 0.344 / 0.313** (clean, zero-point corrected) for λ1260 / λ1193 / λ1190 —
  f·λ = 1487 / 686 / 330. A single-transition "HVC covering fraction" is therefore uncertain
  at the ~22% level purely through the choice of line. The ordering survives every stress
  test applied: matching the fitting windows, the λ1260 zero-point correction, and raw-vs-clean.
* The mechanism is measured, not assumed: the weak transitions recover **+0.215 dex (λ1193)**
  and **+0.408 dex (λ1190)** more column than the saturated λ1260, ordered by f·λ, and their
  completeness relative to λ1260 rises from ~0.15 at logN≈11.9 to ~0.86 at logN≈13.9.
* **v3a and v3b agree to 0.0003–0.0030 per line** (clean sample; the largest is N V at
  0.0030, Si II 1260 is 0.0012). The observables are insensitive to which per-orientation
  v_ISM variant is adopted.

## Si II 1190/1193 fitting window

λ1190 and λ1193 were fit over ±500 km/s while λ1260 used ±800 — a genuine mismatch. It does
not drive the triplet ordering: restricting all three to the common ±500 window gives
**0.425 / 0.354 / 0.320**, a shift of ≤0.007, and the ordering also survives the λ1260
zero-point correction. Quote it, but it is not a confound.

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
