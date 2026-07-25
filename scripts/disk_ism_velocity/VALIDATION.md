# Deep validation of the multi-ion disk-ISM / HVC kinematics pipeline

Validation of the 14,400-sightline v_ISM reference and the 11-ion absorbing-gas catalog
(TNG50-1, 20 SIDs, single QSO J122138+043026 at rho≈25.6 kpc). Scripts:
`scripts/disk_ism_velocity/validation/`; outputs: `outputs/disk_ism_velocity/validation/`
and figures in `outputs/disk_ism_velocity/paper_figures/v{1,2c,3,4,5,6}_*.{pdf,png}`.

**Motivating questions (from the user).** (1) *If O VI is not in the spectra, how do we know
the velocity of O VI absorbers?* → the O VI velocity is the O VI-gas-column-weighted v_rest;
Tier 1 proves this equals the absorption velocity for every in-band ion, so the
identically-computed O VI velocity is legitimate. (2) *Are there ways to test our results?* →
the seven tiers below, all **PASS**.

---

## Tier 0 — data provenance (established before validation)
- The **combined `all_rays_L2Rvir.h5` is authoritative**: self-consistent with the May-23
  mock spectra and the current fixed-observer recipe. The per-SID `original_rays/*.h5` are a
  **stale pre-May-23 artifact** and are read by neither production script.
- **v_ISM (`stage_b_vism.py`) and the absorber catalog (`build_absorber_catalog.py`) both read
  the combined ray** (`combined_path()` → `original_trident_ray_h5/grid`), so
  `dv = v_los − v_ISM` never mixes geometries.
- The 6 native ions reproduce the earlier single-ion Si II run exactly.

## Tier 0b — periodic-wrap / hyper-velocity masking  → **PASS (contamination negligible)**
The combined ray endpoints can poke outside the box (periodic wrap of the far tail) and TNG
holds rare hyper-velocity hot cells. Flagged: `wrapped` (min-image correction ≠ 0) and
`hypervel` (|v_rest| > 700 km/s).
- Flagged **3,824 / 1,294,326 cells (0.295%)** (wrapped 2,258; hypervel 1,576).
- Effect on the headline fractions is tiny: **max ΔHVC = 0.34 pp (O VI)**, **max Δinflow =
  0.74 pp (O VI)**; masking nudges *toward* more inflow / less HVC, i.e. it slightly
  **strengthens** the cool-inflow result.
- Per instruction, the mask is applied to the science figures anyway (clean provenance).
  `make_paper_figures.py::load()` now drops `wrapped|hypervel`; all 9 figures regenerated on
  1,290,502 clean cells. (`validation/tier0b/contamination_by_ion.csv`)

## Tier 1 — gas velocity **=** absorption velocity  → **PASS (decisive; legitimizes O VI)**
For the 6 in-band ions (H I, C II, Si II, Si III, Si IV, N V) over 20 SIDs
(**85,779 sightline-ion measurements**): the spectrum apparent-optical-depth velocity (raw τ)
vs the gas ion-column-weighted v_rest:
- **median(AOD − gas) = +0.86 km/s, robust σ = 0.21 km/s**, 16–84 [+0.5, +1.0].
  Per-ion medians +0.75…+0.92 km/s (all MAD < 1.6). The +0.9 km/s is a uniform velocity
  zero-point, ≪ the 40 / 100 km/s ISM/HVC bin widths.
- The match holds across the full ionization range **up to N V (log T≈5.3)** and is exact
  even for damped H I (τ_max~10⁶·⁸) because raw τ carries the true column-per-velocity-bin.
- Flux-centroid scatter grows with saturation (H I MAD 78; N V MAD 3.8) but AOD — the standard
  observational O VI-velocity method — recovers the gas velocity for all. O VI line-centre
  τ~1–20 (Si IV/N V regime, **not** damped like H I).
- **⇒ Since the O VI velocity is the O VI-column-weighted v_rest, computed identically, it
  equals the absorption velocity an observer would measure.** (`v1_gas_vs_spectrum`)

## Tier 2 — ionization / O VI columns  → **PASS**
- **2a recompute machinery:** recomputing the 6 native ions with the same
  `trident.add_ion_fields` path used for O VI/C IV reproduces the **stored native columns to
  dlogN = 0.0000** (max|·| = 0.0000; 400 sightlines × 20 SIDs). The O VI/C IV pipeline is
  therefore an exactly-verified path.
- **2b oxygen systematic:** Trident scales oxygen by total metallicity × solar pattern
  ((O/Z)⊙ = 0.449 by mass); TNG's true (O/Z) for O VI-phase gas is 0.531 → **our O VI columns
  are low by +0.073 dex** vs a true-oxygen calculation (TNG O is mildly α-enhanced).
- **2c external cross-check (sub-488530, the OVI_CGM_compare galaxy, same HM2012 UVB):** our
  O VI median **logN = 14.78** [16–84 14.53–14.99] vs the independent map **14.74** (radial) /
  14.75 (face-on) / 14.77 (edge-on) → **|ours − map| = 0.036 dex**. Two independent pipelines
  (different ray-casting, LOS depth ±1.85 vs ±1 R200c, and oxygen treatment) agree to
  <0.05 dex. (`v2c_ovi_crosscheck`)
- **2d UVB sensitivity:** fixed-gas O VI column changes by **−0.003 / +0.002 dex** under
  UVB ×2 / ×0.5 (C IV +0.062 / −0.023). The catalog shows **97.7 % of the O VI column is
  collisionally ionized (log T > 5.3)**, so O VI is intrinsically UVB-robust.

## Tier 3 — v_ISM validation  → **PASS**
Direct cool-gas v_ISM (6,366 direct_cool sightlines, 44 %):
- **v_ISM − Si II 1260 dip: median −1.1 km/s, robust σ = 15.5 km/s** (74 % within 20 km/s) —
  lands on the observable absorption dip, unbiased.
- Internally consistent across weightings: cool-density vs Si II / H I / all-gas agree to
  **σ ≤ 5 km/s**.
- **|v_ISM| ≤ v_rot,max for 100 %** of sightlines (median ratio 0.26) — the pre-fix
  ±227 km/s pathology (wrong disk normal) is fully gone.
- Provenance clean: direct_cool ⟺ in_disk; beyond-disk → R95-edge fallback. (`v3_vism_validation`)

## Tier 4 — threshold / parameter robustness  → **PASS**
Re-deriving the two headline trends over HVC cuts 80–150 km/s, ISM/IVC cuts 30/40/50, and ion
floors ×0.3–×3:
- **Inflow-fraction falls with ionization: Spearman ρ ≤ −0.905 at every setting**
  (H I 66–70 % → O VI 48–50 %) — strictly monotone, bulletproof.
- **HVC-fraction rises with ionization:** cool metals ~7 % → warm-hot ~35 %, O VI/H I ratio
  **2.4–2.9×** across all cuts (ρ ≥ +0.79; not strictly monotone only because **H I sits
  elevated**, ~12–15 %, as ubiquitous neutral gas above the cool metal ions). (`v4_robustness`)

## Tier 5 — physical sanity  → **PASS**
- **Escape velocity:** with per-galaxy v_esc(r) = V_circ,max·√(2[1+ln(R200/r)]), HVC absorbers
  are **bound: H I–N V 100 %, O VI 99.7 %** (0.2 % escaping-outflow). Even the warm-hot O VI
  outflow is bound → **galactic fountain / recycling**, not halo-escaping wind.
- **Geometry:** O VI ⟨v_r⟩ turns positive (outflow) off the disk plane (θ > 30°) while Si II
  stays inflowing — the expected biconical/minor-axis outflow (modest, noisy at high θ).
  (`v5_physical`)

## Tier 6 — observational anchoring  → **PASS**
Our galaxies are z=0 ~L* star-forming centrals at ρ≈25.6 kpc (inner COS-Halos range):
- **O VI median logN = 14.53** vs COS-Halos 14.5 (Tumlinson+2011, Werk+2013); our 16–84
  [13.84, 14.96] spans their 14.2–14.9. Covering fraction f_c(>10¹⁴·²) = 0.71.
- **HVC covering fractions**: O VI 0.75 (MW 0.6–0.85, Sembach+2003); H I 0.66 (MW 0.3–0.8,
  Wakker 2004 / Lehner+2012) — every ion within the MW ranges. (`v6_observational`)

---

## Verdict
All seven tiers pass. The two load-bearing science claims are validated:
1. **The gas-based O VI velocity is the absorption velocity** (Tier 1, <1 km/s; O VI is
   collisional and column-verified in Tiers 2a/2c/2d).
2. **The ionization-stratified HVC result is robust** — HVC fraction rising and inflow
   fraction falling from H I to O VI survives masking (Tier 0b), thresholds (Tier 4), and is
   physically (Tier 5) and observationally (Tier 6) consistent.

### Documented systematics / caveats (not failures)
- O VI/C IV/Mg II/Fe II have **no mock spectrum** (COS-G130M 1150–1450 Å only); their
  velocities/columns are gas-based, validated by extension (Tier 1) and cross-check (Tier 2c).
- O VI columns carry a **+0.073 dex** solar-oxygen-vs-true-oxygen systematic (Tier 2b).
- The 14,400 Si II Voigt fits (`all_subhalos_siII1260_voigt_results_*.csv`) are **L4Rvir**
  (±914 kpc) rays and in the plain velocity frame — not used here; our validation uses the
  self-consistent L2Rvir combined rays.
- Tier-5 outflow-geometry signal is real but modest/noisy at high polar angle.
