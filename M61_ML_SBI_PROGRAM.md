# M61 / J1221+0430 — Machine Learning & Simulation-Based Inference Program

**A complete project briefing and research design.**

This document is self-contained. It is written so that another agent (e.g. Claude Science) or a
new collaborator can pick up the project cold, understand the data, the science, the traps, and
the proposed program, and go deeper without re-deriving anything.

**Provenance convention used throughout:**

- **[V]** — verified directly against files on disk during the session that produced this document.
- **[A]** — produced by an analysis agent during the design study; the method is stated, but the
  number should be re-derived before it enters a paper.
- **[D]** — documentation/prior claim inherited from project notes or slides; treat as a hypothesis.

Last updated: 2026-08-08.

---

# PART I — THE OBSERVATION (the anchor)

## 1.1 The system

| quantity | value | source |
|---|---|---|
| Galaxy | **M61 / NGC 4303** | `data/M61_DIISC_Table1_Table2.csv` **[V]** |
| RA, Dec | 185.479, +4.4735 | same |
| Distance | **18.7 Mpc** (0.0907 kpc/arcsec) | same |
| z_gal | **0.005224000196903944** | DIISC slice headers **[V]** |
| cz | ~1560 (VLA) – 1566 (DIISC) km/s | **[V]** / **[D]** |
| log M★ | **11.39** | DIISC table **[V]** |
| log M_halo | **12.7** | same |
| log M_HI | 9.88, R_HI = 28.9 kpc | same |
| log SFR | 0.72 (≈5.2 M⊙/yr) | same |
| Inclination | **23°**, PA = 138°, b/a = 0.93 | same |
| R_vir | **457 kpc** | same |
| QSO | **J122138+043026** (= J1221+0430), z = 0.09 | same |
| Impact parameter | **26 kpc** (sim uses 25.558333340264905) | same |

M61 is a Virgo Cluster member. This matters — see §6.5.

## 1.2 The data on disk

**Nobody in the `m61-tng` project has used this.** `grep -rn 'FIRE_FINESST\|J1221+0430' /home/tsingh65/m61-tng/`
returns nothing. There are **zero FITS files** anywhere under `m61-tng`. **[V]**

```
/scratch/tsingh65/FIRE_FINESST/J1221+0430/J1221+0430/
  J1221+0430_nbin3_coadd.fits                     HST/COS G130M coadd, 10,408 px, 1132.58-1433.86 A
  113_270_z0.005/
    113_270_z0.005_lineresults.txt                EW + logN table, 26 transitions
    fitting/J1221+0430_113_270_z0.005_<LINE>_slice.fits    25 files, continuum-normalised
                                                  velocity slices + Voigt fits
```

Each `*_slice.fits` is a 1-row BinTable with 27 columns. The ones that matter:

| column | shape | meaning |
|---|---|---|
| `VEL` | (1001,) | velocity relative to `ZGAL`, dv ≈ 6.6–7.7 km/s |
| `FNORM`, `ENORM` | (1001,) | continuum-normalised flux and 1σ error |
| `FITNCOMPS` | scalar | number of Voigt components |
| **`FITPARAMS`** | **(4, 10)** | **row 0 = logN, row 1 = b [km/s], row 2 = velocity [km/s], row 3 = 999.99 sentinel** |
| `FITPARAMSERR` | (4, 10) | the corresponding uncertainties (real, not sentinels) **[V]** |
| `FITPROF`, `FITCOMPS` | (1001,), (1001,10) | fitted profile and per-component profiles |

> **Trap.** `FITPARAMS` is a (4,10) array, not a flat list. Reading it flat produces nonsense
> (velocities of 0 and 1000). Row 2 is the velocity row. **[V]**

Also on disk: `/scratch/tsingh65/FIRE_FINESST/J122154+042837-M61/` — VLA H I 21 cm cube, moment
maps, 3D-Barolo products (`M61_mom1_barolo.fits` gives the disk velocity field), and archival imaging.

Sibling system with **matched formats and a NON-DETECTION**: `J1140+1136 / NGC 3810`. This is the
only available second data point (§8.2, Phase 2a).

## 1.3 The observed absorption — the definitive component table **[V]**

Systemic frame (z_gal = 0.005224). Format: `v [logN, b]`.

| ion | components |
|---|---|
| **N V 1238/1242** | **−330.8** [13.51, 27.3] · −200.9 [13.93, 92.9] · +2.9 [14.21, 80.4] |
| **Si IV 1393/1402** | **−308.2** [13.68, 30.7] · −252.0 [12.95, 16.8] · −152.3 [13.11, 61.1] · −115.5 [12.66, 7.6] · −60.0 [13.09, 7.7] · −20.5 [13.73, 45.8] · +73.0 [13.36, 28.7] |
| **Si II 1190/1193/1260/1304** | **−308.2** [13.61, 10.9] · −283.9 [13.41, 54.7] · −175.0 [12.50, 8.2] · −120.0 [13.18, 7.4] · +5.0 [14.72, 53.2] · +65.5 [13.89, 6.4] |
| **Si III 1206** | **−303.0** [13.51, 31.4] · −293.9 [13.51, 57.6] · −177.7 [12.56, 9.9] · −114.3 [12.97, 19.2] · −64.6 [13.33, 32.2] · +30.6 [14.02, 61.4] |
| **C II 1334** | **−306.5** [14.44, 29.7] · −257.1 [13.79, 9.1] · −165.1 [13.74, 66.0] · −112.1 [13.92, 10.2] · +13.1 [15.83, 50.1] |
| O I 1302 | +3.9 [15.41, 27.2] · +63.7 [14.87, 25.9] — *(but see §1.5)* |
| N I 1134/1199/1200 | +0.5 [16.52, 0.7] · +43.3 [13.90, 18.3] |
| Fe II 1144 | −12.7 [14.20, 29.8] · +45.6 [13.89, 26.7] |
| H I 1215 | **`FITNCOMPS = 0`** — there is no published H I Voigt fit **[V]** |

**This is not a single cloud.** It is a blueshifted velocity ladder running from ~0 to −331 km/s,
with the strongest high-velocity component consistently at **−308 km/s** across Si II (±2.0),
Si IV (±6.0), C II (±3.0), Si III, and N V displaced to −331 (±10.3).

### 1.3.1 The fits are TIED MULTIPLETS **[V]**

`FITPARAMS` is **bit-identical** for Si II 1190/1193/1260/1304, for N V 1238/1242, and for
Si IV 1393/1402. The DIISC team fit each multiplet jointly with tied parameters.

> **Consequence for the simulation work.** The mock Voigt fits were done **per transition,
> independently, over different velocity windows** (±500 km/s for Si II 1190/1193 vs ±800 for 1260).
> Therefore the flagship simulation result *"HVC covering fraction = 0.313 / 0.344 / 0.382 for
> Si II λ1190 / λ1193 / λ1260"* **cannot exist in the real data by construction**. It is a
> **fitting-protocol systematic**, not an observational one, and must be presented as such.
> Any mock→real transfer of a Voigt-derived quantity is unvalidated until the protocols match. **[V]**

## 1.4 Velocity frames — resolving 310 vs 345 vs 350

Three numbers circulate. They are three different quantities in two frames with two tracers.

| quantity | value | frame / tracer |
|---|---|---|
| Metal HVC centroid | **−308 ± 3** (N V at −331) | systemic, z_gal **[V]** |
| H I HVC centroid | −284.6 ± 5.6 | systemic, manual fit **[A]** |
| Local co-rotating disk at the sightline | **+62.7** | VLA moment-1 **[A]** |
| **H I HVC − local disk** | **−347.3** | ← **this is the "345"** **[A]** |
| Same, vs the H I kinematic centre (+71.4) | −356.0 | ← this is the "~350" **[A]** |

**Publishable wording:**

> *The high-velocity absorber is centred at v = −308 ± 3 km/s in the low and intermediate metal
> ions and −285 ± 6 km/s in H I, in M61's systemic frame, with N V displaced to −331 km/s. The
> local co-rotating disk at the sightline lies at +63 km/s (VLA 21 cm), so the shear between cloud
> and disk is −347 km/s in H I and −371 km/s in the metals. Earlier quotes of 345–350 km/s are
> this disk-relative H I quantity. No absorption component sits at −345 km/s in the systemic frame.
> Always state tracer and frame.*

**Mandatory caveat.** The coadd carries `VELOFF_G130 = 0.0` and `Z130 = 0.0` as placeholders, and a
Galactic S II fit places the Milky Way at −17.09 ± 1.73 km/s when it should sit near 0. **Every
velocity carries a ≈17 km/s wavelength-solution systematic** **[A]** — comparable to the 24 km/s
H I-vs-metal offset and larger than the significance of the N V offset. Do not do 2 km/s forensics.

## 1.5 CORRECTION: O I *is* marginally detected at the HVC velocity **[V]**

An earlier claim in this project — repeated in conversation — was that O I shows *no* high-velocity
component, implying the HVC contains no neutral gas. Direct integration of `OI1302_slice.fits`:

| window (km/s) | W_r (mÅ) | significance |
|---|---|---|
| **−340 to −280** | **44.0 ± 12.0** | **3.7σ** |
| −360 to −230 | 59.8 ± 18.4 | 3.3σ |
| −400 to −150 | 75.0 ± 25.6 | 2.9σ |
| −900 to −600 (blue control) | −23.2 ± 29.6 | −0.8σ ✓ clean |
| −85 to +150 (published window) | 570.9 ± 18.7 | 30.6σ |

Minimum normalised flux **0.694 at −285.2 km/s**; AOD-weighted centroid **−281 km/s**, coincident
with the metal HVC. The blue-side continuum control is clean, so this is not a continuum artefact.
The published `FITNCOMPS = 2` (at +3.9 and +63.7) is an artefact of the adopted integration window
(−85 to +150) — **nobody looked blueward**.

> **Do not use the red side as a continuum control**: W = 449 ± 23 mÅ at +400..+700 **[V]**, because
> **Si II 1304.37 sits at +507 km/s in the O I 1302 velocity frame**.

**Consequences.**
1. "The HVC contains no neutral gas" becomes "W_r = 44 ± 12 mÅ, a 3–4σ marginal detection,
   with a 3σ limit log N(O I) < 14.3."
2. The HVC is **strongly** ionized but not purely so: at −308 km/s Si II is black (F_norm = 0.089)
   while O I only reaches 0.694 **[V]**.
3. Any analysis that used an "O I veto" to select mock analogues is invalid and must be re-run.

## 1.6 Blend inventory — six of twelve channels are contaminated

| channel | contaminant | offset | evidence |
|---|---|---|---|
| **Si III 1206** | Galactic Lyα damping wing + **geocoronal Lyα** | wing across window; airglow at +709 km/s | `FNORM` = 4.8 / 244 / **457** / 298 at v = +500/600/700/800; `FITCHI2` = **3041** vs 0.3–6.7 elsewhere **[A]** |
| **Si II 1260** | **S II 1259.519** | **−214.8 km/s** | 3.4σ residual absorption at −240..−190 with no fitted component **[A]** |
| Si II 1193 | MW N I 1199.55 / 1200.22 | +6.5 / +175 km/s | 186 mÅ at +100..+185 **[A]** |
| Si II 1190 | S III 1190.203 | −54 km/s | in band |
| C II 1335 | C II* 1335.663 | +254 km/s | expected |
| **O I 1302** | **Si II 1304.37** | **+507 km/s** | W = 449 ± 23 mÅ **[V]** |
| H I 1215 | MW Lyα screen, log N = 20.176 | −1567 km/s + wings | `Manual_fits/HI1215_MW_blend/` **[A]** |

**Decision:** Si III 1206 is simultaneously the strongest HVC channel and the most contaminated.
Truncate it at v < +400 km/s with an explicit MW Lyα damping model, or exclude it. Its published
Voigt decomposition is unusable — component errors reach dlogN = 10.0, db = 227 km/s, dv = 85 km/s **[A]**.

## 1.7 A physics bound worth stating **[A]**

For log M200c = 12.7: R200c ≈ 361 kpc, v200 ≈ 244 km/s, and for NFW c ≈ 8,
**v_esc(26 kpc) ≈ 600–730 km/s**. The observed −308 km/s (−347 disk-relative) cloud is comfortably
**bound**. This rules out an escaping wind while leaving fountain, recycling, and accretion open.

Note the slide's argument — "an outflow at 60° opening angle would require ~700 km/s" **[D]** — has
doubtful deprojection geometry: at i = 23° the line of sight is 67° out of the disk plane at *every*
azimuth, so both major- and minor-axis sightlines enter a 60° polar cone; and with b/a = 0.93 the
photometric PA is poorly determined.

---

# PART II — THE SIMULATION DATASET

## 2.1 Sample construction

20 TNG50-1 subhalos at snapshot 99 (z = 0), selected as the **top 20 by Euclidean distance in
(Δlog M★, Δlog M_HI, Δlog SFR) from M61's observed values**. Source: `data/m61_closest_matches_3d.csv`
(50 rows; the study set is exactly the top 20). dist3d spans 0.125–0.468 dex. **[V]**

> **Re-framing needed.** `GALAXY_PROPERTIES.md` benchmarks the sample against the **Milky Way**
> and concludes "all 20 are more massive than the MW; no strict MW analog." That is true and
> irrelevant — **M61 was the target, not the MW**, and M61 is itself log M★ = 11.39. The sample
> (log M★ = 10.96–11.58, central log M200c = 12.20–12.87) **straddles M61 correctly**. Note the
> irony that sub-488530, chosen for the FINESST figure and for `cgm_qso_488530`, is the *least*
> M61-like by halo mass (12.20 vs 12.7).

### The independence ceiling — the single most important statistical fact

**20 subhalos → 17 FoF groups → effectively 17 independent halos.** **[V]**

- 13 centrals, 7 satellites.
- **SIDs 143881 / 143884 / 143885 / 143886 are four satellites of the SAME host** (grnr = 4,
  M200c = 3.2×10¹³). They contribute **2,880 sightlines = 20% of the dataset from one CGM**.
- 167395, 307487, 342448 are lone satellites in their own groups.

**Every train/test split must be on `grnr`, never on `sid`, and never on sightline.**

## 2.2 Geometry — fixed, and this is the binding confound

| quantity | value |
|---|---|
| Impact parameter ρ | **25.558333340264905 kpc — bit-identical for all 14,400** **[V]** |
| Inclination | **23.0° for all** (M61's value) |
| R_vir used for ray geometry | **457 kpc for all** (M61's value, *not* each subhalo's R200c) |
| Ray length | **914 physical kpc for all** |
| Orientations | α = 0…359° in 1° steps × {flip, noflip} = **720 per galaxy** |
| Total | **20 × 720 = 14,400 sightlines** |

α spins the galaxy about **its own** disk normal (PCA of inner stars within 2 r_half) with the
observer fixed. `flip`/`noflip` is the **observer-side flag** (±normal → +z).

**Consequences of the fixed geometry:**
1. True R200c spans 246–671 kpc, so ρ/R200c spans **0.038–0.104** and the ray spans **1.36–3.71 R200c**.
   Every path-integrated quantity therefore carries an **artificial halo-mass correlation** — this is
   a *leak*, not merely a domain gap.
2. Any `r_gal`, `|z_disk|`, or `v_r` head can obtain its entire apparent skill from a bias term.
3. **FINESST Q2 (co-rotation vs D/R_vir, halo mass) and Q3 (population census) have literally zero
   training support in this dataset.**

**Adjacent-sightline correlation.** Adjacent α are **0.45 kpc apart**. Measured lag-1 ACF of the
binary detection sequence: ρ = 0.23–0.63, coherence ~10–15°, VIF 2.4–6.6 **[V]**. Using the
continuous observable (log N per ion) instead gives τ_int ≈ 12–29° → **n_eff ≈ 250–600 per
configuration** **[A]**. The binary-sequence VIF **understates** the problem. Report both, labelled.

## 2.3 The spectra (x)

Trident synthesis, COS-G130M, **9 transitions only**:

`Si II 1190 · Si II 1193 · Si III 1206 · H I 1216 · N V 1239 · Si II 1260 · O I 1302 · C II 1335 · Si IV 1403`

Stored per sightline per line in the combined ray HDF5:

```
spectrum_by_line/<LINE>/raw/{lambda_A, flux, tau}   <- PRE-LSF, NOISELESS
spectrum_by_line/<LINE>/lsf/{lambda_A, flux, tau}   <- LSF-convolved, NOISELESS
```

Each array is exactly **(30001,) float64**, λ = 1150.0000–1450.0000 Å, Δλ = **0.01 Å uniform**,
dispersion 2.306 km/s at 1300 Å. The grid is identical for every line, sightline, and galaxy. **[V]**

> ### This is a true amortized simulator
> Because noiseless **pre-LSF τ** is cached, *any* instrument LSF, spectral resolution, S/N, binning,
> continuum error, and velocity zero-point can be applied as cheap arithmetic with **no re-raycasting**.
> Unlimited noise realisations are free. The sibling project `/scratch/tsingh65/cgm_qso_488530`
> already uses exactly this pattern (`scripts/degrade.py`).

**Missing ions, and why it is fatal to part of the science.** G130M covers 1150–1450 Å; the real
coadd covers 1132.58–1433.86 Å. At z = 0.005224:

- **O VI 1031.9 → 1037.3 Å: permanently unobtainable for M61 with COS.**
- **C IV 1548/1550 → 1556.3/1558.9 Å: requires G160M (new observations).**

The temperature ladder is {4.0, 4.0, 4.04, 4.09} → 4.33 → 4.72 → **[gap at log T ≈ 5.0]** → 5.27 →
nothing above 5.3. Measured: R² for log T(N V-weighted) rises **0.42 → 0.84** the moment true C IV +
O VI columns are supplied **[A]**. The published result *"inflow fraction falls monotonically
H I → O VI"* has **no observational counterpart on this sightline.**

## 2.4 Truth labels (θ)

**Per-cell ray physics** — `original_trident_ray_h5/grid`, 23 fields × ~1,000 cells per ray
(**14.4M cell records**): `density`, `temperature`, `metallicity`, `velocity_los`,
`relative_velocity_{x,y,z}`, `dl`, `l`, `x`, `y`, `z`, `redshift{,_dopp,_eff}`, `H_nuclei_density`,
`H_number_density`, and six ion densities `H_p0`, `C_p1`, `N_p4`, `Si_p1`, `Si_p2`, `Si_p3`. **[V]**

> **Traps.** Cells are **not sorted** along the ray — sort by `grid/l`. `O_p0` (O I) is **absent**
> even though O I 1302 is synthesised. The ray key under each `alpha=N` node is not constant
> (`ray_000001`, `ray_000003`, …) — enumerate it.

**Absorber catalog** — `outputs/disk_ism_velocity/absorber_catalog/absorbers_sid*.parquet`,
**1,294,326 rows** (cells passing a per-ion column floor ≈ 9% of all cells), 30 columns including
N for **eleven** ions (the extra five — C IV, O I, O VI, Mg II, Fe II — recomputed via
`trident.add_ion_fields` per ray and stored **only here; no spectra exist for them**), plus `dv`,
`v_rest`, **`v_r`** (galactocentric radial; <0 inflow), `v_z`, `R_disk`, `z_disk`, `r_gal`, `s`,
`Zsolar`, `logT`, and `wrapped`/`hypervel` flags. **[V]**

**Voigt component catalog** — `voigt_catalog/voigt_components.parquet`, **310,217 rows**
(299,065 detected), 44 columns; `voigt_line_status.parquet`, 129,600 rows (the covering-fraction
denominator). Built by `build_voigt_catalog.py`; documented in `scripts/disk_ism_velocity/VOIGT_CATALOG.md`.

**v_ISM reference** — `vism_tables_v3/vism_master_v3.csv`, 14,400 rows, with v3a/v3b/v1/v2 variants
and `SiII_dip`.

**Classes:** ISM |Δv| < 40, IVC 40–100, **HVC |Δv| ≥ 100 km/s**, where Δv = v_rest − v_ISM.

## 2.5 Storage and compute

| item | value |
|---|---|
| Total ray/spectrum/fit data | ~681 GB, of which **~442 GB is exact triplication** **[A]** |
| **Compact training-ready re-encoding** | **< 1.5 GB for all 14,400 sightlines** **[V]** |
| /scratch quota | 4.96 TiB used of **100 TiB**; 558 TB free on the filesystem |
| /home | **only 31 GB free of 100 GB** — never install there |
| ML stack | **none installed anywhere** (no torch/sbi/jax). PyPI is reachable. |
| GPU | `public`: 208× A100-80GB, 7-day limit; `htc`: adds H100/L40, 4-hour limit |
| Queue behaviour | **1× A100 backfills in ~20 min; 4× A100 waits ~16 h** → use single-GPU job arrays |
| `general` partition | **inaccessible** (no owned nodes); no `long` QOS → 7 days is the ceiling |
| Interactive session | 20 GiB cgroup, 12 cores, 10 h, no GPU — **never train here** |

> **The absolute-path gotcha.** `conda activate` leaves a numpy-less python on PATH because a
> cluster-wide `PYTHONPATH=/etc/python` is injected. Every script must use an **absolute** python
> path. See §9.3 for the verified sbatch preamble.

**Do not delete the "442 GB triplication"** until re-verified: the inventory does not reproduce
(one sid measures 46–176 GB by `du`), the nominated directories feed the τ-from-grid work, and
there is 558 TB free.

---

# PART III — WHAT IS ALREADY ESTABLISHED

The traditional analysis is **complete and validated** (27 publication figures × 2 v_ISM variants;
`VOIGT_CATALOG.md`, `VISM_V3.md`, `VALIDATION.md`). **ML adds nothing to it and should not be
applied to it.** Key results, all **[V]** unless noted:

- **Ionization stratification.** Cool low ions (H I, Si II) are velocity-confined and ~70% inflowing;
  warm-hot ions are broader and less inflow-dominated. Inflow fraction falls monotonically H I → O VI.
- **Si II transition systematic.** HVC covering fraction 0.382 / 0.344 / 0.313 for λ1260/1193/1190
  — a ~22% line-choice systematic. **But see §1.3.1: this is a fitting-protocol artefact, not
  observational.**
- **HVC column concentration.** Peaks at |Δv| = 100–150 km/s; ~60–71% below 200 km/s.
- **v_ISM robustness.** v3a and v3b agree to < 0.003 in covering fraction (σ ≈ 19 km/s vs the Si II dip).
- **Metallicity.** Warm-hot HVC gas is metal-richer in outflow (+0.17 dex, significant at galaxy level);
  the cool-ion reversal is **not** significant once errors bootstrap over galaxies.
- **Spectrum ≡ gas velocity.** `VALIDATION.md` Tier 1: median(AOD − gas) = +0.86 km/s, robust
  σ = 0.21 km/s over 85,779 measurements. This licenses using spectra as a proxy for gas kinematics.

## 3.1 Data traps that any ML pipeline must respect **[V]**

1. pyGad's `dv_kms` is the **uncertainty** on `v_kms`, not a velocity offset (stored as `v_err_kms`).
2. `sat` and `Chisq` are **per fit region**, constant across a fit's components — never per-component.
3. **H I 1216** was fit over ±1200 km/s with b_max = 150 while metals used ±800 with b_max 50–100.
   59.5% of its components sit at the b ceiling and 60.6% beyond |Δv| > 500 — damping-wing fits, not
   clouds. HVC covering fraction 0.930 raw → **0.486 cleaned**.
4. H I fitted `logN` is the **fitter's bound** for 57.9% of clean components (`fit_logN_max` = 19.50).
5. Si II 1190/1193 were fit over ±500 km/s vs ±800 for 1260.
6. **Si II 1260 carried a +23.76 km/s rest-wavelength zero-point error** (pyGad 1260.522 vs Trident
   1260.422 — exactly 0.1 Å). Corrected in the catalog via `lam0_pygad`/`v_zeropoint_kms`, **not** in
   the raw fit HDF5s. Diagnose any line with `lam0 = lambda_A / (1 + v_obs/c)`.
7. The absorber catalog uses **v1** v_ISM; the Voigt catalog carries v1/v2/v3a/v3b.
8. Default science sample: `~b_at_ceiling & ~beyond_common_window`.

---

# PART IV — THE CENTRAL FINDING: AN IDENTIFIABILITY CEILING

Five independent design tracks measured the recoverability of inflow-vs-outflow from the spectrum,
with different features and protocols. **They agree.** **[A — re-derive before publication]**

| estimator | leave-one-group-out AUC for sign(v_r) |
|---|---|
| Full spectrum (pixels) | 0.46 – 0.50 |
| LOS velocity offset alone | **0.508** |
| Voigt summaries (9 lines × 12) | 0.54 – 0.60 |
| Photoionization **oracle** (true T, Z) | 0.62 – 0.70 |
| **True metallicity alone** | **0.690** |
| Oracle + full 3-D position | 0.68 – 0.74 |
| *Random per-sightline split, same features* | *0.68 – 0.76* ← **leakage** |

Three load-bearing conclusions:

1. **The spectrum is near-chance.** Accuracy is below the majority-class rate. A posterior on flow
   direction will return the training prior for essentially every input.
2. **≥98% of the achievable skill is metallicity** — and metallicity is unrecoverable from G130M at
   z = 0.005 (R² = −0.02 from observables; no Lyβ at 1031 Å, no Lyman limit at 921 Å; 84% of truth
   HVCs are Lyman-limit thick; H I is saturated in the real data). **Origin ≈ f(Z); Z is inaccessible;
   therefore origin is inaccessible.** This chain is the headline.
3. **Random splits inflate AUC by +0.10 to +0.19** on sightlines 0.45 kpc apart. A cautionary
   methodological result the SBI-in-astronomy literature needs.

**Therefore the program's headline is an identifiability law, not an origin claim:** *the information
content of a single CGM absorption sightline about the radial flow direction of high-velocity gas,
measured as a function of inclination, ion ladder, and metallicity access.* The positive deliverables
are a calibrated amortized posterior for what **is** identifiable, plus a quantified saturation correction.

---

# PART V — THE RECOMMENDED METHOD

## 5.1 θ — inference targets

**Rejected, with reasons:**

| rejected | why |
|---|---|
| Anything thresholded on `dv = v_rest − v_ISM` (HVC class, K_hvc, N_SiII^HVC, f_HVC, `in_disk`) | v_ISM is **spectrally derived** (σ ≈ 19 km/s vs the Si II dip) and the column distribution **peaks at the 100 km/s threshold**. A network locating the Si II trough is re-deriving the label operator. Leakage + label noise. |
| `v_ISM` itself | Regression onto a smoothed version of the input. |
| `log Z` | R² = −0.02 at S/N 10. Not identifiable. |
| `log M200c` | 17 discrete values over 0.67 dex, plus the ray-length leak. |
| path length ℓ | N = n_H × ℓ exactly; one prior-returning direction. |
| Categorical K head | Truth clouds = 1.29/sightline vs pyGad median 20; 42% of fitted K = 1. Ill-posed. **Descope (§10.6).** |

**Adopted — Tier A (sightline-level, θ ∈ ℝ⁹), all v_ISM-free, computed on rays truncated at
±1.0 R200c(sid):**

1. `log N_HI` (total)
2. `log T` (H I-column-weighted)
3. `log n_H` (H I-column-weighted)
4. `v_los` (H I-column-weighted, galaxy systemic frame)
5–7. `log N` for Si II, Si III, Si IV in **fixed galaxy-frame velocity bands** ±[0,50), [50,150), [150,∞)
8. `v₅` — 5th percentile of the cumulative N_SiIII(<v) profile (blue velocity extent)
9. `f_in` — **continuous** column-weighted inflow fraction Σ N·1[v_r<0] / Σ N over v < −150 km/s

θ₉ is *expected* to return the prior. **That is the deliverable**, reported with posterior
contraction 1 − Var(post)/Var(prior). Pre-register a drop threshold of 0.2.

## 5.2 x — the observable

**Grid:** the **real DIISC per-line `VEL` grid** (1001 samples, dv ≈ 6.6–7.7 km/s), restricted to
v ∈ [−800, +800] → ~236 samples. **Degrade mocks onto the observed grid; never resample the
observation.** Mismatched sampling is the classic silent SBI failure.

**Channels per line: 4** — (0) normalised flux; (1) saturation-clipped pseudo-optical depth
a = −ln(clip(F, 3σ, 1)); (2) per-pixel 1σ error; (3) validity mask. **Channel 2 is mandatory** —
the augmentation sweeps S/N over an order of magnitude and without σ as input the posterior widths
are wrong at both ends.

**Lines: 12, not 9.** The nine synthesised plus **Si II 1304, Si IV 1393, N V 1242**, re-synthesised
from the cached per-cell ion densities + `temperature` + `velocity_los` (a Voigt sum, milliseconds
per line, no re-raycasting). This restores the doublets — **the only saturation diagnostics the real
data has** — and **Si II 1304, the only unsaturated Si II measurement in the real spectrum**
(log N = 14.79, flag 1).

> **Mandatory gate:** the offline synthesis must reproduce a *stored* line (Si IV 1403) from the same
> grid to **< 1%** before any synthesised partner is used. If Trident's ionization/partition cannot be
> reproduced offline, the recomputed partner is inconsistent with the primary and the diagnostic is void.

**Forward operator — order matters, and getting it wrong corrupts exactly the saturated cores where
the HVC lives:**

```
raw/tau  ->  resample to a 2.0 km/s galaxy-frame grid      [upsample of native 2.14-2.52 km/s: lossless]
         ->  F = exp(-tau)
         ->  convolve F with the TABULATED STScI COS G130M LSF interpolated to each line's
             OBSERVED wavelength, boundary='extend'         [never zero-pad; never convolve tau]
         ->  flux-conserving rebin onto the DIISC VEL grid via a PRECOMPUTED SPARSE MATRIX
         ->  multiply by a continuum nuisance polynomial
         ->  Poisson noise in counts at native sampling, then bin 3, then rescale so the binned
             residual RMS matches 0.75 x ENORM               [ENORM overestimates by ~33%;
                                                              lag-1 corr = 0.185]
         ->  normalise with the same normaliser applied to the data
```

> **Never use the stored `lsf/flux`.** It is Trident's generic wavelength-independent kernel applied
> with zero padding, corrupting the last ~50 pixels of every array. **`raw/tau` is the only true
> primitive** (`lsf/tau` is byte-identical to `raw/tau`). **[A — verify]**

## 5.3 Architecture — a gated ladder, not a fixed design

**No transformers.** Self-attention over 9–12 tokens is a permutation-equivariant MLP with extra
steps; attention over ~10 pooled positions after a CNN whose receptive field already exceeds the
sequence is a no-op. Three design tracks proposed them; three critiques independently killed them.

**Rung 0 — mandatory baseline (CPU, hours).** Per line, apparent-optical-depth column profile
N_a(v) in 20 velocity bins → 12 × 20 features + log S/N + mask → gradient-boosted quantile regression.
Plus τ-weighted centroids, plus the pyGad Voigt summary vector (re-run with `grnr` groups).
**Every headline metric must be reported against all three on the same 17 folds.**

**Rung 1 — primary SBI (~0.35 M params).** ~180 summary features (AOD profiles + doublet ratios +
velocity moments) → `zuko.flows.NSF(features=9, context=190, transforms=6, hidden_features=(192,192), bins=8)`.
Bounded components via a logit transform. NSF rather than MAF because θ marginals are bounded and multimodal.

**Rung 2 — pixel-level (~0.6 M params), built only if it clears the gate.**

```
per-line dilated residual CNN, weights SHARED across all 12 lines (fold line axis into batch)
  Conv1d(4->48, k=7); 6 ResBlocks [Conv1d(48,48,k=5,dil=d) -> GroupNorm(6) -> GELU] x2 + skip
  d in {1,2,4,8,16,32}          (receptive field 505 > 236 -> full context)
  FiLM(gamma,beta) at every block from a 6-vector [log f, log l0, log(f*l0), A, IP_low, IP_up]
  -> per-velocity MLP mixing 12x48 -> 64        [replaces a cross-ion transformer]
  -> attention pool + max pool -> s in R^128
  context c = [s, log sigma_pix (12), log FWHM_LSF, bin factor, mask (12)]
  -> the same NSF head
```

**Weight sharing across lines is the main defence against 17 groups** — it forces one curve of growth
instead of twelve unrelated channels, and it is what makes the Si II f-ratio ladder
(f = 0.250 / 0.499 / 1.007 / 1.09 for λ1190/1193/1260/1304) cooperate.

> **Gate for Rung 2:** it must beat Rung 1 by **more than one between-fold standard deviation** on the
> LOGO-17 mean. Fold sd in AUC is 0.10–0.15. On 17 halos this is a bar it probably cannot clear, and
> finding that out in week 4 is worth more than finding it out in month six.

**Do not build:** a 241-bin × 10-field autoregressive velocity-field head (a 2,410-dim posterior from
~20 resolution elements); a standalone NRE (a calibrated Bernoulli gives the Bayes factor analytically);
any conditioning on a per-galaxy constant such as `log(L / 2R200c)` (that is the halo ID — fix the leak
by truncating rays at ±1.0 R200c instead).

## 5.4 Inference flavour — and a real methodological point

**Amortized NPE, single round.** State the reason in print:

> **θ is a summary of a ray, not an input to the simulator.** You cannot choose θ and generate x —
> the simulator samples galaxies and orientations. **SNPE / SNLE / SNRE all require simulating at a
> proposal q(θ) and are therefore structurally not implementable here.**

NLE is worse: x is thousands of dimensions and p(x|θ) is genuinely broad. **The amortization that
*is* real is over the observation configuration** (S/N, LSF, binning, continuum, velocity zero-point),
because that is exact arithmetic on cached τ. Claim that, and nothing more.

## 5.5 Training

```
splits      leave-one-FoF-group-out on grnr, 17 folds. NEVER sid, NEVER alpha.
            Model/hyperparameter selection and early stopping use 3 INNER validation groups
            drawn from the 16 training groups; the outer group is touched once.
optimizer   AdamW lr 3e-4 -> cosine 1e-5, 5% warmup, wd 1e-2, grad clip 1.0, bf16
batch       256; ~53 steps/epoch; 200 epochs; early stop on inner-validation NLL
loss        -log q(theta | c); no auxiliary heads
prior       train unweighted first. Importance reweighting toward the tail is HIGH RISK -- it
            upweights the highest-dispersion sightlines, which are the grnr=4 satellites. If used,
            compute N_eff,groups = (sum w)^2 / sum w^2 per grnr and ABORT if fewer than 8 groups
            each carry >= 5% of total weight.
```

**Augmentation** (all on GPU inside the training step; fresh draw every visit; store nothing):

- S/N log-uniform per line in **[3, 20]** around a per-sightline base — the **real per-pixel S/N is
  3.9–7.8**, so the existing mock Voigt fits at S/N = 10 are ~2× too clean **[A]**
- LSF sampled from tabulated G130M kernels at LP1–LP4 ± 1 native pixel shift
- binning ∈ {1, 2, 3}
- continuum × (1 + a₁T₁ + a₂T₂), a ~ N(0, 0.02)
- coherent velocity zero-point N(0, 8) km/s + per-line jitter N(0, 2), and with p = 0.05 a single-line
  ±25 km/s offset (**the +23.76 km/s Si II 1260 bug proves these are real**)
- **contaminant injection p = 0.3**, 1–3 random Voigt absorbers, without changing θ
- channel dropout p = 0.15

**Forbidden:** velocity-axis flip (inverts the quantity being inferred); τ amplitude scaling;
mixup/cutmix; per-channel velocity jitter beyond a few km/s (destroys ion-to-ion alignment, which
*is* the multiphase signal); resampling α as augmentation.

**What augmentation does and does not do.** It buys ~300 fresh (x, θ) pairs per sightline per run and
**zero additional independent halos**. It fixes *calibration* — training on the noise-marginalized
posterior p(θ | x_obs) rather than the unobtainable p(θ | τ) — and controls variance. **It does not
touch bias, and every error bar is unchanged by it.** Say this in print; SBI referees check for it.

## 5.6 Validation

| test | specification |
|---|---|
| **Split** | LOGO-17 on `grnr`. Report every metric as a fold mean **with between-fold sd**, noting that grnr=4 holds 2,880 sightlines vs 720 for the rest. |
| **SBC** | Two runs: in-distribution (tests the algorithm) and LOGO (tests transfer). **Thin the test set at 2–3 τ_int (60–90° in α)** → ~160–240 quasi-independent points. 30° leaves ρ ≈ 0.37. ECDF-difference test with simultaneous bands + Bonferroni across 9 dimensions. |
| **Coverage** | TARP + per-dimension coverage. **The usual A < 0.03 gate is unpassable by construction**: E[A]_null = 0.313/√N = 0.014 at N = 480 but **0.076 at N = 17**. Replace with a **parametric-bootstrap null**: draw θ from the fitted posterior per test x, recompute A, gate on A_obs / A_null^(95th) < 1. Report within-group and between-group components separately. **Always report sharpness beside coverage** — a flow can be perfectly calibrated and useless, and for θ₉ it will be. |
| **Contraction** | 1 − Var(post)/Var(prior) per θ dimension. Drop anything < 0.2 from the headline. |
| **PPC** | Nearest-neighbour only (no re-runnable simulator): retrieve the 20 *training-fold* sightlines nearest in standardised θ and check the observation lies inside their band. **Report how many distinct FoF groups those 20 come from.** If the answer is 1, the PPC is meaningless. |
| **Misspecification** | **C2ST** between mock and observed summary vectors, null built by **permuting FoF-group labels**. Replaces the "≥6/9 PPC" rule (99.2% pass rate under the null) and "coverage within 5%" (needs ~348 groups). |
| **Leave-one-line-out** | Run inference 12× masking one line each; report max pairwise Wasserstein-1 between posteriors normalised by mean width, null calibrated on held-out mocks. **Works on real data with no truth — the best diagnostic in the program.** |
| **Memorization** | Two zero-cost controls: (a) **geometry-only baseline** — predict every target from (sid, α, mode) with no spectrum; any target where this matches the full model is not being measured; (b) hold out one FoF group **and** a contiguous 90° α wedge, evaluate on the wedge only. |

## 5.7 The two non-SBI analyses worth doing alongside

1. **The identifiability ladder** (§8.1, Phase 0.3). Pure sklearn, ~1 CPU-hour, and it *is* the
   headline science. It defines the ceiling every architecture competes against and converts the
   referee's most damaging objection into our own result.
2. **The DIISC data paper** (§8.1, Phase 0.4). Frames, O I, blends, tied-fit forensics, escape
   velocity. Zero simulation dependence, publishable standalone, corrects the record.

---

# PART VI — ALTERNATIVE ML AVENUES (ranked)

| # | avenue | verdict |
|---|---|---|
| 1 | **The Voigt information ledger.** Train identical predictors on (i) raw spectrum, (ii) Voigt summaries, (iii) both; compare posterior width / predictive log-likelihood / mutual information on identical folds. Both representations exist in the **same HDF5** for all 14,400 sightlines. | **PURSUE — the cleanest identical-footing comparison available anywhere, and relative paired-fold comparisons are the one result robust to n = 17.** |
| 2 | **The saturation correction.** All five strong M61 low-ion columns carry DIISC flag 9. Mocks show the Voigt/AOD path underestimates high-velocity N(Si II) by ~1 dex (RMSE 1.44 → 0.74 dex; H I 2.76 → 1.04) **[A]**. A multi-transition learned estimator removes the bias. | **PURSUE — a factor-of-15 correction to inferred cloud mass, applicable to the real spectrum today.** |
| 3 | **Interpretable prediction of HVC emergence** (GBT + SHAP on gas conditions). | **PURSUE with care** — must explicitly exclude circular features (`dv` is *defined* from `v_rest` and `v_ISM`). |
| 4 | **Coarse velocity-band tomography** (6–8 bands × {6 ion columns, ⟨log T⟩, Σdl}). | **MAYBE** — the well-posed replacement for the descoped per-cloud head. Must beat a 3-band version first. |
| 5 | **Unsupervised absorber taxonomy** (clustering / VAE on spectra or gas clouds). | **MAYBE** — hard to validate that a cluster is physical rather than an artefact of the fixed geometry. |
| 6 | **OOD / anomaly scoring** for the real spectrum. | **PURSUE as a component**, not a standalone result (§7.3). |
| 7 | **1-D inverse profile recovery** (n_H(l), T(l), Z(l), v(l) from the spectrum). | **REJECT** — ~1,000 cells from ~20 resolution elements; prior-dominated. |
| 8 | **Origin classification from spectra alone.** | **REJECT as a deliverable** — it *is* the null result (Part IV). Keep it as the measurement. |
| 9 | **Spectrum emulation to bypass Trident.** | **REJECT** — raycasting is already done and cached τ makes re-degradation free. |

---

# PART VII — SIM-TO-REAL

## 7.1 Rebuild the prior-predictive check

The existing version ("7/9 metal EWs above the 95th percentile; 0/2400 matching all nine; a ≲1% tail
event") is **not defensible**. Four required fixes **[A]**:

1. **Blend-mask first.** Si III's excess is concentrated at +100..+185 km/s where no other ion absorbs
   — that is the MW Lyα wing. Si II 1193's excess at +175 is MW N I 1200.22. Re-integrate blue-side-only
   (−370..0) with masks.
2. **Leave-one-mock-out control.** Treat each mock as the observation; if the match rate is also ≈0,
   the statistic measures the tolerance, not misspecification.
3. **Report groups, not sightlines.** "k of 17 groups produced any matching sightline," not "10 of 2400"
   (≈2.5 effective events). **Test the environment hypothesis first**: if most matches are grnr=4 — the
   only group-environment analogues — the result is "*the analogue selection omitted environment and
   M61 is in Virgo*", which is better, actionable, and more defensible than "TNG50 is wrong."
4. **Use EWs, not log N.** Seven of nine DIISC log N carry flag 9 (saturated lower limits);
   `frac(mock ≥ obs)` on a lower limit is vacuous. The one valid column-space comparison is H I:
   observed W_r(Lyα) = 2886 mÅ over −550..+420 → **log N ≈ 19.1 if damped**, against a mock median of
   **20.37**. **The real sightline is a sub-DLA; the mocks are DLAs.** That ~1 dex mismatch is a genuine
   geometry failure and is invisible to a flag-9 comparison. **[A — high priority to verify]**

## 7.2 The domain gap, ranked by how badly each biases a posterior

1. **UVB / local ionizing sources.** The cached-τ simulator **cannot vary the UVB** — ion fractions are
   baked in. At 26 kpc inside R_HI in a 5 M⊙/yr galaxy, local sources are plausible and Si III / Si IV /
   N V are the most sensitive. **This is the dominant unmitigated systematic.** Either budget a
   scaled-UVB re-run on a subsample (Phase 2d) or state in print that it is unaddressed.
2. **TNG metallicity.** All 20 analogues have starZ = 1.9–2.5 Z⊙; truth HVC log Z spans only −0.34 to
   +0.21. This can be misread as "metal-rich → outflow."
3. **Environment.** M61 is in Virgo; TNG50-1's 51.7 Mpc box contains no Virgo. The analogues were
   selected on (M★, M_HI, SFR) with no environmental or halo-mass constraint.
4. **Fixed geometry** (§2.2) — testable, and Phase 1 tests it.
5. **Instrumental/continuum systematics** — the ≈17 km/s wavelength solution (§1.4), blends (§1.6).

## 7.3 If the real spectrum is OOD — the likely outcome. Pre-committed protocol.

1. **Do not report a calibrated posterior for the real sightline.** Say so plainly.
2. **Coarsen.** Move from pixels to a ~10-number summary (blue-side-only masked EWs, the Si II f-ratio
   ladder from 1190/1260/1304, N V/Si II, the O I limit, v₅, Δv_HVC−ISM) and report a likelihood ratio
   between hypotheses there. Coarse summaries are far more robust to misspecification than pixels.
3. **Retrain with a nuisance layer** (continuum coefficients; LSF width scale ∈ [0.8, 1.3]; velocity
   zero-point ±20 km/s; per-line metal-column offset ±0.3 dex for the 2× solar TNG metallicity; error
   rescale ∈ [0.8, 1.25]) and re-test.
4. **Report which summary drove the flag, and publish that.**
   > *"TNG50-1 M61 analogues at ρ = 25.6 kpc, i = 23° cannot simultaneously produce a Si II component
   > with b ≈ 11 ± 3 km/s and log N = 13.6 at −308 km/s, a sub-DLA H I column, and N V at log N = 14.4"*

   is a **better** paper than an uncalibrated posterior, and a concrete falsifiable claim about a
   specific simulation.

**Two sharp, cheap falsifiers already available:**

- **The b-distribution test.** The observed Si II HVC has **b = 10.9 ± 2.7 km/s** **[V]**, implying
  T ≤ 2.0 × 10⁵ K, i.e. b_turb ≈ 10.6 km/s at T = 10⁴ K — an ordinary CGM line width, *not* a
  "genuinely cold" cloud. Compare against the mock b-distribution at matched velocity. One afternoon.
- **The N V test.** The observed doublet EW ratio is **1.72 ± 0.46** full-window, **1.65 ± 0.79** in the
  HVC window **[A]** — consistent with the optically thin 2.0, so N V is clean and unsaturated and
  log N = 14.44 is a genuine measurement. Only 6.1% of mocks reach it **[A]** → **a real TNG50 N V
  under-production**, publishable as a result rather than an open question.

---

# PART VIII — STAGED EXECUTION PLAN

## Phase 0 — Audit and identifiability (≈5 weeks, < 100 CPU-h, 0 GPU-h, data already on disk)

Phase 0 produces **two publishable results on its own** and gates everything else.

**0.1 Velocity-convention CI test (1 day) — BLOCKING.** Two design tracks disagreed on the sign of the
mock velocity axis. Adjudicate empirically: τ-weighted centroids of the optically-thin N V / Si IV /
Si III channels vs the ion-column-weighted `v_rest` in `absorbers_sid*.parquet`, on 3 sids × {flip,
noflip} × α ∈ {0, 120, 240}. Require agreement < 2 km/s; cross-check that the mock projected disk
rotation has the same handedness as the VLA moment-1 map. **`v_sys` reaches ±410 km/s per sightline —
a sign error manufactures an 820 km/s artefact.** Wire into CI before any training.

**0.2 The front/back mechanism (2 days).** Confirm from `orient_m61.py:446-453` that `flip`/`noflip` is
the observer-side flag. At matched (sid, α): does `v_los` change sign? does truth `v_r` change?
(`v_r` is galactocentric, invariant under an observer rotation — so only the *selection* of which
clouds land in the blueshifted window can change, which is exactly the front/back degeneracy.)
Report f_inflow conditional on mode, and separately for grnr = 4.
**Deliverable:** whether observer-side conditioning would resolve the M61 question, and how confidently
M61's near side can be determined at i = 23° with b/a = 0.93.

**0.3 The identifiability ladder (3 days, ~2 CPU-h) — GATES THE WHOLE SBI PROGRAM.**
LOGO-17 on `grnr` (all previously published ML-adjacent numbers used `sid` and must be re-run).
Targets: sign(v_r), f_in, log N_HI, log T, log n_H, v_los, band columns. Feature ladder:
geometry-only → AOD columns → τ-weighted centroids → Voigt summaries → oracle(T, Z) → oracle(+3D position).
Report fold-level values, between-fold sd, and a **group-level sign test** — the defensible significance
statement is *"15/17 folds > 0.5, p = 2.4 × 10⁻³ ≈ 3σ"*, not "5.5σ" from std/√17 on 13×-imbalanced folds.
Delete the circular `log N_HI ← [features containing N_HI]` cell.
> **GO/NO-GO:** if the spectrum-level ladder cannot beat AOD by more than one between-fold sd on *any*
> Tier-A target, the SBI build stops at Rung 1 and the program becomes the identifiability paper plus
> the data paper.

**0.4 DIISC data forensics (1 week) — this is Paper I, and it needs no simulation.**
Frames table (§1.4); O I re-measurement (§1.5, **done**); tied-fit forensics (§1.3.1, **done**); the
blend inventory (§1.6); Si III airglow/damping-wing quantification; the N V doublet ratio; the
escape-velocity bound (§1.7); the ≈17 km/s wavelength systematic.

**0.5 Prior-predictive check, rebuilt (3 days).** All four fixes of §7.1.
> **GO/NO-GO:** only if the misspecification survives all four fixes does it go in an abstract.

**0.6 QA and re-encoding (1 week).** Resolve **Σdl = 1,012 vs 914 kpc** (blocking — every column is
Σ n dl). Recompute all θ on rays truncated at ±1.0 R200c(sid) and report how much covering fractions
move (this isolates the fixed-ray-length leak). Synthesise Si II 1304, Si IV 1393, N V 1242 (and
C IV 1548/1550 as a G160M forecast), **gated on reproducing stored Si IV 1403 to < 1%**. Build the
compact HDF5 (§9.2). Build and verify the environment (§9.3).

**Phase 0 deliverables:** Paper I submitted; the identifiability figure that becomes Paper II Figure 1;
a sign-correct, leak-free, RAM-resident ~2.1 GB training store; a go/no-go on the SBI build.

## Phase 1 — Geometry grid + primary SBI (≈8 weeks, ~1,700 core-h, ~60 A100-h)

**The key insight: expand in (ρ, i) on the 20 cutouts already on disk. Do not chase more galaxies yet.**
More galaxies at fixed geometry buy tighter error bars on an artefact — the binding confound is that
ρ, i, PA and R_vir are constants.

```
20 galaxies (17 groups) x rho in {10, 25.56, 50, 100} kpc
                        x i   in {23, 45, 70, 85} deg    (23 exact, so the existing set is a subset)
                        x 24 alpha (15 deg spacing ~ tau_int)
                        x 2 modes
= 15,360 rays
Cost:  24.1 s/ray on 16 cores = 0.107 core-h/ray -> ~1,640 core-h
       as a 20-task array of 16-core tasks: ~5 h wall
Disk:  compact product only, ~2 GB. NO combined/all_rays bundles (1.3 TB of triplication).
       NO Voigt refits (0.32 core-h/sightline = 5,000 core-h saved; fit a 5,000-sightline
       stratified subsample at the TEST S/N for the baseline comparison only).
```

**Two bugs to avoid, both caught in review:**
- Slim-cutout radius must be stated in **absolute kpc** — `r < max(1.05 × ray half-length + softening,
  1.15 R200c) ≈ 500–770 kpc`. 1.15 R200c = 283 kpc for sid488530 would **truncate every 457 kpc ray**.
- Add `O_p0_number_density` to the saved ray grid (O I 1302 is synthesised but has no cell-level truth).

**Deliverable:** train at (25.56 kpc, 23°) only; evaluate **frozen** at all 15 other (ρ, i) cells.
Report ΔAUC(i) and Δbias(log N_SiII)(ρ). **Expected: AUC for flow direction rises monotonically with
inclination**, converting the central negative result into a quotable **geometric identifiability law
directly useful for DIISC target selection.**

Also: Rung-0 baselines, Rung-1 flow, LOGO-17 with inner-validation selection, SBC ×2,
bootstrap-calibrated TARP, contraction per θ, C2ST; the information ledger vs Voigt summaries; and the
L2-vs-L3 truncation null on **200 sightlines** stratified over all 17 groups (~40 core-h). Do **not**
run full L3 (2,900 core-h to change nothing measurable — G130M carries no O VI, the only ion for which
the outer halo matters).

> **GO/NO-GO for Rung 2:** Rung 1 must beat AOD+GBM by > 1 between-fold sd on ≥ 2 Tier-A targets **and**
> pass the bootstrap-calibrated TARP null. Otherwise ship Rung 1 and stop.
> **GO/NO-GO for Phase 2:** AUC(i) must rise monotonically and significantly with inclination. If it
> does not, the null is a fact about absorption spectroscopy rather than about M61's geometry, the
> science content is complete, and no expansion is warranted.

## Phase 2 — Conditional extensions (choose at most two; ~3 months)

- **2a. NGC 3810 / J1140+1136** — the matched **non-detection control**, already on disk in identical
  formats (most blueshifted component anywhere is Si III −135 km/s). A detection plus a matched
  non-detection is a genuine 2-point covering-fraction constraint. **Check NGC 3810's inclination
  first** — if it is more edge-on, the identifiability law predicts the origin *is* recoverable there,
  which is a strong test.
- **2b. Coarse velocity-band tomography** — the well-posed replacement for the descoped K head. Must
  beat a 3-band version before the 8-band version is built. Drop ⟨r_gal⟩, ⟨z_disk⟩, ⟨v_r⟩ as
  deliverables (prior-dominated); keep as prior-sensitivity diagnostics.
- **2c. Galaxy expansion** — only if 2a/2b are unnecessary. **Evaluate the TNG public API cutout route
  first** (~30 GB server-side vs a 2.7 TB chunk re-download; the parent snapshot directory
  `/scratch/tsingh65/TNG50-1_snap99/data/` is **empty**). **Select to *span* (log M200c 11.8–13.6,
  sSFR in 3 bins), not to match** — widening σ_x(log M200c) from 0.20 to 0.50 dex is worth 6× in slope
  precision, more than any plausible galaxy count at fixed properties. Note this changes the project's
  identity from "the M61 paper" to "a CGM population paper."
- **2d. Scaled-UVB / local-source re-run** on a subsample — the only route to the confound the cached-τ
  simulator provably cannot reach.

## Phase 3 — Cross-suite (Year 2, conditional)

FIRE-2 + SIMBA M61-analogue rays at matched ρ and i, using the existing `OVI_CGM_compare` galaxy set.
This is a **new matched-geometry raycasting campaign producing COS sightlines with tied Voigt fits**,
not a port of the existing map pipeline.

> **If Phase 3 is not funded/scoped, delete "leave-one-suite-out" from the proposal text.** With one
> simulation it is a promise the dataset cannot keep.

---

# PART IX — PRACTICAL SETUP

## 9.1 What to build where

| item | location |
|---|---|
| ML env | **`/scratch/tsingh65/envs/sbi311`** — /home has only 31 GB free |
| Training store | `/scratch/tsingh65/m61-sbi/data/` |
| Code | `/home/tsingh65/m61-tng/scripts/` (repo), outputs to `/scratch` |

## 9.2 Compact on-disk encoding

One HDF5 per data version, chunked at ~1.4 MB (BeeGFS prefers ≥ 1 MB reads), lzf:

| dataset | shape (N = 14,400) | size |
|---|---|---|
| `tau_raw` (12 lines, 2.0 km/s, ±1200 km/s) | (N, 12, 1201) f32 | 830 MB |
| `cells` (16 fields, sorted by `grid/l`, padded to 1300) | (N, 1300, 16) f32 | 1.20 GB |
| `theta_A` (§5.1, both full-ray and ±1.0 R200c) | (N, 2, 9) f32 | 1 MB |
| keys: `key_grnr`, `key_sid`, `key_mode`, `key_alpha`, `key_rho`, `key_inc` | (N,) each | < 1 MB |
| `gal_feat` (diagnostics only, **never** targets) | (N, 8) f32 | < 1 MB |
| **total** | | **~2.1 GB** (→ ~4.5 GB post-Phase-1) |

**RAM-resident. No DataLoader, no worker processes, no BeeGFS I/O during training.** `tau_raw` fits in
A100 HBM — put it on the GPU once and run the whole augmentation pipeline as GPU kernels inside the
training step.

**HDF5, not parquet** (pyarrow reports 20.0.0 but imports 13.0.0 in the only env that has it — convert
the parquet catalogs to HDF5 **once**, in the `trident` env, so the ML env never needs pyarrow).
**Not zarr** (hundreds of thousands of inodes for no benefit; 558k of a 20M cap already used).

## 9.3 Environment

**First check whether a build is needed at all:** `/packages/envs/pytorch-gpu-2.3.1-cuda-12.1/bin/python`
exists and carries astropy 6.1.3 / numpy 1.26.4 **[V]**; reportedly also torch 2.3.1 + CUDA 12.1,
pyarrow 16.1, h5py 3.11. If `torch.cuda.is_available()` is True inside a `--qos=debug` job, a
`--system-site-packages` venv + `pip install zuko` is a 10-minute job.

Fallback:

```bash
export PIP_CACHE_DIR=/scratch/tsingh65/.cache/pip
module purge && module load mamba
mamba create -y -p /scratch/tsingh65/envs/sbi311 -c conda-forge python=3.11
PY=/scratch/tsingh65/envs/sbi311/bin/python
$PY -m pip install torch --index-url https://download.pytorch.org/whl/cu126
$PY -m pip install zuko h5py scipy astropy matplotlib tqdm scikit-learn lightgbm
```

**Install `zuko` only, not `sbi`** — we hand-roll NPE; `sbi` pulls a dependency tree that will cost a
day of debugging. Do **not** `module load cuda/cudnn` (their `LD_LIBRARY_PATH` shadows the wheel's
bundled libraries). `unset PYTHONPATH`, `PYTHONNOUSERSITE=1`, absolute python path, never `conda activate`.

**SLURM: single-GPU job arrays only.** `--array=1-17 --gres=gpu:a100:1 -c 12 --mem=48G` on `public`
(~20 min backfill) or `htc` (near-instant, 4 h cap). **Never `gpu:a100:4`** (~16 h queue). GPU jobs
require `--mem ≥ 24G`.

**Six 15-minute `--qos=debug` smoke tests before any array:** GPU sanity (assert bf16 matmul ≥ 100
TFLOP/s, to catch a CPU-only wheel); data contract (assert 17 unique `grnr`, finite τ, ρ constant);
augmentation round-trip (noiseless limit reproduces exp(−τ) to < 1e-5); 50 training steps with loss
decreasing; checkpoint reload bit-identical; **AOD baseline runs and its LOGO score is recorded as the gate.**

## 9.4 Compute totals

A ≤ 1 M-parameter model at batch 256 is ~15 min/fold. LOGO-17 × 8 configs (nested, unbiased) ≈ **34
A100-h**; + 5 seeds on the final config ≈ 21 A100-h. **The entire ML program is under 100 A100-hours.**
GPU is not the bottleneck and should not drive any ranking decision. CPU: Phase 0 < 100 core-h; Phase 1
raycasting ~1,700 core-h.

## 9.5 One rule that prevents most of the statistical failures

> Write one `boot_by_group()` that bootstraps over `grnr`, and **forbid every other error bar in the
> codebase.** Group-level SEs: covering fraction ±0.02–0.05, inflow fraction ±0.04–0.12. Any covering
> fraction quoted to three decimals is wrong.

---

# PART X — PAPERS, RISKS, AND DECISIONS

## 10.1 Papers

**Paper I — "Velocity frames, ionization, and Galactic contamination of the high-velocity absorber
toward NGC 4303."** No simulation, no ML. The frames table settling 310/345/350; the O I re-measurement
(44 ± 12 mÅ, 3.7σ, centroid −281 km/s, log N < 14.3); the tied-multiplet demonstration and its
consequence for the mock covering-fraction systematic; the Galactic Lyα damping wing and geocoronal
Lyα rendering Si III 1206 unusable beyond +400 km/s; the S II 1259 blend at −215 km/s; the ≈17 km/s
wavelength systematic; the N V doublet ratio establishing N V as clean; the escape-velocity bound.
**Submit first — it de-risks everything, corrects the record, and establishes priority.**

**Paper II — "How much does one CGM absorption sightline say about where the gas is going?"** The
headline. Each claim with a measured number and a group-level error bar:
- LOS velocity sign does **not** diagnose radial flow direction: LOGO-17 AUC = **0.51 ± 0.02**.
- The full COS-G130M spectrum adds essentially nothing: AUC 0.46–0.60, below the majority-class rate.
- The ceiling is **physical, not methodological**: a photoionization oracle with exact T and Z reaches
  only 0.62–0.70, and **≥98% of that is metallicity alone (0.690)** — unrecoverable from G130M at
  z = 0.005 because Lyβ and the Lyman limit fall below the bandpass and 84% of HVCs are LLS-thick.
- The information content is a function of **inclination** — AUC(i) from the Phase-1 grid, with a
  threshold below which single-sightline flow direction is unrecoverable. **Directly useful for DIISC
  target selection.**
- **Random per-sightline splits inflate AUC by +0.10 to +0.19** on suites with 0.45 kpc spacing.
- Therefore the ~310 km/s absorber toward NGC 4303 **cannot be attributed to accretion from one
  sightline**; we replace the point claim with a bounded, calibrated null.

**Paper III — "An amortized, calibrated posterior for CGM absorber properties from HST/COS."** Methods.
The amortized-simulator design and the argument for why sequential SBI is structurally impossible here;
the information ledger vs Voigt summaries; the saturation-bias correction reframed as *saturation bias
at COS S/N* (not as a measurement of NGC 4303), with an independent damping-wing curve-of-growth on the
observed Lyα EW as a sanity check; an explicit statement that textbook SBC is not applicable (the
simulator is not an independently samplable prior) and what replaces it.

**Paper IV — conditional.** Either the matched detection/non-detection pair (NGC 4303 + NGC 3810) or
the cross-suite test.

**Deleted headlines:** *"P(inflow) = 0.52, a coin flip"* (manufactured by the now-invalid O I veto);
*"the real observation is a ≲1% tail event in TNG50"* (not defensible until blend-masked,
leave-one-mock-out calibrated, and tested against the environment hypothesis); *"settle the M61 HVC
origin statistically."*

## 10.2 Risk register

| # | risk | P × I | early warning | mitigation |
|---|---|---|---|---|
| R1 | Group-level error bars not used; every result 5–30× too precise | High × Fatal | any covering fraction below ±0.02 or inflow fraction below ±0.04 | one `boot_by_group()`; forbid all others; report grnr=4 separately (20% of the set) |
| R2 | Real spectrum OOD because 6 of 12 channels carry unmodelled blends | High × Severe | per-line z-score flags Si III / Si II 1193; leave-one-line-out Wasserstein exceeds its null | explicit masking + contaminant injection p = 0.3–0.5; truncate Si III at +400; §7.3 |
| R3 | "Unrecoverable" read as a failed project | Triggered × Moderate | — | frame as **the result** from day one; Paper II's title *is* the null; pair with the positive deliverables |
| R4 | Dominant ionization systematic unmitigable with cached τ | High × Severe | Si III/Si IV/N V posteriors shift > 0.2 dex under a ±0.3 dex nuisance | budget the scaled-UVB re-run (2d) or state it is unaddressed |
| R5 | Label leakage via v_ISM-thresholded targets | High × Severe | high accuracy on `cls`, `K_hvc`, `in_disk`, `v_ISM` | fixed in §5.1: all targets v_ISM-free; re-run every prior number |
| R6 | Model selection on the test fold | High × Moderate | best-of-8 reported on a 17-point estimate | 3 inner validation groups; outer group touched once |
| R7 | Fixed-geometry memorization mistaken for physics | High × Severe | geometry-only baseline matches the full model | both controls in Phase 0.3; the (ρ,i) grid is the definitive test |
| R8 | Gates that cannot fire (TARP A < 0.03; ≥6/9 PPC; ±5% coverage) | Certain × Moderate | — | replaced (§5.6) with bootstrap-calibrated nulls, C2ST, contraction |
| R9 | Slim-cutout radius bug truncates every ray | Moderate × Fatal to Phase 1 | new columns fall below existing at matched (ρ,i,α) | absolute kpc radius; 20-ray reproduction test first |
| R10 | Data deletion destroys inputs | Low × Fatal | — | **delete nothing**; 558 TB free |
| R11 | Scooped on "SBI for CGM absorbers" | Low-Mod × Moderate | arXiv | ship Paper I; the moat is the matched-geometry real anchor + the identifiability law |
| R12 | Proposal promises silently unmet | High × Reputational | — | descope explicitly in writing (§10.3) |

## 10.3 Decisions that need the PI

**D1 — Rewrite the Year-1 milestone?** The proposal says "settle the M61 HVC origin statistically."
Five independent measurements say it cannot be settled from one sightline. **Recommended:** rewrite to
*"bound the recoverability of HVC origin from COS-G130M and establish the calibrated posterior and its
information ceiling,"* communicated to the program officer proactively.

**D2 — Invest in determining M61's near side?** If the observer-side flag resolves f_inflow strongly
(Phase 0.2), knowing which face we see would nearly settle the origin. At i = 23° with b/a = 0.93,
near-side determination from arm winding + trailing-arm assumption + H I rotation sense is possible but
weakly constrained. **This is the single decision most likely to change what can be claimed about M61
specifically.** Commit only if the flip/noflip split in f_inflow exceeds ~0.3.

**D3 — Propose for HST/COS G160M?** C IV 1548/1550 → 1556.3/1558.9 Å, above the coadd's cutoff. Payoff:
R² for warm-hot temperature 0.42 → 0.84 with C IV + O VI. **O VI at 1037.3 Å is permanently
unobtainable for M61 with COS — say so plainly in any proposal.** Run Phase 0.6's C IV forecast first
to convert this from a column-space to a spectrum-space argument.

**D4 — Galaxy expansion route?** **Recommended: none in Year 1.** Phase 1's (ρ,i) grid addresses the
binding confound. If any expansion happens it must select to **span** log M200c and sSFR, not to match
M61 — which changes the project's identity.

**D5 — Match environment, or state it as a limitation?** Restricting to log M200c ∈ [12.55, 12.90] *and*
requiring the sightline to pass through its own H I disk at N_HI ≥ 3×10¹⁹ leaves **4 galaxies in 4
groups** — not a sample. **Recommended: keep all 17 groups, report the cuts as sensitivity tests, and
name the environment mismatch in the abstract.**

**D6 — Formally descope the K head and per-cloud posteriors?** Every element of Objective 2b's per-cloud
promise is ill-posed or unidentifiable (truth clouds 1.29/sightline vs pyGad median 20; K ≥ 3 has
n_eff < 10; Z unrecoverable; halo environment is 17 values plus a leak; origin class is the null).
**Recommended: descope explicitly, replacing it with the coarse velocity-band decomposition** — a
well-posed, resolution-matched reformulation of the same scientific content. **Needs PI sign-off
because it is a funded promise.**

**D7 — Second sightline (NGC 3810) vs deeper M61 analysis?** NGC 3810 turns n = 1 into a 2-point
covering-fraction constraint and directly tests the identifiability law. Cost ~1,000–3,000 core-h.
**Ranking: NGC 3810 > UVB re-run > tomography** — but the UVB re-run addresses the largest unmitigated
systematic. Genuinely a PI call about whether the program's weakness is *breadth* or *fidelity*.

**D8 — Refit mocks with tied multiplets, or refit the real data per transition?** The mock→real transfer
of every Voigt-derived quantity is currently unvalidated (§1.3.1). **Recommended: restrict Voigt
comparisons to single, untied transitions (Si III 1206, C II 1335, O I 1302) now; refit the real
spectrum per transition as a cross-check; refit all 14,400 mocks with tied multiplets (~4,600 core-h,
obsoleting the 310,217-row catalog) only if the information ledger becomes a headline result.**

---

# APPENDIX A — Key file paths

```
CODE
/home/tsingh65/m61-tng/scripts/disk_ism_velocity/     45 modules + 8 .md docs
    VOIGT_CATALOG.md      the Voigt product, definitions, all traps
    VISM_V3.md            the per-orientation v_ISM method
    VALIDATION.md         7-tier validation of the traditional analysis
    m61_voigt.py          Voigt catalog API + definitions
    m61_lines.py          gas-catalog loader (ION-level; detection NOT defined here)
    build_voigt_catalog.py
/home/tsingh65/m61-tng/data/M61_DIISC_Table1_Table2.csv   the real geometry table
/home/tsingh65/m61-tng/data/m61_closest_matches_3d.csv    the analogue selection (50 rows)
/home/tsingh65/m61-tng/scripts/orient_m61.py              orientation / alpha convention
/home/tsingh65/finesst-codes/code/figure3/panelC_spectra_m61.py   the ONLY existing real<->mock bridge
/home/tsingh65/finesst-codes/sections/stm.tex             the FINESST proposal (Objective 2b = SBI)

SIMULATION OUTPUTS
/scratch/tsingh65/m61-tng/outputs/sid<SID>/rays_and_spectra_sid<SID>_snap99_L2Rvir/
    combined/all_rays_L2Rvir.h5                           ~11 GB/galaxy: grid + spectra
    fitted_individual_line_spectra_parallel_snr10_bin3/per_spectrum_h5/<mode>/alpha<NNN>/
                                                          per-line Voigt fits + model + components
/scratch/tsingh65/m61-tng/outputs/disk_ism_velocity/
    absorber_catalog/absorbers_sid*.parquet               1.29M cells, 11 ions, v_r/Z/T
    voigt_catalog/voigt_components.parquet                310,217 components
    voigt_catalog/voigt_line_status.parquet               129,600 rows = the denominator
    vism_tables_v3/vism_master_v3.csv                     14,400 sightlines, v3a/v3b
    galaxy_properties/galaxy_properties_tng.csv           20 galaxies incl. grnr
    diagnostics_v3/disk_extent.csv
    paper_figures_v3a/, paper_figures_v3b/                27 figures each

REAL OBSERVATIONS
/scratch/tsingh65/FIRE_FINESST/J1221+0430/J1221+0430/
    J1221+0430_nbin3_coadd.fits
    113_270_z0.005/113_270_z0.005_lineresults.txt
    113_270_z0.005/fitting/*_slice.fits                   25 transitions, VEL/FNORM/ENORM/FITPARAMS
/scratch/tsingh65/FIRE_FINESST/J122154+042837-M61/        VLA HI cube + Barolo moment maps
/scratch/tsingh65/FIRE_FINESST/M61_DIISC.pptx             the founding slide

RELATED PROJECTS
/scratch/tsingh65/cgm_qso_488530/     672 CGM sightlines, OVI/Lyb/CIII, cached-tau + degrade.py
                                      (the amortized-simulator pattern to copy)
/scratch/tsingh65/OVI_CGM_compare/    matched OVI maps FIRE-2/TNG50/SIMBA (Phase 3 galaxy set)
```

# APPENDIX B — Verified numbers quick reference **[V]**

```
Real HVC centroid            -308 +/- 3 km/s (metals), -331 (N V), systemic frame
Real Si II HVC               logN 13.61, b = 10.9 +/- 2.7 km/s  -> T <= 2.0e5 K
O I at the HVC               W_r = 44.0 +/- 12.0 mA (3.7 sigma), F_min = 0.694 at -285 km/s
Si II at the HVC             F_min = 0.089 (black)
z_gal                        0.005224000196903944
DIISC fits                   TIED multiplets (SiII 1190/1193/1260/1304 bit-identical)
H I 1215                     FITNCOMPS = 0 -- no published H I Voigt fit
Sim sample                   20 subhalos, 17 FoF groups, 4 satellites share grnr=4
Sim geometry                 rho = 25.558333340264905 kpc, i = 23.0 deg, Rvir = 457 kpc (all)
Sim sightlines               14,400 = 20 x 720 (360 alpha x 2 modes)
Adjacent alpha separation    0.45 kpc; lag-1 ACF 0.23-0.63; VIF 2.4-6.6
Spectra                      9 lines, (30001,) each, 1150-1450 A, dLambda = 0.01 A
                             raw/ = PRE-LSF NOISELESS tau  <- the amortized simulator primitive
Voigt catalog                310,217 components / 299,065 detected / 129,600 line-fits
Si II covering fraction      0.382 / 0.344 / 0.313 (1260/1193/1190) -- protocol artefact, see 1.3.1
Si II 1260 zero-point bug    +23.76 km/s (pygad 1260.522 vs trident 1260.422); corrected in catalog
H I HVC covering fraction    0.930 raw -> 0.486 cleaned (damping-wing artefact)
H I logN                     at the fitter's 19.50 bound for 57.9% of clean components
Compact training encoding    < 1.5 GB for all 14,400 sightlines
Whole ML program             < 100 A100-hours
```
