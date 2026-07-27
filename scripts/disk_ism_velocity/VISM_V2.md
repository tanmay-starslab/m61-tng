# v_ISM v2 — supervisor's 3-tracer rotation-curve method (corrected)

Rebuild of the disk-ISM velocity following the supervisor's recipe exactly, with the one
requested change (cold gas T<10³ → T<10⁴ K). All outputs are in NEW directories; nothing in
the v1 pipeline was overwritten.

## The correctness fix (why this matters)
The v1 fiducial curve was the equal-weight average of {cold gas, SF gas, young stars}, but
cold gas was selected at **T<10³ K, which is empty in TNG** (the star-forming EoS floors gas
at ~10³·⁸ K). Verified: `n_cold_gas = 0` in all 60 radial bins, `contrib = "sf_gas;young_stars"`
everywhere. So the "3-tracer" average had silently been **SF gas + young stars only**. Using
**T<10⁴ K** restores cold gas — the dominant, most spatially extended ISM tracer (SF gas and
young stars thin out beyond ~20 kpc, exactly where the sightlines probe at ρ≈26 kpc).

## Method — full walkthrough (every component, this version)

v_ISM is the line-of-sight velocity the disk ISM is *expected* to produce in absorption for a
given sightline. It is built in two stages: a galaxy-level rotation curve (Stage A) and a
per-sightline projection (Stage B). One value results per (galaxy, orientation).

**Inputs (per galaxy = SID).** TNG50 cutout particles (gas + stars) within R<40 kpc of the
subhalo centre; the subhalo centre `c` (ckpc/h → physical kpc) and bulk velocity `SubhaloVel`
(km/s) from the TNG subhalo catalog; the orientation table giving, for each of the 720
(mode, α), a unit sightline `los` and the ray anchor.

**Component 1 — the disk frame (ê₁, ê₂, n̂).** The disk normal `n̂` is the axis of the
α-sightline cone, recovered as the smallest-variance SVD direction of the unit-`los` points
(one mode). This is used *instead of* the stored `normal_used_hat`, which is wrong by up to 49°
for several SIDs. ê₁, ê₂ span the disk plane; inclination = 23° for all SIDs.
→ `build_sid_rc.disk_normal_from_los`

**Component 2 — the three tracers ("cold gas + SF gas + young stars").** Selected from the
cutout: **cold gas: T < 10⁴ K** (the fix — T<10³ is empty in TNG); **star-forming gas: SFR>0**;
**young stars: age < 300 Myr**.  → `build_sid_rc_v2.tracer_arrays`

**Component 3 — velocity convention and per-tracer rotation curve v_φ(R).** Each particle
velocity is put in the galaxy rest frame: `v_rel = v_particle·√a − SubhaloVel` (a=1 at z=0;
positive = recession). It is decomposed into disk-frame cylindrical components (R, φ, z); `v_φ`
is the azimuthal (rotation) component. For each tracer, `v_φ(R)` = the **median v_φ in 0.5 kpc
radial bins** (bins with ≥20 particles).  → `dv_core.rotation_curve`

**Component 4 — the 3-tracer fiducial curve v_rot(R).** The **equal-weight, per-bin average**
of the three tracer medians (a tracer contributes to a bin only if it holds ≥5 particles):
`v_rot(R) = mean{ v_φ,cold(R), v_φ,SF(R), v_φ,young(R) }`. This is a property of the galaxy
(α-independent). Built at 0.5 / 1 / 2 kpc; the operative value is bin-insensitive (median
change 0 km/s).  → `build_sid_rc_v2.build_ism_average`

**Component 5 — the systemic velocity v_sys.** `v_sys = SubhaloVel · los` — the galaxy's bulk
motion projected onto that sightline (varies ~323 km/s peak-to-peak across α). It defines the
rest frame: it is already removed from `v_rot` (via `v_rel` above) and is removed from the
spectrum axis `v = −c(λ/λ₀−1) − v_sys`, so both live in the same galaxy-rest frame and are
directly comparable.  → `pm_general.get_geometry`

**Component 6 — sightline geometry: the anchor and R_anchor.** The sightline passes the galaxy
at impact parameter ρ ≈ 25.6 kpc. The **anchor** is the closest-approach point,
`anchor = c + ρ·ρ̂` with `ρ̂ ⟂ los`. Its disk-plane radius
`R_anchor = |projection of (anchor − c) onto the disk plane|` is the radius at which the
rotation curve is read.  → `pm_general.compute_endpoints`, `ray_ism_diagnostic.projection`

**Component 7 — the projection factor proj.** The rotation curve is a circular *speed*; only
its component along the sightline is observable. With `φ̂` the azimuthal (rotation-direction)
unit vector at the anchor: `proj = φ̂ · los`.  → `ray_ism_diagnostic.projection`

**Result — the per-sightline ISM velocity.**
> **v_ISM(sid, mode, α) = v_rot(R_anchor) × proj**  (galaxy rest frame; one value per orientation)

→ `stage_b_vism_v2.py` → `vism_tables_v2/vism_master_v2.csv`

**Downstream use.** For every absorbing gas cell, `Δv = v_los(rest frame) − v_ISM`; kinematic
class by |Δv|: ISM < 40, IVC 40–100, HVC > 100 km s⁻¹. This Δv drives all `paper_figures_v2`.

**Reliability.** v_ISM is trustworthy when the anchor lies inside a well-sampled disk (v_rot
well-defined at R_anchor). For compact galaxies whose disk ends before ρ≈26 kpc, or galaxies
with noisy outer rotation curves, `v_rot(R_anchor)` is an extrapolation and v_ISM is unreliable
(flagged beyond-disk). Overall it is unbiased vs the Si II dip (median −4 km/s) but ~46 km/s
scatter, and lands on a major ISM absorption component ~56% of the time (≈84% in-disk
well-sampled, ~30% beyond-disk).

## Answers to the review questions
- **Systemic velocity.** `v_sys = SubhaloVel·los(α)` — the galaxy's bulk peculiar velocity
  projected onto that orientation's sightline. It is strongly orientation-dependent: the
  per-galaxy peak-to-peak over α has median **≈323 km/s**. Subtracted identically from gas and
  spectrum (Tier-1 validation already confirmed the two agree to <1 km/s with this v_sys).
- **Binning.** Curves built at 0.5 / 1 / 2 kpc. The operative v_φ at the sampled radius is
  **bin-insensitive**: the induced change in v_ISM from 0.5→2 kpc bins has median **0.0 km/s**
  (see `diag_bin_sensitivity.png`; the three bin-width curves overlie in `rotation_curves/`).
- **Averaging.** Equal-weight mean of the three tracer medians per bin (now including cold gas).
  Cross-checked against the mass-weighted mean; the tracers agree within their 16–84 bands.

## How well does v_ISM land on the absorption? (the acid test)
Overlaying v_ISM on the Si II 1260 spectrum and testing whether it sits within 30 km/s of a
major absorption trough (`diag_spectra_overlay.py`, gallery + landing CSV per SID):

- **Well-sampled disks: it works.** SID 342448 → **84%** of sightlines land on a major
  component, median offset **14 km/s** (see `gallery_sid342448.png`; the model line sits on the
  deep Si II/C II trough in nearly every orientation).
- **Beyond-disk / noisy-outer-curve galaxies: it degrades.** Compact galaxies whose disk ends
  before ρ≈26 kpc (e.g. 432106, 46%) and galaxies with turbulent outer curves (360923, 30%)
  land far less often, because there is no ordered disk rotation to project at that radius.

**Full 20-SID aggregate (14,400 sightlines):** the model lands on a major Si II component
**56%** of the time (median offset to nearest trough 26 km/s), ranging **0.27→0.84** across
galaxies (best for well-sampled disks, worst for beyond-disk/noisy). Distance to the Si II dip:
**MODEL median 37 km/s** (53% within 40 km/s, 82% within 100) vs **DIRECT-v1 median 11 km/s**
(91% within 40, 98% within 100).

Overall the model is **unbiased** vs the Si II dip (median offset −4 km/s) but has **~46 km/s
scatter** (vs the v1 direct along-ray method's ~15 km/s), and is nearly **one value per galaxy**
(v_rot(≈26 kpc)×proj), clustering in two ±100 km/s bands — it captures the mean disk rotation
sense but not the per-sightline ISM structure the direct method resolves. So the supervisor
model marks a major ISM component **~56% of the time overall (≈80%+ for in-disk well-sampled
sightlines)**, while the direct method does so ~91% of the time.

## Takeaway
The supervisor's method is now implemented correctly (cold gas included) and is a clean,
unbiased estimate of the *mean* disk-ISM rotation velocity at the impact radius. It reliably
marks the major ISM absorption for **in-disk, well-sampled sightlines**, but should be flagged
as unreliable (or replaced by the direct method) where ρ is **beyond the disk edge** — which is
the same in_disk boundary the v1 work already identified.

## Files (all new)
- `build_sid_rc_v2.py` (+ `build_rc_v2_from_disk.py`) → `rotation_curves_v2/`
- `stage_b_vism_v2.py` → `vism_tables_v2/vism_master_v2.csv` (14,400 sightlines)
- `diag_rotation_curves.py`, `diag_vsys_bins_compare.py`, `diag_spectra_overlay.py` →
  `diagnostics_v2/`
- `make_paper_figures_v2.py` → `paper_figures_v2/` (9 HVC figures, dv from the model v_ISM)
